#!/usr/bin/env python3
"""QLoRA pointer decision model (Kev-style cross-encoder) on a Qwen3 base, one GPU.

One sequence per question: <state + instructions>\n\nOptions:\n- key: text\n- key: text ...
The final-layer hidden state at the last token of each option line goes through a linear
pointer head; softmax over options; soft-target cross-entropy. Options are shuffled in
training. Base weights 4-bit NF4 (bitsandbytes), fp16 compute (Turing), LoRA on all
attention/MLP projections, gradient checkpointing, token-budget batches.

    python pointer_lora.py train --model Qwen/Qwen3-4B --out runs/ptr4b --max-steps 10   # pilot
    python pointer_lora.py eval  --model Qwen/Qwen3-4B --ckpt runs/ptr4b --suites kev_dev td_test ...
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "repo", "src"), os.path.join(HERE, "repo", "train")]
import decision_data  # noqa: E402

# compute dtype: fp16 by default (Turing 2080 Ti); PTR_BF16=1 for bf16 on Ampere (3090), no loss scaling needed
DTYPE = torch.bfloat16 if os.environ.get("PTR_BF16") == "1" else torch.float16


class Encoder:
    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, max_len
        self.opt_hdr = tok("\n\nOptions:", add_special_tokens=False)["input_ids"]

    def ids(self, t):
        return self.tok(t, add_special_tokens=False)["input_ids"]

    def build(self, state_text, keys, cands, order):
        """-> (input ids, pointer positions in the given option order)."""
        lines = [self.ids(f"\n- {keys[j]}: {cands[j]}")[:192] for j in order]
        room = self.max_len - len(self.opt_hdr) - sum(map(len, lines))
        s = self.ids(state_text)
        if room < 32:
            raise ValueError("options too long")
        if len(s) > room:   # keep the head and the tail (instructions sit at the end)
            h = room // 3
            s = s[:h] + s[-(room - h):]
        ids = s + self.opt_hdr
        pos = []
        for ln in lines:
            ids = ids + ln
            pos.append(len(ids) - 1)
        return ids, pos


def load_model(name, ckpt=None, r=16, train=True, adapter="lora", target_set="all", alpha=None):
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    tok = AutoTokenizer.from_pretrained(name)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=DTYPE)
    base = AutoModel.from_pretrained(name, quantization_config=q, dtype=DTYPE, device_map={"": 0})
    is35 = hasattr(base, "language_model") and hasattr(base, "visual")
    if is35:   # Qwen3.5: text-only use; drop the vision tower, LoRA only on the language model
        del base.visual
        torch.cuda.empty_cache()
    hidden = getattr(base.config, "text_config", base.config).hidden_size
    names = (["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"] if target_set == "attn"
             else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "in_proj_qkv", "in_proj_z", "out_proj"])
    if not is35:
        names = [n for n in names if not n.startswith(("in_proj", "out_proj"))]
    targets = rf".*language_model.*\.({'|'.join(names)})" if is35 else names
    head_only = adapter == "none" or (ckpt and not os.path.exists(os.path.join(ckpt, "adapter_config.json")))
    if head_only:   # ablation: frozen base, only the pointer head is trained (no LoRA)
        base.requires_grad_(False)
        base.config.use_cache = False
        base._midm_frozen = True
        model = base
    elif train:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        base.config.use_cache = False
    if head_only:
        pass
    elif ckpt:
        model = PeftModel.from_pretrained(base, ckpt, is_trainable=train)
    else:
        model = get_peft_model(base, LoraConfig(r=r, lora_alpha=alpha or 2 * r, lora_dropout=0.05, bias="none",
                                                target_modules=targets, use_dora=adapter == "dora",
                                                use_rslora=adapter == "rslora"))
    head = torch.nn.Linear(hidden, 1).cuda().float()
    if ckpt and os.path.exists(os.path.join(ckpt, "pointer_head.pt")):
        head.load_state_dict(torch.load(os.path.join(ckpt, "pointer_head.pt")))
    return tok, model, head


def forward(model, head, batch):
    """batch: list of (ids, pos) -> list of logits tensors (one per item)."""
    L = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), L), 0, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for r, (i, _) in enumerate(batch):
        ids[r, :len(i)] = torch.tensor(i); att[r, :len(i)] = 1
    frozen = torch.no_grad() if getattr(model, "_midm_frozen", False) else contextlib.nullcontext()
    with frozen, torch.autocast("cuda", dtype=DTYPE):
        h = model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
    out = []
    for r, (_, pos) in enumerate(batch):
        out.append(head(h[r, pos].float()).squeeze(-1))
    return out


def batches(items, budget, shuffle, rng):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
        # sort within chunks of 64 by length to limit padding
        idx = [j for c in range(0, len(idx), 64) for j in sorted(idx[c:c + 64], key=lambda k: len(items[k][0]))]
    cur, mx = [], 0
    for j in idx:
        L = len(items[j][0])
        if cur and max(mx, L) * (len(cur) + 1) > budget:
            yield cur; cur, mx = [], 0
        cur.append(j); mx = max(mx, L)
    if cur:
        yield cur


@torch.no_grad()
def evaluate(model, head, enc, ex, budget, tta=1):
    """tta = number of option orders averaged (1 = the given order only). Order 0 is always the given
    order; the others are fixed random permutations (seed 0). A causal model lets later options see
    earlier ones but not the reverse, so averaging over orders removes that position asymmetry."""
    model.eval()
    rng = random.Random(0)
    orders = []
    for e in ex:
        k = len(e.keys)
        os_ = [list(range(k))]
        for _ in range(tta - 1):
            o = list(range(k)); rng.shuffle(o); os_.append(o)
        orders.append(os_)
    probs = [torch.zeros(len(e.keys)) for e in ex]
    t0 = time.time()
    for t in range(tta):
        items = [enc.build(e.state_text, e.keys, e.candidates, orders[i][t]) for i, e in enumerate(ex)]
        for b in batches(items, budget, False, None):
            for j, lg in zip(b, forward(model, head, [items[k] for k in b])):
                p = lg.softmax(-1).cpu()
                back = torch.zeros_like(p)
                back[torch.tensor(orders[j][t])] = p   # position i held option orders[j][t][i]
                probs[j] += back / tta
    model.train()
    torch.cuda.empty_cache()
    acc = sum(int(p.argmax()) == e.label for p, e in zip(probs, ex)) / len(ex)
    brier = sum(float(((p - F.one_hot(torch.tensor(e.label), len(p)).float()) ** 2).sum()) for p, e in zip(probs, ex)) / len(ex)
    return {"n": len(ex), "acc": acc, "brier": brier, "sec": round(time.time() - t0, 1)}, probs


def cmd_train(a):
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    tok, model, head = load_model(a.model, ckpt=a.resume, r=a.rank, adapter=a.adapter, target_set=a.targets,
                                  alpha=a.alpha)
    enc = Encoder(tok, a.max_len)
    ex = []
    for s in a.train:
        ex += [(s, e) for e in decision_data.load(s)]
    groups = sorted({e.group for s, e in ex if s == "td_train"})
    random.Random(0).shuffle(groups)
    hold = set(groups[:len(groups) // 10])
    sel = {"td_holdout": [e for s, e in ex if e.group in hold]}
    srcs = set(a.holdout_sources or [])
    train = [e for s, e in ex if e.group not in hold and not (s == "kev_train" and e.workflow in srcs)]
    kd = decision_data.load("kev_dev")
    if srcs:   # proxy transfer: kev_dev rows of sources removed from training
        sel["src_holdout"] = [e for e in kd if e.workflow in srcs]
        kd = [e for e in kd if e.workflow not in srcs]
    random.Random(0).shuffle(kd)
    sel["kev_dev500"] = kd[:500]
    items = []
    for e in train:
        order = list(range(len(e.keys))); rng.shuffle(order)
        try:
            ids, pos = enc.build(e.state_text, e.keys, e.candidates, order)
        except ValueError:
            continue
        items.append((ids, pos, torch.tensor([e.target[j] for j in order])))
    print(f"[ptr] train items {len(items)}, tokens {sum(len(i[0]) for i in items)}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    lora = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(([{"params": lora, "lr": a.lr}] if lora else [])
                            + [{"params": head.parameters(), "lr": a.lr * 10}], weight_decay=0.0)
    steps_per_epoch = sum(1 for _ in batches(items, a.budget, True, random.Random(1)))
    total = a.max_steps or steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=DTYPE == torch.float16)
    step, t0, tok_seen, hist = 0, time.time(), 0, []
    for _ in range(a.skip_steps):   # resume: same seed -> same batch order; optimizer state restarts
        sched.step()
    model.train()
    for ep in range(a.epochs):
        for b in batches(items, a.budget, True, rng):
            if step < a.skip_steps:
                step += 1
                continue
            lgs = forward(model, head, [(items[k][0], items[k][1]) for k in b])
            loss = sum(-(items[k][2].cuda() * F.log_softmax(lg, -1)).sum() for k, lg in zip(b, lgs)) / len(b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step()
            step += 1; tok_seen += sum(len(items[k][0]) for k in b)
            done = step - a.skip_steps
            if step % 10 == 0 or done == 1:
                el = time.time() - t0
                print(f"[ptr] step {step}/{total} loss {loss.item():.4f} tok/s {tok_seen / el:.0f} "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB eta {(total - step) * el / done / 60:.0f}m",
                      flush=True)
            if step % 200 == 0:   # long runs on the 11 GB 2080 Ti drift into shared-memory spill otherwise
                torch.cuda.empty_cache()
            if a.save_every and step % a.save_every == 0 and not (a.eval_every and step % a.eval_every == 0):
                save(model, head, a.out, f"step{step}")
            if a.eval_every and step % a.eval_every == 0:
                m = {k: evaluate(model, head, enc, v, a.budget)[0] for k, v in sel.items()}
                hist.append({"step": step, **{k: v["acc"] for k, v in m.items()}})
                print(f"[ptr] eval step {step}: {hist[-1]}", flush=True)
                save(model, head, a.out, f"step{step}")
            if step >= total:
                break
        if step >= total:
            break
    m = {k: evaluate(model, head, enc, v, a.budget)[0] for k, v in sel.items()}
    hist.append({"step": step, **{k: v["acc"] for k, v in m.items()}})
    print(f"[ptr] final select: {hist[-1]}", flush=True)
    save(model, head, a.out, "final")
    json.dump({"args": {k: v for k, v in vars(a).items() if k != "fn"}, "history": hist, "minutes": (time.time() - t0) / 60,
               "max_mem_gib": torch.cuda.max_memory_allocated() / 2**30},
              open(os.path.join(a.out, "train_summary.json"), "w"), indent=1)


def save(model, head, out, tag):
    d = os.path.join(out, tag)
    os.makedirs(d, exist_ok=True)
    if getattr(model, "_midm_frozen", False):   # head-only ablation: the base is unchanged, save the head only
        open(os.path.join(d, "HEAD_ONLY"), "w").write("frozen base; pointer head only\n")
    else:
        model.save_pretrained(d)
    torch.save(head.state_dict(), os.path.join(d, "pointer_head.pt"))


def cmd_eval(a):
    tok, model, head = load_model(a.model, ckpt=a.ckpt, train=False)
    enc = Encoder(tok, a.max_len)
    res, dump = {}, []
    for s in a.suites:
        ex = decision_data.load(s)
        m, probs = evaluate(model, head, enc, ex, a.budget, tta=a.tta)
        res[s] = m
        if a.save_probs:
            dump += [{"suite": s, "qid": e.qid, "group": e.group, "label": e.label, "probs": p.tolist()}
                     for e, p in zip(ex, probs)]
        print(f"[ptr-eval] {s}: {m}", flush=True)
    if a.save_probs:
        with open(a.save_probs, "w", encoding="utf-8") as f:
            for r in dump:
                f.write(json.dumps(r) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"ckpt": a.ckpt, "model": a.model, "tta": a.tta, "eval": res}, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", default="Qwen/Qwen3-4B")
    t.add_argument("--train", nargs="+", default=["td_train", "kev_train"])
    t.add_argument("--out", required=True)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--alpha", type=int, default=None, help="LoRA alpha (default 2 x rank)")
    t.add_argument("--adapter", choices=["lora", "dora", "rslora", "none"], default="lora",
                   help="none = frozen base, pointer head only (ablation)")
    t.add_argument("--targets", choices=["all", "attn"], default="all")
    t.add_argument("--holdout-sources", nargs="*", default=None,
                   help="Kev train sources left out entirely; their kev_dev rows become the src_holdout select set")
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--budget", type=int, default=2048, help="padded tokens per batch")
    t.add_argument("--max-len", type=int, default=1024)
    t.add_argument("--eval-every", type=int, default=0)
    t.add_argument("--save-every", type=int, default=400, help="checkpoint without eval every N steps (0 = off)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", default=None, help="adapter dir (e.g. runs/x/step400) to continue from")
    t.add_argument("--skip-steps", type=int, default=0, help="batches already trained (with --resume)")
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("eval")
    e.add_argument("--model", default="Qwen/Qwen3-4B")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--suites", nargs="+", required=True)
    e.add_argument("--budget", type=int, default=4096)
    e.add_argument("--max-len", type=int, default=1024)
    e.add_argument("--out", required=True)
    e.add_argument("--save-probs", default=None, help="jsonl of per-example probabilities (for router calibration)")
    e.add_argument("--tta", type=int, default=1, help="option-order permutations averaged at inference (1 = off)")
    e.set_defaults(fn=cmd_eval)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
