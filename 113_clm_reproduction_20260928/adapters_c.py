#!/usr/bin/env python3
"""Stage C: adapter composition for the QLoRA pointer decision model (plan: STAGE_C_PLAN.md).

Reuses pointer_lora.py (Encoder, forward, batches, evaluate, load_model, save) unchanged; the
stage-A recipe (LoRA r16 on all projections, lr 2e-4, head lr x10, cosine, 4096-token batches,
4-bit NF4 base, gradient checkpointing) is the default everywhere.

    dry-run       data/item/token counts for a domain spec (tokenizer only, no model, no GPU)
    train-domain  one LoRA adapter + pointer head on a subset of sources (a "domain")
    merge         PEFT add_weighted_adapter over domain adapters (cat/linear/svd/ties/dare...),
                  head = weighted average, or average + short head tuning, or a fresh head
    continue      adapter A frozen + new adapter B trained on top (stack: exact, 4-bit;
                  dequant: fp16 base with A merged; requant: PEFT 4-bit merge, lossy)
    eval-compose  evaluate a `continue` output (rebuilds base + A + B from compose.json)
    route-eval    per-domain adapters with routing (source tag / nearest-centroid on base hidden
                  states), plus single adapters (merged, stage-A baselines) on the same rows

Selection protocol (all train commands): select sets are exactly pointer_lora's --
td_holdout (10% td_train groups, seed-0 rule), kev_dev500 (seed-0 500 kev_dev rows without the
held-out sources) and src_holdout (kev_dev rows of --holdout-sources). Test/report suites are
only read by route-eval / eval-compose with --final.

Source names: kev_train `workflow` (agnews, yelp, compositional, dbpedia14, banking77, trec,
mnli, imdb, sst5, boolq, amazon, legacy_policy) plus the pseudo-source "typed-decisions" (td_train).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pointer_lora as PL  # noqa: E402  (also puts repo/src, repo/train on sys.path)
from pointer_lora import DTYPE, Encoder, batches, evaluate, forward, load_model  # noqa: E402
import decision_data  # noqa: E402

TD = "typed-decisions"
KEV_SOURCES = ["agnews", "yelp", "compositional", "dbpedia14", "banking77", "trec", "mnli", "imdb",
               "sst5", "boolq", "amazon", "legacy_policy"]
DOMAINS = {
    "classification": ["agnews", "yelp", "dbpedia14", "banking77", "trec", "imdb", "sst5", "amazon"],
    "reasoning_policy": ["mnli", "boolq", "compositional", "legacy_policy", TD],
    "all": KEV_SOURCES + [TD],
}
FINAL_SUITES = {"td_test", "kevT_dev", "kevT_test", "kevT9_dev", "jb_original", "jb_easy", "jb_hard"}
TIES_LIKE = {"ties", "ties_svd", "dare_ties", "dare_ties_svd"}   # PEFT disjoint_merge = mean of agreeing entries
DENSITY = TIES_LIKE | {"dare_linear", "dare_linear_svd", "magnitude_prune", "magnitude_prune_svd"}
METHODS = ["cat", "linear", "svd", "ties", "ties_svd", "dare_ties", "dare_linear", "dare_ties_svd",
           "dare_linear_svd", "magnitude_prune", "magnitude_prune_svd"]
META = "adapter_c.json"


# ----------------------------------------------------------------------------- data

def domain_spec(domain, sources):
    srcs = list(sources) if sources else list(DOMAINS[domain])
    bad = [s for s in srcs if s != TD and s not in KEV_SOURCES]
    if bad:
        raise SystemExit(f"unknown sources {bad}; valid: {KEV_SOURCES + [TD]}")
    return (domain or "custom"), srcs


def td_split():
    """-> (td_train rows used for training, td_holdout rows); same seed-0 group rule as pointer_lora."""
    ex = decision_data.load("td_train")
    groups = sorted({e.group for e in ex})
    random.Random(0).shuffle(groups)
    hold = set(groups[:len(groups) // 10])
    return [e for e in ex if e.group not in hold], [e for e in ex if e.group in hold]


def select_sets(holdout, td_hold):
    """Exactly pointer_lora.cmd_train's select sets (never a test suite)."""
    sel = {"td_holdout": td_hold}
    kd = decision_data.load("kev_dev")
    srcs = set(holdout or [])
    if srcs:
        sel["src_holdout"] = [e for e in kd if e.workflow in srcs]
        kd = [e for e in kd if e.workflow not in srcs]
    random.Random(0).shuffle(kd)
    sel["kev_dev500"] = kd[:500]
    return sel


def train_rows(srcs, holdout, td_train):
    """-> [(source, example)]; td first, then kev_train in file order (same order as pointer_lora)."""
    rows = [(TD, e) for e in td_train] if TD in srcs else []
    kev = set(srcs) - {TD} - set(holdout or [])
    if kev:
        rows += [(e.workflow, e) for e in decision_data.load("kev_train") if e.workflow in kev]
    return rows


def build_items(enc, rows, rng):
    items = []
    for _, e in rows:
        order = list(range(len(e.keys))); rng.shuffle(order)
        try:
            ids, pos = enc.build(e.state_text, e.keys, e.candidates, order)
        except ValueError:
            continue
        items.append((ids, pos, torch.tensor([e.target[j] for j in order])))
    return items


def source_of(suite, e):
    return TD if suite.startswith("td_") else e.workflow


def metrics(probs, ex):
    if not ex:
        return {"n": 0}
    acc = sum(int(p.argmax()) == e.label for p, e in zip(probs, ex)) / len(ex)
    brier = sum(float(((p - F.one_hot(torch.tensor(e.label), len(p)).float()) ** 2).sum())
                for p, e in zip(probs, ex)) / len(ex)
    return {"n": len(ex), "acc": acc, "brier": brier}


# ----------------------------------------------------------------------------- model helpers

def load_base(name):
    """4-bit NF4 base exactly as pointer_lora.load_model builds it, without PEFT / kbit preparation."""
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(name)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=DTYPE)
    base = AutoModel.from_pretrained(name, quantization_config=q, dtype=DTYPE, device_map={"": 0})
    if hasattr(base, "language_model") and hasattr(base, "visual"):
        del base.visual
        torch.cuda.empty_cache()
    return tok, base, getattr(base.config, "text_config", base.config).hidden_size


def lora_cfg(base, r, alpha, target_set="all"):
    """Same targets as pointer_lora.load_model (Qwen3.5 detection by language_model attr: visual is dropped)."""
    from peft import LoraConfig
    is35 = hasattr(base, "language_model")
    names = (["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"] if target_set == "attn"
             else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "in_proj_qkv", "in_proj_z", "out_proj"])
    if not is35:
        names = [n for n in names if not n.startswith(("in_proj", "out_proj"))]
    targets = rf".*language_model.*\.({'|'.join(names)})" if is35 else names
    return LoraConfig(r=r, lora_alpha=alpha or 2 * r, lora_dropout=0.05, bias="none", target_modules=targets)


def new_head(hidden):
    return torch.nn.Linear(hidden, 1).cuda().float()


def load_head(d, hidden):
    h = new_head(hidden)
    h.load_state_dict(torch.load(os.path.join(d, "pointer_head.pt")))
    return h


def read_meta(d):
    for p in (os.path.join(d, META), os.path.join(os.path.dirname(os.path.normpath(d)), META)):
        if os.path.exists(p):
            return json.load(open(p))
    return {}


def write_meta(d, meta):
    os.makedirs(d, exist_ok=True)
    json.dump(meta, open(os.path.join(d, META), "w"), indent=1)


def fp32_adapter(model, name):
    for n, p in model.named_parameters():
        if f".{name}." in n and ("lora_A" in n or "lora_B" in n) and p.dtype != torch.float32:
            p.data = p.data.float()


def freeze_dropout(model, name):
    """Frozen adapter in a stack: no dropout on its path (it is 'merged' in spirit)."""
    from peft.tuners.lora import LoraLayer
    for m in model.modules():
        if isinstance(m, LoraLayer) and name in m.lora_dropout:
            m.lora_dropout[name] = torch.nn.Identity()


@torch.no_grad()
def hidden_at(model, batch):
    """batch [(ids, pos)] -> list of [n_opt, hidden] float tensors (no head, no grad)."""
    L = max(len(i) for i, _ in batch)
    ids = torch.zeros((len(batch), L), dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for r, (i, _) in enumerate(batch):
        ids[r, :len(i)] = torch.tensor(i); att[r, :len(i)] = 1
    with torch.autocast("cuda", dtype=DTYPE):
        h = model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
    return [h[r, pos].float() for r, (_, pos) in enumerate(batch)]


@torch.no_grad()
def mean_embed(model, enc, ex, budget):
    """Mean-pooled, L2-normalised final hidden state of the canonical sequence, adapters disabled."""
    items = [enc.build(e.state_text, e.keys, e.candidates, list(range(len(e.keys)))) for e in ex]
    out = [None] * len(ex)
    model.eval()
    with model.disable_adapter():
        for b in batches(items, budget, False, None):
            L = max(len(items[k][0]) for k in b)
            ids = torch.zeros((len(b), L), dtype=torch.long)
            att = torch.zeros((len(b), L), dtype=torch.long)
            for r, k in enumerate(b):
                ids[r, :len(items[k][0])] = torch.tensor(items[k][0]); att[r, :len(items[k][0])] = 1
            with torch.autocast("cuda", dtype=DTYPE):
                h = model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
            m = att.cuda().unsqueeze(-1).to(h.dtype)
            v = F.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1).cpu()
            for r, k in enumerate(b):
                out[k] = v[r]
    torch.cuda.empty_cache()
    return torch.stack(out)


# ----------------------------------------------------------------------------- selection eval + training

def select_eval(model, head, enc, sel, budget, trained=None, declared=None):
    """Metrics on each select set, plus in-domain slices (no extra forward passes)."""
    res, seen, unseen = {}, [], []
    for k, ex in sel.items():
        if not ex:
            continue
        m, probs = evaluate(model, head, enc, ex, budget)
        res[k] = m
        suite = "td_holdout" if k == "td_holdout" else "kev_dev"
        for p, e in zip(probs, ex):
            s = source_of(suite, e)
            if trained is not None and s in trained:
                seen.append((p, e))
            elif declared is not None and s in declared:
                unseen.append((p, e))
    if trained is not None:
        res["in_domain"] = metrics(*zip(*seen)) if seen else {"n": 0}
        res["in_domain_unseen_src"] = metrics(*zip(*unseen)) if unseen else {"n": 0}
    return res


def flat(step, res):
    return {"step": step, **{k: round(v["acc"], 4) for k, v in res.items() if "acc" in v}}


def train_loop(a, model, head, enc, items, sel, save_fn, trained, declared, hist, rng):
    """pointer_lora.cmd_train's loop (same optimizer groups, schedule, scaler, clipping; rng continues from the
    item shuffles, so an 'all' spec reproduces a stage-A batch order)."""
    lora = [p for p in model.parameters() if p.requires_grad]
    params = lora + list(head.parameters())
    print(f"[c] trainable LoRA params {sum(p.numel() for p in lora) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": list(head.parameters()), "lr": a.lr * 10}],
                            weight_decay=0.0)
    steps_per_epoch = sum(1 for _ in batches(items, a.budget, True, random.Random(1)))
    total = a.max_steps or steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=DTYPE == torch.float16)
    step, t0, tok_seen = 0, time.time(), 0
    model.train()
    for _ in range(a.epochs):
        for b in batches(items, a.budget, True, rng):
            lgs = forward(model, head, [(items[k][0], items[k][1]) for k in b])
            loss = sum(-(items[k][2].cuda() * F.log_softmax(lg, -1)).sum() for k, lg in zip(b, lgs)) / len(b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step()
            step += 1; tok_seen += sum(len(items[k][0]) for k in b)
            if step % 10 == 0 or step == 1:
                el = time.time() - t0
                print(f"[c] step {step}/{total} loss {loss.item():.4f} tok/s {tok_seen / el:.0f} "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB eta {(total - step) * el / step / 60:.0f}m",
                      flush=True)
            if a.eval_every and step % a.eval_every == 0 and step < total:
                hist.append(flat(step, select_eval(model, head, enc, sel, a.budget, trained, declared)))
                print(f"[c] eval {hist[-1]}", flush=True)
                save_fn(f"step{step}")
            if step >= total:
                break
        if step >= total:
            break
    final = select_eval(model, head, enc, sel, a.budget, trained, declared)
    hist.append(flat(step, final))
    print(f"[c] final select: {hist[-1]}", flush=True)
    save_fn("final")
    return final, (time.time() - t0) / 60


def dump_summary(a, out, **kw):
    os.makedirs(out, exist_ok=True)
    json.dump({"args": {k: v for k, v in vars(a).items() if k != "fn"}, **kw,
               "max_mem_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None},
              open(os.path.join(out, "train_summary.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- train-domain

def cmd_dry_run(a):
    from transformers import AutoTokenizer
    name, srcs = domain_spec(a.domain, a.sources)
    td_tr, td_ho = td_split()
    rows = train_rows(srcs, a.holdout_sources, td_tr)
    enc = Encoder(AutoTokenizer.from_pretrained(a.model), a.max_len)
    items = build_items(enc, rows, random.Random(a.seed))
    sel = select_sets(a.holdout_sources, td_ho)
    per = {}
    for s, _ in rows:
        per[s] = per.get(s, 0) + 1
    ntok = sum(len(i[0]) for i in items)
    spe = sum(1 for _ in batches(items, a.budget, True, random.Random(1)))
    print(json.dumps({"domain": name, "sources": srcs, "holdout": a.holdout_sources, "rows": len(rows),
                      "items": len(items), "tokens_per_epoch": ntok, "steps_per_epoch@budget": spe,
                      "rows_per_source": per, "select_sizes": {k: len(v) for k, v in sel.items()}}, indent=1))


def cmd_train_domain(a):
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    name, srcs = domain_spec(a.domain, a.sources)
    hold = list(a.holdout_sources or [])
    tok, model, head = load_model(a.model, r=a.rank, adapter=a.adapter, target_set=a.targets, alpha=a.alpha)
    enc = Encoder(tok, a.max_len)
    td_tr, td_ho = td_split()
    rows = train_rows(srcs, hold, td_tr)
    sel = select_sets(hold, td_ho)
    items = build_items(enc, rows, rng)
    trained = sorted({s for s, _ in rows})
    print(f"[c] domain {name}: sources {trained} (declared {srcs}, held out {hold}) items {len(items)} "
          f"tokens {sum(len(i[0]) for i in items)}", flush=True)
    meta = {"kind": "domain", "domain": name, "domain_sources": srcs, "trained_sources": trained,
            "holdout_sources": hold, "model": a.model, "rank": a.rank, "adapter": a.adapter, "targets": a.targets}

    def save_fn(tag):
        PL.save(model, head, a.out, tag)
        write_meta(os.path.join(a.out, tag), meta)

    hist = []
    final, minutes = train_loop(a, model, head, enc, items, sel, save_fn, set(trained), set(srcs), hist, rng)
    dump_summary(a, a.out, meta=meta, history=hist, final=final, minutes=minutes)


# ----------------------------------------------------------------------------- merge

def train_head(model, head, enc, items, steps, lr, budget, seed):
    """Head only, on frozen (merged) adapter features: forward under no_grad, no backward through the LM."""
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    if not items:
        raise SystemExit("no training rows for head tuning")
    rng, step, model_was = random.Random(seed), 0, model.training
    model.eval()
    while step < steps:
        for b in batches(items, budget, True, rng):
            feats = hidden_at(model, [(items[k][0], items[k][1]) for k in b])
            loss = sum(-(items[k][2].cuda() * F.log_softmax(head(f).squeeze(-1), -1)).sum()
                       for k, f in zip(b, feats)) / len(b)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            if step % 50 == 0 or step == 1:
                print(f"[c] head step {step}/{steps} loss {loss.item():.4f}", flush=True)
            if step >= steps:
                break
    model.train(model_was)
    return head


def cmd_merge(a):
    from peft import PeftModel
    torch.manual_seed(a.seed)   # DARE random pruning
    if a.weights and len(a.weights) != len(a.adapters):
        raise SystemExit("--weights needs one value per adapter")
    metas = [read_meta(d) for d in a.adapters]
    hold = a.holdout_sources if a.holdout_sources is not None else (metas[0].get("holdout_sources") or [])
    if any(set(m.get("holdout_sources") or []) != set(hold) for m in metas if m):
        print("[c] WARNING: input adapters used different --holdout-sources; src_holdout is not a clean proxy", flush=True)
    leak = [(m.get("domain"), s) for m in metas for s in (m.get("trained_sources") or []) if s in hold]
    if leak:
        raise SystemExit(f"holdout sources were trained by an input adapter: {leak}")
    trained = sorted({s for m in metas for s in (m.get("trained_sources") or [])})
    if not all(metas):   # e.g. a stage-A pointer_lora adapter: assume all sources minus the holdout
        print(f"[c] WARNING: some inputs have no {META}; assuming they trained on all sources minus {hold}", flush=True)
        trained = sorted(set(trained) | (set(DOMAINS["all"]) - set(hold)))
    declared = sorted({s for m in metas for s in (m.get("domain_sources") or [])} | set(trained))
    tok, base, hidden = load_base(a.model)
    names = [f"a{i}" for i in range(len(a.adapters))]
    model = PeftModel.from_pretrained(base, a.adapters[0], adapter_name=names[0])
    for n, d in zip(names[1:], a.adapters[1:]):
        model.load_adapter(d, adapter_name=n)
    heads = [load_head(d, hidden) for d in a.adapters]
    enc = Encoder(tok, a.max_len)
    td_tr, td_ho = td_split()
    sel = select_sets(hold, td_ho)
    items = None
    if a.head in ("tune", "fresh"):
        items = build_items(enc, train_rows(trained, hold, td_tr), random.Random(a.seed))
    summary = {}
    for meth in a.methods:
        w = a.weights or [1.0 if meth in TIES_LIKE else 1.0 / len(names)] * len(names)
        kw = {}
        if meth in DENSITY:
            kw["density"] = a.density
        if meth.endswith("svd"):
            kw.update(svd_rank=a.svd_rank, svd_full_matrices=False)
        t0 = time.time()
        model.add_weighted_adapter(names, w, "default", combination_type=meth, **kw)
        model.set_adapter("default")
        sec = time.time() - t0
        tw = torch.tensor(w).abs() / torch.tensor(w).abs().sum()
        head = new_head(hidden)
        with torch.no_grad():
            head.weight.copy_(sum(t * h.weight for t, h in zip(tw, heads)))
            head.bias.copy_(sum(t * h.bias for t, h in zip(tw, heads)))
        res0 = select_eval(model, head, enc, sel, a.budget, set(trained), set(declared)) if a.head == "tune" else None
        if a.head == "fresh":
            head = new_head(hidden)
        if a.head in ("tune", "fresh"):
            head = train_head(model, head, enc, items, a.head_steps, a.head_lr, a.budget, a.seed)
        res = select_eval(model, head, enc, sel, a.budget, set(trained), set(declared))
        d = os.path.join(a.out, meth)
        model.save_pretrained(d, selected_adapters=["default"])
        torch.save(head.state_dict(), os.path.join(d, "pointer_head.pt"))
        meta = {"kind": "merge", "method": meth, "weights": w, "density": kw.get("density"),
                "svd_rank": kw.get("svd_rank"), "inputs": [os.path.abspath(x) for x in a.adapters],
                "head": a.head, "head_steps": a.head_steps if a.head != "avg" else 0, "domain": f"merge_{meth}",
                "domain_sources": declared, "trained_sources": trained, "holdout_sources": hold, "model": a.model}
        write_meta(d, meta)
        summary[meth] = {"merge_sec": round(sec, 1), "select": res, "select_avg_head": res0}
        print(f"[c] merge {meth} w={w}: {flat(0, res)}", flush=True)
        model.delete_adapter("default")
        model.set_adapter(names[0])
        torch.cuda.empty_cache()
    dump_summary(a, a.out, merges=summary)


# ----------------------------------------------------------------------------- continue (A frozen, B trained)

def delta_retention(model, adapter, n_layers):
    """Requant diagnostic: how much of dW survives Q(Q(W) + dW)? kept = <W1 - W0, dW> / |dW|^2."""
    import bitsandbytes.functional as bnbF
    from peft.tuners.lora import LoraLayer
    from peft.utils.integrations import dequantize_bnb_weight
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, LoraLayer) and adapter in m.lora_A]
    pick = layers[:: max(1, len(layers) // n_layers)][:n_layers]
    out = []
    for n, m in pick:
        try:
            w = m.get_base_layer().weight
            W0 = dequantize_bnb_weight(w, state=w.quant_state).float()
            dW = m.get_delta_weight(adapter).float()
            q, st = bnbF.quantize_4bit((W0 + dW).to(DTYPE).contiguous(), blocksize=getattr(w, "blocksize", 64),
                                       compress_statistics=getattr(w, "compress_statistics", True),
                                       quant_type=getattr(w, "quant_type", "nf4"))
            W1 = bnbF.dequantize_4bit(q, st).float()
            out.append({"layer": n, "kept": float(((W1 - W0) * dW).sum() / (dW * dW).sum()),
                        "rel_err": float((W1 - W0 - dW).norm() / dW.norm()),
                        "dW_over_W": float(dW.norm() / W0.norm())})
        except Exception as ex:   # diagnostic only
            out.append({"layer": n, "error": repr(ex)})
    return out


def build_continue(a):
    """-> tok, model (A frozen + B trainable, or B on a merged base), head, extra info."""
    from peft import PeftModel, get_peft_model
    info = {}
    if a.mode in ("stack", "requant"):
        tok, model, head = load_model(a.model, ckpt=a.base_adapter, train=True)   # A = "default", prepared for kbit
        inner = model.base_model.model
        if a.mode == "stack":
            model.add_adapter("B", lora_cfg(inner, a.rank, a.alpha, a.targets))
            fp32_adapter(model, "B")
            # LoraModel.set_adapter(list) activates both and makes both trainable; PeftModel.active_adapter stays
            # the string "default", so PeftModel.forward / disable_adapter keep working.
            model.base_model.set_adapter(["default", "B"])
            for n, p in model.named_parameters():
                if ".default." in n and "lora_" in n:
                    p.requires_grad = False
            freeze_dropout(model, "default")
        else:
            info["delta_retention"] = delta_retention(model, "default", a.retention_layers)
            kept = [r["kept"] for r in info["delta_retention"] if "kept" in r]
            print(f"[c] requant: mean fraction of dW kept {sum(kept) / max(1, len(kept)):.3f} "
                  f"({len(kept)} layers)", flush=True)
            inner = model.merge_and_unload()   # PEFT Linear4bit.merge: dequant -> + dW -> re-quantize NF4
            model = get_peft_model(inner, lora_cfg(inner, a.rank, a.alpha, a.targets), adapter_name="B")
    else:   # dequant: exact merge on an fp16/bf16 copy of Q(W); base stays unquantized while B trains
        tok, base, hidden = load_base(a.model)
        deq = base.dequantize(dtype=DTYPE)
        base = deq if deq is not None else base
        pm = PeftModel.from_pretrained(base, a.base_adapter)
        base = pm.merge_and_unload()
        for p in base.parameters():
            p.requires_grad = False
        base.config.use_cache = False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        model = get_peft_model(base, lora_cfg(base, a.rank, a.alpha, a.targets), adapter_name="B")
        head = load_head(a.base_adapter, hidden)
        info["weights_gib"] = torch.cuda.memory_allocated() / 2**30
        print(f"[c] dequant: base + merged A in {DTYPE} occupies {info['weights_gib']:.2f}GiB", flush=True)
    if a.head == "fresh":
        head = new_head(head.in_features)
    return tok, model, head, info


def cmd_continue(a):
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    name, srcs = domain_spec(a.domain, a.sources)
    hold = list(a.holdout_sources or [])
    ameta = read_meta(a.base_adapter)
    leak = [s for s in (ameta.get("trained_sources") or []) if s in hold]
    if leak and not a.allow_leak:
        raise SystemExit(f"adapter A was trained on held-out sources {leak}; src_holdout would not be a transfer proxy")
    if not ameta and hold:
        print("[c] WARNING: adapter A has no adapter_c.json; cannot verify it did not train on --holdout-sources", flush=True)
    tok, model, head, info = build_continue(a)
    enc = Encoder(tok, a.max_len)
    td_tr, td_ho = td_split()
    rows = train_rows(srcs, hold, td_tr)
    sel = select_sets(hold, td_ho)
    items = build_items(enc, rows, rng)
    trained = sorted({s for s, _ in rows} | set(ameta.get("trained_sources") or []))
    declared = sorted(set(srcs) | set(ameta.get("domain_sources") or []) | set(trained))
    meta = {"kind": "continue", "mode": a.mode, "domain": f"{ameta.get('domain', 'A')}+{name}",
            "domain_sources": declared, "trained_sources": trained, "b_sources": srcs, "holdout_sources": hold,
            "model": a.model, "a": os.path.abspath(a.base_adapter), "rank_b": a.rank}
    hist = []
    if not a.no_eval_start:   # step 0 = A on this base (stack: exactly A; requant: A after NF4 re-quantization)
        hist.append(flat(0, select_eval(model, head, enc, sel, a.budget, set(trained), set(declared))))
        print(f"[c] start (A only) select: {hist[-1]}", flush=True)

    def save_fn(tag):
        d = os.path.join(a.out, tag)
        model.save_pretrained(d, selected_adapters=["B"])   # -> d/B/
        torch.save(head.state_dict(), os.path.join(d, "pointer_head.pt"))
        write_meta(d, meta)
        json.dump({"mode": a.mode, "model": a.model, "a": os.path.abspath(a.base_adapter), "b": "B"},
                  open(os.path.join(d, "compose.json"), "w"), indent=1)

    final, minutes = train_loop(a, model, head, enc, items, sel, save_fn, set(trained), set(declared), hist, rng)
    if a.mode == "stack":   # exact single-adapter export: cat(A, B) with weights 1, 1 == A + B
        model.add_weighted_adapter(["default", "B"], [1.0, 1.0], "AB", combination_type="cat")
        d = os.path.join(a.out, "final")
        model.save_pretrained(d, selected_adapters=["AB"])   # -> final/AB/ (pointer_lora eval --ckpt final/AB)
        shutil.copy(os.path.join(d, "pointer_head.pt"), os.path.join(d, "AB", "pointer_head.pt"))
        write_meta(os.path.join(d, "AB"), {**meta, "kind": "continue_cat_export"})
    dump_summary(a, a.out, meta=meta, info=info, history=hist, final=final, minutes=minutes)


# ----------------------------------------------------------------------------- evaluation

def load_suites(names, hold, td_ho, final):
    out = {}
    for s in names:
        if s == "select":
            out.update({f"{k}": v for k, v in select_sets(hold, td_ho).items()})
            continue
        if s in FINAL_SUITES and not final:
            raise SystemExit(f"{s} is a report/test suite: pass --final (read once, after the arm is chosen)")
        if s in ("kev_dev500", "src_holdout"):
            ss = select_sets(hold, td_ho)
            if s not in ss:
                raise SystemExit("src_holdout needs --holdout-sources")
            out[s] = ss[s]
        else:
            out[s] = td_ho if s == "td_holdout" else decision_data.load(s)
    return out


suite_src = source_of   # td_* suites -> typed-decisions; kev/kevT/jb -> workflow / family


def eval_suites_json(a, model, head, enc, extra):
    td_tr, td_ho = td_split()
    suites = load_suites(a.suites, a.holdout_sources or [], td_ho, a.final)
    res, dump = {}, []
    for s, ex in suites.items():
        m, probs = evaluate(model, head, enc, ex, a.budget)
        res[s] = m
        print(f"[c-eval] {s}: {m}", flush=True)
        if a.save_probs:
            dump += [{"suite": s, "qid": e.qid, "group": e.group, "label": e.label, "source": suite_src(s, e),
                      "probs": p.tolist()} for e, p in zip(ex, probs)]
    if a.save_probs:
        with open(a.save_probs, "w", encoding="utf-8") as f:
            for r in dump:
                f.write(json.dumps(r) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"model": a.model, **extra, "eval": res}, open(a.out, "w"), indent=1)


def cmd_eval_compose(a):
    from peft import PeftModel
    comp = json.load(open(os.path.join(a.ckpt, "compose.json")))
    tok, model, head = load_model(a.model, ckpt=comp["a"], train=False)
    if comp["mode"] in ("stack", "dequant"):   # dequant: Q(W)+A in 4-bit compute == fp16(Q(W))+A up to rounding
        model.load_adapter(os.path.join(a.ckpt, comp["b"]), adapter_name="B")
        model.base_model.set_adapter(["default", "B"])
    else:
        model = PeftModel.from_pretrained(model.merge_and_unload(), os.path.join(a.ckpt, comp["b"]))
    head = load_head(a.ckpt, head.in_features)
    eval_suites_json(a, model, head, Encoder(tok, a.max_len), {"ckpt": a.ckpt, "compose": comp})


def parse_named(specs):
    out = []
    for s in specs or []:
        n, _, d = s.partition("=")
        if not d:
            d = n
            n = read_meta(d).get("domain") or os.path.basename(os.path.normpath(os.path.dirname(os.path.normpath(d))))
        out.append((n.replace(".", "_").replace("+", "_"), d))   # PEFT adapter names are ModuleDict keys
    return out


def cmd_route_eval(a):
    from peft import PeftModel
    doms = parse_named(a.adapters)
    singles = parse_named(a.single)
    names = [n for n, _ in doms + singles]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate adapter names {names}; use NAME=DIR")
    metas = {n: read_meta(d) for n, d in doms + singles}
    dsrc = {}
    for n, _ in doms:
        m = metas[n]
        dsrc[n] = set(m.get("domain_sources") or DOMAINS.get(n) or [])
        if not dsrc[n]:
            raise SystemExit(f"no domain sources for {n} (no {META}, not a preset name)")
    hold = a.holdout_sources if a.holdout_sources is not None else next(
        (metas[n].get("holdout_sources") for n, _ in doms if metas[n].get("holdout_sources") is not None), [])
    tok, base, hidden = load_base(a.model)
    (n0, d0), rest = (doms + singles)[0], (doms + singles)[1:]
    model = PeftModel.from_pretrained(base, d0, adapter_name=n0)
    for n, d in rest:
        model.load_adapter(d, adapter_name=n)
    heads = {n: load_head(d, hidden) for n, d in doms + singles}
    enc = Encoder(tok, a.max_len)
    td_tr, td_ho = td_split()
    suites = load_suites(a.suites, hold, td_ho, a.final)
    probs = {n: {} for n in names}
    for n in names:
        model.set_adapter(n)
        for s, ex in suites.items():
            probs[n][s] = evaluate(model, heads[n], enc, ex, a.budget)[1]
            print(f"[route] {n} on {s}: acc {metrics(probs[n][s], ex)['acc']:.4f}", flush=True)
    # routing tags: the domain adapter whose declared sources contain the row's source
    tags = {s: [next((n for n, _ in doms if suite_src(s, e) in dsrc[n]), None) for e in ex] for s, ex in suites.items()}
    need_centroid = a.router in ("centroid", "both") or any(t is None for v in tags.values() for t in v)
    cent, router_info = {s: [None] * len(ex) for s, ex in suites.items()}, {}
    if need_centroid:
        rng = random.Random(a.seed)
        src2dom, fit_ex, fit_src = {}, [], []
        for n, _ in doms:
            tr = set(metas[n].get("trained_sources") or dsrc[n]) - set(hold)
            for s in tr:
                src2dom.setdefault(s, n)
        pool = {s: [] for s in src2dom}
        for s, e in train_rows(sorted(src2dom), hold, td_tr):
            pool[s].append(e)
        for s, ex in pool.items():
            for e in rng.sample(ex, min(a.router_per_source, len(ex))):
                fit_ex.append(e); fit_src.append(s)
        srcs = sorted(pool)
        Z = mean_embed(model, enc, fit_ex, a.budget)
        C = F.normalize(torch.stack([Z[[i for i, s in enumerate(fit_src) if s == src]].mean(0) for src in srcs]), dim=-1)
        for s, ex in suites.items():
            nearest = (mean_embed(model, enc, ex, a.budget) @ C.T).argmax(-1).tolist()
            cent[s] = [src2dom[srcs[j]] for j in nearest]
        router_info = {"per_source": a.router_per_source, "sources": {s: src2dom[s] for s in srcs}}
        if a.router_save:
            torch.save({"sources": srcs, "domains": [src2dom[s] for s in srcs], "centroids": C}, a.router_save)
    res, dump = {}, []
    for s, ex in suites.items():
        r = {n: metrics(probs[n][s], ex) for n in names}
        route_t = [probs[t or c][s][i] if (t or c) else None for i, (t, c) in enumerate(zip(tags[s], cent[s]))]
        if all(p is not None for p in route_t):
            r["route_tag"] = {**metrics(route_t, ex), "n_centroid_fallback": sum(t is None for t in tags[s])}
        if need_centroid:
            r["route_centroid"] = metrics([probs[c][s][i] for i, c in enumerate(cent[s])], ex)
            known = [(t, c) for t, c in zip(tags[s], cent[s]) if t is not None]
            r["router_agree_with_tag"] = sum(t == c for t, c in known) / len(known) if known else None
        dn = [n for n, _ in doms]
        r["ensemble_mean"] = metrics([sum(probs[n][s][i] for n in dn) / len(dn) for i in range(len(ex))], ex)
        r["oracle_domain_adapter(diagnostic)"] = sum(any(int(probs[n][s][i].argmax()) == e.label for n in dn)
                                                     for i, e in enumerate(ex)) / len(ex)
        by = {}
        for i, (e, t) in enumerate(zip(ex, tags[s])):
            by.setdefault(t or "untagged", []).append(i)
        r["by_tag_domain"] = {t: {n: metrics([probs[n][s][i] for i in ix], [ex[i] for i in ix])["acc"] for n in names}
                              | {"n": len(ix)} for t, ix in by.items()}
        res[s] = r
        print(f"[route] {s}: " + json.dumps({k: round(v["acc"], 4) for k, v in r.items()
                                              if isinstance(v, dict) and "acc" in v}), flush=True)
        if a.save_probs:
            dump += [{"suite": s, "qid": e.qid, "group": e.group, "label": e.label, "source": suite_src(s, e),
                      "tag": tags[s][i], "centroid": cent[s][i],
                      "probs": {n: probs[n][s][i].tolist() for n in names}} for i, e in enumerate(ex)]
    if a.save_probs:
        with open(a.save_probs, "w", encoding="utf-8") as f:
            for row in dump:
                f.write(json.dumps(row) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"model": a.model, "adapters": dict(doms), "single": dict(singles), "domains": {n: sorted(v) for n, v in dsrc.items()},
               "holdout_sources": hold, "router": router_info, "eval": res}, open(a.out, "w"), indent=1)


# ----------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, train=True):
        p.add_argument("--model", default="Qwen/Qwen3-1.7B")
        p.add_argument("--budget", type=int, default=4096, help="padded tokens per batch (train and select eval)")
        p.add_argument("--max-len", type=int, default=1024)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--holdout-sources", nargs="*", default=None,
                       help="Kev sources never trained; their kev_dev rows form src_holdout (transfer proxy)")
        if train:
            p.add_argument("--rank", type=int, default=16)
            p.add_argument("--alpha", type=int, default=None)
            p.add_argument("--targets", choices=["all", "attn"], default="all")
            p.add_argument("--lr", type=float, default=2e-4)
            p.add_argument("--epochs", type=int, default=2)
            p.add_argument("--max-steps", type=int, default=None, help="overrides epochs (cosine over this many steps)")
            p.add_argument("--eval-every", type=int, default=400)

    def domain_args(p):
        p.add_argument("--domain", default=None, help=f"preset: {', '.join(DOMAINS)} (or a label for --sources)")
        p.add_argument("--sources", nargs="*", default=None, help=f"explicit sources; '{TD}' = td_train")

    p = sub.add_parser("dry-run", help="data/items/tokens for a domain spec; tokenizer only, no GPU")
    common(p, train=False); domain_args(p); p.set_defaults(fn=cmd_dry_run)

    p = sub.add_parser("train-domain")
    common(p); domain_args(p)
    p.add_argument("--adapter", choices=["lora", "dora", "rslora"], default="lora")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_train_domain)

    p = sub.add_parser("merge")
    common(p, train=False)
    p.add_argument("--adapters", nargs="+", required=True, help="adapter dirs (e.g. runs/C/C1_cls/final)")
    p.add_argument("--methods", nargs="+", default=["cat"], choices=METHODS,
                   help="one output dir per method; all work on the 4-bit base (LoRA-factor / B@A math only)")
    p.add_argument("--weights", type=float, nargs="*", default=None,
                   help="default: 1/n each (average of dW) for cat/linear/svd/dare_linear/magnitude_prune; "
                        "1 each for ties-family (PEFT ties already averages agreeing entries)")
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--svd-rank", type=int, default=None, help="default: max input rank")
    p.add_argument("--head", choices=["avg", "tune", "fresh"], default="tune",
                   help="avg: weighted head average; tune: avg then --head-steps of head-only training; fresh: new head")
    p.add_argument("--head-steps", type=int, default=300)
    p.add_argument("--head-lr", type=float, default=2e-3)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("continue", help="adapter A frozen (merged in spirit) + fresh adapter B trained on top")
    common(p); domain_args(p)
    p.add_argument("--base-adapter", required=True, help="adapter A dir (train-domain / merge / pointer_lora output)")
    p.add_argument("--mode", choices=["stack", "dequant", "requant"], default="stack",
                   help="stack: 4-bit Q(W) + A(frozen) + B, exact, no extra memory; dequant: fp16/bf16 base with A "
                        "merged (memory = full-precision weights); requant: PEFT 4-bit merge_and_unload (NF4 "
                        "re-quantization, loses most of dA; diagnostic printed)")
    p.add_argument("--head", choices=["from-a", "fresh"], default="from-a")
    p.add_argument("--retention-layers", type=int, default=12)
    p.add_argument("--no-eval-start", action="store_true")
    p.add_argument("--allow-leak", action="store_true", help="allow A trained on --holdout-sources (not a clean proxy)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_continue)

    def eval_args(p):
        p.add_argument("--suites", nargs="+", default=["select"],
                       help="'select' = td_holdout + kev_dev500 (+ src_holdout); decision_data suites need --final "
                            f"for {sorted(FINAL_SUITES)}")
        p.add_argument("--final", action="store_true", help="allow test/report suites (read once, arm already chosen)")
        p.add_argument("--out", required=True, help="results json")
        p.add_argument("--save-probs", default=None, help="jsonl of per-row probabilities")

    p = sub.add_parser("eval-compose", help="evaluate a `continue` output dir (contains compose.json)")
    common(p, train=False); eval_args(p)
    p.add_argument("--ckpt", required=True)
    p.set_defaults(fn=cmd_eval_compose)

    p = sub.add_parser("route-eval")
    common(p, train=False); eval_args(p)
    p.add_argument("--adapters", nargs="+", required=True, help="domain adapters, DIR or NAME=DIR")
    p.add_argument("--single", "--merged", dest="single", nargs="*", default=None,
                   help="adapters evaluated on every row without routing (merged, stage-A baselines), DIR or NAME=DIR")
    p.add_argument("--router", choices=["tag", "centroid", "both"], default="both",
                   help="tag: source tag (unknown sources fall back to centroid); centroid: nearest source centroid "
                        "of base (adapter-disabled) mean-pooled hidden states")
    p.add_argument("--router-per-source", type=int, default=200)
    p.add_argument("--router-save", default=None)
    p.set_defaults(fn=cmd_route_eval)

    a = ap.parse_args()
    if getattr(a, "domain", None) is None and getattr(a, "sources", None) is None and a.cmd in ("dry-run", "train-domain", "continue"):
        ap.error("give --domain PRESET or --sources ...")
    if getattr(a, "domain", None) and a.domain not in DOMAINS and not getattr(a, "sources", None):
        ap.error(f"--domain {a.domain} is not a preset; add --sources")
    a.fn(a)


if __name__ == "__main__":
    main()
