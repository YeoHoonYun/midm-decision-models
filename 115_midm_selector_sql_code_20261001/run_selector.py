#!/usr/bin/env python3
"""Score candidate pools with a MiDM checkpoint (best-of-N selector). Offline: reads data/<split>.jsonl built by
build_pools.py (stored candidates + stored correctness); never touches a database or runs code.

    python run_selector.py --split spider_dev --rag 0 --ckpt ../113_clm_reproduction_20260928/runs/E/q35_4b_e1_breadth_mc/final

Writes results/probs_<split>_<tag>_rag<k>.jsonl: {"id", "probs": [...], "correct": [...], "n_votes": [...]}.
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP113 = os.path.join(HERE, "..", "113_clm_reproduction_20260928")
sys.path.insert(0, EXP113)
import pointer_lora as P  # noqa: E402

INSTR = {"sql": "Several candidate SQL queries were written for this question. Which candidate returns the correct answer?",
         "code": "Several candidate programs were written for this problem. Which candidate is a correct solution?"}


def state(rec, kind, rag):
    parts = []
    if rag and rec.get("rag"):
        ex = "\n".join(f"Q: {r['question']}\nSQL: {r['sql']}" for r in rec["rag"][:rag])
        parts.append(f"Similar solved examples:\n{ex}")
    parts.append(rec["state_text"])
    parts.append(INSTR[kind])
    return "\n\n".join(parts)


def option_text(o, kind):
    body = o["sql_or_code"].strip()
    if kind == "sql" and o.get("result_preview"):
        body += f"\n  result: {o['result_preview']}"
    return body


def build(enc, stext, keys, cands, max_len, opt_cap):
    lines = [enc.ids(f"\n- {k}: {c}")[:opt_cap] for k, c in zip(keys, cands)]
    room = max_len - len(enc.opt_hdr) - sum(map(len, lines))
    s = enc.ids(stext)
    if room < 64:   # too many / too long options: shrink options evenly
        cap = max(48, (max_len - len(enc.opt_hdr) - 256) // max(1, len(lines)))
        lines = [ln[:cap] for ln in lines]
        room = max_len - len(enc.opt_hdr) - sum(map(len, lines))
    if len(s) > room:   # keep head (question) and tail (instruction)
        h = room // 2
        s = s[:h] + s[-(room - h):]
    ids, pos = s + enc.opt_hdr, []
    for ln in lines:
        ids = ids + ln
        pos.append(len(ids) - 1)
    return ids, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--tag", default="bx")
    ap.add_argument("--rag", type=int, default=0, help="number of RAG examples in the state (0 = off)")
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()
    kind = "code" if a.split.startswith("lcb") else "sql"
    opt_cap = 1200 if kind == "code" else 320
    recs = [json.loads(l) for l in open(os.path.join(HERE, "data", f"{a.split}.jsonl"), encoding="utf-8")]
    tok, model, head = P.load_model(a.base, ckpt=os.path.abspath(a.ckpt), train=False)
    model.eval()
    enc = P.Encoder(tok, a.max_len)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"probs_{a.split}_{a.tag}_rag{a.rag}.jsonl")
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f, torch.no_grad():
        for i, r in enumerate(recs):
            opts = r["options"]
            if len(opts) == 1:
                probs = [1.0]
            else:
                keys = [chr(65 + j) if j < 26 else f"C{j}" for j in range(len(opts))]
                ids, pos = build(enc, state(r, kind, a.rag), keys, [option_text(o, kind) for o in opts], a.max_len, opt_cap)
                probs = P.forward(model, head, [(ids, pos)])[0].float().softmax(-1).tolist()
            f.write(json.dumps({"id": r["id"], "probs": probs, "correct": [bool(o["correct"]) for o in opts],
                                "n_votes": [o.get("n_votes", 1) for o in opts], "db_id": r.get("db_id"),
                                "executable": [o.get("executable", True) is not False for o in opts],
                                "excluded": bool(r.get("label_excluded_nondeterministic"))}) + "\n")
            if i % 200 == 0:
                print(f"[sel] {a.split} {i}/{len(recs)} {time.time() - t0:.0f}s", flush=True)
    print(f"[sel] wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
