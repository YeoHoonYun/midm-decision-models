#!/usr/bin/env python3
"""Input length of JevBench public items under the MiDM input format (Qwen3.5 tokenizer) and how many exceed the
1,024-token training/eval limit. Reads only the public item files; no model is run.

    python jb_input_lengths.py      # -> results/diag/jb_input_lengths.{json,md}
"""
import json
import os
from collections import defaultdict

import decision_data
import pointer_lora as P

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, "ext", "jevbench", "datasets", "public")
LIMIT = 1024


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base")
    enc = P.Encoder(tok, 10 ** 6)   # no truncation: measure the full length
    fam = {}
    for l in open(os.path.join(PUB, "hard.jsonl"), encoding="utf-8"):
        r = json.loads(l); fam[r["id"]] = r.get("family", "")
    stats = defaultdict(list)
    for suite in ("jb_original", "jb_easy", "jb_hard"):
        for e in decision_data.load(suite):
            ids, _ = enc.build(e.state_text, e.keys, e.candidates, list(range(len(e.keys))))
            key = suite if suite != "jb_hard" else f"jb_hard:{fam.get(e.group, '?')}"
            stats[key].append(len(ids))
    out = {}
    for k, v in sorted(stats.items()):
        s = sorted(v)
        out[k] = {"n": len(v), "median_tokens": s[len(s) // 2], "max_tokens": s[-1], "over_1024": sum(x > LIMIT for x in v)}
    lp, mh = out.get("jb_hard:long_policy", {}), out.get("jb_hard:multi_hop", {})
    out["_summary"] = {"long_policy_plus_multi_hop_n": lp.get("n", 0) + mh.get("n", 0),
                       "long_policy_plus_multi_hop_over_1024": lp.get("over_1024", 0) + mh.get("over_1024", 0),
                       "definition": "full MiDM input (state + instructions + 'Options:' + all option lines), Qwen3.5-4B-Base tokenizer"}
    os.makedirs(os.path.join(HERE, "results", "diag"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "diag", "jb_input_lengths.json"), "w"), indent=1)
    md = ["# JevBench public: MiDM input length vs the 1,024-token limit", "", "| subset | n | median tokens | max tokens | over 1,024 |", "|---|---|---|---|---|"]
    md += [f"| {k} | {v['n']} | {v['median_tokens']} | {v['max_tokens']} | {v['over_1024']} |" for k, v in out.items() if not k.startswith("_")]
    s = out["_summary"]
    md += ["", f"long_policy + multi_hop: {s['long_policy_plus_multi_hop_over_1024']} of {s['long_policy_plus_multi_hop_n']} exceed 1,024 tokens ({s['definition']})."]
    open(os.path.join(HERE, "results", "diag", "jb_input_lengths.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
