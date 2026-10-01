"""Uniform decision examples across benchmarks, built with the repo's clm.schema.build_pairs
(the same state/candidate texts the CLM server embeds at inference).

Suites (name -> file, role):
  td_train / td_test         LocalLLaMA/typed-decisions 'all' (gold probabilities as soft targets)
  kev_train / kev_dev        jaredpalmer/kev-suites decision-v7 train / development (in-domain)
  kev_cal                    decision-v7 calibration
  kevT_dev / kevT_test       transfer-v4 development / test (never-trained sources; test read once)
  kevT9_dev                  transfer-v9 development
  jb_original/easy/hard      JevBench public items (one question per row)
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "repo", "src"), os.path.join(HERE, "repo", "train")]
import adapters  # noqa: E402
import finetune  # noqa: E402
from adapters import ChoiceExample, _key_of  # noqa: E402
from clm.schema import build_pairs  # noqa: E402

KEV = os.path.join(HERE, "data", "kev-suites")
JB = os.path.join(HERE, "ext", "jevbench", "datasets", "public")
FILES = {
    "kev_train": f"{KEV}/v7/decision-v7/train.jsonl", "kev_dev": f"{KEV}/v7/decision-v7/development.jsonl",
    "kev_cal": f"{KEV}/v7/decision-v7/calibration.jsonl",
    "kevT_dev": f"{KEV}/v4/transfer-v4/development.jsonl", "kevT_test": f"{KEV}/v4/transfer-v4/test.jsonl",
    "kevT9_dev": f"{KEV}/v9/transfer-v9/development.jsonl",
    "jb_original": f"{JB}/original.jsonl", "jb_easy": f"{JB}/easy.jsonl", "jb_hard": f"{JB}/hard.jsonl",
}
JB_YESNO = {"yes": "true", "no": "false"}


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _kev(rows, suite):
    for i, r in enumerate(rows):
        qs = {k: {kk: vv for kk, vv in q.items() if kk not in ("label", "src")} for k, q in r["questions"].items()}
        for qid, (stext, keys, cands) in build_pairs(r["state"], qs).items():
            lab = _key_of(r["questions"][qid]["label"])
            if lab not in keys:
                continue
            yield ChoiceExample(qid=f"{suite}:{qid}", state_text=stext, keys=keys, candidates=cands,
                                target=[float(k == lab) for k in keys], label=keys.index(lab),
                                group=(r.get("_meta") or {}).get("group_id", f"{suite}/{i}"),
                                workflow=(r.get("_meta") or {}).get("source"))


def _jb(rows, suite):
    for r in rows:
        q = r["question"]
        (stext, keys, cands), = build_pairs(r["state"], {"q": q}).values()
        lab = str(r["expected"])
        if q["type"] == "noul":
            lab = JB_YESNO.get(lab, lab)
        if lab not in keys:
            raise ValueError(f"{r['id']}: expected {lab!r} not in {keys}")
        yield ChoiceExample(qid=f"{suite}:{r['family']}", state_text=stext, keys=keys, candidates=cands,
                            target=[float(k == lab) for k in keys], label=keys.index(lab),
                            group=r["group"], workflow=r["family"])


PP6_NEW = ("arc", "openbookqa", "csqa")   # public-pool-v6 sources absent from decision-v7 and all eval suites


def load(suite):
    if suite == "breadth_v1":   # balanced breadth (sentence-pair / affect / toxicity / commonsense), data/breadth_v1
        return list(_kev(_rows(os.path.join(HERE, "data", "breadth_v1", "train.jsonl")), suite))
    if suite in ("long_v1", "long_v1_dev"):   # long-context rows (1k-3.8k tokens), data/long_v1
        fn = "train.jsonl" if suite == "long_v1" else "dev.jsonl"
        return list(_kev(_rows(os.path.join(HERE, "data", "long_v1", fn)), suite))
    if suite == "pp6_new":   # breadth experiment: extra training sources (never an eval suite)
        rows = [r for r in _rows(f"{KEV}/public-pool-v6/train.jsonl")
                if (r.get("_meta") or {}).get("source") in PP6_NEW]
        return list(_kev(rows, suite))
    if suite == "td_holdout":   # the 10% td_train row groups pointer_lora.py holds out for selection
        import random
        ex = load("td_train")
        groups = sorted({e.group for e in ex})
        random.Random(0).shuffle(groups)
        hold = set(groups[:len(groups) // 10])
        return [e for e in ex if e.group in hold]
    if suite in ("td_train", "td_test"):
        rows = finetune.load_typed_rows(os.path.join(HERE, "data", "typed-decisions"), suite[3:], "all", None)
        return list(adapters.typed_decision_examples(rows))
    rows = _rows(FILES[suite])
    return list(_jb(rows, suite) if suite.startswith("jb_") else _kev(rows, suite))


ALL = ["td_train", "td_test", "kev_train", "kev_dev", "kev_cal", "kevT_dev", "kevT_test", "kevT9_dev",
       "jb_original", "jb_easy", "jb_hard"]

if __name__ == "__main__":
    from embed_utils import Recipe
    rec = Recipe("Qwen/Qwen3-4B", 2048)
    seen, tot = set(), 0
    for s in ALL:
        ex = load(s)
        texts = [t for e in ex for t in (e.state_text, *e.candidates)]
        new = [t for t in dict.fromkeys(texts) if t not in seen]
        toks = sum(len(rec.text_ids(t, keep="tail")) for t in new)
        seen.update(new); tot += toks
        print(f"{s:12s} examples {len(ex):6d} new unique texts {len(new):6d} tokens {toks:9d}")
    print("total unique texts", len(seen), "tokens", tot)
