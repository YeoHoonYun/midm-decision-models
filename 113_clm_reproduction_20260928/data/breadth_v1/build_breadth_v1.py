"""Build data/breadth_v1: ~1,000 typed decision questions per new public source, balanced across the
task types diluted in Breadth D1 (sentence-pair, affect, toxicity) plus one commonsense reasoning set.

Run from the experiment dir with PYTHONPATH='' and HF_HOME=<exp>/hf_cache (CPU only).
Rows use the Kev System One format; every row is checked against all eval suites and dropped on overlap.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import re
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

EXP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(EXP, "data", "breadth_v1")
sys.path.insert(0, EXP)
os.chdir(EXP)
import decision_data as dd  # noqa: E402
import finetune  # noqa: E402  (path set up by decision_data)
from clm.schema import build_pairs, to_text  # noqa: E402

SEED = 20261001
N_PER_SOURCE = 1000
EVAL_SUITES = ["kevT_dev", "kevT_test", "kevT9_dev", "jb_original", "jb_easy", "jb_hard", "td_test", "kev_dev"]
SHINGLE_SUITES = ["kevT_dev", "kevT_test", "kevT9_dev", "jb_original", "jb_easy", "jb_hard"]
FORBIDDEN_SOURCES = {"mmlu", "mmlu_pro", "sciq", "emotion", "tweet_offensive", "qnli", "paws"}
FORBIDDEN_REPOS = {"cais/mmlu", "TIGER-Lab/MMLU-Pro", "allenai/sciq", "dair-ai/emotion", "cardiffnlp/tweet_eval",
                   "google-research-datasets/paws"}
FORBIDDEN_CONFIGS = {("nyu-mll/glue", "qnli")}
K = 13  # shingle size (words)

# (repo, revision on main, parquet file) -- the train split only
SPECS = {
    "mrpc": ("nyu-mll/glue", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "mrpc/train-00000-of-00001.parquet"),
    "qqp": ("nyu-mll/glue", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "qqp/train-00000-of-00001.parquet"),
    "rte": ("nyu-mll/glue", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "rte/train-00000-of-00001.parquet"),
    "go_emotions": ("google-research-datasets/go_emotions", "add492243ff905527e67aeb8b80c082af02207c3",
                    "simplified/train-00000-of-00001.parquet"),
    "civil_comments": ("google/civil_comments", "f2970eb3a55777454c94069077cc8d9b5866312d",
                       "data/train-00000-of-00002.parquet"),
    "winogrande": ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
                   "winogrande_xl/train-00000-of-00001.parquet"),
}
TASK_TYPE = {"mrpc": "sentence_pair/paraphrase", "qqp": "sentence_pair/paraphrase", "rte": "sentence_pair/entailment",
             "go_emotions": "affect", "civil_comments": "toxicity", "winogrande": "reasoning/coreference"}
QTYPE = {"mrpc": "noul", "qqp": "choice", "rte": "noul", "go_emotions": "choice", "civil_comments": "noul",
         "winogrande": "choice"}

GOEMO = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
         "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
         "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness",
         "surprise", "neutral"]


def table(name):
    repo, rev, fn = SPECS[name]
    assert repo not in FORBIDDEN_REPOS and (repo, fn.split("/")[0]) not in FORBIDDEN_CONFIGS
    return pq.read_table(hf_hub_download(repo, fn, repo_type="dataset", revision=rev))


def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


# --------------------------------------------------------------------------- row builders
# Each yields (row_index, label_for_balance, state, {qid: question}).

def gen_mrpc(t):
    for s1, s2, lab, idx in zip(*(t.column(c).to_pylist() for c in ("sentence1", "sentence2", "label", "idx"))):
        yield idx, lab, {"sentence_a": clean(s1), "sentence_b": clean(s2)}, {"equivalent": {
            "type": "noul", "instructions": "Do sentence A and sentence B report the same thing?",
            "criteria": {"true": "Yes, they state the same facts in different words",
                         "false": "No, one adds, drops or changes information"},
            "label": bool(lab == 1), "src": "mrpc"}}


def gen_qqp(t):
    for q1, q2, lab, idx in zip(*(t.column(c).to_pylist() for c in ("question1", "question2", "label", "idx"))):
        yield idx, lab, {"question_1": clean(q1), "question_2": clean(q2)}, {"duplicate": {
            "type": "choice", "instructions": "Would one answer satisfy both of these questions?",
            "criteria": {"duplicate": "Same question, just phrased differently",
                         "distinct": "Different questions that need different answers"},
            "label": "duplicate" if lab == 1 else "distinct", "src": "qqp"}}


def gen_rte(t):
    for s1, s2, lab, idx in zip(*(t.column(c).to_pylist() for c in ("sentence1", "sentence2", "label", "idx"))):
        yield idx, lab, {"passage": clean(s1)}, {"follows": {
            "type": "noul", "instructions": f"Claim: \"{clean(s2)}\" Does the passage support this claim?",
            "criteria": {"true": "The claim follows from the passage",
                         "false": "The passage does not establish the claim"},
            "label": bool(lab == 0), "src": "rte"}}   # glue/rte: 0 = entailment


def gen_go_emotions(t):
    names = GOEMO
    for i, (text, labs) in enumerate(zip(t.column("text").to_pylist(), t.column("labels").to_pylist())):
        if len(labs) != 1:          # single-label rows only
            continue
        lab = names[labs[0]]
        yield i, lab, {"comment": clean(text)}, {"feeling": {
            "type": "choice", "instructions": "Which feeling does this comment mainly express?",
            "criteria": {n: None for n in names}, "label": lab, "src": "go_emotions"}}


def gen_civil_comments(t, rng):
    tox = t.column("toxicity").to_pylist()
    txt = t.column("text").to_pylist()
    pos = [i for i, v in enumerate(tox) if v >= 0.5]
    neg = [i for i, v in enumerate(tox) if v == 0.0]     # unambiguous: no rater marked it toxic
    rng.shuffle(pos), rng.shuffle(neg)
    pick = sorted(pos[:8000] + neg[:8000])
    for i in pick:
        s = clean(txt[i])
        if not 30 <= len(s) <= 700:
            continue
        lab = tox[i] >= 0.5
        yield i, lab, {"comment": s}, {"toxic": {
            "type": "noul", "instructions": "Would most readers find this comment rude, disrespectful or hateful?",
            "criteria": {"true": "Toxic: insulting, demeaning, threatening or hateful",
                         "false": "Civil, even if it disagrees or criticizes"},
            "label": bool(lab), "src": "civil_comments"}}


def gen_winogrande(t, rng):
    cols = [t.column(c).to_pylist() for c in ("sentence", "option1", "option2", "answer")]
    idx = list(range(len(cols[0])))
    rng.shuffle(idx)
    for i in sorted(idx[:16000]):
        s, o1, o2, a = (c[i] for c in cols)
        if a not in ("1", "2"):
            continue
        yield i, a, {"sentence": clean(s)}, {"blank": {
            "type": "choice", "instructions": "Who or what belongs in the blank (_)?",
            "criteria": {"opt_1": clean(o1), "opt_2": clean(o2)}, "label": f"opt_{a}", "src": "winogrande"}}


# --------------------------------------------------------------------------- contamination sets

def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def leaves(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from leaves(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from leaves(v)


def shingles(s):
    w = re.findall(r"\w+", s.lower())
    return {" ".join(w[i:i + K]) for i in range(len(w) - K + 1)}


def raw_eval_rows(suite):
    if suite == "td_test":
        rows = finetune.load_typed_rows(os.path.join(EXP, "data", "typed-decisions"), "test", "all", None)
        out = []
        for r in rows:
            st, qs = r["state"], r["questions"]
            st = json.loads(st) if isinstance(st, str) and st[:1] in "{[" else st
            qs = json.loads(qs) if isinstance(qs, str) else qs
            out.append((st, qs))
        return out
    rs = dd._rows(dd.FILES[suite])
    if suite.startswith("jb_"):
        for r in rs:
            src = str(r.get("family", ""))
            assert "paws" not in src
        return [(r["state"], {"q": r["question"]}) for r in rs]
    return [(r["state"], r["questions"]) for r in rs]


def eval_index():
    idx = {}
    for s in EVAL_SUITES:
        states, cands, sh = set(), set(), set()
        for st, qs in raw_eval_rows(s):
            texts = [to_text(st)] + [x for x in leaves(st)]
            for q in qs.values():
                texts.append(to_text(q.get("instructions")))
            try:
                for stext, _k, cs in build_pairs(st, {k: {kk: vv for kk, vv in q.items() if kk not in ("label", "src")}
                                                      for k, q in qs.items()}).values():
                    texts.append(stext)
                    cands.update(c for c in cs if len(c) > 20)
            except Exception:
                pass
            for t in texts:
                t = t.strip()
                if len(t) > 20:
                    states.add(t)
        idx[s] = {"exact": {h(t) for t in states}, "norm": {h(norm(t)) for t in states},
                  "cand_exact": {h(t) for t in cands}, "cand_norm": {h(norm(t)) for t in cands}}
        for t in states:
            sh |= shingles(t)
        idx[s]["shingles"] = sh
        idx[s]["n_texts"], idx[s]["n_cands"] = len(states), len(cands)
    return idx


def row_texts(state, q):
    (stext, _keys, cands), = build_pairs(state, {"q": {k: v for k, v in q.items() if k not in ("label", "src")}}).values()
    units = [to_text(state), stext, *leaves(state)]
    if isinstance(q.get("instructions"), str):
        m = re.search(r'"(.+)"', q["instructions"])
        if m:
            units.append(m.group(1))
    units = [u.strip() for u in units if len(u.strip()) > 20]
    return stext, units, [c for c in cands if len(c) > 20]


def check(state, q, idx):
    """-> {suite: [kinds of overlap]} for this row (empty if clean)."""
    stext, units, cands = row_texts(state, q)
    ue, un = {h(u) for u in units}, {h(norm(u)) for u in units}
    ce, cn = {h(c) for c in cands}, {h(norm(c)) for c in cands}
    sh = shingles(stext)
    hits = {}
    for s, ix in idx.items():
        k = []
        if ue & ix["exact"] or ce & ix["exact"]:
            k.append("exact_state")
        if un & ix["norm"] or cn & ix["norm"]:
            k.append("norm_state")
        if (ue | ce) & ix["cand_exact"] or (un | cn) & ix["cand_norm"]:
            k.append("eval_candidate")
        if sh & ix["shingles"]:
            k.append("shingle13")
        if k:
            hits[s] = k
    return hits


def balanced(items, n, rng):
    by = collections.defaultdict(list)
    for it in items:
        by[it["_bal"]].append(it)
    for v in by.values():
        rng.shuffle(v)
    out, labs = [], sorted(by, key=str)
    while len(out) < n and any(by.values()):
        for l in labs:
            if by[l] and len(out) < n:
                out.append(by[l].pop())
    return out


def main():
    rng = random.Random(SEED)
    print("indexing eval suites ...", flush=True)
    idx = eval_index()
    # also report (not filter) overlap with existing training data
    train_norm = set()
    for p in (dd.FILES["kev_train"], f"{dd.KEV}/public-pool-v6/train.jsonl"):
        for r in dd._rows(p):
            train_norm.add(h(norm(to_text(r["state"]))))

    rows, report, label_maps = [], {}, {}
    gens = {"mrpc": lambda t: gen_mrpc(t), "qqp": lambda t: gen_qqp(t), "rte": lambda t: gen_rte(t),
            "go_emotions": lambda t: gen_go_emotions(t), "civil_comments": lambda t: gen_civil_comments(t, rng),
            "winogrande": lambda t: gen_winogrande(t, rng)}
    for name, gen in gens.items():
        repo, rev, fn = SPECS[name]
        print("building", name, flush=True)
        t = table(name)
        cands = list(gen(t))
        if name == "qqp":   # 364k rows: pre-sample a balanced pool before the (slow) checks
            rng.shuffle(cands)
            cands = [c for c in cands if c[1] == 1][:8000] + [c for c in cands if c[1] == 0][:8000]
        rep = collections.Counter(considered=len(cands))
        per_suite = collections.defaultdict(collections.Counter)
        seen, kept = set(), []
        for ridx, bal, state, qs in cands:
            (qid, q), = qs.items()
            key = h(norm(build_pairs(state, {"q": {k: v for k, v in q.items() if k not in ("label", "src")}})["q"][0]))
            if key in seen:
                rep["duplicate_within"] += 1
                continue
            seen.add(key)
            hits = check(state, q, idx)
            if hits:
                rep["dropped_contamination"] += 1
                for s, kinds in hits.items():
                    for kd in kinds:
                        per_suite[s][kd] += 1
                continue
            stext = to_text(state)
            rid = f"{name}/train/{ridx}"
            body = {"state": state, "questions": qs}
            meta = {"row": ridx, "text_sha256": h(stext), "row_sha256": h(json.dumps(body, sort_keys=True, ensure_ascii=False)),
                    "source": name, "repo": repo, "config": fn.split("/")[0], "revision": rev, "split": "train",
                    "id": rid, "group_id": rid, "variant": "clean", "task_type": TASK_TYPE[name]}
            kept.append({**body, "_meta": meta, "_bal": bal, "_trainoverlap": h(norm(stext)) in train_norm})
        sel = balanced(kept, N_PER_SOURCE, rng)
        sel.sort(key=lambda r: r["_meta"]["row"])
        rep["clean_available"] = len(kept)
        rep["selected"] = len(sel)
        rep["selected_overlap_existing_train_norm"] = sum(r["_trainoverlap"] for r in sel)
        labc = collections.Counter(str(r["questions"][next(iter(r["questions"]))]["label"]) for r in sel)
        report[name] = {**rep, "drop_hits_by_suite": {s: dict(c) for s, c in per_suite.items()},
                        "label_counts": dict(sorted(labc.items()))}
        for r in sel:
            r.pop("_bal"), r.pop("_trainoverlap")
        rows += sel
        print(name, dict(rep), flush=True)

    label_maps = {
        "mrpc": {"1 (equivalent)": True, "0 (not_equivalent)": False},
        "qqp": {"1 (duplicate)": "duplicate", "0 (not_duplicate)": "distinct"},
        "rte": {"0 (entailment)": True, "1 (not_entailment)": False},
        "go_emotions": {"rule": "simplified config; rows with exactly one label; key = label name", "names": GOEMO},
        "civil_comments": {"toxicity >= 0.5": True, "toxicity == 0.0": False,
                           "dropped": "0 < toxicity < 0.5 (ambiguous); text length outside 30-700 chars"},
        "winogrande": {"answer '1'": "opt_1 (= option1)", "answer '2'": "opt_2 (= option2)"},
    }
    rng.shuffle(rows)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "name": "breadth_v1", "seed": SEED, "created": "2026-10-01", "split_used": "train",
        "sources": {n: {"repo": SPECS[n][0], "config": SPECS[n][2].split("/")[0], "revision": SPECS[n][1],
                        "file": SPECS[n][2], "task_type": TASK_TYPE[n],
                        "question_type": QTYPE[n]} for n in SPECS},
        "counts": {n: report[n]["selected"] for n in SPECS}, "total_rows": len(rows),
        "label_maps": label_maps,
        "excluded_eval_sources": sorted(FORBIDDEN_SOURCES), "excluded_repos": sorted(FORBIDDEN_REPOS),
        "excluded_configs": ["nyu-mll/glue:qnli"],
        "contamination": {
            "suites_checked": EVAL_SUITES, "shingle_suites": SHINGLE_SUITES, "shingle_k_words": K,
            "note": ("A row is dropped on any hit against any suite: exact or normalized (lowercase, collapsed "
                     "whitespace) sha256 of state text / state leaves / quoted hypothesis vs eval state texts, "
                     "state leaves and instructions (>20 chars); the same vs eval candidate texts (>20 chars); and "
                     "any shared 13-word shingle vs eval state texts. Shingles were checked against all listed "
                     "suites (superset of kevT_* and jb_*)."),
            "eval_index_sizes": {s: {"texts": idx[s]["n_texts"], "candidates": idx[s]["n_cands"],
                                     "shingles": len(idx[s]["shingles"])} for s in EVAL_SUITES},
            "per_source": report,
        },
    }
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("wrote", len(rows), "rows")


if __name__ == "__main__":
    main()
