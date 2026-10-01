"""Study115 - offline best-of-N candidate pools for a selector experiment (deterministic, CPU only).

Reads ONLY stored generations, stored execution results and stored labels. It never opens, copies or
executes a SQLite file, never executes SQL or candidate code, and never touches Study111.
Spider schemas and Spider TRAIN examples are read from the JSON members of spider_data.zip (the
.sqlite members are never read). Holdout schemas are the DDL strings the holdout prompts used
(101/data/schemas.json for the 265 cohort; the SCHEMAS literals in 101/source/domains_d1.py, read with
`ast` - not imported - for the 676 cohort).

Run:  $env:PYTHONPATH=''; <113 venv python> build_pools.py
"""

from __future__ import annotations

import ast
import hashlib
import os
import json
import math
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
EXP = HERE.parent
S86 = EXP / "86_spider_full_oss_expansion_20260915"
S87 = EXP / "87_qwen32b_off_full_spider_20260915"
S88 = EXP / "88_oss_medium_repeats_full_spider_20260915"
S90 = EXP / "90_qwen_off_repeats_full_spider_20260915"
S92 = EXP / "92_solar_pro4_direct_full_spider_20260917"
S101 = EXP / "101_private_synthetic_holdout_20260918"
S104 = EXP / "104_coder32b_comparison_20260919"
S108 = EXP / "108_local_remaining_d1_20260921"
S109 = EXP / "109_audit_resolution_20260922"
S112 = EXP / "112_livecodebench_hard_20260927"
DEV = S109 / "spider_dev"
ARCHIVE = Path(os.environ.get("SPIDER_DATA_ZIP", "spider_data.zip"))   # official Spider release zip (JSON members only)
TOKENIZER = (EXP / "113_clm_reproduction_20260928/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots"
             / "c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json")
HOLDOUT_LABEL = "corrected_strict"
RAG_K = 5
STATE_MAX_TOK = 2500
CODE_MAX_TOK = 1200
PREVIEW_CHARS = 300

assert "111_coding" not in str(HERE)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")


def keys_for(n):
    out = []
    for i in range(n):
        s, j = "", i
        while True:
            s = chr(65 + j % 26) + s
            j = j // 26 - 1
            if j < 0:
                break
        out.append(s)
    return out


# ----------------------------------------------------------------------------- tokenizer ---------
from tokenizers import Tokenizer  # noqa: E402

TOK = Tokenizer.from_file(str(TOKENIZER))


def ntok(text: str) -> int:
    return len(TOK.encode(text, add_special_tokens=False).ids) if text else 0


def truncate_tokens(text: str, limit: int) -> tuple[str, bool]:
    enc = TOK.encode(text, add_special_tokens=False)
    if len(enc.ids) <= limit:
        return text, False
    end = enc.offsets[limit - 1][1]
    return text[:end] + "\n...[truncated]", True


# ----------------------------------------------------------------------------- schemas -----------
def spider_schema(entry: dict) -> str:
    tables = entry["table_names_original"]
    cols = entry["column_names_original"]
    lines = []
    for t_index, t in enumerate(tables):
        names = [c for ti, c in cols if ti == t_index]
        lines.append(f"{t}({', '.join(names)})")
    def q(ci):
        ti, c = cols[ci]
        return f"{tables[ti]}.{c}"
    pks = []
    for pk in entry["primary_keys"]:
        pks.extend(pk if isinstance(pk, list) else [pk])
    if pks:
        lines.append("Primary keys: " + ", ".join(q(c) for c in pks))
    if entry["foreign_keys"]:
        lines.append("Foreign keys: " + ", ".join(f"{q(a)} -> {q(b)}" for a, b in entry["foreign_keys"]))
    return "\n".join(lines)


def ddl_schema(ddl: str) -> str:
    lines, pks, fks = [], [], []
    for m in re.finditer(r"CREATE TABLE\s+(\w+)\s*\((.*?)\);", ddl, re.S):
        table, body = m.group(1), m.group(2)
        parts, depth, cur = [], 0, ""
        for ch in body + ",":
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur.strip())
                cur = ""
            else:
                cur += ch
        names = []
        for p in parts:
            up = p.upper()
            if up.startswith("FOREIGN KEY"):
                f = re.search(r"FOREIGN KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\s*\((\w+)\)", p, re.I)
                if f:
                    fks.append(f"{table}.{f.group(1)} -> {f.group(2)}.{f.group(3)}")
            elif up.startswith("PRIMARY KEY"):
                pks.extend(f"{table}.{c}" for c in re.findall(r"\w+", p.split("(", 1)[1]))
            elif p:
                name = p.split()[0]
                names.append(name)
                if "PRIMARY KEY" in up:
                    pks.append(f"{table}.{name}")
        lines.append(f"{table}({', '.join(names)})")
    if pks:
        lines.append("Primary keys: " + ", ".join(pks))
    if fks:
        lines.append("Foreign keys: " + ", ".join(fks))
    return "\n".join(lines)


def d1_schemas() -> dict:
    """SCHEMAS of 101/source/domains_d1.py, recovered with ast (the module is not imported)."""
    tree = ast.parse((S101 / "source/domains_d1.py").read_text(encoding="utf-8"))
    strings = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    strings[t.id] = node.value.value
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                and getattr(node.value.func, "id", None) == "domain":
            args = node.value.args
            db, schema = args[0].value, args[2]
            out[db] = strings[schema.id] if isinstance(schema, ast.Name) else schema.value
    return out


def state_sql(db_id, schema, question):
    return f"Database: {db_id}\nSchema:\n{schema}\nQuestion: {question}"


# ----------------------------------------------------------------------------- TF-IDF RAG --------
def words(text):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return w + [a + " " + b for a, b in zip(w, w[1:])]


class TfIdf:
    def __init__(self, docs):
        import numpy as np
        self.np = np
        self.n = len(docs)
        tfs = [Counter(words(d)) for d in docs]
        df = Counter(t for tf in tfs for t in tf)
        self.idf = {t: math.log((1 + self.n) / (1 + c)) + 1.0 for t, c in df.items()}
        post = defaultdict(lambda: ([], []))
        for i, tf in enumerate(tfs):
            vec = {t: (1 + math.log(c)) * self.idf[t] for t, c in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for t, v in vec.items():
                post[t][0].append(i)
                post[t][1].append(v / norm)
        self.post = {t: (np.array(a, dtype=np.int64), np.array(b, dtype=np.float64)) for t, (a, b) in post.items()}

    def top(self, query, k):
        np = self.np
        tf = Counter(t for t in words(query) if t in self.idf)
        scores = np.zeros(self.n)
        if tf:
            vec = {t: (1 + math.log(c)) * self.idf[t] for t, c in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for t, v in vec.items():
                idx, w = self.post[t]
                scores[idx] += w * (v / norm)
        order = sorted(range(self.n), key=lambda i: (-scores[i], i))
        return order, scores


def spider_rag(index, train, question):
    order, scores = index.top(question, RAG_K)
    out, seen = [], set()
    for i in order:
        key = (train[i]["question"].strip().lower(), train[i]["query"].strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(question=train[i]["question"], sql=train[i]["query"], db_id=train[i]["db_id"],
                        score=round(float(scores[i]), 4)))
        if len(out) == RAG_K:
            break
    return out


# ----------------------------------------------------------------------------- clustering --------
def row_key(execution):
    if execution and execution.get("status") == "ok":
        return tuple(sorted(Counter(json.dumps(r, sort_keys=True) for r in execution["rows"]).items()))
    return None


def preview(execution):
    if execution is None:
        return None
    if execution.get("status") != "ok":
        msg = f"ERROR[{execution.get('status')}]: {execution.get('error') or ''}"
        return msg[:PREVIEW_CHARS]
    rows = execution["rows"]
    text = f"{len(rows)} rows" + ("" if not rows else ": " + " | ".join(
        json.dumps(r, ensure_ascii=False) for r in rows[:3]) + (" | ..." if len(rows) > 3 else ""))
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS - 3] + "..."


def norm_sql(sql: str) -> str:
    """Text key for SQL when no result rows are stored: case/whitespace/quoting/alias-insensitive."""
    s = (sql or "").strip().rstrip(";").strip()
    pieces = re.split(r"('(?:[^']|'')*')", s)
    out = []
    for i, p in enumerate(pieces):
        if i % 2 == 1:
            out.append(p)
            continue
        p = p.lower().replace('"', "").replace("`", "")
        p = re.sub(r"\s+as\s+[a-z_][a-z0-9_]*", "", p)
        p = re.sub(r"\s+", " ", p)
        p = re.sub(r"\s*([(),=<>+\-*/])\s*", r"\1", p)
        out.append(p.strip())
    return " ".join(x for x in out if x)


def cluster(cands, mode):
    """cands: list (arm order) of dict(arm, sql, execution, correct). mode 'rows' or 'text'.
    -> options list; failed executions (rows mode) cluster by normalized SQL text and are flagged."""
    groups, order = {}, []
    for c in cands:
        if mode == "rows":
            rk = row_key(c["execution"])
            key = ("rows", rk) if rk is not None else ("fail", norm_sql(c["sql"]))
        else:
            key = ("text", norm_sql(c["sql"]))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(c)
    options = []
    for k, key in zip(keys_for(len(order)), order):
        members = groups[key]
        rep = members[0]
        labels = [m["correct"] for m in members]
        known = [x for x in labels if x is not None]
        options.append(dict(
            key=k, sql_or_code=rep["sql"], result_preview=preview(rep["execution"]) if mode == "rows" else None,
            n_votes=len(members), arms=[m["arm"] for m in members], correct=rep["correct"],
            executable=(key[0] != "fail"), mixed_correctness=len(set(known)) > 1,
            n_correct_members=sum(1 for x in known if x)))
    return options


def vote_pick(options):
    """Largest executable cluster; ties -> the cluster whose first member comes first in arm order
    (options are already in that order). No executable cluster -> the first option."""
    usable = [o for o in options if o.get("executable", True)]
    if not usable:
        return options[0]
    return max(usable, key=lambda o: (o["n_votes"], -options.index(o)))


# ----------------------------------------------------------------------------- Spider test -------
TEST_ARMS = [("oss_low_0", S86, "direct_low", "86__direct_low"), ("oss_low_a", S88, "low_a", "88__low_a"),
             ("oss_low_b", S88, "low_b", "88__low_b"), ("oss_medium_0", S86, "direct_medium", "86__direct_medium"),
             ("oss_medium_a", S88, "medium_a", "88__medium_a"), ("oss_medium_b", S88, "medium_b", "88__medium_b"),
             ("qwen_off_0", S87, "off", "87__off"), ("qwen_off_a", S90, "off_a", "90__off_a"),
             ("qwen_off_b", S90, "off_b", "90__off_b")]


def gen_index(study, arm):
    index = {}
    base = study / "generation" / arm
    if base.is_dir():
        for execution in sorted(base.rglob("execution.json")):
            index.setdefault(execution.parent.name, execution.parent)
    return index


def resolve(cell, index, committed):
    """Same rule as _ops/score_full_spider.resolve."""
    if (cell / "execution.json").exists():
        return cell
    pointer = committed.get("reused_from")
    if not pointer:
        return index.get(cell.name)
    source = Path(pointer)
    if source.name == "committed.json":
        if (source.parent / "execution.json").exists():
            return source.parent
        source_study, source_arm = source.parents[3], source.parent.parent.name
        found = sorted((source_study / "generation" / source_arm / "cells").glob(f"*/{cell.name}/execution.json"))
        return found[0].parent if found else None
    if source.parent.parent.name == "outcomes":
        study = source.parents[2]
        return study / "pipelines" / committed["conversion_effort"] / "cells" / committed["sql_effort"] / committed["id"]
    return source.parent


def load_test_arm(study, arm, cache_key, log):
    cache = read(S109 / "spider" / "cache" / f"{cache_key}.json")
    index = gen_index(study, arm)
    out, disagree = {}, 0
    for cell in sorted((study / "cells" / arm).iterdir()):
        committed = read(cell / "committed.json")
        folder = resolve(cell, index, committed)
        pred = read(folder / "final_prediction.json") if folder and (folder / "final_prediction.json").exists() else {}
        execution = read(folder / "execution.json") if folder and (folder / "execution.json").exists() else None
        label = bool(cache["per_q"][cell.name]["f"][0])
        study_score = read(study / "scores" / arm / f"{cell.name}.json")["correct"]
        disagree += int(bool(study_score) != label)
        out[cell.name] = dict(sql=pred.get("sql") or "", execution=execution, correct=label,
                              study_correct=bool(study_score))
    log[cache_key] = dict(n=len(out), cache_vs_study_score_disagreements=disagree,
                          missing_execution=sum(1 for v in out.values() if v["execution"] is None))
    return out


def build_spider_test(z, index, train, log):
    tasks = json.loads(z.read("spider_data/test.json"))
    schemas = {s["db_id"]: spider_schema(s) for s in json.loads(z.read("spider_data/test_tables.json"))}
    arms = {name: load_test_arm(study, arm, ck, log) for name, study, arm, ck in TEST_ARMS}
    solar_cache = read(S109 / "spider" / "cache" / "92__solar_direct.json")["per_q"]
    records = []
    for idx, t in enumerate(tasks):
        qid = f"spider_test_{idx:04d}"
        cands = [dict(arm=name, **arms[name][qid]) for name, *_ in TEST_ARMS if qid in arms[name]]
        options = cluster(cands, "rows")
        schema = schemas[t["db_id"]]
        records.append(dict(
            id=qid, split="spider_test", db_id=t["db_id"], question=t["question"], schema=schema,
            state_text=state_sql(t["db_id"], schema, t["question"]), options=options,
            rag=spider_rag(index, train, t["question"]),
            baseline=dict(solar=bool(solar_cache[qid]["f"][0]),
                          per_arm={c["arm"]: c["correct"] for c in cands}),
            _rows={c["arm"]: (c["correct"], row_key(c["execution"])) for c in cands}))
    return records


# ----------------------------------------------------------------------------- Spider dev --------
DEV_ARMS = ["k0", "k3", "k5_s73", "opt_k5", "k7", "k10"]


def build_spider_dev(z, index, train, log):
    sample = read(DEV / "data" / "sample_400.json")
    schemas = {s["db_id"]: spider_schema(s) for s in json.loads(z.read("spider_data/tables.json"))}
    records, missing = [], Counter()
    for t in sample:
        qid = t["id"]
        cands = []
        for cond in DEV_ARMS:
            sp = DEV / "scores" / cond / f"{qid}.json"
            if not sp.exists():
                missing[cond] += 1
                continue
            score = read(sp)
            ep = (DEV / "executions" / cond / f"{qid}.json") if cond == "opt_k5" else \
                 (DEV / "generation" / cond / "cells" / "low" / qid / "execution.json")
            execution = read(ep) if ep.exists() else None
            if execution is None:
                missing[cond + "_execution"] += 1
            cands.append(dict(arm=cond, sql=score["sql"] or "", execution=execution, correct=bool(score["correct"])))
        options = cluster(cands, "rows")
        schema = schemas[t["db_id"]]
        records.append(dict(
            id=qid, split="spider_dev", db_id=t["db_id"], question=t["question"], schema=schema,
            state_text=state_sql(t["db_id"], schema, t["question"]), options=options,
            rag=spider_rag(index, train, t["question"]),
            baseline=dict(solar=None, per_arm={c["arm"]: c["correct"] for c in cands}),
            hardness=t.get("hardness")))
    log["spider_dev_missing"] = dict(missing)
    return records


# ----------------------------------------------------------------------------- holdout -----------
HOLDOUT_ARMS = ["oss_low", "oss_medium", "qwen_off", "coder32b",
                "oss_low_rag_bankv2", "oss_medium_rag_bankv2", "qwen_off_rag_bankv2", "coder32b_rag_bankv2"]


def holdout_folder(cohort, arm):
    if arm.endswith("_bankv2"):
        return S109 / "rag_v2" / "cells" / cohort / arm[: -len("_bankv2")]
    if arm == "coder32b":
        return (S104 / "cells" / "coder32b") if cohort == "265" else (S108 / "cells_d1" / "coder32b")
    return S101 / ("cells" if cohort == "265" else "cells_d1") / arm


def build_holdout(log):
    labels = read(S109 / "holdout" / "RESCORE_PER_TASK.json")
    ddl = {"265": read(S101 / "data" / "schemas.json"), "676": d1_schemas()}
    cohorts = {"265": read(S101 / "data" / "cohort_dedup.json"),
               "676": read(S101 / "data" / "cohort_expansion_d1_labelled.json")}
    banks = {"265": read(S109 / "bank_v2" / "examples_265.json"),
             "676": read(S109 / "bank_v2" / "examples_676.json")}
    records, missing = [], Counter()
    for cohort, tasks in cohorts.items():
        for t in tasks:
            qid = t["id"]
            cands = []
            for arm in HOLDOUT_ARMS:
                f = holdout_folder(cohort, arm) / f"{qid}.json"
                lab = labels.get(f"{cohort}/{arm}", {}).get(qid)
                if not f.exists() or lab is None:
                    missing[f"{cohort}/{arm}"] += 1
                    continue
                cands.append(dict(arm=arm, sql=read(f).get("sql") or "", execution=None,
                                  correct=lab[HOLDOUT_LABEL], unified_strict=lab["unified_strict"]))
            options = cluster(cands, "text")
            # per-option secondary label (unified strict, defined on all 941) for the excluded items
            for o in options:
                o["correct_unified_strict"] = next(c["unified_strict"] for c in cands if c["arm"] == o["arms"][0])
            schema = ddl_schema(ddl[cohort][t["db_id"]])
            sol = labels[f"{cohort}/solar_direct"].get(qid)
            records.append(dict(
                id=f"{cohort}/{qid}", split="holdout", cohort=cohort, db_id=t["db_id"], question=t["question"],
                schema=schema, state_text=state_sql(t["db_id"], schema, t["question"]), options=options,
                rag=[dict(question=e["question"], sql=e["sql"]) for e in banks[cohort][qid][:RAG_K]],
                baseline=dict(solar=None if sol is None else sol[HOLDOUT_LABEL],
                              solar_unified_strict=None if sol is None else sol["unified_strict"],
                              per_arm={c["arm"]: c["correct"] for c in cands}),
                label_excluded_nondeterministic=all(c["correct"] is None for c in cands),
                family=t.get("family"), difficulty=t.get("difficulty")))
    log["holdout_missing"] = dict(missing)
    return records


# ----------------------------------------------------------------------------- LiveCodeBench -----
LCB_ARMS = ["oss_low_direct_p32k", "oss_low_idea_k5_p32k", "oss_low_full_k5_p32k"]


def extract_code(raw: str) -> str:
    """112/source/grader.py extract(): last fenced block, else from the first import/def/class."""
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    blocks = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", raw, re.I)
    if blocks:
        return blocks[-1].strip()
    m = re.search(r"^(?:from |import |def |class )[\s\S]*", raw, re.M)
    return m.group(0).strip() if m else ""


def build_lcb(log):
    problems = read(S112 / "data" / "problems.json")
    solved = {a: {p["id"]: bool(p["solved"]) for p in read(S112 / "results" / f"SCORE_{a}.json")["per_problem"]}
              for a in LCB_ARMS + ["solar_direct", "solar_direct_rmedium"]}
    records, trunc = [], Counter()
    for p in problems:
        body = f"Title: {p['title']}\n\n{p['statement'].rstrip()}"
        if p["kind"] == "functional" and p.get("starter_code"):
            body += f"\n\nComplete this (starter code):\n```python\n{p['starter_code']}\n```"
        state, cut = truncate_tokens(body, STATE_MAX_TOK)
        trunc["state"] += cut
        options = []
        for key, arm in zip(keys_for(len(LCB_ARMS)), LCB_ARMS):
            cell = S112 / "cells" / arm / f"{p['id']}.json"
            if not cell.exists() or p["id"] not in solved[arm]:
                trunc[f"missing_{arm}"] += 1
                continue
            code = extract_code(read(cell).get("answer") or "")
            code_t, cut = truncate_tokens(code, CODE_MAX_TOK)
            trunc["code"] += cut
            options.append(dict(key=key, sql_or_code=code_t, result_preview=None, n_votes=1, arms=[arm],
                                correct=solved[arm][p["id"]], code_truncated=cut, empty_code=not code))
        records.append(dict(
            id=p["id"], split="lcb_hard", db_id=None, question=p["title"], schema=None, state_text=state,
            options=options, rag=[],
            baseline=dict(solar=solved["solar_direct"].get(p["id"]),
                          solar_rmedium=solved["solar_direct_rmedium"].get(p["id"]),
                          per_arm={o["arms"][0]: o["correct"] for o in options}),
            kind=p["kind"], platform=p["platform"], state_truncated=cut))
    log["lcb"] = dict(trunc)
    return records


# ----------------------------------------------------------------------------- stats -------------
def render_options(r):
    parts = []
    for o in r["options"]:
        s = f"[{o['key']}] (votes {o['n_votes']})\n{o['sql_or_code']}"
        if o.get("result_preview"):
            s += f"\nResult: {o['result_preview']}"
        parts.append(s)
    return "\n\n".join(parts)


def render_rag(r):
    return "\n\n".join(f"Q: {e['question']}\nSQL: {e['sql']}" for e in r["rag"])


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)] if xs else None


def split_stats(records, arms, solar_key="solar"):
    def acc(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None, len(vals)
    labelled = [r for r in records if any(o["correct"] is not None for o in r["options"])]
    oracle, n_or = acc([any(o["correct"] for o in r["options"]) for r in labelled])
    vote, n_v = acc([vote_pick(r["options"])["correct"] for r in labelled])
    single = {a: acc([r["baseline"]["per_arm"].get(a) for r in labelled])[0] for a in arms}
    solar = acc([r["baseline"][solar_key] for r in labelled])
    tok_state = [ntok(r["state_text"]) for r in records]
    tok_opts = [ntok(render_options(r)) for r in records]
    tok_rag = [ntok(render_rag(r)) for r in records]
    both = [a + b for a, b in zip(tok_state, tok_opts)]
    allt = [a + b + c for a, b, c in zip(tok_state, tok_opts, tok_rag)]
    mixed = sum(1 for r in records for o in r["options"] if o.get("mixed_correctness"))
    return dict(
        n_questions=len(records), n_labelled=len(labelled),
        mean_options=round(statistics.mean(len(r["options"]) for r in records), 3),
        pct_exactly_one_option=round(100 * sum(len(r["options"]) == 1 for r in records) / len(records), 2),
        oracle=oracle, majority_vote=vote, single_arm=single, solar=solar[0], solar_n=solar[1],
        mixed_correctness_clusters=mixed,
        questions_with_mixed_cluster=sum(1 for r in records if any(o.get("mixed_correctness") for o in r["options"])),
        tokens_qwen3=dict(
            state=dict(median=statistics.median(tok_state), p90=pct(tok_state, .9), max=max(tok_state)),
            options=dict(median=statistics.median(tok_opts), p90=pct(tok_opts, .9), max=max(tok_opts)),
            state_plus_options=dict(median=statistics.median(both), p90=pct(both, .9), max=max(both)),
            state_options_rag=dict(median=statistics.median(allt), p90=pct(allt, .9), max=max(allt))))


def test_sanity(records):
    """Reproduce _ops/selection_full.py on the per-arm (correct, row key) pairs."""
    def vote(results):
        groups = {}
        for pos, (c, k) in enumerate(results):
            if k is not None:
                groups.setdefault(k, []).append((pos, c))
        if not groups:
            return results[0][0]
        best = max(groups.values(), key=lambda m: (len(m), -m[0][0]))
        return best[0][1]
    sets = {"low x3": ["oss_low_0", "oss_low_a", "oss_low_b"],
            "medium x3": ["oss_medium_0", "oss_medium_a", "oss_medium_b"],
            "low+medium x6": ["oss_low_0", "oss_low_a", "oss_low_b", "oss_medium_0", "oss_medium_a", "oss_medium_b"],
            "qwen x3": ["qwen_off_0", "qwen_off_a", "qwen_off_b"],
            "all x9": [a[0] for a in TEST_ARMS]}
    out = {}
    for name, members in sets.items():
        res = [[r["_rows"][m] for m in members] for r in records]
        out[name] = dict(vote=round(sum(vote(x) for x in res) / len(res), 4),
                         oracle=round(sum(any(c for c, _ in x) for x in res) / len(res), 4))
    out["solar"] = round(sum(r["baseline"]["solar"] for r in records) / len(records), 4)
    out["known"] = {"low x3": dict(vote=0.7569, oracle=0.7918), "medium x3": dict(vote=0.7443, oracle=0.7806),
                    "low+medium x6": dict(vote=0.7555, oracle=0.8179), "solar": 0.7932}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = {}
    with zipfile.ZipFile(ARCHIVE) as z:
        names = ["spider_data/train_spider.json", "spider_data/train_others.json"]
        train = json.loads(z.read(names[0])) + json.loads(z.read(names[1]))
        index = TfIdf([x["question"] for x in train])
        dev = build_spider_dev(z, index, train, log)
        test = build_spider_test(z, index, train, log)
        train_dbs = {x["db_id"] for x in train}
        log["rag_train_db_overlap"] = dict(dev=len(train_dbs & {r["db_id"] for r in dev}),
                                           test=len(train_dbs & {r["db_id"] for r in test}))
    hold = build_holdout(log)
    lcb = build_lcb(log)

    sanity = test_sanity(test)
    for r in test:
        r.pop("_rows")
    stats = dict(
        spider_dev=split_stats(dev, DEV_ARMS),
        spider_test=split_stats(test, [a[0] for a in TEST_ARMS]),
        holdout=split_stats(hold, HOLDOUT_ARMS),
        holdout_265=split_stats([r for r in hold if r["cohort"] == "265"], HOLDOUT_ARMS),
        holdout_676=split_stats([r for r in hold if r["cohort"] == "676"], HOLDOUT_ARMS),
        lcb_hard=split_stats(lcb, LCB_ARMS))
    stats["lcb_hard"]["solar_rmedium"] = round(
        sum(bool(r["baseline"]["solar_rmedium"]) for r in lcb if r["baseline"]["solar_rmedium"] is not None)
        / sum(1 for r in lcb if r["baseline"]["solar_rmedium"] is not None), 4)
    stats["spider_test_sanity_vs_selection_full"] = sanity
    stats["build_log"] = log
    stats["settings"] = dict(holdout_label=HOLDOUT_LABEL, rag_k=RAG_K, state_max_tokens=STATE_MAX_TOK,
                             code_max_tokens=CODE_MAX_TOK, token_counter=str(TOKENIZER.parts[-4]),
                             vote_rule="largest executable cluster; ties -> earliest arm; none executable -> first option")

    for name, recs in (("spider_dev", dev), ("spider_test", test), ("holdout", hold), ("lcb_hard", lcb)):
        write_jsonl(OUT / f"{name}.jsonl", recs)
        stats[name]["sha256"] = hashlib.sha256((OUT / f"{name}.jsonl").read_bytes()).hexdigest()
    (OUT / "POOL_STATS.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("tokens_qwen3",)} if isinstance(v, dict) else v
                      for k, v in stats.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
