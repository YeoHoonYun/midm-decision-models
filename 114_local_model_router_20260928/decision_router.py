#!/usr/bin/env python3
"""Decision router: one System One endpoint (/v1/systemone, /v1/rank) over local decision models.

model = a registry MiDM name (e.g. MiDM-4B-q35-e1) -> forwarded to that backend unchanged.
model = "MiDM-Auto" (alias decider-auto; + optional "domain"): known domains -> per-question cascade (below); unknown or
untagged -> straight to the strongest live tier in registry.cascade.direct (DECISIONS.md D1a).
Cascade over registry.cascade.tiers: every question goes to
the first live tier; questions whose top-option probability is below that tier's min_confidence are
re-asked to the next tier; the last tier may be "llm-judge" (chat model through the LiteLLM gateway,
asked for a JSON distribution over the option keys). Each answer carries "routed_to".

    python decision_router.py --registry registry.yaml --port 8800 --gateway http://127.0.0.1:4000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "113_clm_reproduction_20260928", "repo", "src"))
from clm.client import question_to_dict  # noqa: E402
from clm.schema import answer_from_probs, build_pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--registry", default=os.path.join(HERE, "registry.yaml"))
ap.add_argument("--gateway", default="http://127.0.0.1:4000")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8800)
args, _ = ap.parse_known_args()

REG = yaml.safe_load(open(args.registry, encoding="utf-8"))
DECIDERS = {m["name"]: m for m in REG.get("decision", [])}
LLMS = {m["name"]: m for m in REG.get("llm", [])}
CASCADE = REG.get("cascade", {})
AUTO_NAMES = {"MiDM-Auto", "midm-auto", "decider-auto"}   # MiDM-Auto; decider-auto kept as an alias
client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=2.0))
app = FastAPI(title="local decision router")


def confidence(ans: dict) -> float:
    if ans.get("type") == "noul":
        p = float(ans["noul"])
        return max(p, 1 - p)
    probs = ans.get("probabilities") or {}
    return max(probs.values()) if probs else 0.0


async def alive(url: str) -> bool:
    try:
        return (await client.get(url + "/health", timeout=2.0)).status_code == 200
    except httpx.HTTPError:
        return False


async def ask_decider(name: str, state, questions: dict, temperature: float) -> dict:
    m = DECIDERS[name]
    r = await client.post(m["url"] + "/v1/systemone",
                          json={"state": state, "questions": questions, "model": "clm-latest" if m["backend"] == "clm"
                                else None, "temperature": temperature})
    r.raise_for_status()
    return r.json()["answers"]


LLM_PROMPT = """You are a decision model. Read the context and answer the question by giving a probability
for every option key. Reply with JSON only, mapping each key to a probability, summing to 1.

{state}

Options:
{options}
"""


async def ask_llm(name: str, state, questions: dict) -> dict:
    model = LLMS[name]["litellm_model"]
    pairs = build_pairs(state, questions)
    out = {}
    for qid, (stext, keys, cands) in pairs.items():
        opts = "\n".join(f"- {k}: {c}" for k, c in zip(keys, cands))
        body = {"model": model, "temperature": 0,
                "messages": [{"role": "user", "content": LLM_PROMPT.format(state=stext, options=opts)}]}
        r = await client.post(args.gateway + "/v1/chat/completions", json=body)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        m = re.search(r"\{.*\}", text, flags=re.S)
        probs = {}
        if m:
            try:
                probs = {str(k): float(v) for k, v in json.loads(m.group(0)).items()}
            except (ValueError, AttributeError):
                probs = {}
        p = [max(0.0, probs.get(k, 0.0)) for k in keys]
        z = sum(p)
        p = [v / z for v in p] if z > 0 else [1 / len(keys)] * len(keys)
        out[qid] = answer_from_probs(questions[qid], keys, p)
    return out


async def route(state, questions: dict, temperature: float, domain: str | None) -> dict:
    """D1a: cascade for known domains; unknown / untagged domains go to the strongest live tier."""
    if domain in set(CASCADE.get("known_domains", [])):
        tiers, mins = CASCADE.get("tiers", []), CASCADE.get("min_confidence", [])
        if tiers and tiers[-1] in DECIDERS and not await alive(DECIDERS[tiers[-1]]["url"]):
            tiers, mins = CASCADE.get("fallback_tiers", tiers), CASCADE.get("fallback_min_confidence", mins)
        return await cascade(state, questions, temperature, tiers, mins)
    for name in CASCADE.get("direct", []):
        if name in DECIDERS and await alive(DECIDERS[name]["url"]):
            got = await ask_decider(name, state, questions, temperature)
            return {q: {**a, "routed_to": name, "route": "direct:unknown_domain"} for q, a in got.items()}
    raise HTTPException(503, "no decision backend reachable")


async def cascade(state, questions: dict, temperature: float, tiers=None, mins=None) -> dict:
    tiers = tiers if tiers is not None else CASCADE.get("tiers", [])
    mins = mins if mins is not None else CASCADE.get("min_confidence", [])
    pending, answers = dict(questions), {}
    for i, tier in enumerate(tiers):
        if not pending:
            break
        last = i == len(tiers) - 1
        try:
            if tier in DECIDERS:
                if not await alive(DECIDERS[tier]["url"]):
                    continue
                got = await ask_decider(tier, state, pending, temperature)
            elif tier in LLMS:
                got = await ask_llm(tier, state, pending)
            else:
                continue
        except (httpx.HTTPError, KeyError, ValueError):
            continue
        thr = mins[i] if i < len(mins) else 0.0
        for qid, ans in got.items():
            if last or confidence(ans) >= thr or qid not in pending:
                answers[qid] = {**ans, "routed_to": tier, "confidence_gate": thr}
                pending.pop(qid, None)
            else:   # keep the low-confidence answer in case no later tier is reachable
                answers[qid] = {**ans, "routed_to": tier, "confidence_gate": thr, "escalation_failed": True}
    for qid in list(pending):
        if qid in answers:
            pending.pop(qid)
    if pending:
        raise HTTPException(503, f"no decision backend reachable for {sorted(pending)}")
    for a in answers.values():
        if a.get("escalation_failed") and a["routed_to"] == tiers[-1]:
            a.pop("escalation_failed")
    return answers


@app.get("/health")
async def health():
    live = {n: await alive(m["url"]) for n, m in DECIDERS.items()}
    return {"ok": True, "embedder": any(live.values()), "backends": live}


@app.get("/v1/models")
async def models():
    rows = [{"name": "MiDM-Auto", "description": f"MiDM cascade {CASCADE.get('tiers')}; alias decider-auto"}]
    rows += [{"name": n, "description": f"{m['backend']} {m.get('base', '')}".strip(), "quality": m.get("quality"),
              "residency": m.get("residency")} for n, m in DECIDERS.items()]
    return {"models": rows}


@app.post("/v1/systemone")
async def systemone(request: Request):
    body = await request.json()
    if not isinstance(body, dict) or "state" not in body or not isinstance(body.get("questions"), dict):
        raise HTTPException(422, "body must be {state, model, questions}")
    t0 = time.perf_counter()
    model = body.get("model") or "MiDM-Auto"
    temp = float(body.get("temperature", 1.0))
    try:
        qs = {k: question_to_dict(q) for k, q in body["questions"].items()}
        build_pairs(body["state"], qs)   # validate before routing
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(422, f"invalid request: {e}") from e
    if model in AUTO_NAMES:
        answers = await route(body["state"], qs, temp, body.get("domain"))
    elif model in DECIDERS:
        try:
            answers = await ask_decider(model, body["state"], qs, temp)
        except httpx.HTTPError as e:
            raise HTTPException(502, f"{model} unreachable: {e}") from e
    elif model in LLMS:
        answers = await ask_llm(model, body["state"], qs)
    else:
        raise HTTPException(422, f"unknown model {model!r}; see /v1/models")
    ms = (time.perf_counter() - t0) * 1000
    return JSONResponse({"model": model, "answers": answers, "usage": {"billing_units": len(qs)}},
                        headers={"X-CLM-Latency-Ms": f"{ms:.1f}"})


@app.post("/v1/rank")
async def rank(request: Request):
    body = await request.json()
    answers = body.get("answers") or []
    if not answers or not all(isinstance(a, str) and a for a in answers):
        raise HTTPException(422, "body must be {context, question, answers: [..]}")
    q = {"rank": {"type": "choice", "instructions": body.get("question"),
                  "criteria": {str(i): a for i, a in enumerate(answers)}}}
    res = await systemone_inner(body.get("context") or "", q, body.get("model") or "MiDM-Auto", body.get("domain"))
    probs = res["rank"]["probabilities"]
    order = sorted(probs.items(), key=lambda kv: -kv[1])
    return {"model": body.get("model") or "MiDM-Auto", "routed_to": res["rank"].get("routed_to"),
            "ranked": [{"rank": r + 1, "candidate": answers[int(i)], "prob": p} for r, (i, p) in enumerate(order)]}


async def systemone_inner(state, qs, model, domain=None):
    if model in AUTO_NAMES:
        return await route(state, qs, 1.0, domain)
    if model in DECIDERS:
        return await ask_decider(model, state, qs, 1.0)
    return await ask_llm(model, state, qs)


if __name__ == "__main__":
    import uvicorn
    print(f"[router] {len(DECIDERS)} deciders, cascade {CASCADE.get('tiers')} at http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
