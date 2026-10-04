"""MiDM (Minimal Decision Model): typed decisions (choice / noul / score) with a LoRA-tuned causal LM
and a pointer head. Self-contained inference for the Hugging Face release.

    from midm import MiDM
    m = MiDM.from_pretrained("path/or/repo-id")           # base model is downloaded from its own repo
    m.predict(
        state={"order_total": 1240, "po_total": 1420},
        questions={"action": {"type": "choice", "instructions": "What should happen to this invoice?",
                              "criteria": {"approve": "Approve and pay.",
                                           "hold": "Hold for discrepancy review.",
                                           "reject": "Reject as fraud."}}})

Request/answer format follows the "System One" typed-decision wire shape (state + typed questions ->
probability distributions). The text rendering helpers (to_text, state_text, candidates,
answer_from_probs, confidence) are adapted from github.com/Contrastive-LM/CLM (Apache-2.0).
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import torch

QUESTION_TYPES = ("noul", "choice", "score")
NOUL_KEYS = ("false", "true")


# ---------------------------------------------------------------- request rendering (from CLM, Apache-2.0)
def to_text(x: Any, indent: int = 0) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return str(x)
    pad = " " * indent
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{pad}{k}:\n{to_text(v, indent + 2)}")
            else:
                parts.append(f"{pad}{k}: {to_text(v)}")
        return ("\n\n" if indent == 0 else "\n").join(parts)
    if isinstance(x, (list, tuple)):
        parts = []
        for v in x:
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{pad}-\n{to_text(v, indent + 2)}")
            else:
                parts.append(f"{pad}- {to_text(v)}")
        return "\n".join(parts)
    return json.dumps(x, ensure_ascii=False)


def state_text(state: Any, instructions: Any) -> str:
    s, i = to_text(state).strip(), to_text(instructions).strip()
    return f"{s}\n\n{i}" if s and i else (s or i)


def candidates(q: dict) -> tuple[list[str], list[str]]:
    t, crit, ins = q.get("type"), q.get("criteria"), to_text(q.get("instructions")).strip()
    if t not in QUESTION_TYPES:
        raise ValueError(f"unknown question type {t!r}; expected one of {QUESTION_TYPES}")
    if t == "choice":
        if not isinstance(crit, dict) or not crit:
            raise ValueError("choice question needs a non-empty 'criteria' object")
        keys = list(crit)
        return keys, [to_text(crit[k]) if crit[k] not in (None, "") else k for k in keys]
    if t == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise ValueError("score question needs 'criteria' as an ordered list of >= 2 levels")
        return [str(i) for i in range(len(crit))], [to_text(c) for c in crit]
    crit = crit or {}
    texts = []
    for k in NOUL_KEYS:
        d = crit.get(k) if isinstance(crit, dict) else None
        if d in (None, ""):
            d = (f"Yes. This is true: {ins}" if k == "true" else f"No. This is false: {ins}") if ins else k
        texts.append(f"{k}: {to_text(d)}")
    return list(NOUL_KEYS), texts


def confidence(probs: list[float]) -> float:
    if len(probs) < 2:
        return 1.0
    j = max(range(len(probs)), key=probs.__getitem__)
    rest = [p for i, p in enumerate(probs) if i != j]
    return max(0.0, min(1.0, probs[j] - sum(rest) / len(rest)))


def answer_from_probs(q: dict, keys: list[str], probs: list[float]) -> dict:
    t = q["type"]
    probs = [float(p) for p in probs]
    dist = dict(zip(keys, probs))
    if t == "noul":
        return {"type": "noul", "noul": dist["true"]}
    if t == "choice":
        j = max(range(len(probs)), key=probs.__getitem__)
        return {"type": "choice", "choice": keys[j], "confidence": confidence(probs), "probabilities": dist}
    legend = {str(i): c if isinstance(c, str) else to_text(c) for i, c in enumerate(q["criteria"])}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)), "confidence": confidence(probs),
            "legend": legend, "probabilities": dist}


# ---------------------------------------------------------------- model
class MiDM:
    def __init__(self, tok, model, head, cfg, device):
        self.tok, self.model, self.head, self.cfg, self.device = tok, model, head, cfg, device
        self.opt_hdr = tok("\n\nOptions:", add_special_tokens=False)["input_ids"]

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda", load_in_4bit: bool = True, dtype: str = "bfloat16"):
        """path: local folder or Hugging Face repo id of this MiDM release. The base model named in
        midm_config.json is loaded from its own repo. load_in_4bit=True matches training (NF4 base)."""
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        local = path if os.path.isdir(path) else snapshot_download(path)
        cfg = json.load(open(os.path.join(local, "midm_config.json"), encoding="utf-8"))
        dt = getattr(torch, dtype)
        kw = {"dtype": dt, "device_map": {"": device}}
        if load_in_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt)
        base = AutoModel.from_pretrained(cfg["base_model"], **kw)
        if hasattr(base, "language_model") and hasattr(base, "visual"):   # Qwen3.5: text-only
            del base.visual
        model = PeftModel.from_pretrained(base, local).eval()
        head = torch.nn.Linear(cfg["hidden_size"], 1).to(device).float()
        head.load_state_dict(load_file(os.path.join(local, "pointer_head.safetensors")))
        tok = AutoTokenizer.from_pretrained(cfg["base_model"])
        return cls(tok, model, head, cfg, device)

    def _build(self, stext, keys, cands):
        ids_ = lambda t: self.tok(t, add_special_tokens=False)["input_ids"]
        lines = [ids_(f"\n- {k}: {c}")[:192] for k, c in zip(keys, cands)]
        room = self.cfg["max_len"] - len(self.opt_hdr) - sum(map(len, lines))
        if room < 32:
            raise ValueError("options too long")
        s = ids_(stext)
        if len(s) > room:
            h = room // 3
            s = s[:h] + s[-(room - h):]
        ids, pos = s + self.opt_hdr, []
        for ln in lines:
            ids = ids + ln
            pos.append(len(ids) - 1)
        return ids, pos

    @torch.no_grad()
    def predict(self, state: Any, questions: dict[str, dict], temperature: float = 1.0) -> dict:
        """-> {question_id: Answer}; all questions of the request run in one padded batch."""
        pairs = {qid: (state_text(state, q.get("instructions")), *candidates(q)) for qid, q in questions.items()}
        items = [self._build(*v) for v in pairs.values()]
        L = max(len(i) for i, _ in items)
        ids = torch.zeros((len(items), L), dtype=torch.long)
        att = torch.zeros((len(items), L), dtype=torch.long)
        for r, (i, _) in enumerate(items):
            ids[r, :len(i)] = torch.tensor(i)
            att[r, :len(i)] = 1
        dt = next(self.model.parameters()).dtype if not hasattr(self.model, "dtype") else self.model.dtype
        with torch.autocast("cuda", dtype=torch.bfloat16 if dt == torch.bfloat16 else torch.float16):
            h = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device)).last_hidden_state
        out = {}
        for r, ((qid, (_, keys, _)), (_, pos)) in enumerate(zip(pairs.items(), items)):
            logits = self.head(h[r, pos].float()).squeeze(-1) / temperature
            out[qid] = answer_from_probs(questions[qid], keys, logits.softmax(-1).tolist())
        return out


if __name__ == "__main__":
    import sys
    m = MiDM.from_pretrained(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
    print(json.dumps(m.predict(
        "Refund policy: a receipt is required and the purchase must be within 30 days. "
        "The customer bought the item 12 days ago and has no receipt.",
        {"permitted": {"type": "noul", "instructions": "Is a refund permitted under the policy?"},
         "action": {"type": "choice", "instructions": "What should the agent do?",
                    "criteria": {"refund": "Issue the refund.", "deny": "Politely deny the refund.",
                                 "escalate": "Escalate to a human."}}}), indent=1))
