# 114 Local model router (OpenRouter-style gateway), 2026-09-28

Goal: one local entry point for three kinds of models. The first is decision/selection models
(typed choice / noul / score in the TypeSafe "System One" wire format). The second is
analysis/feature models (embeddings, later rerankers). The third is general LLMs. Each is
addressed by model name, with fallbacks, load balancing and logging.

```
client ──► LiteLLM proxy :4000  (OpenAI-compatible)
            ├─ /v1/chat/completions ─► shared Ollama (exp 34) :11439 / :11443  qwen3 0.6-8B, gpt-oss-20b, sqlcoder
            ├─ /v1/embeddings       ─► embed_server (exp 113) :8090           Qwen3-Embedding / Qwen3
            └─ /v1/systemone, /v1/rank, /v1/decision-models (pass-through)
                    └─► decision_router.py :8800
                          ├─ decider-0.6b / 1.7b / 4b  ─► serve_pointer.py (exp 113 QLoRA pointer models)
                          ├─ clm-typed-0.6b            ─► serve_clm.py (bi-encoder heads)
                          └─ decider-auto: per-question cascade 0.6b → 4b → llm-judge (gateway chat)
```

## Files
- `registry.yaml`: catalogue of decision, feature and LLM models with URL, VRAM, residency
  (resident / on_demand / shared), exp-113 quality numbers, and cascade tiers and thresholds.
- `litellm_config.yaml`: gateway model list. Ollama replicas are load-balanced (least-busy), with
  fallbacks (qwen3-8b → qwen3-4b, gpt-oss-20b → qwen3-8b) and pass-through routes to the decision router.
- `decision_router.py`: System One endpoints and the cascade. Every answer is tagged `routed_to`.
  Backends that are down are skipped. The LLM judge is asked for a JSON distribution over the option keys.
- `mock_decider.py`: fixed-probability backend for GPU-free tests.

## Run
```
.venv\Scripts\python.exe decision_router.py --port 8800
.venv\Scripts\litellm.exe --config litellm_config.yaml --port 4000 --host 127.0.0.1
```
Decision backends come from experiment 113, e.g.
`python serve_pointer.py --model Qwen/Qwen3-4B --ckpt runs/ptr4b/final --port 8712`.

## Verified (2026-09-28, mocks only, no GPU)
Gateway to router to mocks: 0.6b mock (p=0.55) escalated to 4b mock (p=0.95) because 0.55 is under
the 0.80 gate. Tags were correct. /v1/rank and /v1/decision-models passed through. The /v1/models
list shows the 6 LLM/feature names. Added latency was 11.6 ms for two hops. No real model was called:
calling Ollama would load models into experiment 34's shared instances on the RTX 3090.

## Next (after the exp-113 GPU search finishes, ~18:30)
1. Replace mocks with the real pointer servers. Decide which GPU serves: resident models hold VRAM
   and conflict with the GPU queue (`experiments/_ops/GPU_QUEUE_SPEC.md`).
2. Set cascade thresholds on dev data only (kev_dev, the td_train holdout). Build the curve of
   accuracy against escalation rate and latency, then test once.
3. LLM judge: measure its accuracy on dev before trusting it as the top tier; qwen3-8b vs gpt-oss-20b.
4. Feature models: embeddings now; Qwen3-Reranker later; per-key auth and usage logs in LiteLLM.
5. On-demand start/stop of on_demand models through the queue (service slot policy).
