# [초안] 81.6%인가, 3과제인가: CLM DeepSWE verifier 감사, 그리고 ≤8B frozen-encoder head와 QLoRA cross-encoder의 typed-decision 비교

작성일: 2026-09-28 (stage B 결과와 cascade 재계산 반영: 2026-09-29)
상태: **내부 초안** (arXiv tech report 또는 workshop 후보)

모든 수치는 디스크의 결과 파일에서 옮겼다.

**명칭(2026-10-01):** 본 연구의 QLoRA pointer cross-encoder 결정 모델을 **MiDM (Minimal Decision Model)** 이라 부른다. 아키텍처명과 모델 계열명이 같다.
- 모델 이름 규칙은 `MiDM-{크기}-{base}-e{epoch}`이다. 대표 모델은 `MiDM-4B-q35-e1`(Qwen3.5-4B-Base, 1 epoch)과 `MiDM-8B-q3-e2`다.
- cascade 라우터는 `MiDM-Auto`다.
- 전체 목록은 `experiments/114_local_model_router_20260928/MODELS.md`에 있다.

**2026-10-01 업데이트 요약** (상세는 §4.5b, 표는 `paper/analysis/`, 관련 연구는 `paper/related_work/related_work_update_20261001.md`)
- **논문 방향.** "새 아키텍처"가 아니라 **실증·재현 논문**으로 간다. pointer head 구조는 jina-reranker-v3(arXiv:2509.25085)와 사실상 같아서 신규성을 주장하지 않는다. 기여는 (1) CLM 감사, (2) 무엇이 효과 있고 무엇이 없는지에 대한 통제 실험, (3) confidence cascade다.
- **효과 있음.**
  - base 세대(Qwen3 → Qwen3.5)가 4B에서 Kev T test +5.0pp다(p = 0.004). **단, 0.8B와 2B에서는 transfer에서 유의하지 않다**(+0.7pp, p = 0.78; +2.1pp, p = 0.29). 세대 효과는 크기와 상호작용한다.
  - 데이터 구성(breadth_v1 + 지식형 MC, "-bx")은 Kev T test +3.4pp(p = 0.042), T-v9 dev +3.0pp(p = 0.009)다. 대신 td_test는 −1.7pp다(p = 0.024). seed 재현은 진행 중이다.
- **효과 없음.** breadth만 추가, TTA(옵션 순서 4개), 온도 보정, 앙상블(비용 대비), DoRA, rsLoRA, r64, attention-only, 어댑터 병합과 라우팅.
- **해로움.** Qwen3.5의 두 번째 epoch(Kev T test −2.9pp, p = 0.053; T-v9 −2.8pp, p = 0.002).
- **Qwen3.5 크기 사다리.** 2B → 4B에서 Kev T test가 +10.2pp로 크게 뛴다. 0.8B → 2B는 +7.1pp다.
- test-read ledger(SHA-256)가 생겼고, 47개 파일을 기록했다(§6.5 갱신).

**2026-10-01 저녁 업데이트** (한계 해소 실험. 표는 `paper/analysis/`, 계획은 `paper/PLAN_limitations_20261001.md`)

1. **Seed (한계 3 해소)**
   - Qwen3.5-4B e1: 3 seed, Kev T test 0.763 ± 0.005.
   - -bx: 3 seed, 0.796 ± 0.002.
   - 같은 seed끼리 짝지은 -bx 이득은 +3.4 / +3.8 / +2.7pp(p = 0.042 / 0.014 / 0.062)다.
   - typed-decisions 손실은 3 seed 평균 −0.5pp로 잡음 수준이다. seed 0의 −1.7pp는 우연이었다.
2. **정밀도 교란 해소**
   - fp16에서 bf16으로 바꿔도 성능 차이는 유의하지 않다(4B, 2B 모두).
   - bf16끼리 비교한 세대 효과(4B)는 Kev T dev +5.9pp(p < 0.001), T-v9 +3.9pp(p = 0.001), Kev T test +3.4pp(p = 0.066)다. 효과는 있지만 처음 추정한 +5.0pp보다 작다.
   - bf16끼리 비교한 2B → 4B는 Kev T test +10.2pp(p < 0.001)다.
3. **크기 하한**: -bx의 효과는 크기에 따라 다르다.

   | 크기 | Kev T test |
   |---|---|
   | 0.8B | −1.7pp (ns) |
   | 2B | +5.8pp (p = 0.001) |
   | 4B | +3.3pp (3 seed) |

   세대 효과는 2B 이하에서 유의하지 않다. 따라서 실용 하한은 4B다.
4. **요인 분리 (논의 6.1(1) 해소)**: Qwen3-4B, 같은 데이터로 비교했다.

   | 구조 | Kev T test |
   |---|---|
   | frozen 양방향 head | 0.411 |
   | 옵션 공동 읽기, LoRA 없음 | 0.630 |
   | 옵션 공동 읽기 + LoRA | 0.715 |

   transfer 이득의 약 70%는 옵션을 함께 읽는 구조에서 온다.
5. **DeepSWE 누출 (한계 10 해소)**
   - 학습 풀에 held-out 38과제(132,352 step)가 들어 있다.
   - 목록으로 제외하고 재학습한 5 seed의 결과는 81.6% ± 1.9(30–32/38)다.
   - 따라서 81.6%는 누출이나 운 좋은 seed 때문이 아니다. 우리의 비판은 "decidable 13과제로는 우위가 약하다"는 점으로 좁힌다(seed별 p는 0.014–0.18).
6. **Jev와 문항별 짝지은 비교**: JevBench가 공개한 문항별 결과를 썼다.
   - 차이는 hard 등급에 몰려 있다: 0.450 vs 0.730, Jev만 맞힌 문항 32개, MiDM만 맞힌 문항 1개. original 등급(−6.9pp, CI가 0을 포함)과 easy 등급(동률)은 사실상 대등하다.
   - 입력을 4096으로 늘려도 hard는 +0.9pp에 그친다.
   - 긴 문서 학습셋 long_v1(ContractNLI, HotpotQA yes/no, 문서 덧붙이기 증강)으로 재학습한 -bxL은 long dev(215문항)에서 0.856이다. 같은 dev에서 -bx는 입력 1,024 토큰일 때 0.656, 재학습 없이 4,096 토큰으로 늘렸을 때 0.786이었다. 그러나 JevBench hard는 0.441이고 transfer는 −2.0pp(ns)여서 채택하지 않는다.
   - 결론: 남은 격차는 입력 길이가 아니라 한 번의 forward로 하는 점수화가 다단계 추론에 갖는 한계다.
7. **SQL·코드 선택기로의 전이 (Study115, 오프라인)**: 저장된 후보와 정답 표시만 썼고, DB는 열지 않았다. λ는 Spider dev로 정했다.

   | 평가 | RAG 없음 | RAG 있음 | Solar |
   |---|---|---|---|
   | Spider test, 다수결 0.773 | 0.756 | 0.783 | 0.793 (RAG 있음 대비 −1.0pp, ns) |
   | holdout, 다수결 0.799 | 0.805 | 0.817 | 0.798 (RAG 있음 대비 +1.8pp, ns) |

   LiveCodeBench hard 80문항에서는 다수결 0.188 대비 0.288이다(+10pp, p = 0.004).
   RAG가 선택기를 강화한다. 다만 zero-shot으로는 Solar를 유의하게 넘지 못한다.
8. **속도와 cascade 비용 (한계 9 수정)**: `paper/analysis/speed/` 참조.
   - 묶음 처리 시 배포 모델 4B-q35는 70 ms/문항이다.
   - batch 1 latency는 0.6B 151 / 1.7B 156 / 4B-q35 309 / 8B 196 ms다.
   - 실측 기준으로 cascade는 8B 단독보다 빠르지 않다(237–309 ms).
   - Jev API는 p50 0.665 s다(네트워크 포함).
   - HF 패키지 검증은 통과했다(td_holdout 0.835, 학습 저장소 값 0.833).

**코드·모델 공개(v0.1.0):** Zenodo DOI 10.5281/zenodo.23084584 · GitHub YeoHoonYun/midm-decision-models · HF yunicro/MiDM-4B-q35-e1-bx (모두 public)

주 출처:
- 실험 113: `experiments/113_clm_reproduction_20260928/README.md`, `results/`, `runs/search/*/train_summary.json`
- 실험 114: `experiments/114_local_model_router_20260928/README.md`, `DECISIONS.md`, `results/cascade*_*.json`, `ood/`
- 선행연구: 같은 폴더의 `README.md`, `references.json`

---

## 1. Abstract

Stanford/NVIDIA의 CLM-8B(frozen Qwen3-8B + contrastive state/action head)는 DeepSWE held-out 38과제의 best-of-4 선택에서 81.6%를 보고했다. 공개 데이터와 released head로 이 수치를 **정확히 재현**했다(31/38).

그러나 이 81.6%는 보이는 것보다 약하다.
- 38과제 중 21개는 모든 후보가 통과하고 4개는 모든 후보가 실패한다. selector가 결과를 바꿀 수 있는 과제(decidable)는 **13개**뿐이다.
- 81.6%와 random 기대치 73.7%의 차이는 **3과제**다.
- DeepSWE fine-tuning 전 base head(CLM-v0.1-8B)는 27/38 = 71.1%로 **random보다 낮다**.

이어서 ≤8B 공개 모델을 소비자 GPU 두 장(RTX 3090, RTX 2080 Ti; Windows, vLLM 없음)에서 두 방식으로 학습해 typed-decision 벤치마크 여러 개로 비교했다.
- **(a) CLM식 frozen encoder + 작은 head.** z-score 입력 표준화를 더하면 typed-decisions test가 0.766(3-seed 평균)이다. 그러나 학습하지 않은 source에서는 0.37–0.41로 붕괴한다.
- **(b) Kev식 QLoRA pointer cross-encoder.** 8B, 2 epoch에서 typed-decisions 0.805, Kev transfer-v4 test 0.772, JevBench 공개 231문항 0.706이다.

마지막으로 pointer 모델들을 작은 모델 → 큰 모델 cascade로 묶었다. in-distribution에서는 최상위 모델과 0.15pp 이내의 정확도를 test 비용 75%로 낸다. 반면 학습하지 않은 source에서는 작은 모델의 과신 때문에 약 3pp를 잃는다.

모든 선택은 dev에서만 했고, 각 test는 한 번 읽었다. 예외(cascade 분석을 위한 재읽기)는 따로 밝힌다.

---

## 2. 기여 (신규성 순)

**C1. CLM DeepSWE verifier 주장의 독립 재현과 감사** (가장 방어하기 쉬운 기여)

- **재현.** released DeepSWE head로 31/38 = 81.579%를 정확히 재현했다. 공개 데이터셋이 이전 이름(`tarsur385/deepswe-prm-embeddings-8k`)으로 남아 있는 것을 찾아 사용했고, head와 held-out 목록의 SHA-256이 `verification.json`과 일치한다.
- **decidable subset 분해.** 21 all-pass, 4 all-fail이라 decidable은 13개다. released head는 10/13, random 기대치는 약 7/13, oracle은 13/13이다. 즉 헤드라인 차이 7.9pp는 3과제다. Bo-N selector 평가에서 decidable n을 따로 보고하는 규범은 선행연구 검토에서 명시적으로 찾지 못했다. 학습 쪽에서는 DAPO가 비슷한 논리를 쓴다(§7).
- **base head < random.** DeepSWE fine-tuning 전 CLM-v0.1-8B head는 27/38(71.1%)로 random 73.7%보다 낮다. decidable 기준으로는 6/13이다. 즉 성능은 과제 특화 fine-tuning에서 나온다.
- **데이터 위험 두 가지.**
  - (i) eval 보상(통과 여부)이 공개 데이터셋에 포함되어 있다.
  - (ii) train 임베딩 풀에 held-out 과제가 섞여 있고, 과제 목록 필터로만 분리된다.
  - 이것은 **누출 위험**이지, 입증된 누출이 아니다. leave-task-out 재학습으로 확인해야 한다(§8).

**C2. 같은 데이터와 하드웨어에서 frozen-encoder head와 QLoRA pointer cross-encoder를 여러 벤치마크로 비교**

- **frozen head(z-score 포함).** in-distribution에서는 경쟁력이 있다: 4B 기준 typed-decisions test 0.759, Kev decision-v7 dev 0.715. 그러나 학습하지 않은 source에서는 붕괴한다: Kev transfer-v4 test 0.411, JevBench 공개 0.411.
- **QLoRA pointer.** 같은 base와 학습 데이터로 transfer를 크게 복구한다.
  - 1 epoch: 4B 0.716, 8B 0.764.
  - 2 epoch(stage B): 4B 0.725, 8B 0.772.
  - 0.6B→8B 크기 사다리에서 모든 suite가 단조 증가한다.
- **부수 관찰(증분적).** CLM 공식 레시피 앞에 z-score 한 줄을 넣으면 CV가 0.738에서 0.774로 오르고(40 epoch, l2 대비), test는 0.766이다. CLM 저자 PR #2가 보고한 0.685보다 약 8pp 높다. 방법 자체는 알려져 있다(Timkey & van Schijndel 2021 등). 이 맥락에서의 효과 크기만 새롭다.

**C3. Cascade: in-distribution 비용 절감, out-of-distribution 과신**

- dev(in-distribution)에서 임계값을 고른 cascade(stage-B tier, 0.6B → 4B → 8B, 0.80/0.50)는 td_test에서 8B 단독과 0.15pp 차이다. test 비용은 8B의 75%이고, dev에서는 42%였다.
- Kev transfer test에서는 cascade가 최상위 단일 모델보다 약 3pp 낮다. 작은 tier가 새 source에서 과신해 충분히 escalate하지 않기 때문이다.
- 운영 결론(D1a): cascade는 등록된 학습 domain에만 쓰고, 미지 domain은 최상위 tier로 바로 보낸다.
- kNN 임베딩 거리 기반 OOD gate는 tag 없이 이 결정을 근사한다(§4.6, 제안 단계).

**(부) 엔지니어링.**
- vLLM 없이 Windows에서 CLM 서버(repo 코드 무수정)를 transformers 임베더로 구동했다. warm head-only p50은 0.9 ms다.
- 공유 GPU 큐(Huey)와 로컬 게이트웨이(LiteLLM + decision router)를 만들었다.

---

## 3. 실험 설정

### 3.1 하드웨어·소프트웨어
- **GPU.** RTX 3090 24 GB(다른 실험의 Ollama와 공유)와 RTX 2080 Ti 11 GB.
- **소프트웨어.** Windows 11, Python 3.12, torch 2.14.0+cu126, transformers 5.17.0, peft 0.21.0, bitsandbytes 0.50.2. vLLM은 Linux 전용이라 쓰지 않았다.
- **CLM repo(`bb42c6c`)는 수정하지 않았다.** `run_choice_hf.py`로 감싸 in-process transformers 임베더를 쓰게 했다. 토큰 레시피는 같다(special token 없음, tail 유지, 최종층 last-token, L2).
- **정밀도.** 2080 Ti(Turing)는 fast bf16이 없어 fp16을 썼다. 3090 학습(stage B 8B, Qwen3.5)은 bf16이다. CLM released head는 vLLM bf16 임베딩으로 학습됐다.

### 3.2 모델
- **Frozen head.** CLM `finetune.py --task choice` 계열이다. 최종 설정은 z-score(train 통계), InfoNCE, width 1536, depth 3, 40 epoch.
- **Pointer(QLoRA).**
  - 입력: 질문당 한 시퀀스(state와 지시, 이어서 `- key: text` 옵션 줄).
  - 점수: 각 옵션 줄 끝의 hidden state → linear pointer head → softmax. 학습은 soft-target CE이고, 옵션 순서를 섞는다.
  - 설정: 4-bit NF4, LoRA r16(모든 attention/MLP projection), lr 2e-4, 4096-token 배치.
  - epoch: 1 epoch(cycle 5), 2 epoch(stage B). 체크포인트 규칙은 사전에 `final`로 고정했다.

### 3.3 데이터

| suite | 설명 | n (평가 단위) |
|---|---|---|
| DeepSWE held-out | Opus-5 rollout, 과제당 4개, Qwen3-8B 임베딩(공개) | 38 과제 / 151 rollout |
| typed-decisions | 합성, teacher 라벨, **test gold 공개** | test 400 rows = 2000 결정 |
| Kev decision-v7 | 10개 공개 데이터셋과 생성 policy 예제 | dev 1468 |
| Kev transfer-v4 | 학습에 없는 source | dev 764, test 764 |
| Kev transfer-v9 | 학습에 없는 source | dev 1264 |
| JevBench 공개 | original 72, easy 48, hard 111 | 231 |

pointer 학습 데이터는 td_train(10% 행 holdout 제외)과 Kev decision-v7 train을 합친 20,976 질문, 4.47M 토큰이다.

### 3.4 선택 프로토콜
- **선택은 dev에서만 했다.** 각 단계의 dev는 다음과 같다.
  - frozen head: row 단위 5-fold CV
  - multi-benchmark head: Kev dev와 td holdout
  - pointer: 사전 고정 규칙
  - stage A: src_holdout 포함 dev
- **각 test는 한 번 읽었다.** 기록된 예외는 세 가지다.
  - 0.6B smoke run이 td_test를 읽었다(0.7225). 선택에는 쓰지 않았다.
  - 8B 1 epoch 최종 평가는 외부 프로세스 kill로 중단돼 재실행했다.
  - cascade와 OOD 분석을 위해 확률 덤프로 test suite를 다시 읽었다.
- **ledger.** (2026-10-01) 기계 판독용 test-read ledger(SHA-256, 47개 파일)를 `results/ledger/`에 만들었다.

### 3.5 Stage A (adapter 탐색, Qwen3-1.7B proxy)
- `--holdout-sources boolq mnli sst5`로 세 Kev source를 학습에서 뺐다. 해당 kev_dev 276행은 미지 source의 대리 지표 `src_holdout`으로 썼다.
- 선택 점수는 td_holdout, kev_dev500, src_holdout의 평균이다. test는 읽지 않는다.

### 3.6 운영 교훈
- 오늘 기록된 nvlddmkm event 153을 처음에는 driver reset이나 TDR로 진단했지만, **틀린 진단이었다.** 모든 event가 CUDA 프로세스 강제 종료 시각과 일치한다.
- 15:45에는 외부에서 광범위한 kill이 일어나 두 GPU의 작업과 큐 worker가 동시에 죽었다.
- 교훈: 프로세스는 PID로 정확히 지정해서만 종료하고, 큐에 retry, lost 감지, resume을 둔다(`experiments/_ops/GPU_DRIVER_RESET_PROCEDURE.md`).

---

## 4. 결과

### 4.1 DeepSWE held-out 38 (Bo4, last-12 mean)
출처: `results/r1_deepswe_heldout38_released_head.json`, `results/r2_deepswe_heldout38_zeroshot_v0.1_head.json`

| selector | 해결 / 38 | 정확도 | decidable 13 |
|---|---|---|---|
| released DeepSWE head | 31 | **0.8158** (CLM 보고와 일치) | 10/13 |
| CLM-v0.1-8B base head, zero-shot | 27 | 0.7105 | 6/13 |
| random pick (기대값) | 28 | 0.7368 | 약 7/13 |
| oracle | 34 | 0.8947 | 13/13 |

보도된 Jev 71.1%는 우리 base head와 같은 값이다. 38과제 기준 1과제는 2.6pp다.

### 4.2 typed-decisions: frozen encoder head
출처: `results/choice_summary.md` (기본 레시피, 3 seeds)

| encoder | patience 5 | patience 20 |
|---|---|---|
| Qwen3-0.6B | 0.6430 | 0.6508 |
| Qwen3-1.7B | 0.5938 | 0.6390 |
| Qwen3-4B | 0.6937 | 0.6910 |
| Qwen3-8B (scratch) | – | 0.6973 |
| Qwen3-8B, released CLM head warm start | – | 0.6793 |
| Qwen3-8B, released CLM head zero-shot | – | 0.3580 |

Head 탐색(Qwen3-8B, 5-fold CV). 출처: `results/search/`

| 설정 | best CV |
|---|---|
| 기본 레시피, 20 ep (l2) | 0.7003 |
| 40 ep (l2) | 0.7380 |
| width 3072 / depth 2 | 0.7405 |
| **z-score, 40 ep** | **0.7743** |
| z-score, 80 ep 변형 | 0.7665–0.7728 |
| last-token layer 12 / 18 / 24 / 35 | 0.7195 / 0.7700 / 0.7725 / 0.7737 |
| 16–28층 평균 | 0.7770 |

최종 결과(설정 고정 후 test 1회, `results/search/FINAL_test_r1_08.json`)는 다음과 같다.
- seed별 0.7690 / 0.7675 / 0.7620, 평균 **0.7662**, 3-seed 앙상블 **0.7685**.
- 참고 수치(모두 비짝지음):
  - Jev 0.727, meraGPT 0.768: zero-shot
  - Laya-FT 0.766, openJev-verdict 0.771, od1 0.796: 학습 모델
  - CLM PR #2: 0.685

### 4.3 Multi-benchmark: frozen head vs QLoRA pointer
출처: `results/multi/FINAL_qwen3-4b_e40.json`, `results/pointer/FINAL_*.json`

| suite (n) | frozen 4B head | 4B pointer, 1 ep | 참고 (보고값) |
|---|---|---|---|
| typed-decisions test (2000) | 0.759 | **0.789** | Jev 0.727, od1 0.796 |
| Kev decision-v7 dev (1468) | 0.715 | **0.854** | Kev-9B 0.872 |
| Kev transfer-v4 dev (764) | 0.366 | **0.700** | Jev 0.857, Kev-9B 0.822 |
| Kev transfer-v4 test (764) | 0.411 | **0.716** | Kev-9B 0.852 |
| Kev transfer-v9 dev (1264) | 0.370 | **0.634** | – |
| JevBench 공개 (231) | 0.411 | **0.688** | Jev 0.866, Kev 8B 0.714, CLM-8B 0.407 |

Pointer 크기 사다리(1 epoch):

| base | typed-dec. | Kev dev | Kev T dev | Kev T test | T-v9 dev | JevBench 공개 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.736 | 0.785 | 0.597 | 0.585 | 0.525 | 0.576 |
| Qwen3-1.7B | 0.782 | 0.817 | 0.624 | 0.641 | 0.553 | 0.632 |
| Qwen3-4B | 0.789 | 0.854 | 0.700 | 0.716 | 0.634 | 0.688 |
| Qwen3-8B | 0.7945 | 0.857 | 0.726 | 0.764 | 0.657 | 0.697 |

### 4.4 Stage A: adapter 탐색 (Qwen3-1.7B proxy, test 미사용)
출처: `runs/search/A*/train_summary.json`

| run | 설정 | td_holdout | src_holdout | kev_dev500 | 평균 |
|---|---|---|---|---|---|
| A1 | LoRA r16 all, lr 2e-4 | 0.825 | 0.543 | 0.848 | 0.739 |
| A2 | DoRA r16 | 0.798 | 0.533 | 0.866 | 0.732 |
| A3 | rsLoRA r64 (α128, fp16) | – | – | – | 발산(NaN) |
| A4 | attention-only | 0.793 | 0.511 | 0.850 | 0.718 |
| A5 | lr 1e-4 | 0.812 | 0.554 | 0.850 | 0.739 |
| A6 | lr 4e-4 | 0.820 | 0.511 | 0.856 | 0.729 |
| **A7** | **2 epoch** | **0.848** | **0.583** | **0.880** | **0.771** |
| A8 | r64 | 0.803 | 0.478 | 0.852 | 0.711 |

### 4.5 Stage B: A7 레시피(2 epoch), 전체 source
출처: `results/pointer/FINAL_B_qwen3-{4b,8b}-2ep.json`. 8B는 3090에서 bf16, 4B는 2080 Ti에서 fp16으로 학습했다.

| base | typed-dec. | Kev dev | Kev T dev | Kev T test | T-v9 dev | JevBench 공개 (orig/easy/hard) |
|---|---|---|---|---|---|---|
| Qwen3-4B, 2 ep | 0.797 | 0.848 | 0.703 | 0.725 | 0.633 | 0.684 |
| **Qwen3-8B, 2 ep** | **0.805** | **0.864** | **0.729** | **0.772** | **0.661** | **0.706** (0.944 / 1.000 / 0.423) |

- 두 번째 epoch의 효과는 +0.3~1.1pp다. 1.7B proxy에서 본 src_holdout +4pp는 4B·8B transfer로 이전되지 않았다.
- typed-decisions 0.805는 od1 0.796보다 높지만 차이가 SE(약 1pp) 안이다.
- Kev(0.838–0.852)와 Jev(JevBench 0.866)와의 남은 격차는 epoch보다 학습 source의 폭 때문으로 보인다.
- **Qwen3.5-4B-Base, 1 epoch**(bf16, 3090): typed-dec. 0.805, Kev dev 0.861, Kev T dev 0.755, Kev T test 0.764, T-v9 dev 0.675, JevBench 공개 0.710.
  - 같은 크기·같은 epoch의 Qwen3-4B보다 미지 source에서 +4~5.5pp 높다.
  - Qwen3-8B 2 epoch와 비슷한 수준이다.
  - 즉 base 세대가 epoch나 LoRA 변형보다 큰 요인이다.
- **A9**(1.7B, 3 epoch): dev 평균 0.748로 2 epoch(0.771)보다 낮다. src_holdout이 0.583에서 0.518로 떨어졌으므로 과적합이다.
- **Qwen3.5-4B, 2 epoch**(fp16, 2080 Ti, step 1600에서 optimizer를 새로 만들어 재개, 학습 중 메모리 spill 발생): td_holdout 0.868, Kev dev 0.866, typed-dec. 0.802, Kev T dev 0.709, Kev T test 0.734, T-v9 dev 0.646, JevBench 0.706.
  - 1 epoch 대비 in-distribution dev는 올랐지만 미지 source는 −3 ~ −4.6pp 떨어졌다. A9와 같은 방향이다.
  - 단, 정밀도(fp16 vs bf16), optimizer 재시작, spill이 섞여 있어 epoch 효과로 단정할 수 없다.
  - in-distribution dev로는 transfer를 고를 수 없다는 점이 여기서도 다시 확인된다.
- **Qwen3.5-4B, 2 epoch 재실행**(B5: 3090 bf16, 재시작·spill 없음, 1 epoch와 같은 조건): Kev T dev 0.734, Kev T test 0.736, T-v9 0.647, JevBench 0.684. 1 epoch보다 −2.1 ~ −2.8pp로, **두 번째 epoch가 transfer를 떨어뜨린다는 것이 교란 없이 확인됐다.** Qwen3.5는 1 epoch를 쓴다.
- **학습 source 확장**(D1: Qwen3-4B 1 epoch, public-pool-v6의 arc/openbookqa/csqa 3,000문항 추가, eval과 텍스트 중복 0):
  - 전체: Kev T test +3.0pp, T-v9 +1.9pp, JevBench +0.5pp.
  - source별로 보면 근거리 전이다. mmlu +18.1, mmlu_pro +6.5, sciq +6.9로 오른 반면 paws −7.5, qnli −5.0, composition −6.2로 떨어졌다.
  - 즉 한 과제 유형을 더하면 그 유형만 오르고 문장쌍 과제는 희석된다. transfer를 넓히려면 유형을 균형 있게 넓혀야 한다.
  - 이 run은 Kev보다 넓은 데이터를 쓴 것이므로 Kev와의 동일 조건 비교가 아니다.

### 4.5b Qwen3.5 확장: 데이터 구성, 크기, TTA (2026-10-01 추가)
출처: `paper/analysis/results_master.md`, `per_source.md`, `stats/stats.md`. 모두 3090 bf16, 1 epoch, 같은 레시피다. Δ는 paired row-clustered bootstrap 결과다.

| 비교 | td_test Δ (p) | Kev T test Δ (p) | 해석 |
|---|---|---|---|
| Qwen3-8B e2 → Qwen3.5-4B e1 | +0.0 (1.00) | −0.8 (0.69) | 절반 크기로 동등 |
| Qwen3.5-4B e1 → e2 | −0.4 (0.48) | −2.9 (0.053) | 두 번째 epoch가 transfer를 해침 (T-v9 −2.8, p = 0.002) |
| + breadth_v1 (E1) | −1.4 (0.053) | +1.4 (0.41) | 효과 없음 |
| + breadth_v1 + arc/obqa/csqa (E2, "-bx") | −1.7 (0.024) | **+3.4 (0.042)** | transfer↑, in-distribution↓ (trade-off) |
| TTA 4 orders | −0.4 (0.15) | +0.1 (0.88) | 효과 없음 (shuffle 학습이 이미 순서 편향을 제거) |
| Qwen3-0.6B → Qwen3.5-0.8B | +3.5 (<0.001) | +0.7 (0.78) | 작은 크기에선 세대 이득이 in-distribution에만 |
| Qwen3-1.7B → Qwen3.5-2B | +0.6 (0.39) | +2.1 (0.29) | 유의하지 않음 |
| Qwen3.5 0.8B → 2B | +1.8 (0.007) | +7.1 (0.001) | |
| Qwen3.5 2B → 4B | +1.7 (0.014) | **+10.2 (<0.001)** | transfer는 4B에서 크게 뜀 |

- **-bx의 source별 이득**(Kev T test, 4B-q35-e1 → -bx): mmlu 0.560 → 0.629, composition 0.698 → 0.854, deadline 0.700 → 0.725, paws 0.800 → 0.812. sciq는 0.974 → 0.940으로 떨어진다. td_test에서는 customer_service(−3.6)와 invoice_processing(−5.0)에서 잃는다.
- **kevT9_dev는 kevT_dev를 포함한다**(공통 source의 수치가 동일). 독립 transfer 셋이 아니라 확장으로 보고한다.
- **진행 중.** q35-4B seed 1·2, -bx seed 1·2(약 16:15 완료 예정), -bx 2B·0.8B. -bx의 +3.4pp가 seed 간에 재현되는지가 핵심 확인 사항이다.

### 4.6 Cascade (실험 114)
출처: `results/cascade_*.json`(1 epoch tier), `results/cascadeB_*.json`(stage-B tier), `DECISIONS.md`

- 비용은 **잠정 가중치**다(0.6B=1, 1.7B=2, 4B=6, 8B=12).
- 규칙: dev 정확도가 최선 단일 tier의 0.5pp 이내인 설정 중 평균 비용이 가장 낮은 것.
- dev = kev_dev + td_holdout, 2,068행.
- 첫 계산에는 행 키 중복 버그가 있었고(§6.7), 아래는 수정 후 재계산한 값이다.

| tier 세트 | cascade | thr | dev acc (최선) | dev 비용 | td_test (최상위 단독) | td_test 비용 | Kev T test (최상위 단독) |
|---|---|---|---|---|---|---|---|
| 1 ep | 0.6B → 4B | 0.62 | 0.842 (0.846) | 53% | 0.785 (0.788) | 80% | 0.690 (0.715) |
| 1 ep | 0.6B → 8B | 0.62 | 0.845 (0.849) | 45% | 0.793 (0.795) | 71% | 0.728 (0.764) |
| 1 ep | 1.7B → 8B | 0.70 | 0.844 (0.849) | 54% | 0.794 (0.795) | 88% | 0.736 (0.764) |
| 1 ep | 0.6B → 4B → 8B | 0.64 / 0.30 | 0.846 | 28% | 0.783 (0.795) | 42% | 0.691 (0.764) |
| stage B | 0.6B → 4B → 8B | **0.80 / 0.50** | 0.852 (0.857) | 42% | **0.8035 (0.805)** | **75%** | 0.741 (0.772) |
| stage B | 0.6B → 8B | 0.76 | 0.853 | 59% | 0.803 (0.805) | 87% | 0.770 (0.772) |

관찰:
- (i) in-distribution에서는 정확도 유지가 dev에서 test로 이전됐다. 반면 비용 절감은 dev보다 작다. test의 escalation 비율이 더 높기 때문이다.
- (ii) 새 source에서는 작은 tier의 과신으로 2.5–3.7pp(1 ep)와 약 3pp(stage B)를 잃는다.
- (iii) OOD gate(`ood/`)의 설정과 결과는 다음과 같다.
  - 방법: base Qwen3-0.6B state 임베딩의 kNN-10 거리. 임계값은 dev 95% 분위수로 정했다.
  - 결과: tag 없이 td_test를 모두 cascade로 보내고, 미지 source의 81%를 8B로 직행시켰다. 8B 단독보다 약 1.5pp 낮다.
  - 한계: 1 ep tier와 수정 전 임계값 0.58로 계산했고, 인코더와 점수 방식은 eval 수치를 본 뒤에 골랐다. 재계산이 필요하다.

### 4.7 서빙과 T-Rex
- **서버 재생**(400 test rows → `/v1/systemone`, 0.6B head): 정확도 0.6515(오프라인 0.6490). head-only cold p50 4.0 ms, warm 0.9 ms. 8 동시 요청에서 500 req/s. head 없는 raw cosine은 0.3485다.
- **Chrome T-Rex**(4B multi frozen head, 5 seeds × 60 s): 생존 **0/5**, planner 일치 0.014–0.030, latency p50 16.5 ms.
  - CLM-8B 공개 보고는 5/5, 일치 0.658, shield 개입 4,883회다. 우리 16.5 ms p50은 60 FPS 프레임 간격을 따라가는 것으로 보인다.

---

## 5. 음성·무효 결과
1. **T-Rex 실패.** head가 "Unsafe ... Collision." 옵션에 0.99를 준다. head 없는 raw cosine은 "jump"를 고른다.
2. **Frozen head의 transfer 붕괴.** in-distribution 0.715–0.759가 미지 source에서 0.366–0.411로 떨어진다. Brier 0.88–0.93, ECE 0.33–0.38이다.
3. **중간층 이득 없음.** 16–28층 평균 0.777 vs 최종층 0.774. Skean et al.(ICML 2025)과 반대 방향이고, gavel-decide-4b의 자기 보고와는 일치한다.
4. **DoRA, r64, rsLoRA, attention-only는 LoRA r16보다 낫지 않다.** rsLoRA r64(α128)는 fp16에서 발산했다.
5. **워크플로별 specialist가 단일 head보다 나쁘다.** 4B 0.631 vs 0.691.
6. **Released CLM head의 warm start는 해롭다.** 0.679 vs 0.697, zero-shot은 0.358(prior 0.470 미만).
7. **2 epoch의 transfer 이득이 크기에 따라 사라진다.** 1.7B src_holdout +4pp, 4B·8B Kev transfer +0.9·+0.8pp.

---

## 6. 타당성 위협
1. **typed-decisions는 합성, teacher 라벨이고 test gold가 공개되어 있다.** Laya 카드는 teacher self-agreement ceiling을 0.735로 언급한다. 우리 0.79–0.80은 teacher 분포를 모사하는 데 과적합했을 수 있다.
2. **n이 작다.**
   - DeepSWE는 decidable 13개다.
   - JevBench 공개 231문항 중 easy 48은 모든 pointer가 1.000으로 포화했다.
   - src_holdout은 276문항이다.
   - typed-decisions는 400 case로 cluster되어 있다.
   - pointer는 크기별 **seed 1개**다.
3. **비교 모델 수치는 리더보드와 카드 수치다.** 비짝지음이고 대부분 자기 보고다. zero-shot(Jev, meraGPT)과 fitted 모델의 비교는 비대칭이다.
4. **임베딩 정밀도가 다르다.** 우리는 fp16 transformers, CLM head는 vLLM bf16 임베딩으로 학습됐다. released head의 zero-shot과 warm-start 수치는 우리 쪽에 불리하게 편향됐을 수 있다.
5. **test를 다시 읽었다.**
   - cascade와 OOD 분석이 확률 덤프로 test를 재읽기했다. 이 재평가 수치는 1회 평가와 소수 셋째 자리에서 다르다(예: 4B Kev T test 0.7160 vs 0.7147).
   - 0.6B smoke run이 td_test를 읽었다.
   - ~~기계 판독 ledger는 없다.~~ (2026-10-01) `results/ledger/test_read_ledger.{jsonl,md}`에 held-out을 읽은 결과 파일 47개의 SHA-256을 기록했다. 사본은 `paper/analysis/ledger/`에 있다.
6. **1–2 epoch만 학습했다.** 4B 1 epoch run은 step 400에서 optimizer 상태를 새로 만들어 재개했다.
7. **(수정 완료) cascade 행 키 중복.** 첫 `calibrate_cascade.py`는 `(suite, qid, group)`으로 키잉해 행이 덮어써졌다. dev는 2,068 중 1,720, Kev T는 764 중 544만 쓰였다. 키를 `(suite, 위치)`로 바꾸고 tier 정렬 검사를 넣어 재계산했다. §4.6은 수정 후 수치다.
8. **(수정 완료) 선택 기록 부정확.** multi head 선택에서 README는 m1_06(0.753)을 최선으로 적었지만, m1_09(0.755)가 0.002 높았다. 더 단순한 기본값을 택한 것인데 당시 기록하지 않았다. 차이는 잡음 수준이고, 기록은 정정했다.
9. **비용은 잠정 가중치다.** 실측 latency가 아니고, Windows와 공유 GPU 환경이다.
10. ~~Qwen3 계열만 완료했다.~~ (2026-10-01) Qwen3.5 0.8B, 2B, 4B를 완료했다(§4.5b). Qwen3.5-9B는 8B 제한을 넘어 하지 않았다.
11. **pointer seed.** Qwen3-4B e1은 3 seed(Kev T test 0.723 ± 0.006)다. Qwen3.5-4B와 -bx는 seed 실험이 진행 중이다.
12. **아키텍처 신규성.** 옵션별 마지막 토큰 점수는 jina-reranker-v3, LS-LLaMA, FIRST와 겹친다. 신규성은 적용 영역과 통제 실험에 둔다.

---

## 7. 관련 연구 (검증된 참고문헌만; [PR] 동료 심사, [arXiv] 프리프린트, [GH] GitHub·HF, [blog] 블로그·뉴스)

**CLM과 Jev 계열 생태계**
- **재현 대상.** CLM repo와 모델 카드 [GH], PR #2 [GH], Issue #15(state embedding collapse) [GH]. CLM 논문은 찾지 못했다.
- **보도.** VentureBeat, MarkTechPost, AI타임스 [blog].
- **경쟁 시스템.** TypeSafe Jev [blog/docs], Kev repo와 kev-9b 카드 [GH].
- **기타 시스템** [GH]: Open-Jev, JevK5, openJev-verdict-2.0, Laya, od1-typed-decisions, gavel-decide-4b(frozen Qwen3-4B + CPU MLP head), typecastlm, RYOTIDE.
- **벤치마크.** JevBench [GH], typed-decisions 카드 [GH], AIMultiple [blog].
- **동료 심사 논문이 없다.** 이 생태계에는 동료 심사 논문이 없고, 관련 arXiv는 응용뿐이다. 그중 JEV-as-a-Judge(2609.26550)의 "확신하면 수락, 아니면 escalate"가 우리 cascade와 가장 가까운 선행이다.

**frozen LLM 표현과 작은 head**
- LLM2Vec [arXiv], NV-Embed [PR, ICLR 2025], E5-Mistral [PR, ACL 2024], Qwen3 Embedding [arXiv]
- Layer by Layer [PR, ICML 2025]
- 소형 LLM + LR [arXiv 2408.03414]
- Reusing Embeddings [arXiv 2502.04357]

**이방성과 표준화**
- All-but-the-Top [PR, ICLR 2018], BERT-whitening [arXiv], SimCSE [PR, EMNLP 2021]
- **All Bark and No Bite** [PR, EMNLP 2021]: z-score 효과의 가장 가까운 선행
- Massive Activations [arXiv]

**bi-encoder와 cross-encoder**
- Poly-encoders [PR, ICLR 2020], CLIP [PR, ICML 2021], Contrastive RL [PR, NeurIPS 2022]

**Verifier와 평가 방법론**
- Let's Verify Step by Step, Generative Verifiers, Large Language Monkeys, R2E-Gym, RewardBench 2 [arXiv]
- DAPO [arXiv]: decidable subset에 해당하는 학습 쪽 대응물
- Adding Error Bars to Evals, Quantifying Variance [arXiv]
- SWE-Bench Illusion [PR, ICSE-SEIP 2026]
- 보정: Guo et al. [PR, ICML 2017]

---

## 8. 출판 전에 필요한 실험
1. **통계 검정.**
   - DeepSWE: exact permutation과 paired bootstrap(released vs random, vs base).
   - typed-decisions: case 단위 cluster bootstrap.
   - 모델 간 비교: paired McNemar와 bootstrap.
2. **DeepSWE 누출 확인.** held-out 과제를 임베딩 풀에서 물리적으로 제거하는 leave-task-out 재학습.
3. **같은 데이터의 LoRA·Kev 기준선.**
   - Kev 레시피(Qwen3.5, shared state cache, fitted temperature)로 같은 데이터를 재학습해 짝지은 비교를 만든다.
   - z-score + 로지스틱 회귀, Qwen3-Reranker, full FT 기준선도 추가한다.
   - 학습 source를 넓히는 실험(Kev와의 transfer 격차 원인 검증)도 필요하다.
4. **Linux/vLLM 실측 latency.** 이 값으로 cascade 비용 가중치를 바꾼다.
5. **Sealed 평가.** JevBench sealed 문항으로 평가한다(공개 231 중 easy 48은 포화).
6. **Cascade 보강.**
   - OOD gate를 stage-B tier와 수정된 임계값으로 재평가한다.
   - source-held-out dev에서 temperature scaling을 적용한다.
7. **z-score ablation.** none / L2 / center / z-score / whitening / all-but-top-k를 크기 3종 이상, 벤치마크 3개 이상에서 비교한다.
8. **재현성 산출물.**
   - 해시가 달린 test-read ledger
   - pointer 다중 seed
   - Qwen3.5 결과(진행 중)
   - stage C(어댑터 구성) 결과(진행 중)
