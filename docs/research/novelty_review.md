# CLM 재현 + frozen-encoder decision head 실험: 선행연구·신규성·출판 가능성 검토 (2026-09-28)

대상 실험: `113_clm_reproduction_20260928\README.md`
참고문헌 목록(기계 판독용): `references.json` (같은 폴더)

표기 규칙: **[PR]** peer-reviewed, **[arXiv]** 프리프린트, **[GH]** GitHub/HF model·dataset card, **[blog]** 블로그·뉴스.
모든 항목은 이번 조사에서 직접 열었거나 검색 결과로 확인한 URL만 사용했다. 직접 본문을 확인하지 못한 것은 "(미확인)"으로 표시했다.
검색은 2026-09-28 기준 웹 검색·페치만 사용했고, 어떤 곳에도 게시·연락하지 않았다.

---

## 0. 결론 요약 (한 문단)

"frozen LLM 임베딩 + 작은 학습 head"라는 방법 자체는 **새롭지 않다**(CLM-8B 자체, gavel-decide-4b, 학계의 linear probing·embedding 기반 reward model). typed-decisions 0.7685도 **SOTA가 아니다**: 같은 공개 train split에 맞춘 specialist들이 이미 0.766(Laya-typed-decisions), 0.771(openJev-verdict-2.0), 0.796(od1-typed-decisions)을 보고했다. z-score(표준화)가 이상치 차원을 보정한다는 사실도 이미 알려져 있다(Timkey & van Schijndel 2021, BERT-whitening, all-but-the-top). 다만 **CLM식 InfoNCE head 앞에서 z-score 하나로 공식 레시피보다 +7~8pp**(CLM 저자 PR #2의 0.685 대비 0.766)라는 수치는 찾은 범위에서 누구도 보고하지 않았다. 가장 방어하기 쉬운 기여는 **CLM DeepSWE verifier 주장에 대한 재현·감사(audit)**다. 38개 중 13개만 selector가 결과를 바꿀 수 있다는 점, fine-tuning 전 base head가 random보다 낮다는 점, train 임베딩 풀에 held-out 과제가 섞여 있다는 점, eval 보상 데이터가 공개되어 있다는 점은 찾은 범위에서 아무도 지적하지 않았다. 현실적인 출판 형태는 **arXiv 기술 보고서 + 워크숍**(재현성·평가 방법론 트랙)이다. 메인 학회는 다중 벤치마크, 강한 baseline(LoRA/full FT/cross-encoder/로지스틱 회귀), 통계 검정을 갖춰도 어렵다.

---

## 1. 생태계 (A): "Jev-class typed decision model" 현황

전부 2026-09 중순 이후 나온 것이다. Jev는 2026-09-15에 공개됐다. **대부분 GitHub/HF 카드와 블로그이고 동료 심사된 논문은 없다.** 관련 arXiv 프리프린트는 Jev를 "응용"한 것들(§1.3)뿐이다.

### 1.1 주요 시스템

| 시스템 | 방법 | 데이터 | 보고 수치 | 논문 |
|---|---|---|---|---|
| **TypeSafe Jev** 1.13 [blog/docs] | 비공개. Archer Hume의 역공학 글은 "shared state prefix + 질문별 병렬 branch + prediction head" 구조로 추정(RLCD) | 비공개 | typed-decisions 0.727, JevBench v1.4.2.2 63.29(4위), Kev transfer-v4 0.857 | 없음 |
| **CLM-v0.1-8B** (Stanford/NVIDIA) [GH/HF] | frozen Qwen3-8B, last-token, state/action projection head 2개, bidirectional InfoNCE | 60M Nemotron QA → 30M 합성 hard negative → 1M agent trajectory | DeepSWE 81.6%(held-out 38, Bo4), TB 2.1 87.6%(held-out 30). typed-decisions는 PR #2에서 **0.685±0.001**(warm start, 3 seeds). JevBench 8.6(#47) | Notion 블로그뿐, arXiv 없음. Issue #15: README 예제 재현 불가, "state embeddings collapse" 보고 |
| **gavel-decide-4b** (JevBench #118) [GH] | **frozen Qwen3-4B-Base, L36 last token, pairwise MLP 2560→512→256→1 (1.44M), CPU 학습** + meta-calibration. 정규화 언급 없음. L36 > L20(0.840 vs 0.747) | JevBench 공개 항목 689쌍 + MNLI 1,720쌍 | 공개 231문항 74.5%. 단 92.2%가 학습과 겹침. 겹치지 않는 18문항은 3/18, MNLI만 쓰면 0.529. 저자 스스로 "sealed 점수는 훨씬 낮을 것"이라고 명시 | 없음 |
| **Kev** (jaredpalmer) [GH/HF] | Qwen3.5/3.8 + LoRA r16 + pointer head, 질문별 row + shared state cache, fitted temperature | decision-v7 (10개 공개 데이터셋 1만 + 생성 policy/rule 예제). 생성분은 비공개 | transfer-v4(새 source) Kev-27B 0.848 vs Jev 0.857, Kev-9B 0.822. MMLU 0.74 vs Jev 0.90 | 없음 |
| **Open-Jev** (Zefan Cai) [GH/site] | Qwen 2B/9B/27B + LoRA(r8) + scalar head + temperature | 자체 데이터(HF 공개) | JevBench 공개 231: 27B 85.3% vs Jev 86.6%. 리더보드 종합 10~11점(#77-78) | 없음 |
| **OpenJev** (razorback16) [GH] | DiffusionGemma 26B-A4B, frozen, option 확률 readout | – | BANKING77 66.9%(SOTAAZ 측정) | 없음 |
| **JevK5** [GH/HF] | Qwen3.5-4B + distilled LoRA(merge), option-letter logit readout + temperature | distillation | JevBench v1.4 62.04(Jev 63.29 다음) | 없음 |
| **Laya** (Convai) [GH/HF] | ModernBERT-large 421M, RLCD(REINFORCE + proper scoring rule + soft CE) | – | zero-shot typed-decisions **0.362**. **typed-decisions train에 fine-tune하면 0.766**(Brier 0.062). AIMultiple 브라우저 과제 0/50 | 없음 |
| **openJev-verdict-2.0** [GH] | ModernBERT-base 150M full FT, 별도 confidence head | typed-decisions train | **0.771**, Brier 0.0636, ECE 1.44%. 체크포인트는 받을 수 없음(issue #2) | 없음 |
| **od1-typed-decisions** (mvbalaji) [HF] | Qwen3.5-4B **full FT**(Open Decider) + typed head, typed-decisions로 추가 FT | typed-decisions train | **0.796**, KL 0.082, Brier 0.045 | 없음 |
| **meraGPT Decider 1** [site/dataset card] | 비공개. Qwen3.5 fine-tune 계열로 추정(미확인) | – | typed-decisions **zero-shot** 0.768 | 없음 |
| **RYOTIDE** [GH] | frozen 공개 LLM에서 한 번의 forward로 option-marker logit readout. 47회 실행 | 학습 없음 | JevBench 공개 231 | 없음 |
| typecastlm / Decision-0 / Cygnet [GH issues] | frozen Qwen3.5-4B/Gemma, gradient 없는 "computed head" 또는 logit readout | – | JevBench | 없음 |

벤치마크·리더보드:
- **JevBench** (fstandhartinger) [GH]: Intelligence, Calibration, Speed, Cost 4축의 기하평균. hard 220(공개 111 + held-out 109)을 포함해 534 결정. **공개 절반은 학습이나 선택에 쓸 수 있다고 README가 명시**하고, 공개→sealed 붕괴 사례를 기록했다(decider-4b v2: 83.5% → 34.7%).
- **LocalLLaMA/typed-decisions** [HF]: 합성 데이터. teacher endpoint로 3회 샘플(T=0.7)한 평균 분포를 gold로 쓴다. **test gold가 공개**되어 있다. 400 test case / 2,000 결정. Laya 카드는 "teacher self-agreement ceiling 0.735"를 언급한다.
- **AIMultiple decision-models** [blog/HF]: 브라우저 50과제, 단일 시도. Jev 17/50, Kev-9B 20/50, Laya 0/50, GPT-6 Astra 47/50.
- **SOTAAZ 측정** [blog]: BANKING77/TREC/AG News에서 **MiniLM 임베딩 + 로지스틱 회귀(CPU)가 모든 decision model을 이기거나 비슷**했다(TREC 90.4%, BANKING77 90.3%). 라벨 이름만 준 CLM-8B는 세 데이터셋 모두 chance 수준이었다.
- **fornewchallenge.tistory.com** 비교글 (2026-09-27) [blog]: 5종(CLM/Laya/OpenJev/Kev/Jev)을 비교했지만 자체 실험은 없다. 벤더 수치를 모은 것이고 "독립 재현 부족"을 경고한다.

### 1.2 DeepSWE 수치의 공개 맥락
- CLM README는 38 held-out 과제에서 Opus 5 후보 중 선택해 81.6%라고만 쓴다. pass@1이나 oracle은 README에 없다.
- 보도(VentureBeat, MarkTechPost)에는 **pass@1 baseline 73.7%**와 **Jev 71.1%**가 나온다. 즉 "random = 73.7%" 자체는 저자 측이 공개한 것이다. 페치 요약끼리 서로 맞지 않아 정확한 원문 표는 미확인이다.
- 우리 base head(zero-shot)도 27/38 = 71.1%로 보도된 Jev와 같은 값이다(우연일 수 있다). 38개 기준으로 한 과제 차이가 2.6pp라는 점을 보여 주는 사례다.

### 1.3 생태계 관련 arXiv (모두 프리프린트, 2026-09)
Visual Jev (2609.25845), Jev 범죄·교통 서술 코딩 (2609.24052), CSS 주석 평가 (2609.24574: decision model이 15개 중 14개 과제에서 최고 LLM보다 macro-F1 중앙값 11.6pt 낮지만 비용은 44배 싸다), JEV-as-a-Judge (2609.26550), pentest (2609.28940), Jev-Mem (2609.23986, 미확인), 방사선 판독 (2609.27607, 미확인). **frozen encoder head 방법론이나 CLM 감사를 다룬 논문은 찾지 못했다.**

---

## 2. 학술 선행연구 (B, C)

**frozen LLM 표현 + 작은 head / probe**
- LLM2Vec [arXiv 2404.05961, COLM'24 채택으로 알려짐(미확인)], NV-Embed [ICLR'25, arXiv 2405.17428], E5-Mistral [ACL'24, 2401.00368], Qwen3-Embedding [arXiv 2506.05176]. 모두 decoder LLM을 임베딩 모델로 쓰는 방향이다. 우리와 달리 encoder를 대조학습으로 **fine-tune**한다.
- Skean et al. "Layer by Layer" [PR, ICML 2025]: 중간층이 최종층보다 나은 경우가 많다고 보고했다. **우리 결과(16~28층 평균 0.777 vs 최종층 0.774, 차이 없음)와 gavel(L36 > L20)은 반례 또는 과제 의존성**을 보여 준다. 새롭지는 않지만 보고할 가치는 있는 음성 결과다.
- 로지스틱 회귀 on small LLM [arXiv 2408.03414], Code correctness linear probe (Qwen3-4B, AUC 0.88) [arXiv 2606.14530], classification-head FT of tiny LMs (Qwen3 0.6~8B, LoRA) [arXiv 2607.03801].
- **Reusing Embeddings** (Sun et al., arXiv 2502.04357): 임베딩 기반 reward model로 GPU 없이 재현 가능한 RM 연구를 하자는 주장. 우리 "CPU에서 head 학습·병렬 sweep" 설정과 거의 같은 동기라 반드시 인용해야 한다.

**이방성·표준화·이상치 차원**
- All-but-the-top [PR, ICLR 2018], BERT-whitening [arXiv 2103.15316], SimCSE 이방성 분석 [PR, EMNLP 2021], **Timkey & van Schijndel "All Bark and No Bite"** [PR, EMNLP 2021]: 1~3개의 rogue dimension이 유사도를 지배하고 **standardization이 이를 교정한다**. Massive activations [arXiv 2402.17762, COLM'24(미확인)].
- 따라서 "z-score가 frozen LLM 임베딩 품질을 드러낸다"는 기존 지식이다. z-score는 MLP 입력 전처리의 표준 관행이기도 하다. 새로운 것은 "CLM 공식 레시피가 이를 빠뜨렸고 그 비용이 약 7~8pp"라는 **경험적 관찰** 정도다. CLM issue #15의 "state embedding collapse" 보고와 연결하면 설명력이 커진다.

**dual-encoder vs cross-encoder, 대조 state-action**
- Poly-encoders [PR, ICLR 2020]: bi-encoder는 후보를 캐시할 수 있어 빠르지만 cross-encoder보다 부정확하다. CLM 구조의 트레이드오프가 이미 정리된 문헌이다. CLIP [PR, ICML 2021], Contrastive RL (Eysenbach et al.) [PR, NeurIPS 2022]: inner product가 goal-conditioned value가 된다.

**Verifier / best-of-N / PRM**
- Let's Verify Step by Step [arXiv 2305.20050, ICLR'24(미확인)], Generative Verifiers [arXiv 2408.15240], Large Language Monkeys [arXiv 2407.21787]: 자동 verifier가 없으면 선택이 plateau한다. DeepSWE/R2E-Gym hybrid verifier [blog + arXiv 2504.07164], RewardBench 2 (Bo4, random 25%) [arXiv 2506.01937].
- **Terminal-Bench verifier 전용 논문은 찾지 못했다.**

**평가 방법론 (C)**
- "decidable subset"과 같은 생각: DAPO [arXiv 2503.14476]는 정확도 0 또는 1인 prompt를 학습 신호가 없다며 걸러낸다. RL 학습 쪽 개념이지만 "모든 후보가 통과하거나 모두 실패하면 selector를 평가할 수 없다"는 논리는 같다. **Bo-N selector 평가에서 decidable subset만 따로 보고하는 규범을 명시한 논문은 찾지 못했다.** 부분적으로 새로운 지점이다.
- 통계: Miller "Adding Error Bars to Evals" [arXiv 2411.00640](paired test, power analysis), Madaan et al. "Quantifying Variance in Evaluation Benchmarks" [arXiv 2406.10229 / NeurIPS'24 workshop].
- 오염: "The SWE-Bench Illusion" [PR, ICSE-SEIP 2026, arXiv 2506.12286]. JevBench README의 공개/sealed 분리와 gavel의 자기 고지도 생태계 안의 관련 사례다.
- 보정: Guo et al. temperature scaling [PR, ICML 2017].

---

## 3. 우리 발견별 신규성 판정 (회의적으로)

| 발견 | 판정 | 근거 / 선행 |
|---|---|---|
| CLM DeepSWE 81.6% 정확히 재현 | **재현 자체는 가치 있지만 신규 아님** | 독립 재현이 없다고 여러 블로그가 지적(systemonemodels.tech, tistory). 첫 독립 재현이라는 점은 기술 보고서 수준에서 의미가 있다 |
| random 73.7% | **이미 공개** | 보도의 pass@1 baseline 73.7% |
| 38개 중 decidable 13개, oracle 89.5%, 차이는 3과제 | **새로운 분석(찾은 범위)** | 개념은 DAPO·Miller·Madaan에서 알려져 있지만 CLM 주장에 적용한 곳은 없다. 단 통계적으로 약하다. 대략 계산하면 10/13 vs 기대 ~7/13은 p≈0.1 수준으로 보이며, 과제별 통과 비율로 정확한 순열 검정을 다시 해야 한다 |
| base head(FT 전) 71.1% < random | **새로운 관찰** | 보도는 Jev만 random 아래라고 강조했다. CLM의 zero-shot도 그렇다는 것은 보고되지 않았다 |
| train 임베딩 풀에 held-out 과제 포함(과제 목록 필터로만 분리), 보상 포함 eval 데이터 공개 | **새로운 감사 발견**(오염 가능성 지적) | 필터가 정확하면 누출이 없을 수 있다. "위험"이지 "누출 확인"은 아니므로 표현에 주의 |
| frozen Qwen3 + head가 typed-decisions에서 Jev(0.727) 초과 | **신규 아님** | gavel(frozen+MLP), Laya-FT 0.766, verdict-2.0 0.771, od1 0.796이 모두 Jev 초과. Jev 0.727은 zero-shot이고 우리는 fitted라 비교 자체가 비대칭 |
| z-score + 40 epoch로 CV 0.700 → 0.774, test 0.766 | **증분적**(방법은 알려짐, CLM 맥락의 효과 크기는 미보고) | Timkey 2021, BERT-whitening, all-but-the-top. CLM 공식 0.685(PR #2) 대비 +8pp라는 수치는 새롭다 |
| 중간층 이득 없음 | 증분적 음성 결과 | Skean 2025와 반대 방향, gavel과 일치 |
| raw cosine < prior | 알려진 현상과 일치 | 이방성·rogue dimension 문헌. SOTAAZ도 CLM zero-shot이 chance 수준이라고 측정 |
| released head zero-shot 0.358, warm start가 해로움 | 새 데이터 포인트, 작다 | Laya zero-shot 0.362, CLM PR #2 warm start 0.685와 일치 |
| 워크플로별 specialist < pooled head | 알려진 경향(데이터 양 효과) | 신규성 낮음 |
| Windows/no-vLLM 서빙, 0.9 ms warm | 엔지니어링 | 하드웨어가 표준이 아니라(2080 Ti/3090 공유) 비교 불가 |

---

## 4. 출판 가능성

- **현재 상태: arXiv 기술 보고서(reproducibility report) 수준.** 워크숍은 가능하다(예: 평가·재현성, efficient inference, agent evaluation 워크숍. 2026-27 개최 여부는 미확인). ML Reproducibility Challenge 형식도 맞는다.
- **메인 학회는 비현실적**이다. 이유는 다음과 같다.
  - 방법 기여가 사실상 표준 전처리(z-score)다.
  - typed-decisions 수치가 기존 specialist(0.771/0.796)보다 낮다.
  - 핵심 벤치마크가 합성 데이터(teacher-labeled, test 라벨 공개)이거나 38문항짜리다.
  - 비교 대상 대부분이 동료 심사를 받지 않은 GitHub 주장이다.
- 빠진 것: (1) paired bootstrap/순열 검정과 CI. (2) 강한 baseline: 로지스틱 회귀 on z-scored 임베딩, Qwen3-Embedding + LR, LoRA(Kev 레시피), full FT(od1), cross-encoder/reranker(Qwen3-Reranker 등), ModernBERT FT. 모두 같은 split·같은 test 1회 조건이어야 한다. (3) 표준화 ablation: none / L2 / mean-center / z-score / whitening / all-but-top-k / 헤드 내부 LayerNorm. 이상치 차원 분석(massive activation 차원 제거만으로 재현되는지)도 필요하다. (4) 비합성·비공개 라벨 평가: JevBench sealed(유지관리자 제출), Kev transfer-v4 test. (5) 표준 하드웨어 latency(단일 전용 GPU, vLLM/Linux)와 처리량-정확도 곡선. (6) 보정: ECE/Brier + temperature scaling.

---

## 5. 후보 프레이밍 (정직한 최대 기여 순)

1. **"81.6% or 3 tasks? Auditing a best-of-N verifier claim for a contrastive System One model"** (재현·평가 방법론)
   - 기여: 정확한 재현, decidable-subset 분해, zero-shot < random, 임베딩 풀·공개 보상 데이터 오염 위험 감사, 권고(decidable n, oracle, paired test 동시 보고).
   - 필요한 실험:
     - 과제별 통과 비율 기반 정확한 순열/부트스트랩 p값
     - leave-task-out 재학습으로 held-out 과제를 풀에서 **물리적으로 제거**한 뒤 81.6%가 유지되는지
     - Terminal-Bench 2.1 데이터가 공개되면 같은 분석
     - 다른 공개 Bo-N verifier 결과(RewardBench 2 등)에 같은 decidable 분해를 적용해 일반화 보여 주기
2. **"Standardize before you contrast: a z-score is worth ~8 points for frozen-LLM decision heads"** (짧은 empirical note)
   - 필요한 실험:
     - 4개 encoder 크기(0.6~8B) × 3개 이상 벤치마크(typed-decisions, Kev transfer-v4, BANKING77/TREC 같은 비합성)에서 전처리 ablation
     - 차원별 분산·massive activation 분석
     - CLM 원래 과제(DeepSWE)에서도 z-score가 도움이 되는지
     - 로지스틱 회귀 baseline과 비교해 "head의 대조 구조가 필요한가"를 검증
3. **"How far can a frozen ≤8B encoder go? Frozen heads vs LoRA vs full FT for typed decisions"** (현재 진행 중인 3단계 연장)
   - 필요한 실험:
     - 동일 데이터(decision-v7 + typed-decisions)로 Kev-LoRA와 full FT를 직접 재학습해 쌍대 비교
     - transfer-v4 test와 JevBench sealed 평가
     - 정확도-latency-비용 Pareto 곡선
     - 통계 검정
   - 결과가 "frozen이 LoRA에 근접하면서 10배 싸다"로 나오면 워크숍 이상도 노릴 수 있다. 반대로 나와도 음성 결과로 쓸 수 있다.
4. (보조) **"What do Jev-class leaderboards measure?"**
   - 다룰 내용: test 라벨 공개, teacher ceiling 0.735를 넘는 점수들, 공개→sealed 붕괴, 합성 라벨 노이즈
   - 필요한 실험: teacher 재샘플링 일치도, gold 불확실성 가중 정확도, 문항 수 대비 검정력 분석

추천: 1번을 주 기여로, 2번을 부 기여로 묶어 arXiv 기술 보고서 1편으로 내고, 3번은 결과를 보고 별도로 판단한다.

---

## 6. 위험 목록

1. **생태계 근거가 대부분 비심사 자료다.** GitHub/HF 카드, 블로그, X 게시물이고 수치가 자기 보고다. 리뷰어가 "baseline이 논문이 아니다"라고 지적할 수 있다. 인용할 때 전부 "self-reported"로 표기해야 한다.
2. **수치를 서로 비교할 수 없다.** Jev·meraGPT는 zero-shot, 우리·Laya-FT·verdict·od1은 train에 맞춘 fitted 모델이다. 리더보드 수치는 항목별로 짝지을 수 없다(우리가 Jev를 재실행할 수 없음). 하드웨어·런타임도 제각각이다.
3. **합성 벤치마크다.** typed-decisions는 teacher LLM 라벨이다. 여러 모델이 teacher self-agreement 0.735를 넘는 것은 teacher 분포를 흉내 내는 데 과적합했을 가능성을 시사한다.
4. **test 라벨이 공개되어 있다.** typed-decisions test gold와 DeepSWE eval 보상이 공개돼 있다. 우리는 test를 한 번만 읽었지만 외부에서는 이를 검증할 수 없으므로 실행 로그·해시 공개가 필요하다.
5. **표본이 작다.** DeepSWE 38(decidable 13), typed-decisions 2,000 결정이지만 case 400개(결정끼리 상관이 있어 cluster SE 필요). 3-seed 차이 0.3pp 수준은 잡음이다.
6. **오염 주장은 신중해야 한다.** 임베딩 풀 포함은 "누출 위험"이지 입증된 누출이 아니다. 저자 측이 반박할 여지를 남기고 leave-task-out 실험으로 확정해야 한다.
7. **변화 속도가 빠르다.** 생태계가 주 단위로 바뀐다. CLM 논문이 나오거나 z-score가 반영된 공식 패치가 나오면 신규성이 즉시 사라질 수 있다(2026-09-28 기준 검색 결과).
8. **재현 조건이 다르다.** 우리 임베딩은 transformers fp16이고 CLM은 vLLM bf16이다. zero-shot·warm-start 수치가 불리하게 편향됐을 수 있다.
9. **이름·출처 혼동.** "OpenJev"는 여러 프로젝트(razorback16/DiffusionGemma, Heman10x Verdict, zhangcy122, SemIf 개명)가 쓰고 "Decider"도 여러 fork가 있다. 인용 시 URL로 특정해야 한다.

---

## 7. 확인하지 못한 것 (정직한 공백)
- CLM Notion 블로그(contrastive-lm.notion.site) 본문은 페치가 안 됐다. CLM 논문·arXiv 존재 여부는 README와 모델 카드 기준으로 "없음"이다.
- Terminal-Bench 2.1 CLM 설정·데이터, meraGPT Decider 1의 방법 세부사항, AIMultiple 페이지의 CLM 포함 여부(대상 모델 목록에 없음).
- MarkTechPost 표의 원문 수치는 페치 요약이 뒤섞였다. DeepSWE pass@1 73.7%와 Jev 71.1%는 VentureBeat·검색 요약 기준이다.
- "z-score before InfoNCE head"에 대한 선행 보고는 목표 검색 2회로 찾지 못했다. 부재를 증명한 것은 아니다.
