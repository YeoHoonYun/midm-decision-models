# MiDM 배포 정보 (논문 "Code and model availability" 근거)

최종 갱신: 2026-10-01

## 1. Hugging Face

| 항목 | 값 |
|---|---|
| repo | https://huggingface.co/yunicro/MiDM-4B-q35-e1-bx |
| 계정 | `yunicro` |
| 공개 여부 | **private** (공개 전에 학습 데이터 이용 조건 확인 필요: Yelp, Amazon reviews 등) |
| 라이선스 | Apache-2.0 (어댑터와 코드만. base 모델과 데이터셋은 각자의 조건을 따름) |
| base 모델 | `Qwen/Qwen3.5-4B-Base` (4-bit NF4, 사용자가 별도로 내려받음) |
| 로컬 원본 | `huggingface/MiDM-4B-q35-e1-bx/` (`build_hf_package.py --repo-id yunicro/MiDM-4B-q35-e1-bx`로 생성) |
| 학습 run | `experiments/113_clm_reproduction_20260928/runs/E/q35_4b_e1_breadth_mc/final` |
| 평가 파일 | `results/pointer/FINAL_E_q35_4b_e1_breadth_mc.json` (ledger 등록) |

커밋 이력

| 날짜 (UTC) | 커밋 | 내용 |
|---|---|---|
| 2026-09-30 23:30 (KST 10-01 08:30) | `e02cfa1` | v0.1.0 최초 업로드 (7개 파일) |
| 2026-09-30 23:34 (KST 10-01 08:34) | `680de33` | 모델 카드에 CLM-8B·TypeSafe Jev 비교 표와 DeepSWE 감사 요약 추가 |

파일 (SHA-256, 로컬 기준. 원격 LFS 해시는 두 가중치 파일에서 일치를 확인함)

| 파일 | bytes | sha256 |
|---|---|---|
| adapter_model.safetensors | 121,954,608 | 7b165d1d0d57a29e9c9ae39f639f77edc1ef8d2084174eb2afcc364eb8f00055 |
| pointer_head.safetensors | 10,380 | 7e9e1f0f94d00ad5835d7829c22ef74827f4a95dcbf335e9ceac3a6d60bd9d50 |
| adapter_config.json | 1,326 | 50ba2db8ff05e6d6a1f8bdecd8005a3f76e8cf8e926ec84e3d7e5666d07ffc11 |
| midm_config.json | 427 | fc14e2b67032a6815856c65418d152222bb0cdf580d801b470d7e040959273d2 |
| midm.py | 9,008 | 2a3985c09b38802d240acec138b507358ee4bd30e413c82dd3215ee7ac0681fd |
| README.md (모델 카드) | 5,107 | 37ed244fc7592d9179c23b934a29403972d0c14674c5c2f9595e750bab127236 |
| requirements.txt | 106 | 550b986e177232471ed3dee7efacb0d8b0dacf0cbc7044b9b81a58dc38d808c2 |

모델 카드 수치 (각 suite 1회 평가)

| suite | MiDM-4B-q35-e1-bx | MiDM-4B-q35-e1 |
|---|---|---|
| typed-decisions test | 0.788 | 0.805 |
| Kev decision-v7 dev | 0.856 | 0.861 |
| Kev transfer-v4 dev | 0.791 | 0.755 |
| Kev transfer-v4 test | 0.798 | 0.764 |
| Kev transfer-v9 dev | 0.705 | 0.675 |
| JevBench public (231) | 0.710 | 0.710 |

**검증 상태.** 패키지 경로 검증(queue job 102)은 논문 seed 실험 뒤로 보류돼 있다. 이 검증은 `verify_hf_package.py`로 패키지의 `midm.py`만 써서 모델을 불러 데모를 실행하고, td_holdout을 다시 채점해 0.8333 ±0.005가 나오는지 확인한다. 끝나면 결과를 여기에 기록한다. 결과 파일은 `huggingface/verify_MiDM-4B-q35-e1-bx.json`이다.

## 2. GitHub

| 항목 | 값 |
|---|---|
| 로컬 repo | `github/midm-decision-models/` (branch `main`) |
| 원격 | https://github.com/YeoHoonYun/midm-decision-models (**private**, 2026-10-01 생성, branch `main`, `gh` CLI로 푸시) |
| 라이선스 | Apache-2.0, `NOTICE`(CLM 스키마 헬퍼 Apache-2.0 표기) |
| 인용 | `CITATION.cff`: YeoHoon Yoon, Seoul School of Integrated Sciences and Technologies, ORCID 0009-0007-2669-8127 |
| 포함 | 실험 113과 114의 코드, 결과 JSON, stats, ledger, 문서 |
| 제외 | 데이터, 확률 덤프, 가중치 |
| 주요 커밋 | `cae10c6` v0.1.0 → `7de0918` → `61f14ab` → `70d7ae2` → `b1744d4` → `977f60c` CLM·Jev 비교와 HF 링크 |

## 3. Zenodo (DOI)
- 아직 발급되지 않았다.
- 절차: 저장소를 public으로 전환 → Zenodo에서 GitHub 계정 연결 → repo 토글 ON → GitHub Release(`v0.1.0`) 생성 → DOI 자동 발급.
- Zenodo 연동은 public repo에서만 동작한다. 발급 후 DOI를 `CITATION.cff`, GitHub README, HF 모델 카드, 논문에 반영한다.
- 메타데이터는 `.zenodo.json`에 준비돼 있다.

## 4. 비교 대상 수치의 출처 (논문 표에 쓰는 값, 모두 비짝지음)

| 시스템 | 값 | 출처 |
|---|---|---|
| TypeSafe Jev | typed-dec. 0.727, Kev T dev 0.857, JevBench public 0.866, DeepSWE 71.1% | 제품 문서, 리더보드, 보도 (동료 심사 논문 없음) |
| CLM-8B | DeepSWE 81.6% (주장), typed-dec. 0.685 (CLM PR #2), JevBench 0.407 (보드) | HF 모델 카드 `Contrastive-LM/CLM-v0.1-8B`, GitHub PR #2, 보도 (논문 없음) |
| Kev-9B | Kev dev 0.872, Kev T dev 0.822, Kev T test 0.852 | kev-9b 모델 카드와 Kev repo |
| 우리 재현 | CLM DeepSWE 31/38 재현, base head 27/38, CLM 레시피 재학습 0.766 | `results/r1_*`, `results/r2_*`, `results/search/FINAL_test_r1_08.json` |

## 5. 새 버전 배포 절차
1. dev 기준으로 선택한 뒤 test를 1회 평가한다. `FINAL_*.json`이 생성되면 `build_ledger.py`를 실행한다.
2. 패키지를 만든다:
   `python build_hf_package.py --adapter <run>/final --name <이름> --base <base> --final <FINAL json> --baseline <이전 FINAL> --baseline-name <이전 이름> --train-suites ... --repo-id yunicro/<이름>`
3. 검증한다(GPU 큐 사용):
   `gpuq.py submit --queue gpu0 --mem 10 --priority 12 --env PTR_BF16=1 -- python verify_hf_package.py <패키지> --suite td_holdout --expect <dev acc>`
4. 업로드한다:
   `hf repos create yunicro/<이름> --private`, 이어서 `hf upload yunicro/<이름> huggingface/<이름> .`
5. 이 파일, GitHub README, 연구 폴더 README를 갱신하고 GitHub repo에 커밋한다.

후보: `-bx-td2`(queue job 99–100, 보류 중). 모든 주요 suite에서 MiDM-4B-q35-e1 이상이면 새 버전으로 배포한다.
