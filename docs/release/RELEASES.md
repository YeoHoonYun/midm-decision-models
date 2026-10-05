# Current release: v0.2.1

Benchmark/architecture analysis update; weights unchanged from v0.2.0. [Release notes](../releases/v0.2.1/README.md). DOI: https://doi.org/10.5281/zenodo.23164651 .

The v0.1.0 record below is retained as historical release history.

# MiDM 배포 정보 (논문 "Code and model availability" 근거)

최종 갱신: 2026-10-01 23:30 — **v0.1.0 공개 배포 완료**

## 0. 최종 공개 상태 (v0.1.0)

| 채널 | 주소 | 상태 |
|---|---|---|
| **Zenodo DOI** | **https://doi.org/10.5281/zenodo.23084584** (전체 버전 concept DOI: 10.5281/zenodo.23084583) | 공개, open access, 압축본 1개(393 KB) |
| GitHub | https://github.com/YeoHoonYun/midm-decision-models | **public**. 릴리스는 https://github.com/YeoHoonYun/midm-decision-models/releases/tag/v0.1.0 |
| Hugging Face | https://huggingface.co/yunicro/MiDM-4B-q35-e1-bx | **public** (commit 957e982, 모델 카드에 DOI·견고성 절 포함) |

- 릴리스 대상 커밋은 `b3dcb96`(태그 v0.1.0)이다. 이후 DOI를 반영한 커밋은 `f12bb13`, README 링크 정리는 `72fda54`다.
- 기계 판독용 기록: `release_v0.1.0.json`. 릴리스 노트: `release_notes_v0.1.0.md`. HF 패키지 검증 결과: `verify_MiDM-4B-q35-e1-bx.json`(td_holdout 0.835, 일치).
- 커밋 작성자 이메일은 모두 GitHub noreply다. 공개 전에 비밀값, 개인 이메일, 로컬 절대경로를 스캔했고 0건이었다.
- **논문 인용 문구:** Yoon, Y. (2026). *MiDM: Minimal Decision Models and an audit of CLM-8B* (v0.1.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.23084584
- **이후 버전:** 수정할 내용은 GitHub에 반영한 뒤 `publish_release.py --version 0.1.1`을 실행한다. 새 버전 DOI가 발급되고, concept DOI는 그대로 유지된다.
- **권장:** 채팅에 노출된 HF 토큰과 Zenodo 토큰은 각 사이트에서 폐기하고 새로 발급한다.
## 1. Hugging Face

| 항목 | 값 |
|---|---|
| repo | https://huggingface.co/yunicro/MiDM-4B-q35-e1-bx |
| 계정 | `yunicro` |
| 공개 여부 | **public** (2026-10-01 전환. 학습 데이터 이용 조건은 모델 카드에 명시) |
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
| 2026-10-01 (KST 23:2x) | `957e982` | 견고성 절(3 seed, 패키지 검증, latency, 약점)과 DOI 인용 추가, **public 전환** |

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

**검증 상태.** 통과했다(2026-10-01 23:11, queue job 102). 패키지의 `midm.py`만으로 불러 td_holdout 0.835를 얻었고, 학습 저장소 값 0.8333과 ±0.005 이내로 일치한다. 결과 파일은 `verify_MiDM-4B-q35-e1-bx.json`이다.

## 2. GitHub

| 항목 | 값 |
|---|---|
| 로컬 repo | `github/midm-decision-models/` (branch `main`) |
| 원격 | https://github.com/YeoHoonYun/midm-decision-models (**public**, 2026-10-01 생성·공개, branch `main`, 태그 `v0.1.0`) |
| 라이선스 | Apache-2.0, `NOTICE`(CLM 스키마 헬퍼 Apache-2.0 표기) |
| 인용 | `CITATION.cff`: YeoHoon Yoon, Seoul School of Integrated Sciences and Technologies, ORCID 0009-0007-2669-8127 |
| 포함 | 실험 113과 114의 코드, 결과 JSON, stats, ledger, 문서 |
| 제외 | 데이터, 확률 덤프, 가중치 |
| 주요 커밋 | noreply로 기록을 재작성한 뒤 `db00c0e`, 최종 결과 `6f7a622`, 경로 정리 `b3dcb96`(태그 v0.1.0), DOI 반영 `f12bb13`, README 링크 `72fda54` |

## 2b. 공개 배포 절차 (2026-10-01 결정, 실험 완료 후 실행)

사용자 결정: GitHub와 HF를 **공개**로 전환하고 Zenodo DOI를 발급한다.

- **커밋 이메일:** 공개 전에 커밋 18개의 작성자 이메일을 GitHub noreply(`45550361+YeoHoonYun@users.noreply.github.com`)로 바꿔 강제 푸시했다(`main` = `db00c0e`). 이후 커밋도 이 저장소에서는 noreply로 기록된다.
- **Zenodo:** GitHub 연동 대신 **API 업로드**를 쓴다. 토큰은 `%USERPROFILE%\.config\zenodo\token`에 소유자 전용 ACL로 보관하고, 어떤 저장소에도 넣지 않는다. 같은 계정에 기존 레코드 1건(NSR 연구, 10.5281/zenodo.22927634)이 있으며, 이번 릴리스는 별도의 새 레코드다.
- **실행:** `experiments/113_clm_reproduction_20260928/publish_release.py --version 0.1.0` (`--dry-run` 확인 완료: 압축본 354KB)
  1. GitHub: 태그 `v0.1.0`을 푸시하고, 저장소를 공개로 전환하고, GitHub Release를 만든다.
  2. Zenodo: `.zenodo.json` 메타데이터와 태그 압축본을 업로드하고 publish해 DOI를 받는다. **영구적이며 삭제할 수 없다.**
  3. CITATION.cff와 README에 DOI(배지 포함)를 넣고 커밋·푸시한다.
  4. HF 모델 카드에 DOI와 GitHub 링크를 넣고 공개로 전환한다.
  5. `paper/release/release_v0.1.0.json`에 기록을 남긴다.

## 3. Zenodo (DOI)
- **발급 완료:** 10.5281/zenodo.23084584 (record 23084584), concept DOI 10.5281/zenodo.23084583. 2026-10-01에 API 업로드로 발급했고, 대상은 태그 v0.1.0의 git archive다.
- 메타데이터는 `.zenodo.json`에서 가져왔다(저자, ORCID, 소속, Apache-2.0, 관련 식별자: GitHub 태그와 HF 모델).

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
