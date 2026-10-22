# 시나리오 분석 원고 재검토 2026-10-21

검토 대상은 업로드한 `2_Manuscript_anonymized.docx`이며 본문 제목은 **Coverage Is Not a Decision: Separating Scenario Classification, Numerical Comprehension and Economic Value in a Local Neural–Language-Model Pipeline**이다. 기존 *From Financial Scenario Coverage…* 시나리오 논문의 제목을 변경한 원고로 관리한다. MiDM 범용 논문이나 *Coincident, Not Leading* 검토가 아니다. 원본 DOCX는 수정하지 않았다.

## 판단

분류·수치이해·행동·경제가치를 구분하고 부정 결과와 한계를 공개한 방향은 타당하다. 다만 현재파일을 수정 없이 제출하는 것은 권하지 않는다. 확정적인 익명화 결함, 과장·단정 표현, 노출률 분모 오류, 미실행 기준선이 남아 있다. 아래 문제는 실측을 새로 만들라는 요구와 문장만 고치면 되는 요구를 구분한다. 어떤 저널의 심사 통과를 보장하는 판단은 아니다.

## 제출 전 수정할 사항

1. **익명화 불완전 — 파일 자체에서 확인.** `docProps/core.xml`의 creator에 저자 두 명의 이름이 남아 있고 내부title은 예전 제목이다. `word/_rels/document.xml.rels`에는 저자 GitHub/HF/Pages와 선행논문 DOI 관계가 남아 있다. 본문에는 활성 hyperlink 요소가0개이므로 클릭 가능한 저자 링크가 남았다는 뜻은 아니지만 ZIP패키지에는 식별주소가 잔존한다. 메타데이터·사용하지 않는관계·customXml/그림부속정보를 정리하고 익명사본을 다시 검사해야 한다. tracked insert/delete와 comments part는 확인되지 않았다. 저자명 삭제만으로 모든 그림·부속패키지의 익명성이 확인된 것은 아니다.

2. **§5.2의 'MiDM’s 98% exposure rate'는 잘못된 분모.** §4.5의630/642와602/612는 flat 상태의 진입 선택 비율(약98.13/98.37%)이지 전체날짜의 실제자본노출률이 아니다. 'BUY-choice rate among flat-state decisions'로 바꾸고 실제노출률은 정책원장집계를 별도로 인용해야 한다. 새로운수치를추정해채우지 않는다.

3. **초록의 'five-session policies turn negative at20bp per side'는 범위가 넓다.** 기존E6에서 h5 B3-1=-8.6332%,B3-2=-14.2497%지만 B3-3=+4.9346%다. §4.8은B3-1/B3-2라고정확히구분한다. 초록도두정책으로한정한다. 비용민감도표를본문에올리는A07은이미승계초안이있다.

4. **§5.3의 'the null action result therefore survives this threat'는 근거 초과.** 사전학습오염이정확도를한방향으로만올린다고보장하지않으며 식별자지식이방해를줄수도있다. 권장문장: 'No action improvement is established under the observed protocol; the direction and magnitude of pretraining contamination remain unresolved.' 마스킹A04는필요하지만성능유지/하락만으로암기유무를입증하지않는다. Glasserman/Lin 원문은식별자제거후성능이더좋아지는 distraction effect도보고한다.

5. **§5.2 'transfer failure rather than a capacity failure'는 실험으로 구분하지 못했다.** 같은과제·데이터·훈련으로크기별모델을통제하지않았으므로 capacity원인을배제할수없다. 'The tested adaptation procedures did not improve action selection at the evaluated capacity'처럼실행범위로한정한다. §1의pairhead가activecomponent로'identified'됐다는표현도특정CE+rank 대비관측효과로한정한다. 표13의3.69pp와원래B3전체개선5.91pp는동일분해가아니다.

6. **§4.11의 seed SD로 효과 불확정을 판정하는 논리는 수정.** 5.50pp seed SD가2.65pp 평균차보다크다는것만으로통계적효과를판정하지못한다. Table20은동일fresh generic pass를쓴점은좋지만paired seed×date CI와balanced accuracy/regret도같이제시하는편이적절하다. 새계산전에는'Across-seed variation is large; no inferential conclusion is based on this SD comparison'로쓴다. 기존다른checkpoint의CI를대신쓰지않는다.

7. **Backward holdout 계획의 학습교집합 오류.** §5.3은2008/2020/2022를untouched earlier periods라묶지만 B3는2022까지학습하고행동모델도2017–2022를쓴다. 2020/2022는시나리오·정책평가도미접촉이라고부를수없다. A03처럼모델별실제학습·선택기간감사후미접촉만holdout으로명명한다. 나머지는훈련내stress diagnostic이다. 'Both markets rose throughout the window'도'Both markets had large positive cumulative returns over the window'로고쳐중간하락이없었다는뜻을피한다.

8. **AI 사용선언의 절대문장 수정.** AI가실험코드·분석·그림코드를도왔다고쓴뒤 'AI tools did not generate or alter any reported result'라고하면모호하다. 실제역할에맞게 'Reported numerical results were computed by the documented experimental scripts from recorded data and model outputs; AI assistance was not used to fabricate or manually replace results. The authors verified…' 같은문장으로정리하되저자가실제검증한범위를넘겨쓰지않는다.

9. **부록B의 'no private examples are rescored'는 §4.11과 충돌.** §4.11은generic을single-item batch와고정옵션순열로재평가했다고명시한다. 'The archived independent B3-policy audit did not rerun model inference; the separate continuation experiments included the matched re-evaluation described in Section4.11'로범위를분리한다.

10. **증거패키지 표번호를 현재원고에 다시 맞출 것.** 로컬이전v5.1 EVIDENCE_MAP은옛표번호를쓴다. 현재표13은이전표14,현재표18은이전paired표19계열이며F1은이전표13에서이동했다. 제출companion이이전매핑이라면심사자가잘못된파일을연다. 현재21개표및3개그림의표번호→원본파일→해시→모수/단위→본절을재작성한다. 원고는package가accompany한다고쓰지만이번업로드에는DOCX만있어실제첨부여부는검증하지못했다.

## 새실험이 필요한 부분

- A01 동일훈련표본 GBM/다항로지스틱과A02 추세·모멘텀규칙은여전히미실행기준선으로본문에명시돼있다. 기존TabICLv2/tree탐색을동일표본·동일선택예산기준선의대체물로간주하지않는다.
- A04 날짜/자산/가격마스킹과A03 가용한미접촉과거구간감사는별도실행이필요하다.
- A05 DSR/시행수감사는현재Bonferroni6대비로대체되지않는다. 412산식은72+50+50+144+96=412로맞지만군별fit인지선택시도인지별도로정의한다. 현재표13 E4의30회등추가fit의포함여부도lineage로확인한다. 전체독립시행412회로해석하지않는다.
- A10 conformal집합비교가아직없다. 특히2023을모델선택과보정에같이쓰는문제는등록설계대로독립보정과재사용진단을분리한다. 보정평균크기2와평가평균크기2를혼동하지않는다.
- A06 h라벨실험과A09 현금금리는기제·가정민감도이며현재결론을확정해주는실험으로미리쓰지않는다. A08은기존weighted CE결과중복확인후진행한다.

## 근거 대조 결과와 한계

업로드21개표중13개는이전v5.1의표셀과완전히일치한다. 이것은승계값확인이며독립재현은아니다. 현재Table10의12정책행×4지표(수익,MDD,거래수,노출)48칸과Table12의6행×2수익12칸,Table3의B3두coverage를기존집계RESULTS.json 및344/677·362/647과대조했다. 반올림범위안에서일치했다. 새수치조작이나이수치들의원자료불일치는확인되지않았지만모든실험의원자료재산출을마친뜻은아니다. 특히Table11/18/20을이번작업에서원시일별수익/문항별예측으로재계산하지않았다.

검토는본문·표·집계·OOXML익명화검사다. Word페이지PNG를새로렌더링해전페이지레이아웃검수한것은아니다. 원본DOCX는그대로보존했다. 신규실험·GPU추론·백테스트·공개배포없음.

참고 원문: https://arxiv.org/abs/2309.17322 (Glasserman and Lin). 이원문은식별정보가항상유리한방향으로작용한다는가정을지지하지않는다.
