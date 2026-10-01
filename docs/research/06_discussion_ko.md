# 6. 논의

결과는 세 가지로 요약된다.
- 결정 모델은 옵션을 함께 읽어야 한다.
- 학습 레시피를 조정하는 것보다 base 모델과 학습 데이터가 중요하다.
- in-distribution dev 정확도는 transfer 성능을 대신하지 못한다.

## 6.1 해석

**(1) 옵션을 함께 읽는 구조.** 같은 base와 같은 데이터에서 비교하면 frozen head는 MiDM에 뒤진다(4B 기준). 학습 분포 안에서는 3–14pp, 학습하지 않은 source에서는 26–33pp 차이다. 두 모델의 차이는 두 가지다.
- MiDM에서는 옵션이 attention으로 state와 다른 옵션을 함께 본다.
- MiDM은 LoRA로 base 표현 자체를 조정한다.

두 요인을 분리하면(Qwen3-4B, 같은 데이터) Kev T test는 frozen 양방향 head 0.411, 옵션을 함께 읽고 LoRA는 쓰지 않을 때 0.630, 둘 다 쓸 때 0.715다. transfer 이득의 약 70%는 옵션을 함께 읽는 구조에서 온다.

**(2) 레시피보다 base와 데이터.** DoRA, rsLoRA, rank 확대, 추가 epoch, TTA는 transfer를 올리지 못했다. 추가 epoch는 오히려 transfer를 낮췄다. 같은 4B에서 transfer를 올린 것은 두 가지뿐이다.
- base 세대 교체(Qwen3 → Qwen3.5): 정밀도를 맞추면(bf16끼리) transfer suite에서 +3.4–5.9pp다(Kev T test +3.4pp, p = 0.066; Kev T dev +5.9pp, p < 0.001).
- 데이터 구성(-bx): Kev T test +3.3pp(3 seed 평균; seed별 +3.4 / +3.8 / +2.7pp)이고, typed-decisions 손실은 −0.5pp로 잡음 수준이다.

그 결과 4B 모델이 이전 세대 8B와 동등해진다(p = 0.69).

단, 세대 효과는 크기에 따라 다르다. 0.8B와 2B에서는 transfer 이득이 유의하지 않았고(+0.7pp, +2.1pp), Qwen3.5 안에서는 2B → 4B 구간에서 transfer가 +10.2pp로 크게 뛰었다. "base 세대가 가장 큰 요인"이라는 결론은 4B 이상에 한정된다.

**(3) 선택 지표.** 두 번째 epoch와 -bx는 in-distribution dev와 transfer가 반대 방향으로 움직인 사례다.
- 두 번째 epoch: td_holdout은 오르고 transfer는 내려갔다.
- -bx: td_holdout은 내려가고(0.844 → 0.832, 3 seed) transfer는 올랐다.

따라서 transfer가 목표라면 학습에서 제외한 source로 dev를 따로 구성해야 한다. 이는 Kumar et al.(2022), Gulrajani & Lopez-Paz(2021)와 같은 방향의 관찰이다.

**(4) Cascade의 조건.** 확신도에 따라 escalate하는 방식은 확신도가 믿을 만한 domain에서만 작동한다. 미지 source에서는 작은 tier가 과신하므로, 이를 먼저 다루어야 한다. 본 연구는 미지 domain을 최상위 모델로 바로 보내는 규칙으로 대응했다.

**(5) CLM 주장의 평가.** 81.6%는 정확히 재현된다(31/38). 그러나 이 주장이 성립하는 범위는 "38과제 부분집합에서, DeepSWE 특화 fine-tuning 뒤"다.
- 결과를 바꿀 수 있는 decidable 과제는 13개다.
- random 대비 이득은 3과제이고, exact p = 0.062다.
- fine-tuning 전 base head는 random보다 낮다.

best-of-N selector를 평가할 때는 decidable 과제 수, random 기대값, oracle, 검정 결과를 함께 보고할 것을 권한다.

## 6.2 한계

**데이터와 표본**
1. **합성 벤치마크.** typed-decisions는 teacher가 라벨을 붙인 합성 데이터이고, test gold가 공개돼 있다. 0.79–0.80이라는 수치는 teacher 분포를 모사하는 데 과적합한 결과일 수 있다.
2. **작은 n.**
   - DeepSWE의 decidable 과제는 13개다.
   - jb_public 231문항 중 easy 48문항은 모든 모델이 만점이다.
   - proxy 탐색에 쓴 src_holdout은 276행이다.
3. **Seed.** (해소, 2026-10-01) MiDM-4B-q3-e1, Qwen3.5-4B e1, -bx를 각각 3 seed로 학습했다(Kev T test 0.723 ± 0.006 / 0.763 ± 0.005 / 0.796 ± 0.002). -bx 이득은 seed별로 +3.4 / +3.8 / +2.7pp다. 다른 크기(0.8B, 2B)와 ablation은 seed가 1개다.

**평가 설계**
4. **kevT_dev와 kevT9_dev의 독립성.** 미지 domain용 모델은 kevT_dev로 골랐다. 따라서 -bx의 kevT_dev 수치는 선택에 쓰인 값이다. kevT9_dev는 kevT_dev의 source를 포함하므로 역시 독립이 아니다. 독립 확인은 kevT_test만 맡는다.
5. **held-out 재읽기.** cascade와 통계 분석을 위해 확률 덤프로 held-out suite를 다시 읽었다. 이 재읽기는 선택에 쓰지 않았고, 모든 읽기는 ledger(SHA-256)에 기록했다.

**비교 대상**
6. **참고 수치.** Jev, Kev, od1, CLM-8B의 수치는 자기 보고이고, 같은 문항으로 짝지은 비교가 아니다. 또한 zero-shot 모델(Jev)과 학습 모델(MiDM)의 비교는 비대칭이다.
7. **Kev와의 조건 차이.** -bx는 Kev보다 넓은 학습 source를 쓴다. 따라서 Kev와 동일 조건 비교가 아니다.
8. **정밀도와 하드웨어.**
   - 일부 모델은 2080 Ti에서 fp16으로, 나머지는 3090에서 bf16으로 학습했다. 정밀도 통제(4B, 2B를 bf16으로 재학습)에서는 정밀도만으로 생기는 차이가 유의하지 않았다.
   - CLM 공개 head는 vLLM bf16 임베딩으로 학습됐지만, 우리는 transformers 임베더를 썼다. 따라서 공개 head의 zero-shot과 warm-start 수치는 CLM 쪽에 불리하게 편향됐을 수 있다.

**기타**
9. **비용.** (실측으로 수정) 3090, bf16, batch 1 기준 tier별 지연은 0.6B 151 ms, 1.7B 156 ms, 4B-q35 309 ms, 8B 196 ms다. 호출당 고정 비용이 커서, 잠정 가중치(1 : 6 : 12)가 가정한 절감은 나타나지 않는다. 실측 기준으로는 cascade(237–309 ms)가 8B 단독(196 ms)보다 빠르지 않다. Qwen3.5-4B가 느린 것은 linear-attention 가속 커널이 없어 참조 구현으로 동작하기 때문이다. 따라서 cascade의 이점은 이 환경에서는 비용이 아니라 메모리(작은 tier 상주) 쪽에 한정된다.
10. **누출.** (해소) 공개 학습 풀에 held-out 과제가 있지만, 목록으로 제외하고 재학습한 5 seed가 81.6% ± 1.9를 재현했다. 남는 문제는 평가 정답이 공개 데이터에 함께 실려 있다는 배포 방식뿐이다.
11. **신규성.** 옵션별 마지막 토큰에 pointer head를 두는 구조는 선행 연구와 겹친다 [7–9]. 본 연구의 기여는 이 구조를 typed decision에 적용한 것, 통제 실험, 그리고 CLM 감사에 있다.
