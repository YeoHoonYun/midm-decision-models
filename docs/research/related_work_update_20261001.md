# Related work update (2026-10-01)

This update was compiled by a literature-search agent for the MiDM findings: base-model generation, epochs, breadth + knowledge MC, PEFT null results, cascade and CLM audit.
- **[V]** means checked on arXiv, ACL Anthology or a proceedings page during the search.
- **[M]** means from well-established memory: the ID and venue are believed correct but were not re-fetched. Spot-check these before submission.
- **UNVERIFIED** means not confirmed.

This supplements `novelty_review.md` and `references.json`.

## (a) LLMs as classifiers/rankers; listwise and cross-encoder rerankers; MCQ scoring
- Nogueira & Cho (2019). Passage Re-ranking with BERT. arXiv:1901.04085 [M]. The original cross-encoder reranker. MiDM is a cross-encoder over (state, question, options).
- Li et al. (2023). Label Supervised LLaMA Finetuning (LS-LLaMA). arXiv:2310.01208 [V]. A LoRA decoder with a last-token classification head in place of generation. This is MiDM's head design at the single-label level.
- Reddy et al. (2024). FIRST: Faster Improved Listwise Reranking with Single Token Decoding. EMNLP 2024, pp. 8642–8652 [V]. Listwise one-pass scoring from first-token logits.
- Wang, Li & Xiao (2025). jina-reranker-v3: Last but Not Late Interaction for Listwise Document Reranking. arXiv:2509.25085 [V]. A 0.6B Qwen3 model: the query and all documents sit in one causal context, and each document is scored from its final-token embedding. **This is the closest architecture to MiDM's pointer head.**
- Liu et al. (2025). E2Rank: Your Text Embedding can Also be an Effective and Efficient Listwise Reranker. arXiv:2510.22733 [V]. It bridges CLM-style bi-encoders and MiDM-style listwise scoring.
- Robinson & Wingate (2023). Leveraging Large Language Models for Multiple Choice Question Answering. ICLR 2023. arXiv:2210.12353 [M].

## (b) Option-order and selection bias
- Zheng et al. (2024). Large Language Models Are Not Robust Multiple Choice Selectors (PriDe). ICLR 2024. arXiv:2309.03882 [M]. It motivates shuffled-option training and the permutation TTA, which gave no gain.
- Pezeshkpour & Hruschka (2024). Large Language Models Sensitivity to the Order of Options in Multiple-Choice Questions. Findings of NAACL 2024. arXiv:2308.11483 [M].
- Wang et al. (2024). Look at the Text: Instruction-Tuned Language Models are More Robust Multiple Choice Selectors than You Think. COLM 2024. arXiv:2404.08382 [V]. First-token probabilities are brittle. MiDM scores the option content instead.
- Choi et al. (2025). Mitigating Selection Bias with Node Pruning and Auxiliary Options. ACL 2025. arXiv:2409.18857 [V]. Its CKLD bias metric could be reported for MiDM.
- (optional) Quantifying and Mitigating Selection Bias in LLMs: A Transferable LoRA Fine-Tuning and Efficient Majority Voting Approach. arXiv:2511.21709. The title and ID were checked; the authors and venue are UNVERIFIED.

## (c) Data mixture and task breadth
- Sanh et al. (2022). T0: Multitask Prompted Training Enables Zero-Shot Task Generalization. ICLR 2022. arXiv:2110.08207 [M].
- Wang et al. (2022). Super-NaturalInstructions. EMNLP 2022. arXiv:2204.07705 [M].
- Chung et al. (2024). Scaling Instruction-Finetuned Language Models (Flan). JMLR 2024. arXiv:2210.11416 [M].
- Xie et al. (2023). DoReMi. NeurIPS 2023. arXiv:2305.10429 [M].
- Ye et al. (2025). Data Mixing Laws. ICLR 2025. arXiv:2403.16952 [M].

## (d) PEFT variants
- Hu et al. (2022). LoRA. ICLR 2022. arXiv:2106.09685 [M].
- Dettmers et al. (2023). QLoRA. NeurIPS 2023. arXiv:2305.14314 [M].
- Liu et al. (2024). DoRA. ICML 2024. arXiv:2402.09353 [M].
- Kalajdzievski (2023). rsLoRA. arXiv:2312.03732 [V].
- Meng et al. (2024). PiSSA. NeurIPS 2024. arXiv:2404.02948 [M]. Not tested in MiDM.
- Huang et al. (2024). LoraHub. COLM 2024. arXiv:2307.13269 [M].
- Chronopoulou et al. (2023). AdapterSoup. Findings of EACL 2023. arXiv:2302.07027 [V].
- Wortsman et al. (2022). Model Soups. ICML 2022. arXiv:2203.05482 [M].

## (e) Cascades, routing and selective prediction
- Chen, Zaharia & Zou (2024). FrugalGPT. TMLR 2024. arXiv:2305.05176 [M].
- Ding et al. (2024). Hybrid LLM. ICLR 2024. arXiv:2404.14618 [V].
- Ong et al. (2025). RouteLLM. ICLR 2025. arXiv:2406.18665 [V].
- Aggarwal, Madaan et al. (2024). AutoMix. NeurIPS 2024. arXiv:2310.12963 [V]. It escalates by self-verification; MiDM uses its own softmax confidence instead.
- Geifman & El-Yaniv (2017). Selective Classification for Deep Neural Networks. NeurIPS 2017. arXiv:1705.08500 [M].
- Stepanov et al. (2026). SCX Router: Streaming Zero-Shot Model Selection with a Decoder-KV Classifier and a Real-World Task Ontology. arXiv:2609.02292 [V], preprint. It is a Qwen3 decoder with a scorer that ranks model labels without generating text. This is close to the router story.

## (f) Calibration
- Guo et al. (2017). On Calibration of Modern Neural Networks. ICML 2017. arXiv:1706.04599 [M].
- Kadavath et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221 [M].
- Tian et al. (2023). Just Ask for Calibration. EMNLP 2023. arXiv:2305.14975 [M].
- Jiang et al. (2021). How Can We Know When Language Models Know? TACL 2021. arXiv:2012.00955 [M].

## (g) Model selection under shift and ID–OOD correlation
- Miller et al. (2021). Accuracy on the Line. ICML 2021. arXiv:2107.04649 [M].
- Gulrajani & Lopez-Paz (2021). In Search of Lost Domain Generalization. ICLR 2021. arXiv:2007.01434 [M].
- Baek et al. (2022). Agreement-on-the-Line. NeurIPS 2022. arXiv:2206.13089 [M].
- Kumar et al. (2022). Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution. ICLR 2022. arXiv:2202.10054 [V]. A mechanism for why the 2nd epoch hurts transfer.

## (h) Contamination and statistical reporting
- Sainz et al. (2023). NLP Evaluation in Trouble. Findings of EMNLP 2023. arXiv:2310.18018 [M].
- Oren et al. (2024). Proving Test Set Contamination in Black-Box Language Models. ICLR 2024. arXiv:2310.17623 [M].
- Berg-Kirkpatrick, Burkett & Klein (2012). An Empirical Investigation of Statistical Significance in NLP. EMNLP-CoNLL 2012 [M].
- Dror et al. (2018). The Hitchhiker's Guide to Testing Statistical Significance in NLP. ACL 2018 [M].
- Madaan et al. (2024). Quantifying Variance in Evaluation Benchmarks. arXiv:2406.10229 [V].
- Miller (2024). Adding Error Bars to Evals. arXiv:2411.00640 [V].
- Colas, Sigaud & Oudeyer (2018). How Many Random Seeds? arXiv:1806.08295 [M].
- Dodge et al. (2020). Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping. arXiv:2002.06305 [M].

## (i) Close work since 2025 (novelty threats)
- **CLM-8B.** Kwok et al., from Stanford Scaling Intelligence Lab, Hazy Research and NVIDIA, released 2026-09-23. The model card exists at huggingface.co/Contrastive-LM/CLM-v0.1-8B [V]. **No arXiv paper or peer-reviewed paper was found.** Cite it as a model card or blog. The full author list is UNVERIFIED.
- **TypeSafe Jev.** A commercial "System One" model. **No technical report was found.** Cite it as product documentation.
  - Downstream arXiv papers: Barbosa (2026). Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers. arXiv:2609.28940 [V].
  - Jev-Mem, arXiv:2609.23986, and JevVibe, arXiv:2609.34963: the titles were checked; the authors are UNVERIFIED.
  - "Laya" and AgentJev-0.6B (GitHub malevrigns/agent-jev) are UNVERIFIED.
- jina-reranker-v3 (arXiv:2509.25085) and SCX Router (arXiv:2609.02292), both covered above.
- Zhang et al. (2025). Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv:2506.05176. The paper's existence was checked; the author list is from memory. Its reranker is a pointwise Qwen3 cross-encoder that scores from logits.

---

## Draft Related Work (~560 words)

**Decoder LLMs as scorers and rankers.** Cross-encoders that read a query and candidate jointly have long dominated reranking [Nogueira2019]. With decoder LLMs, a last-token classification head trained with LoRA can replace generation [Li2023]. Listwise reranking has moved from generating permutations to single-pass scoring: FIRST reads candidate order from first-token logits [Reddy2024]. jina-reranker-v3 places the query and all documents in one causal context and scores each document from its final-token embedding [Wang2025]. E2Rank shows that an embedding model can itself act as a listwise reranker [Liu2025]. MiDM applies this listwise, last-token-per-option design to typed agent decisions rather than document relevance. It trains with soft targets and compares directly against a bi-encoder alternative, CLM [Kwok2026], and against a commercial System One model, Jev.

**Selection bias in multiple-choice scoring.** LLMs prefer particular option IDs and positions [Zheng2024; Pezeshkpour2024], and first-token probabilities are less robust than the model's textual answer [Wang2024]. Debiasing ranges from permutation-based priors to pruning the parameters that carry the bias [Choi2025]. MiDM avoids label tokens by pointing at option lines and shuffles the options during training. After that, permutation test-time augmentation brings no further gain.

**Data breadth and mixtures.** Multitask instruction mixtures drive zero-shot transfer [Sanh2022; Wang2022; Chung2022], and mixture weights can be optimized or predicted [Xie2023; Ye2025]. Our results sharpen this picture. Adding task breadth alone had no effect. Adding knowledge multiple-choice data (ARC/OBQA/CSQA) improved transfer by about 3 pp but cost about 1.7 pp in-distribution.

**Parameter-efficient fine-tuning.** We build on LoRA and QLoRA [Hu2022; Dettmers2023]. We also tested several proposed improvements to them:
- weight decomposition [Liu2024];
- rank-stabilized scaling [Kalajdzievski2023];
- higher rank;
- adapter averaging and composition [Wortsman2022; Chronopoulou2023; Huang2024].

None of them improved transfer in our setting. Moving to a newer base-model generation, by contrast, yielded 4–5 pp.

**Cascades and routing.** Cost-aware cascades and routers send queries between small and large models using learned scorers, preference data or self-verification [Chen2024; Ding2024; Ong2025; Aggarwal2024]. Our confidence cascade (0.6B→4B→8B) is a form of selective prediction [Geifman2017]. It uses the decision head's own probability, so it needs no separate router or verifier call. Recent routers that score without generating text take a similar approach [Stepanov2026].

**Calibration.** Temperature scaling is the standard post-hoc fix [Guo2017], and LLM self-knowledge and verbalized confidence have been studied extensively [Kadavath2022; Tian2023; Jiang2021]. As expected, temperature scaling did not change MiDM's argmax accuracy. It matters only for the cascade threshold.

**Model selection under distribution shift.** In-distribution (ID) and out-of-distribution (OOD) accuracy are often linearly related [Miller2021; Baek2022]. Selecting models on ID validation data can still mislead under shift [Gulrajani2021], and extra fine-tuning can distort the pretrained features that OOD performance relies on [Kumar2022]. Two of our findings fit the second view: a second epoch hurts transfer, and ID dev accuracy does not select for it.

**Evaluation hygiene.** Contamination can inflate benchmark results [Sainz2023; Oren2024], and small differences call for paired tests and seed-variance estimates [Berg-Kirkpatrick2012; Dror2018; Dodge2020; Madaan2024; Miller2024]. We follow these practices and use them to audit CLM's DeepSWE claim. Only 13 of the 38 held-out tasks are decidable, and the advantage is not significant (exact p = 0.062).

---

## Novelty risks (how to frame the paper)
1. **jina-reranker-v3** already uses a shared causal context with a last-token embedding per candidate, on a 0.6B Qwen3. Do not present the pointer head as a new architecture. Frame the contribution as three things: applying it to typed decisions, the training recipe, and the controlled empirical findings.
2. **Last-token heads on LoRA decoders are standard** (LS-LLaMA, AutoModelForSequenceClassification).
3. **One-pass listwise scoring is established in IR** (FIRST).
4. **Jev and CLM occupy the same niche but have no peer-reviewed papers.** State this explicitly, and say which of their claims we reproduced.
5. **SCX Router (Sept 2026)** is close to the router and cascade story.
6. **The negative PEFT results and "base model matters most" are plausible and partly reported elsewhere.** Present them as careful controlled evidence, not as surprises.
7. **The mismatch between ID dev and transfer echoes Kumar2022 and Gulrajani2021.** Cite them rather than claiming a new phenomenon.
8. **The CLM audit is original but targets a blog claim.** Judge it against the claim as actually stated: 81.6% on a 38-task DeepSWE subset after task-specific fine-tuning.
