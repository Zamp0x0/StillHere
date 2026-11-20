# Limitations

StillHere is a research-grade system released as version 0.1.0. Known limits:

## Evaluation

- **Personalisation:** evaluated on **one real persona**, with AI-verified (not human) reference answers and an AI judge. Evidence
  level: low.
- **Memory benchmarks:** one persona per study, 100 questions each. Confidence intervals are wide, and the answer model and judge
  are the same model family.
- **P2:** the accuracy gain on the independent persona is **not statistically significant**. The recall gain is.
- **P2 ablation:** there is no per-module ablation of P2.
- **Benchmark vs. product:** the benchmark uses one shared answer function. The product's Reader p1 and safety gates, and the
  product's question-answering gain from P2, were not evaluated end to end on real chats.

## Persona

- **Adapters:** trained adapters did not reliably outperform the prompt-only persona. One collapsed into repetitive replies under
  the Skill prompt, so adapters are off by default.
- **Language:** the persona statistics and examples target Chinese WeChat conversations.

## Memory

- **Rule-based P2:** P2's question forms are rule-based. Wordings it does not recognise fall back to the first pass, and records
  that retrieval cannot find at all are not recovered.
- **Token budget:** extra second-pass evidence can push other evidence out of the token budget. In the P2 study this caused most
  of the new errors.
- **Calendar notes:** plans and actual events are distinguished only where the records say so (calendar notes). StillHere does not
  infer whether a plan happened.
- **Writer:** the semantic memory writer is optional and experimental, and the product writer mode is EVIDENCE_ONLY.

## Platform

- **Local models:** only Qwen3-8B (and compatible Qwen3 chat templates) are supported locally.
- **Local training:** local adapter training needs at least 20 GiB VRAM and was not verified on local hardware by the project.
- **Remote training:** remote GPU training is package-only; there is no remote scheduler.
- **Colab notebook:** the training notebook's stage notes are in Chinese.
- **Studio:** the local Studio web UI is a minimal tool, not a polished product interface.
- **Import formats:** only WeFlow JSON exports are supported.
