# FAQ

**Do I need a GPU?**
No. Build the persona and memory on CPU, then chat through the DeepSeek API. A GPU is only needed to run Qwen3-8B locally or to
train an adapter.

**Do I need to train anything?**
No. The default persona is prompt-only (Persona Skill), and the adapter is off. Training is an optional experiment.

**Is my chat uploaded anywhere?**
Not unless you choose to. An API provider receives the persona prompt, the retrieved evidence and your message, and only with
`--allow-external-api`. Training on Colab uploads the training data you put in Drive. See [privacy.md](privacy.md).

**Which chat formats are supported?**
WeFlow JSON exports (WeChat). Other formats are not supported yet.

**Does it work for English chats?**
The pipeline runs on any text, but the persona statistics, examples and evaluation target Chinese chats. English-specific tuning
is on the roadmap.

**What does `ACTIVE_MEMORY_PIPELINE: V2L_P2_FALLBACK` mean?**
The product pipeline (dense + lexical retrieval + P2) is configured, but dense retrieval is unavailable (no embedding model or
index), so memory runs lexically plus P2. Build the dense index to get `V2D_P2`. See [memory.md](memory.md).

**Will the persona make things up?**
It is designed not to. Facts must come from retrieved records, the reader's gates reject unsupported or misattributed answers,
and with no evidence it says it does not remember. No system is perfect: in the benchmark, about 1 in 10 answers was still
wrong without a refusal.

**Is StillHere better than Mem0?**
On the 100-question LifeBench pilot the two were on par (63% vs. 62%, not significant), at about 16× lower estimated API cost for
memory construction plus answering. See [benchmark.md](benchmark.md).

**Can I have several personas?**
Yes. Each profile has its own card, memory and configuration; use `--profile` to choose. `stillhere persona status` lists them.

**Can I use another local model?**
Only Qwen3-8B (or a directory with a compatible Qwen3 chat template) is supported for local chat and training. Any
OpenAI-compatible API can be used through `openai_compatible_custom`.

**How do I update a persona when I have a newer export?**
`stillhere persona memory-build --profile <id> --source <newer-export.json>`. Repeated imports never duplicate messages.
