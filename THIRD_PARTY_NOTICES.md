# Third-party notices

StillHere's own code is under the license in [LICENSE](LICENSE). The following third-party works are **used with** StillHere. Unless
stated otherwise they are **not redistributed** in this repository; each keeps its own license, which you must follow when you
download or use it.

| Work | Used for | In this repository? | License (check the upstream source for the authoritative text) |
|---|---|---|---|
| Qwen/Qwen3-8B (revision `b968826d`) | local chat model, QLoRA base | no, downloaded by you (`persona download-model base`) | Apache-2.0 (model card) |
| Qwen/Qwen3-Embedding-0.6B (revision `97b0c614`) | dense memory index | no, downloaded by you (`persona download-model embedding`) | Apache-2.0 (model card) |
| DeepSeek API (`deepseek-flash`) | optional answer / judge provider | no, a remote service | DeepSeek terms of service |
| LLaMA-Factory | QLoRA training inside the Colab / GPU package | no, installed at training time | Apache-2.0 |
| Hugging Face `transformers`, `peft`, `bitsandbytes`, `accelerate`, `huggingface_hub`; PyTorch; NumPy | optional extras | no, installed with pip | Apache-2.0 / MIT / BSD (per package) |
| LifeBench 2.0 dataset (`C1754955896/Lifebenchv2.0`, revision `3d06052d`) | benchmark | no, downloaded at a pinned revision; question ids only appear in evaluation code | Apache-2.0 (dataset card) |
| LifeBench_eval (`1754955896/LifeBench_eval`, commit `2bfda547`) | official judge prompt, cloned at evaluation time | no | no license stated upstream: not redistributed |
| Mem0 OSS (`mem0ai` 2.2.1) | baseline in the LifeBench pilot | no, a pip dependency of the evaluation only | Apache-2.0 |
| Montserrat Bold (logo, icon and wordmark) and IBM Plex Sans (social-preview tagline) | the StillHere brand assets in `assets/brand/`, converted to outlines | outlines only, no font files | SIL Open Font License 1.1 |

The synthetic examples in `examples/synthetic/` (including the Alice / Bob WeFlow export) are invented for this project.
"WeFlow" names a third-party WeChat export tool; StillHere only reads its JSON format and is not affiliated with it.
