# Installation

## Requirements

| | Minimum | Notes |
|---|---|---|
| Python | 3.10 | tested on 3.10 and 3.12 (CI), Windows and Linux |
| OS | Windows 10/11, macOS, Linux | macOS is expected to work for the CPU / API paths but is not in CI |
| Disk | ~50 MB | + 1.2 GB for dense retrieval, + 16 GB for a local Qwen3-8B |
| GPU | none | only for local chat (CUDA) or local training (>= 20 GiB VRAM) |

## Install from source

```bash
git clone https://github.com/Zamp0x0/StillHere.git StillHere
cd StillHere
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1   (cmd: .venv\Scripts\activate.bat)
python -m pip install --upgrade pip
pip install -e .
stillhere --help
```

On Windows PowerShell, if script activation is blocked, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `cmd` and `.venv\Scripts\activate.bat`.

Use an **editable install from a clone** (`pip install -e .`). Building a training package ships the repository's own `src/` and
`training/` folders, so it needs the source tree, not just an installed wheel.

## Optional extras

| Extra | Install | Adds |
|---|---|---|
| `embedding` | `pip install -e ".[embedding]"` | dense (V2D) memory with Qwen3-Embedding-0.6B on CPU or GPU (`torch`, `transformers`, `numpy`) |
| `download` | `pip install -e ".[download]"` | `stillhere persona download-model` (`huggingface_hub`) |
| `gpu` | `pip install -e ".[gpu]"` | local Qwen3-8B chat with NF4 quantisation (`torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`) |
| `tokenizer` | `pip install -e ".[tokenizer]"` | exact Qwen3 token counts instead of an estimate |
| `dev` | `pip install -e ".[dev]"` | tests (`pytest`, `numpy`) |

For CUDA, install the PyTorch build that matches your driver first. Follow the selector at pytorch.org, then install the extra.

## Check the installation

```bash
stillhere version
stillhere persona doctor              # what this machine can do (GPU, VRAM, bf16, NF4, library versions)
python -m pytest -q                   # with the dev extra: the CPU test suite
```

## Where data goes

Profiles are stored in `~/.stillhere/personas` by default. Change this with `STILLHERE_PERSONA_ROOT` or `--root`. The root must
be outside any git work tree, unless git ignores it. See [privacy.md](privacy.md).
