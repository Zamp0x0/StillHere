# Training on Google Colab

## 1. Build the package (on your computer)

```bash
stillhere persona package --profile <profile-id> --target colab-l4 --confirm-cloud     # or --target colab-a100
```

The command prints `package_id`, `package_dir` and `notebook`. The package directory contains:

| File | What |
|---|---|
| `persona_code.zip` | StillHere's code needed on the GPU (`src/stillhere`, `training/...`, `pyproject.toml`) |
| `persona_data.zip` | the profile's train / dev data and the training config: **your private chat data** |
| `RELEASE_MANIFEST.json`, `SHA256SUMS.txt` | versions and hashes, checked by the notebook |
| `StillHere_Persona_L4.ipynb` | the notebook |
| `README_COLAB_CN.md` | short instructions (Chinese) |

`--confirm-cloud` is required because the data leaves your computer. Without it, cloud targets are refused.

## 2. Upload to Google Drive

- Put the four data and code files in `MyDrive/StillHere/persona/<package-id>/input/`.
- Open the notebook in Colab, for example by uploading it to Drive and opening it with Colab.
- The notebook's `DRIVE_DIR` and `PACKAGE_ID` (STAGE 00) must match.

## 3. Run

1. **Runtime:** choose *Runtime → Change runtime type → GPU → L4*, or A100 for an A100 package.
2. **Confirm the upload:** in STAGE 00, set `CONFIRM_PRIVATE_DATA_ON_CLOUD = True` only if you accept that the training data is
   in your Google Drive and Colab.
3. **Run all.** The stages:
   - 00–04: configuration, environment, data import, sha256 checks, pinned installs;
   - 05–06: base model and tokenizer checks, GPU preflight;
   - 07: loss-mask check;
   - 08: a 2-step smoke test;
   - 09–10: training;
   - 11: checkpoint selection on dev;
   - 12–14: export.
4. **Resume:** if Colab disconnects, reconnect and *Run all* again with the same `RUN_ID`. The stage status and checkpoints live on
   Drive, so finished work is not repeated. A different dataset or config under the same `RUN_ID` is refused.

Each stage prints `PASS / FAIL / BLOCKED / SKIP`. The notebook's stage notes are written in Chinese.

## 4. Import the adapter

- Download the adapter ZIP from `MyDrive/StillHere/persona/<package-id>/runs/<run-id>/exports/`, then:

  ```bash
  stillhere persona import-adapter --profile <profile-id> --file <adapter-zip>
  stillhere persona finalize --profile <profile-id> --adapter on
  stillhere persona chat --profile <profile-id> --base-dir <qwen3-8b-dir>
  ```

- Running an adapter needs a local CUDA GPU with the same base model (Qwen3-8B @ `b968826d`). API models cannot load adapters.

## Troubleshooting

| Problem | What to do |
|---|---|
| `CONFIRM_PRIVATE_DATA_ON_CLOUD` assertion | set it to `True` in STAGE 00 if you accept the upload |
| files not found in STAGE 02 | check `DRIVE_DIR / PACKAGE_ID / input/` holds the four files |
| sha256 mismatch | re-upload the files from the same package; do not mix packages |
| wrong GPU | choose L4 (or A100 for an A100 package); `ALLOW_OTHER_GPU = True` is for experiments only |
| disconnected | reconnect, *Run all* with the same `RUN_ID` |
| no checkpoint passes the selection rule | nothing is exported by default; read the STAGE 11 report (`EXPORT_UNQUALIFIED = True` exports the last one marked UNQUALIFIED, for diagnosis only) |
