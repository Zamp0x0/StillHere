"""``stillhere persona ...`` sub-commands (argparse; standard library only). Parsing, prompting and printing only: every rule lives in the
service modules (``service``, ``card``, ``package``, ``training``, ``adapters``, ``evaluation``, ``doctor``, ``runtime``).

    stillhere persona quickstart --input <weflow.json> [--speaker ...] [--dense auto|on|off]   import -> clean -> card -> memory (V2D) -> finalize
    stillhere persona pipeline --input <weflow.json> --target colab-l4 [--speaker 1|2|contact|self|<id>] [--confirm-cloud]   (all of the below)
    stillhere persona inspect  --input <weflow.json> [--json]                       read-only aggregate audit
    stillhere persona init     --input <weflow.json> [--speaker ...] [--profile <id>] [--display-name <name>]
    stillhere persona prepare  --profile <id>                                       dataset + automatic persona card / prompt
    stillhere persona review   --profile <id> [--edit | --from-file <card.yaml> | --approve]
    stillhere persona doctor   --target cpu-only|local-cuda|colab-l4|colab-a100 [--base-dir <Qwen3-8B>]
    stillhere persona package  --profile <id> --target colab-l4 [--confirm-cloud] [--base-dir <Qwen3-8B>]
    stillhere persona train    --profile <id> --package <pk-...> [--device cuda] [--dry-run] [--base-dir ...]
    stillhere persona import-adapter --profile <id> --file <persona_candidate_*.zip>
    stillhere persona evaluate --profile <id> [--split dev|test] [--groups A,B,C,D,E] [--adapter <a-...>] [--backend hf|fake] [--blind B,C]
    stillhere persona chat     --profile <id> [--backend hf|fake] [--message <text>]          (runtime_config.json; --adapter / --memory override)
                               [--answer-provider deepseek_api|openai_compatible_custom|huggingface_local --answer-model <id> --allow-external-api]
    stillhere persona provider show|check [--role answer|card_assist|memory_extraction|router|judge] [--allow-external-api]
    stillhere persona card list|show|activate|assist --profile <id> [--card <c-...>]       one active card; versions are never deleted
    stillhere persona memory-build --profile <id> [--writer none|hf|deepseek] [--source <newer export>]   evidence archive (+ semantic memory)
    stillhere persona memory-audit --profile <id>      stillhere persona memory-query --profile <id> --q <question> [--at <iso time>]
    stillhere persona refine   --profile <id> [--package --confirm-cloud | --generate --backend hf | --import <zip> | --select]
    stillhere persona finalize --profile <id> [--adapter on|off|auto] [--memory on|off|auto] [--memory-pipeline product|product-v2d|research-final|v1]
    stillhere persona studio   [--port 8765] [--backend fake|hf] [--base-dir <Qwen3-8B>]          local web UI (127.0.0.1 only)

Local processing needs no form: choosing the person and running ``prepare`` is enough. Nothing is uploaded; a cloud package is only
written after you confirm (``--confirm-cloud`` or an interactive yes) that a private chat will leave this machine when YOU upload it.
Exit codes: 0 ok, 1 cancelled, 2 bad input, 3 refused (cloud not confirmed / GPU insufficient / test sealed), 4 identity, 5 profile / store,
6 dataset / gates / card / package / adapter problem. Standard output never contains chat text, except the display names on the
interactive selection screen and ``review`` / ``chat`` (both local, both asked for).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from . import adapters as AD
from . import card as K
from . import dataset as D
from . import doctor as DR
from . import evaluation as EV
from . import identity as I
from . import memory_store as MS
from . import package as P
from . import pipeline as PL
from . import quality as Q
from . import service as svc
from . import store as S
from . import training as TR
from . import weflow as W
from .prompt_writer import PromptError as PromptErr
from stillhere.providers.llm import ProviderError

CODES = [(ProviderError, 6), (PromptErr, 6), (W.WeFlowError, 2), (I.IdentityError, 4), (TR.TrainingRefused, 3), (MS.MemoryError_, 6), (S.StoreError, 5), (D.DatasetError, 6), (K.CardError, 6),
         (P.PackageError, 6), (AD.AdapterError, 6), (EV.EvalError, 6), (ValueError, 2)]
REFUSED = {"CLOUD_NOT_CONFIRMED", "TEST_SEALED", "TEST_ALREADY_LOCKED", "GPU_INSUFFICIENT", "DOCTOR_BLOCKED", "DEVICE", "EXTERNAL_API_NOT_ALLOWED", "NO_GPU", "OWNER_REVIEW_REQUIRED", "NO_API_KEY", "ADAPTER_NEEDS_LOCAL_BASE"}
TARGET_HELP = "cpu-only (data + prompt), local-cuda (check this GPU), colab-l4 (default), colab-a100"


def register(sub) -> None:
    p = sub.add_parser("persona", help="persona fine-tuning pipeline (WeFlow import -> dataset -> persona prompt -> QLoRA package -> adapter -> evaluation)")
    ps = p.add_subparsers(dest="persona_cmd", required=True)

    def common(x, profile=True):
        if profile:
            x.add_argument("--profile", required=True)
        x.add_argument("--root", help=f"private profile root (default: ${S.ROOT_ENV} or ~/.stillhere/personas)")
        x.add_argument("--json", action="store_true")

    pl = ps.add_parser("pipeline", help="one command: choose the person, clean, dataset, persona card / prompt, label audit, training package")
    pl.add_argument("--input", required=True)
    pl.add_argument("--target", default="colab-l4", choices=P.TARGETS, help=TARGET_HELP)
    _speaker_args(pl)
    pl.add_argument("--confirm-cloud", action="store_true", help="I will take this private chat package to a cloud GPU myself (colab targets)")
    pl.add_argument("--base-dir", help="local Qwen3-8B snapshot (pinned revision) for the CPU label audit")
    pl.add_argument("--tokenizer")
    pl.add_argument("--writer", default="none", choices=["none", "local", "hf", "deepseek"], help="semantic memory extraction (default: evidence only, SEMANTIC_EXTRACTION_PENDING)")
    pl.add_argument("--allow-external-api", action="store_true", help="allow the deepseek writer to send chat text to an external API")
    pl.add_argument("--accept-auto-verified-by", "--reviewed-by", dest="reviewed_by", help="your name: you accept AUTO-VERIFIED memories from this real chat into your private store (recorded as auto-verified, not human-reviewed)")
    pl.add_argument("--only", help="comma-separated stage numbers to (re)run, e.g. 08,09,10,11")
    common(pl, profile=False)
    qs = ps.add_parser("quickstart", help="the basic flow: import, choose the person, clean, persona card, memory (V2D, V2L fallback), finalize")
    qs.add_argument("--input", required=True)
    _speaker_args(qs)
    qs.add_argument("--dense", default="auto", choices=["auto", "on", "off"], help="product dense index (auto: when the local embedding model exists)")
    qs.add_argument("--memory-pipeline", choices=["product", "product-v2d", "research-final", "v1"], help="default: product for a new profile, else kept")
    qs.add_argument("--tokenizer")
    common(qs, profile=False)
    pv = ps.add_parser("provider", help="model providers per role (answer, card_assist, memory_extraction, router, judge); secrets masked")
    pv.add_argument("action", choices=["show", "check"])
    pv.add_argument("--role", default="answer")
    pv.add_argument("--profile")
    pv.add_argument("--root")
    pv.add_argument("--allow-external-api", action="store_true", help="check sends ONE tiny request to the configured API")
    pv.add_argument("--json", action="store_true")
    cd = ps.add_parser("card", help="persona card versions: list / show (prompt preview) / activate (or roll back) / assist (LLM draft)")
    cd.add_argument("action", choices=["list", "show", "activate", "assist"])
    common(cd)
    cd.add_argument("--card", help="card id (c-...) for show / activate")
    cd.add_argument("--allow-external-api", action="store_true")
    cd.add_argument("--send-examples", action="store_true", help="assist: also send example replies (private chat lines) to the provider")
    cd.add_argument("--activate", action="store_true", help="assist: make the new draft the active card")
    i = ps.add_parser("inspect", help="read-only aggregate audit of an export (writes nothing)")
    i.add_argument("--input", required=True)
    i.add_argument("--config")
    i.add_argument("--tokenizer", help="pinned Qwen3 tokenizer.json for exact token counts (else STILLHERE_QWEN3_TOKENIZER, else estimated)")
    i.add_argument("--json", action="store_true")
    n = ps.add_parser("init", help="choose the target person and create a private profile")
    n.add_argument("--input", required=True)
    _speaker_args(n)
    common(n, profile=False)
    r = ps.add_parser("prepare", help="clean, rebuild dialogues, versioned train / dev / test, automatic persona card + prompt")
    common(r)
    r.add_argument("--config")
    r.add_argument("--tokenizer")
    v = ps.add_parser("review", help="show the persona card; edit or approve it (a new version, the old one is kept)")
    common(v)
    v.add_argument("--edit", action="store_true", help="open the card in an editor ($EDITOR, else notepad / nano) and save the result as a new version")
    v.add_argument("--from-file", help="an edited persona_card.yaml to save as a new USER_REVIEWED version")
    v.add_argument("--approve", action="store_true", help="mark the current card USER_REVIEWED without changes")
    v.add_argument("--show-prompt", action="store_true", help="print the full persona prompt (private, local terminal only)")
    d = ps.add_parser("doctor", help="what this machine can do: GPU, VRAM, bf16, NF4, versions, base files, disk")
    d.add_argument("--target", default="local-cuda", choices=DR.TARGETS)
    d.add_argument("--base-dir")
    d.add_argument("--json", action="store_true")
    k = ps.add_parser("package", help="build the training package (+ Colab notebook) for a target")
    common(k)
    k.add_argument("--target", default="colab-l4", choices=P.TARGETS, help=TARGET_HELP)
    k.add_argument("--confirm-cloud", action="store_true")
    k.add_argument("--base-dir")
    k.add_argument("--run-id")
    k.add_argument("--config", help="training config template (default: training/persona/configs/persona_qlora.yaml; r2: persona_qlora_r2_a100.yaml)")
    k.add_argument("--non-interactive", action="store_true")
    t = ps.add_parser("train", help="train on THIS machine's CUDA GPU (refused below 20 GiB VRAM; use the Colab notebook then)")
    common(t)
    t.add_argument("--package", required=True)
    t.add_argument("--device", default="cuda")
    t.add_argument("--base-dir")
    t.add_argument("--run-id")
    t.add_argument("--dry-run", action="store_true")
    t.add_argument("--smoke-only", action="store_true")
    t.add_argument("--gpu", type=int, help="physical GPU index: CUDA_VISIBLE_DEVICES=<gpu>, seen as cuda:0 inside the trainer")
    a = ps.add_parser("import-adapter", help="verify and install a persona adapter package ZIP")
    common(a)
    a.add_argument("--file", required=True)
    e = ps.add_parser("evaluate", help="A-E ablation on dev (or the sealed test, once)")
    common(e)
    e.add_argument("--split", default="dev", choices=["dev", "test"])
    e.add_argument("--groups", default="A,B,C,D,E")
    e.add_argument("--adapter", help="installed adapter id (a-...) for groups C / E")
    e.add_argument("--backend", default="fake", choices=["hf", "hf-lowvram", "fake"], help="hf: real Qwen3-8B on CUDA; fake: CPU dry run (DRY_RUN_ONLY)")
    e.add_argument("--base-dir")
    e.add_argument("--limit", type=int)
    e.add_argument("--unseal-test", action="store_true")
    e.add_argument("--blind", help="also write a blind-review CSV for a pair, e.g. B,C")
    e.add_argument("--score-blind", help="score a filled BLIND_REVIEW.csv (its BLIND_KEY.json next to it)")
    c = ps.add_parser("chat", help="talk to the persona (runtime_config.json; or explicit prompt [+ adapter] [+ memory])")
    common(c)
    c.add_argument("--adapter", help="override: an installed adapter id")
    c.add_argument("--memory", choices=["production", "eval", "off"], help="override the memory view")
    c.add_argument("--backend", choices=["auto", "hf", "hf-lowvram", "fake"], help="default: STILLHERE_LOCAL_BACKEND / .env (auto)")
    c.add_argument("--base-dir")
    c.add_argument("--message", help="one message, one reply (non-interactive)")
    c.add_argument("--answer-provider", choices=["deepseek_api", "openai_compatible_custom", "huggingface_local"], help="default: STILLHERE_ANSWER_PROVIDER (huggingface_local)")
    c.add_argument("--answer-model", help="e.g. deepseek-flash, or your endpoint's model id")
    c.add_argument("--answer-base-url", help="openai_compatible_custom: the endpoint base URL")
    c.add_argument("--allow-external-api", action="store_true", help="allow the persona prompt, retrieved evidence and the message to be sent to the API provider")
    c.add_argument("--router", choices=["rules", "llm"], help="default: STILLHERE_ROUTER_MODE (rules)")
    mb = ps.add_parser("memory-build", help="WeFlow -> conversation evidence archive (+ semantic memory with a writer model)")
    common(mb)
    mb.add_argument("--source", help="a newer export of the same chat (incremental import)")
    mb.add_argument("--writer", default="none", choices=["none", "local", "hf", "deepseek"])
    mb.add_argument("--allow-external-api", action="store_true")
    mb.add_argument("--accept-auto-verified-by", "--reviewed-by", dest="reviewed_by", help="your name: you accept AUTO-VERIFIED memories from this real chat into your private store (recorded as auto-verified, not human-reviewed)")
    mb.add_argument("--base-dir")
    mb.add_argument("--limit", type=int, help="process at most N sessions this run (resumable)")
    mb.add_argument("--dense", action="store_true", help="also build / update the product dense index (local Qwen3-Embedding-0.6B)")
    ma = ps.add_parser("memory-audit", help="memory quality report: imported / searchable / extracted / verified / persisted / retrievable")
    common(ma)
    mq = ps.add_parser("memory-query", help="EF1 retrieval over the production memory (local terminal: shows private text)")
    common(mq)
    mq.add_argument("--q", required=True)
    mq.add_argument("--at", help="ask time (ISO); default now")
    rf = ps.add_parser("refine", help="quality diagnosis + prompt-first refinement on DEV (never TEST)")
    common(rf)
    rf.add_argument("--package", action="store_true", help="build the Colab evaluation package for the candidates")
    rf.add_argument("--confirm-cloud", action="store_true")
    rf.add_argument("--generate", action="store_true", help="generate the candidates locally (--backend hf on a CUDA GPU)")
    rf.add_argument("--backend", default="hf", choices=["hf", "hf-lowvram", "fake"])
    rf.add_argument("--base-dir")
    rf.add_argument("--adapter")
    rf.add_argument("--import", dest="import_zip", help="import the Colab refine results ZIP")
    rf.add_argument("--select", action="store_true", help="apply the pre-registered rule and adopt the best candidate (or keep the original)")
    rf.add_argument("--blind", help="write a QUICK blind review (5-10 pairs) for a pair, e.g. B,C")
    rf.add_argument("--non-interactive", action="store_true")
    fz = ps.add_parser("finalize", help="persona quality report + runtime_config.json (adapter / memory selection)")
    common(fz)
    fz.add_argument("--adapter", choices=["on", "off", "auto"], default="auto")
    fz.add_argument("--memory", choices=["on", "off", "auto"], default="auto")
    fz.add_argument("--memory-pipeline", choices=["product", "product-v2d", "research-final", "v1"], help="default: keep the profile's pipeline (product for a new profile)")
    so = ps.add_parser("studio", help="local Persona Studio web UI (127.0.0.1 only)")
    so.add_argument("--root")
    so.add_argument("--port", type=int, default=8765)
    so.add_argument("--backend", choices=["auto", "fake", "hf", "hf-lowvram"], help="default: STILLHERE_LOCAL_BACKEND / .env (auto)")
    so.add_argument("--base-dir")
    sts = ps.add_parser("status", help="local profiles; or one profile: card, messages, memory, dense index, ACTIVE_MEMORY_PIPELINE, adapter, setup notes")
    sts.add_argument("--profile", help="default: list every profile under the root")
    sts.add_argument("--root", help="private profile root (default: $STILLHERE_PERSONA_ROOT or ~/.stillhere/personas)")
    sts.add_argument("--json", action="store_true")
    dm = ps.add_parser("download-model", help="download a pinned model from Hugging Face: embedding (Qwen3-Embedding-0.6B, ~1.2 GB) or base (Qwen3-8B, ~16 GB)")
    dm.add_argument("what", choices=["embedding", "base"])
    dm.add_argument("--dir", help="target directory (default: the Hugging Face cache, where the embedding model is found automatically)")
    dm.add_argument("--json", action="store_true")
    cf = ps.add_parser("config", help="show the effective local model / prompt-provider configuration (secrets masked)")
    cf.add_argument("--profile")
    cf.add_argument("--root")
    cf.add_argument("--json", action="store_true")
    pw = ps.add_parser("prompt", help="Persona Prompt Generator: provider prompt, or an authored prompt file, saved as a validated candidate version")
    common(pw)
    pw.add_argument("--provider", choices=["deterministic", "openai_compatible"], help="default: STILLHERE_PROMPT_PROVIDER (deterministic)")
    pw.add_argument("--from-file", help="a prompt text written from the persona card (e.g. by a person or an assistant session)")
    pw.add_argument("--author", default="manual", help="who wrote the --from-file prompt (recorded in the card version)")
    pw.add_argument("--activate", action="store_true", help="make the new prompt version current (default: keep the current prompt)")
    fp = ps.add_parser("final-package", help="build the ONE Colab L4 input package of the final Style + Memory evaluation (outside git)")
    common(fp)
    fp.add_argument("--old-card", required=True, help="the current prompt's card id (B1 / C1)")
    fp.add_argument("--new-card", required=True, help="the candidate prompt's card id (B2 / C2)")
    fp.add_argument("--adapter", required=True, help="the installed adapter id (C1 / C2 / E)")
    fp.add_argument("--out-dir", help="default: <profile>/packages/final_colab_l4")
    f2 = ps.add_parser("final-package-v2", help="build the second-round Colab evaluation package (r1 vs r2, Memory V1 vs V2, Writer gate; outside git)")
    common(f2)
    f2.add_argument("--skill-card", required=True, help="the runtime persona prompt's card id (r2 was trained with it)")
    f2.add_argument("--r1-adapter", required=True)
    f2.add_argument("--r2-adapter", required=True)
    f2.add_argument("--fast-results", required=True, help="the frozen FAST results ZIP (StillHere_FAST_L4_Results.zip)")
    f2.add_argument("--freeze-manifest", required=True, help="FREEZE_MANIFEST.json written by scripts/freeze_baselines.py")
    f2.add_argument("--out-dir", help="default: <profile>/packages/eval_v2_colab")
    ic = ps.add_parser("import-colab-results", help="verify and import StillHere_Final_L4_Results.zip (memory snapshot + reports)")
    common(ic)
    ic.add_argument("--file", required=True)
    ic.add_argument("--replace-memory", action="store_true", help="replace a local store that already holds OTHER memories (backed up first)")


def _speaker_args(p) -> None:
    p.add_argument("--speaker", "--speaker-id", dest="speaker", help="1 / 2 (list order), contact / self, or a speaker id (u_...)")
    p.add_argument("--profile", help="profile id (default: derived from file + speaker; same input -> same profile)")
    p.add_argument("--display-name", help="the name to use in the persona card (stored in the private profile only)")
    p.add_argument("--non-interactive", action="store_true", help="never prompt; missing choices are errors")
    p.add_argument("--accept-anomalies", action="store_true", help="accept blocking identity anomalies after reviewing them")


def _ask(prompt: str) -> str:
    return input(prompt).strip()


def _interactive(a) -> bool:
    return sys.stdin.isatty() and not getattr(a, "non_interactive", False)


def _print(obj, as_json: bool) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1) if as_json else _plain(obj))


def _plain(obj, indent=0) -> str:
    pad = "  " * indent
    if isinstance(obj, dict):
        return "\n".join(f"{pad}{k}: " + (("\n" + _plain(v, indent + 1)) if isinstance(v, (dict, list)) and v else str(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return "\n".join(f"{pad}- {json.dumps(x, ensure_ascii=False)}" for x in obj)
    return f"{pad}{obj}"


def _choose(a) -> str:
    """Show the two people (real names only on an interactive screen) and return the choice."""
    if not Path(a.input).is_file():
        raise W.WeFlowError("NOT_FOUND", f"input file not found: {a.input}")
    interactive = _interactive(a)
    sp = svc.speakers(a.input)
    out = sys.stderr if getattr(a, "json", False) else sys.stdout           # --json: stdout carries only the JSON result
    print("StillHere Persona Setup\n\nFormat: WeFlow JSON", file=out)
    print(f"Messages: {sum(s['messages'] for s in sp)} (non-system)\n\nDetected speakers:", file=out)
    for s in sp:
        name = s["display_name"] if interactive else s["id"]
        print(f"[{s['index']}] {name} — {'联系人 (contact)' if s['role'] == 'contact' else '导出账号 (self)'} — {s['text_messages']} text — {s['first']} … {s['last']}",
              file=out)
    if a.speaker:
        return a.speaker
    if not interactive:
        raise ValueError("--speaker is required in non-interactive mode (1 / 2 / contact / self / speaker id)")
    for _ in range(3):
        v = _ask(f"\nSelect target speaker [1-{len(sp)}] (q to quit): ")
        if v.lower() == "q":
            return ""
        if v.isdigit() and 1 <= int(v) <= len(sp):
            return v
        print("invalid choice")
    return ""


def _confirm_cloud(a) -> bool:
    if getattr(a, "confirm_cloud", False) or a.target not in P.CLOUD:
        return bool(getattr(a, "confirm_cloud", False))
    print("\n" + S.CLOUD_NOTICE)
    if not _interactive(a):
        return False
    return _ask("Build the cloud training package? [y/N]: ").lower() in ("y", "yes")


def _init(a) -> int:
    choice = _choose(a)
    if not choice:
        print("cancelled")
        return 1
    res = svc.init_profile(a.input, speaker=choice, root=a.root, profile_id=a.profile, display_name=a.display_name, accept_anomalies=a.accept_anomalies)
    print(f"\nprofile: {res['profile_id']} ({'existing, unchanged' if res['reused'] else 'created'})\npath: {res['profile_dir']}")
    print(f"target: {res['speaker']['id']} role={res['speaker']['role']} isSend={res['speaker']['is_send']}")
    print(f"next: stillhere persona prepare --profile {res['profile_id']}")
    return 0


def _pipeline(a) -> int:
    choice = _choose(a)
    if not choice:
        print("cancelled")
        return 1
    confirm = _confirm_cloud(a)
    res = PL.run(a.input, speaker=choice, root=a.root, target=a.target, confirm_cloud=confirm, base_dir=a.base_dir, tokenizer=a.tokenizer,
                 writer=a.writer, allow_external=a.allow_external_api, reviewer=a.reviewed_by, profile_id=a.profile, display_name=a.display_name,
                 accept_anomalies=a.accept_anomalies, only=a.only.split(",") if a.only else None, log=lambda s: print(s, file=sys.stderr))
    out = {"profile_id": res["profile_id"], "overall": res["overall"], "runtime_config": res["runtime_config"], "state_file": res["state_file"],
           "stages": {k: {"stage": v["stage"], "status": v["status"], **{x: v[x] for x in ("reason", "error", "package_id", "notebook", "adapters", "result") if x in v}}
                      for k, v in sorted(res["stages"].items())}}
    _print(out, a.json)
    if not a.json:
        print(f"\nnext: stillhere persona chat --profile {out['profile_id']}   |   stillhere persona studio")
    return 0 if res["overall"] in ("READY", "PARTIAL_READY") else 6


def _review(a) -> int:
    root = Path(a.root) if a.root else S.default_root()
    pdir = S.profile_dir(root, a.profile)
    cur = K.current(pdir)
    if cur is None:
        raise K.CardError("NO_CARD", "no card yet: run stillhere persona prepare")
    if a.edit or a.from_file:
        if a.from_file:
            edited = K.from_yaml(Path(a.from_file).read_text(encoding="utf-8"))
        else:
            tmp = pdir / "cards" / "_edit_persona_card.yaml"
            S.atomic_write(tmp, K.to_yaml(cur["card"]).encode("utf-8"))
            editor = os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "nano")
            print(f"editing {tmp} with {editor}; only {', '.join(K.EDITABLE)} may change. Save and close the editor to continue.")
            subprocess.call([editor, str(tmp)])
            edited = K.from_yaml(tmp.read_text(encoding="utf-8"))
            tmp.unlink()
        res = K.review(pdir, edited=edited)
    elif a.approve:
        res = K.review(pdir, approve=True)
    else:
        c, m = cur["card"], cur["manifest"]
        out = {"card_id": cur["card_id"], "review_status": m["review_status"], "dataset_version": m["dataset_version"], "prompt_sha256": m["prompt_sha256"],
               "card_dir": cur["dir"], "display_name": c["display_name"], "display_name_needs_review": c["display_name_needs_review"],
               "reply_length": c["reply_length"], "common_expressions": [x["text"] for x in c["common_expressions"]],
               "frequent_short_replies": [x["text"] for x in c["frequent_short_replies"]], "addressing_candidates": [x["text"] for x in c["addressing"]["candidates"]],
               "examples": len(c["examples"]), "user_confirmed_facts": c["user_confirmed_facts"], "editable": list(K.EDITABLE)}
        _print(out, a.json)
        if a.show_prompt:
            print("\n----- persona_prompt.txt -----\n" + cur["prompt"])
        print(f"\nedit: stillhere persona review --profile {a.profile} --edit   |   approve: --approve")
        return 0
    _print({"card_id": res["card_id"], "review_status": res["review_status"], "parent_card": res["parent_card"], "prompt_sha256": res["prompt_sha256"],
            "reused": res["reused"], "card_dir": res["card_dir"], "next": _apply_card_hint(pdir, a.profile)}, a.json)
    return 0


def _apply_card_hint(pdir: Path, profile: str) -> str:
    """A changed card is used by chat only after ``finalize`` re-selects the runtime (chat refuses a stale config, CONFIG_STALE)."""
    if (pdir / "runtime" / "runtime_config.json").is_file():
        return f"stillhere persona finalize --profile {profile}   (makes chat use the active card; until then chat stops with CONFIG_STALE)"
    return f"stillhere persona finalize --profile {profile}"


def _evaluate(a) -> int:
    if a.score_blind:
        csv = Path(a.score_blind)
        _print(EV.score_blind(csv, csv.parent / "BLIND_KEY.json"), a.json)
        return 0
    res = svc.evaluate(a.root, a.profile, split=a.split, groups=a.groups.split(","), adapter_id=a.adapter, backend=a.backend, base_dir=a.base_dir,
                       limit=a.limit, unseal_test=a.unseal_test, blind=a.blind.split(",") if a.blind else None, log=lambda s: print(s, file=sys.stderr))
    _print(res, a.json)
    return 0


def _chat(a) -> int:
    from .runtime import PersonaRuntime
    cfgp = S.profile_dir(Path(a.root) if a.root else S.default_root(), a.profile) / "runtime" / "runtime_config.json"
    answer = {k: v for k, v in (("provider", a.answer_provider), ("model", a.answer_model), ("base_url", a.answer_base_url)) if v}
    ext = True if a.allow_external_api else None
    if a.adapter is None and a.memory is None and cfgp.is_file():
        rt = PersonaRuntime.from_config(a.root, a.profile, backend=a.backend, base_dir=a.base_dir, answer=answer, allow_external_api=ext, router=a.router)
    else:
        mem = None if a.memory in (None, "off") else a.memory
        rt = PersonaRuntime(a.root, a.profile, backend=a.backend, base_dir=a.base_dir, adapters=[a.adapter] if a.adapter else [], memory=mem,
                            answer=answer, allow_external_api=ext, router=a.router or "rules")
        rt.use_adapter(a.adapter)
    print(f"persona chat: card {rt.card['card_id']} ({rt.card['manifest']['review_status']}), adapter {rt.active or 'off'}, memory {rt.memory_view or 'off'}"
          f" ({rt.retrieval}{'+P2' if getattr(rt, 'p2', False) else ''}), generator {rt.gen.label}" + (" (SYNTHETIC / DRY_RUN_ONLY: not a model)" if a.backend == "fake" else ""))
    print(f"ACTIVE_MEMORY_PIPELINE: {getattr(rt, 'active_memory_pipeline', None) or ('OFF' if rt.memory_view is None else rt.retrieval)}")
    for n in rt.memory_notes:
        print(f"note: {n}")
    if not a.message:
        print("(type a message; /q to quit)")
    history = []
    if a.message:
        history.append(("other", a.message))
        print(rt.reply(history)["text"])
        return 0
    while True:
        try:
            msg = _ask("你：")
        except EOFError:
            return 0
        if msg in ("/q", "/quit", "exit"):
            return 0
        history.append(("other", msg))
        r = rt.reply(history)
        history.append(("me", r["text"]))
        print(r["text"])


def _refine(a) -> int:
    if a.import_zip:
        r = P.import_refine_results(a.root, a.profile, a.import_zip)
        _print({**r, "select": Q.refine_select(a.root, a.profile, r["refine_id"])}, a.json)
        return 0
    plan = Q.refine_prepare(a.root, a.profile)
    out = {"refine_id": plan["refine_id"], "failures": plan["diagnosis_failures"], "candidates": [{k: c[k] for k in ("name", "options", "why", "card_id")} for c in plan["candidates"]]}
    if a.blind:
        from . import evaluation as EVL
        pdir = S.profile_dir(Path(a.root) if a.root else S.default_root(), a.profile)
        run = Q.latest_eval_run(pdir)
        items = EVL.read_eval_set(Q._dev_set(pdir, P.latest_dataset(pdir), K.current(pdir)))
        out["blind"] = Q.make_blind_review(run, items, a.blind.split(","))
    if a.package and plan["candidates"]:
        a.target = "colab-l4"
        aid = a.adapter or next((x["adapter_id"] for x in AD.installed(S.profile_dir(Path(a.root) if a.root else S.default_root(), a.profile))), None)
        out["package"] = P.build_refine(a.root, a.profile, plan["refine_id"], adapter_id=aid, confirm_cloud=_confirm_cloud(a))
    if a.generate and plan["candidates"]:
        out["generation"] = Q.refine_generate(a.root, a.profile, plan["refine_id"], backend=a.backend, base_dir=a.base_dir, adapter_id=a.adapter,
                                              log=lambda s: print(s, file=sys.stderr))
    if a.select or a.generate:
        out["select"] = Q.refine_select(a.root, a.profile, plan["refine_id"])
    _print(out, a.json)
    return 0


def _quickstart(a) -> int:
    """The basic user flow, no training and no cloud: every stage is idempotent and resumable (re-run the same command after a stop)."""
    choice = _choose(a)
    if not choice:
        print("cancelled")
        return 1
    if not a.profile:
        older = svc.same_chat_profiles(a.input, speaker=choice, root=a.root)
        if older:
            ids = ", ".join(o["profile_id"] for o in older)
            print(f"\nnote: this person already has a profile from ANOTHER export of the same chat: {ids}\n"
                  f"  to UPDATE that profile with this file:  stillhere persona memory-build --profile {older[-1]['profile_id']} --source {a.input}\n"
                  "  continuing creates a NEW, separate profile (its own card and memory).", file=sys.stderr)
            if _interactive(a) and _ask("Create a new profile anyway? [y/N]: ").lower() not in ("y", "yes"):
                print("cancelled")
                return 1
    res = PL.run(a.input, speaker=choice, root=a.root, target="cpu-only", tokenizer=a.tokenizer, writer="none", profile_id=a.profile,
                 display_name=a.display_name, accept_anomalies=a.accept_anomalies, dense=a.dense, memory_pipeline=a.memory_pipeline,
                 only=[c for c, _ in PL.STAGES if c not in ("12", "13", "14", "15")], log=lambda s: print(s, file=sys.stderr))
    root = Path(a.root) if a.root else S.default_root()
    pdir = S.profile_dir(root, res["profile_id"])
    out = {"profile_id": res["profile_id"], "profile_dir": str(pdir), "overall": res["overall"], "dense": res.get("dense"),
           "runtime_config": res["runtime_config"], "stages": {k: v["status"] for k, v in sorted(res["stages"].items())}}
    _print(out, a.json)
    if not a.json:
        from . import status_report as SR
        notes = SR.profile(a.root, res["profile_id"])["setup_notes"]
        if notes:
            print("\nnot PASS (chat still works unless the stage is 06 / 07 / 16):")
            for n in notes:
                print(f"  {n['stage']}: {n['status']}" + (f" - {n['why']}" if n["why"] else "") + (f"  [{n['meaning']}]" if n["meaning"] else ""))
        print(f"\nprofile data (stored only on this machine; sent nowhere unless you choose an API provider): {pdir}")
        print(f"check it any time:    stillhere persona status --profile {res['profile_id']}")
        print(f"try without a model:  stillhere persona chat --profile {res['profile_id']} --backend fake --message \"你好\"   (labelled dry run)")
        print(f"next (local model):   stillhere persona chat --profile {res['profile_id']} --base-dir <Qwen3-8B>")
        print(f"next (DeepSeek API): set DEEPSEEK_API_KEY, then stillhere persona chat --profile {res['profile_id']} --answer-provider deepseek_api "
              "--answer-model deepseek-flash --allow-external-api   (sends the persona prompt, retrieved evidence and your message to DeepSeek)")
    return 0 if res["overall"] in ("READY", "PARTIAL_READY") else 6


def _provider(a) -> int:
    from stillhere.providers import llm as PLM
    from stillhere.providers import roles as PR
    pdir = S.profile_dir(Path(a.root) if a.root else S.default_root(), a.profile) if a.profile else None
    roles = PR.ROLES if a.action == "show" else (a.role,)
    if a.action == "show":
        _print({r: PR.resolve(r, profile_dir=pdir).public() for r in roles}, a.json)
        return 0
    cfg = PR.resolve(a.role, profile_dir=pdir)
    if cfg.provider not in PLM.API_PROVIDERS:
        _print({"role": a.role, "provider": cfg.provider, "check": "local provider: run persona doctor --target local-cuda"}, a.json)
        return 0
    gen = PR.api_generator(a.role, cfg, allow_external=True if a.allow_external_api else None)
    _print({"role": a.role, **PLM.check_connection(gen.client)}, a.json)
    return 0


def _card(a) -> int:
    from . import card_admin as CA
    pdir = S.profile_dir(Path(a.root) if a.root else S.default_root(), a.profile)
    if a.action == "list":
        _print(CA.list_versions(pdir), a.json)
    elif a.action == "show":
        r = CA.preview(pdir, a.card)
        if a.json:
            _print(r, True)
        else:
            print(f"card {r['card_id']} ({r['source']}), prompt sha256 {r['prompt_sha256'][:12]}\n----- prompt (private, local) -----\n{r['prompt']}")
    elif a.action == "activate":
        if not a.card:
            raise ValueError("--card <c-...> is required")
        _print({**CA.activate(pdir, a.card), "next": _apply_card_hint(pdir, a.profile)}, a.json)
    else:
        r = CA.assist(a.root, a.profile, allow_external=True if a.allow_external_api else None, send_examples=a.send_examples, activate_new=a.activate)
        _print({**r, "next": _apply_card_hint(pdir, a.profile)} if a.activate and isinstance(r, dict) else r, a.json)
    return 0


def run(a) -> int:
    try:
        cmd = a.persona_cmd
        if cmd == "pipeline":
            return _pipeline(a)
        if cmd == "inspect":
            _print(svc.inspect(a.input, cfg=svc.load_config(a.config), tokenizer=a.tokenizer), a.json)
            return 0
        if cmd == "quickstart":
            return _quickstart(a)
        if cmd == "provider":
            return _provider(a)
        if cmd == "card":
            return _card(a)
        if cmd == "init":
            return _init(a)
        if cmd == "prepare":
            res = svc.prepare_all(a.root, a.profile, config_path=a.config, tokenizer=a.tokenizer)
            out = {"profile_id": res["profile_id"], "dataset_version": res["dataset_version"], "dataset_dir": res["dataset_dir"], "reused": res["reused"],
                   "samples": {k: v["samples"] for k, v in res["splits"].items()}, "supervised_tokens": {k: v["supervised_tokens"] for k, v in res["splits"].items()},
                   "gates_pass": all(res["gates"].values()), "failed_gates": [k for k, v in res["gates"].items() if not v], "card": res["card"]}
            _print(out, a.json)
            return 0 if out["gates_pass"] else 6
        if cmd == "review":
            return _review(a)
        if cmd == "doctor":
            rep = DR.doctor(a.target, base_dir=a.base_dir)
            _print(rep if a.json else {"status": rep["status"], "blocked": rep["blocked"], "gpu": rep["gpu"], "capabilities": rep["capabilities"],
                                       "versions": rep["versions"], "base": rep["base"], "disk_free_gib": rep["disk_free_gib"]}, a.json)
            return 0 if rep["status"] == "PASS" else 3
        if cmd == "package":
            res = P.build(a.root, a.profile, target=a.target, confirm_cloud=_confirm_cloud(a), base_dir=a.base_dir, run_id=a.run_id,
                          config_template=a.config)
            _print(res, a.json)
            return 0
        if cmd == "train":
            res = TR.train(a.root, a.profile, package_id=a.package, run_id=a.run_id, base_dir=a.base_dir, device=a.device, dry_run=a.dry_run, smoke_only=a.smoke_only,
                           gpu=a.gpu)
            _print(res, a.json)
            return 0 if res["ok"] else 6
        if cmd == "import-adapter":
            _print(AD.import_adapter(a.root, a.profile, a.file), a.json)
            return 0
        if cmd == "evaluate":
            return _evaluate(a)
        if cmd == "chat":
            return _chat(a)
        if cmd == "memory-build":
            ev = MS.build_evidence(a.root, a.profile, source=a.source)
            sem = MS.build_semantic(a.root, a.profile, writer=a.writer, base_dir=a.base_dir, allow_external=a.allow_external_api, reviewer=a.reviewed_by, limit=a.limit,
                                    log=lambda s: print(s, file=sys.stderr))
            dense = None
            if a.dense:
                from . import dense_memory as DM
                dense = DM.build(a.root, a.profile)
            _print({"evidence": {k: ev[k] for k in ("added", "unchanged", "edited_conflicts_kept_original", "absent_from_this_export_kept", "archive_messages_total",
                                                     "archive_sessions", "manifest")}, "memory": {k: v for k, v in sem.items() if k != "run"}, "writer_run": sem.get("run"),
                    "dense": dense}, a.json)
            return 0
        if cmd == "memory-audit":
            r = MS.audit(a.root, a.profile)
            _print({k: r[k] for k in ("status", "semantic_extraction", "layers", "writer_operations", "retrieval_probes", "path")}, a.json)
            return 0
        if cmd == "memory-query":
            from datetime import datetime
            r = MS.query(a.root, a.profile, a.q, ask_time=datetime.fromisoformat(a.at) if a.at else None)
            _print({"question": r["question"], "ask_time": r["ask_time"], "items": r["items"]} if a.json else r["evidence"], a.json)
            return 0
        if cmd == "refine":
            return _refine(a)
        if cmd == "download-model":
            from . import model_download as MD
            try:
                _print(MD.download(a.what, dest=a.dir), a.json)
            except MD.DownloadError as e:
                print(f"error: {e}", file=sys.stderr)
                return 5
            return 0
        if cmd == "status":
            from . import status_report as SR
            if a.profile:
                _print(SR.profile(a.root, a.profile), a.json)
            else:
                ps_ = SR.profiles(a.root)
                _print({"root": str(Path(a.root) if a.root else S.default_root()), "profiles": ps_} if a.json or ps_ else
                       {"root": str(Path(a.root) if a.root else S.default_root()), "profiles": "none yet (create one with: stillhere persona quickstart --input <export.json>)"},
                       a.json)
            return 0
        if cmd == "config":
            from . import config as CF
            pdir = S.profile_dir(Path(a.root) if a.root else S.default_root(), a.profile) if a.profile else None
            c = CF.load(profile_dir=pdir)
            out = c.public()
            try:
                out["resolved_backend"] = CF.resolve_backend(c)
            except RuntimeError as e:
                out["resolved_backend"] = str(e)
            _print(out, a.json)
            return 0
        if cmd == "prompt":
            from . import config as CF
            from . import prompt_writer as PW
            pdir = S.profile_dir(Path(a.root) if a.root else S.default_root(), a.profile)
            if a.from_file:
                res = PW.save_candidate(pdir, Path(a.from_file).read_text(encoding="utf-8"), author=a.author, activate=a.activate)
            else:
                res = PW.generate(a.root, a.profile, cfg=CF.load(profile_dir=pdir, overrides={"prompt_provider": a.provider}), activate=a.activate)
            _print(res, a.json)
            return 0
        if cmd == "final-package":
            from . import final_package as FP
            r = FP.build(a.root, a.profile, old_card=a.old_card, new_card=a.new_card, adapter_id=a.adapter, out_dir=Path(a.out_dir) if a.out_dir else None)
            _print({"package_dir": r["package_dir"], "zip": r["zip"], "zip_mb": r["zip_mb"], "package_id": r["manifest"]["package_id"],
                    "counts": r["manifest"]["counts"], "git_commit": r["manifest"]["git_commit"], "git_dirty": r["manifest"]["git_dirty"]}, a.json)
            return 0
        if cmd == "final-package-v2":
            from . import final_package_v2 as FP2
            r = FP2.build(a.root, a.profile, skill_card=a.skill_card, r1_adapter=a.r1_adapter, r2_adapter=a.r2_adapter, fast_results_zip=Path(a.fast_results),
                          freeze_manifest=Path(a.freeze_manifest), out_dir=Path(a.out_dir) if a.out_dir else None)
            _print({"package_dir": r["package_dir"], "zip": r["zip"], "zip_mb": r["zip_mb"], "package_id": r["manifest"]["package_id"],
                    "plan": r["manifest"]["counts"]["plan"], "git_commit": r["manifest"]["git_commit"], "git_dirty": r["manifest"]["git_dirty"]}, a.json)
            return 0
        if cmd == "import-colab-results":
            from . import results_import as RI
            _print(RI.import_results(a.root, a.profile, a.file, replace_memory=a.replace_memory), a.json)
            return 0
        if cmd == "finalize":
            ov = {"on": True, "off": False, "auto": None}
            d = Q.diagnose(a.root, a.profile) if Q.latest_eval_run(S.profile_dir(Path(a.root) if a.root else S.default_root(), a.profile)) else None
            c = Q.select_runtime(a.root, a.profile, override_adapter=ov[a.adapter], override_memory=ov[a.memory], memory_pipeline=a.memory_pipeline)
            _print({"quality_report": d["path"] if d else "NOT_EVALUATED (no DEV evaluation yet)", "verdicts": {k: v.get("verdict") for k, v in d["sections"]["base_vs_adapter"].items()} if d else None,
                    "failures": {g: [f["type"] for f in v] for g, v in d["failures"].items()} if d else None, "runtime_config": c["path"],
                    "adapter": c["adapter"], "memory": c["memory"], "memory_pipeline": c.get("memory_pipeline"), "selection_reason": c["selection_reason"]}, a.json)
            return 0
        if cmd == "studio":
            from .studio import serve
            httpd, _ = serve(a.root, a.port, a.backend, a.base_dir)
            print(f"Persona Studio: http://127.0.0.1:{a.port}/  (local only; Ctrl+C to stop; generator {a.backend}"
                  + (", SYNTHETIC" if a.backend == "fake" else "") + ")")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass
            return 0
    except tuple(c for c, _ in CODES) as e:                     # noqa: B030
        code = 3 if getattr(e, "code", None) in REFUSED else next(v for k, v in CODES if isinstance(e, k))
        print(f"error: {e}", file=sys.stderr)
        return code
    except RuntimeError as e:
        if not str(e).startswith(("NO_GPU", "BASE_NOT_FOUND", "CUDA")):
            raise
        print(f"error: {e}", file=sys.stderr)
        prof = getattr(a, "profile", None) or "<profile-id>"
        print("no local model can run here. Options:\n"
              f"  API (no GPU):  stillhere persona chat --profile {prof} --answer-provider deepseek_api --answer-model deepseek-flash --allow-external-api\n"
              f"  dry run:       stillhere persona chat --profile {prof} --backend fake --message \"你好\"\n"
              "  local GPU:     stillhere persona doctor --target local-cuda", file=sys.stderr)
        return 3
    return 2
