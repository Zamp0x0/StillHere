"""U1: one import -> one person choice -> persona chain + memory chain -> evaluation / refinement / runtime selection -> ready to chat.

Stages (every one idempotent: the services underneath reuse identical results; nothing is overwritten):

    01 import  02 identify  03 select (profile)                                  common
    04 clean   05 dataset   06 card   07 prompt                                  persona chain
    08 evidence archive   09 semantic memory (Writer)   10 index + EF1   11 memory audit      memory chain
    12 training package   13 adapter   14 evaluation   15 refinement   16 runtime selection   17 ready

A failure in one chain never invalidates the other: the state records per stage PASS / SKIPPED / PENDING / BLOCKED / FAILED, and the
overall status is READY (prompt + memory + an adapter decision), PARTIAL_READY (chat possible, something pending) or NOT_READY.
``<profile>/PIPELINE_STATE.json`` + ``<profile>/logs/pipeline.log``; ``only=`` runs single stages again.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

from . import store as S

STAGES = [("01", "import"), ("02", "identify"), ("03", "select"), ("04", "clean"), ("05", "dataset"), ("06", "card"), ("07", "prompt"),
          ("08", "evidence"), ("09", "semantic_memory"), ("10", "index"), ("11", "memory_audit"), ("12", "package"), ("13", "adapter"),
          ("14", "evaluation"), ("15", "refinement"), ("16", "runtime"), ("17", "ready")]
PERSONA_CHAIN = {"04", "05", "06", "07", "12", "13", "14", "15"}
MEMORY_CHAIN = {"08", "09", "10", "11"}


class State:
    def __init__(self, pdir: Path):
        self.pdir = pdir
        self.path = pdir / "PIPELINE_STATE.json"
        self.data = S.read_json(self.path) if self.path.is_file() else {"stages": {}}
        (pdir / "logs").mkdir(parents=True, exist_ok=True)

    def set(self, code: str, status: str, **info):
        name = dict(STAGES)[code]
        self.data["stages"][code] = {"stage": name, "status": status, "at": S.now_iso(), **info}
        S.write_json(self.path, self.data)
        with open(self.pdir / "logs" / "pipeline.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": S.now_iso(), "stage": f"{code} {name}", "status": status, **{k: v for k, v in info.items() if k != "trace"}}, ensure_ascii=False) + "\n")

    def get(self, code: str) -> Optional[str]:
        return (self.data["stages"].get(code) or {}).get("status")


def run(input_path, *, speaker: str, root=None, target: str = "cpu-only", confirm_cloud: bool = False, base_dir: Optional[str] = None,
        tokenizer: Optional[str] = None, writer: str = "none", allow_external: bool = False, reviewer: Optional[str] = None, profile_id: Optional[str] = None,
        display_name: Optional[str] = None, accept_anomalies: bool = False, only: Optional[Sequence[str]] = None, log: Callable = lambda s: None,
        dense: str = "auto", embedder=None, memory_pipeline: Optional[str] = None) -> dict:
    """``dense``: ``auto`` builds the product dense index when the local Qwen3-Embedding-0.6B is available (else SKIPPED: the product
    pipeline then runs V2L), ``on`` requires it, ``off`` never builds it. ``memory_pipeline``: see ``quality.select_runtime``."""
    from . import adapters as AD
    from . import card as K
    from . import identity as I
    from . import memory_store as MS
    from . import package as P
    from . import quality as Q
    from . import service as svc
    from . import weflow as W
    root = Path(root) if root else S.default_root()
    exp = W.load(input_path)                                    # 01 (read-only) and 02 always run: they decide the profile
    ident = I.resolve(exp)
    ini = svc.init_profile(input_path, speaker=speaker, root=root, profile_id=profile_id, display_name=display_name, accept_anomalies=accept_anomalies)
    pid = ini["profile_id"]
    pdir = S.profile_dir(root, pid)
    st = State(pdir)
    st.set("01", "PASS", source_sha256=exp.sha256, messages=exp.report["messages"])
    st.set("02", "PASS", speakers=len(ident.speakers), blocking_anomalies=len(ident.blocking))
    st.set("03", "PASS", profile_id=pid, reused=ini["reused"])
    want = set(only or [c for c, _ in STAGES])
    out: Dict[str, dict] = {}

    def stage(codes, fn):
        codes = [c for c in codes if c in want]
        if not codes:
            return None
        log(f"STAGE {'/'.join(codes)} ...")
        try:
            res = fn()
            return res
        except Exception as e:  # noqa: BLE001 - recorded; the other chain continues
            for c in codes:
                st.set(c, "FAILED", error=f"{type(e).__name__}: {str(e)[:300]}", trace=traceback.format_exc()[-800:])
            return None

    def persona():
        r = svc.prepare_all(root, pid, tokenizer=tokenizer)
        st.set("04", "PASS", **{k: r["funnel"][k] for k in ("text_events", "media_events", "sessions", "turns")} if "funnel" in r else {"reused": True})
        st.set("05", "PASS" if all(r["gates"].values()) else "FAILED", dataset_version=r["dataset_version"], samples={k: v["samples"] for k, v in r["splits"].items()},
               failed_gates=[k for k, v in r["gates"].items() if not v])
        st.set("06", "PASS", card_id=r["card"]["current_card"], review_status=r["card"]["current_review_status"])
        st.set("07", "PASS", prompt_sha256=K.current(pdir)["manifest"]["prompt_sha256"])
        return r
    out["persona"] = stage(["04", "05", "06", "07"], persona)

    def evidence():
        r = MS.build_evidence(root, pid)
        st.set("08", "PASS", archive_messages=r["archive_messages_total"], added=r["added"], sessions=r["archive_sessions"])
        return r
    out["evidence"] = stage(["08"], evidence)

    def semantic():
        if st.get("08") != "PASS":
            st.set("09", "SKIPPED", reason="no evidence archive")
            return None
        r = MS.build_semantic(root, pid, writer=writer, base_dir=base_dir, allow_external=allow_external, reviewer=reviewer, log=log)
        st.set("09", "PENDING" if r["semantic_extraction"] == MS.PENDING else ("PASS" if r["semantic_extraction"] == "DONE" else "PARTIAL"),
               semantic_extraction=r["semantic_extraction"], memories_active=r["memories_active"], writer=writer)
        return r
    out["semantic"] = stage(["09"], semantic)

    def index():
        s = MS.status(root, pid)
        pr = MS.probe(root, pid)
        st.set("10", "PASS" if pr["ok"] else "FAILED", searchable=s["searchable_text_messages"], memories=s["memories_active"], ef1_probe=pr)
        a = MS.audit(root, pid, probes=20)
        p = a["retrieval_probes"]["exact_message_recall_at_8"].split("/")
        st.set("11", "PASS" if int(p[1]) and int(p[0]) / int(p[1]) >= 0.9 else "FAILED", **a["retrieval_probes"], semantic=a["semantic_extraction"], report=a["path"])
        if dense != "off":
            from . import dense_memory as DM
            if embedder is None and DM.resolve_model_dir() is None:
                if dense == "on":
                    raise S.StoreError("NO_EMBEDDING_MODEL", f"--dense on needs a local Qwen3-Embedding-0.6B (set {DM.MODEL_ENV})")
                st.data["dense"] = {"status": "SKIPPED", "reason": "no local Qwen3-Embedding-0.6B: the product memory runs V2L (lexical)"}
            else:
                try:
                    st.data["dense"] = {"status": "PASS", **{k: v for k, v in DM.build(root, pid, embedder=embedder).items() if k in ("dense_status", "chunks", "dir", "build")}}
                except Exception as e:  # noqa: BLE001 - the lexical memory is ready; dense is an addition
                    if dense == "on":
                        raise
                    st.data["dense"] = {"status": "FAILED", "reason": f"{type(e).__name__}: {str(e)[:200]} (the product memory runs V2L)"}
            S.write_json(st.path, st.data)
        return a
    out["memory_audit"] = stage(["10", "11"], index)

    def package():
        if st.get("05") != "PASS":
            st.set("12", "SKIPPED", reason="dataset not ready")
            return None
        if target in P.CLOUD and not confirm_cloud:
            st.set("12", "BLOCKED", reason="cloud package needs the operator's confirmation (--confirm-cloud); prompt + memory are unaffected")
            return None
        r = P.build(root, pid, target=target, confirm_cloud=confirm_cloud, base_dir=base_dir)
        st.set("12", "PASS", package_id=r["package_id"], notebook=r["notebook"], label_audit=r["label_audit"]["status"], target=target)
        return r
    out["package"] = stage(["12"], package)

    def adapter():
        inst = AD.installed(pdir)
        st.set("13", "PASS" if inst else "PENDING", adapters=[a["adapter_id"] for a in inst],
               reason=None if inst else "train with the package notebook (Colab L4) or a >= 20 GiB local GPU, then persona import-adapter")
        return inst
    out["adapter"] = stage(["13"], adapter)

    def evaluation():
        run_dir = Q.latest_eval_run(pdir)
        if run_dir is None or not AD.installed(pdir):
            st.set("14", "PENDING", reason="needs an adapter and a GPU evaluation (notebook STAGE 11 or persona evaluate --backend hf)")
            st.set("15", "SKIPPED", reason="no evaluation yet")
            return None
        d = Q.diagnose(root, pid, run_dir=run_dir)
        st.set("14", "PASS", eval_run=run_dir.name, quality_status=d["status"], verdicts={k: v.get("verdict") for k, v in d["sections"]["base_vs_adapter"].items()})
        r = Q.refine_prepare(root, pid)
        sel = S.read_json(Path(r["dir"]) / "PROMPT_REFINEMENT.json") if (Path(r["dir"]) / "PROMPT_REFINEMENT.json").is_file() else None
        st.set("15", ("PASS" if sel else "PENDING") if r["candidates"] else "PASS", refine_id=r["refine_id"], candidates=[c["name"] for c in r["candidates"]],
               result=(sel or {}).get("status", "PENDING_GPU_EVALUATION" if r["candidates"] else "NO_CANDIDATES_NEEDED"))
        return d
    out["evaluation"] = stage(["14", "15"], evaluation)

    def runtime():
        c = Q.select_runtime(root, pid, memory_pipeline=memory_pipeline)
        st.set("16", "PASS", adapter=c["adapter"]["enabled"], memory=c["memory"]["enabled"], reason=c["selection_reason"], config=c["path"])
        return c
    out["runtime"] = stage(["16"], runtime)

    s = {c: st.get(c) for c, _ in STAGES if c != "17"}
    chat_ok = s.get("07") == "PASS" and s.get("16") == "PASS"
    full = chat_ok and s.get("11") == "PASS" and s.get("09") == "PASS" and s.get("14") == "PASS" and s.get("15") in ("PASS",)
    overall = "READY" if full else ("PARTIAL_READY" if chat_ok else "NOT_READY")
    st.set("17", overall, pending=[f"{c} {dict(STAGES)[c]}: {v}" for c, v in s.items() if v not in ("PASS",)])
    return {"profile_id": pid, "overall": overall, "stages": st.data["stages"], "dense": st.data.get("dense"), "state_file": str(st.path), "log": str(pdir / "logs" / "pipeline.log"),
            "runtime_config": (out.get("runtime") or {}).get("path")}


def state(root, profile_id: str) -> dict:
    pdir = S.profile_dir(Path(root) if root else S.default_root(), profile_id)
    return S.read_json(pdir / "PIPELINE_STATE.json") if (pdir / "PIPELINE_STATE.json").is_file() else {"stages": {}}
