"""M1: one WeFlow export -> the EXISTING temporal memory system (no second memory store, no new schema).

    Layer 1  Conversation Evidence Archive : every non-system message as a ``CanonicalMessage`` in ``SQLiteMemoryStore.messages``
             (speaker role, time, session = conversation_id, quote reference / message type / source in the locator, source sha).
             Media without text are kept as typed events ("[图片]") and never interpreted. Searchable by EF1 (BM25, time window).
    Layer 2  Structured Temporal Memory    : the existing Memory Writer (``writer.pipeline.Writer``: extract -> deterministic verify ->
             resolve (duplicate / temporal change / correction / retraction / contradiction) -> one atomic commit through MemoryAPI with
             a ``writer_batches`` row). The target person is the store's persona (本人); the other person is ``<pid>.other``; the
             verifier rejects a "self" fact the target did not say, keeps plans apart from completed events, holds uncertain wording back.

Two memory VIEWS that never mix:
* PRODUCTION full-history memory: ``<profile>/memory/memory.sqlite`` (this module; chat runtime);
* EVALUATION train-cutoff memory: ``persona.memory.build_api`` (in memory, messages <= last train target only; A-E ablation).

Semantic extraction needs a model: ``hf`` (local Qwen3-8B NF4 on CUDA), ``deepseek`` (external API, only with ``allow_external``) or an
injected ``complete`` (tests). Without one the evidence layer is complete and the status is SEMANTIC_EXTRACTION_PENDING; nothing is
pretended. Re-imports are idempotent (same message id + content: skipped); new messages are added; an edited message (same id,
other content) is never overwritten (recorded as a conflict); a message missing from a newer export stays in the archive.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import cleaning as C
from . import identity as I
from . import store as S
from . import weflow as W

TZ = timezone(timedelta(hours=8))
MEMORY_PIPELINE_VERSION = "stillhere.persona.memory_pipeline/1.0"
WRITER_ENV = {"MEMORY_WRITER_V1_ENABLED": "1", "MEMORY_WRITER_M22_ENABLED": "1", "MEMORY_WRITER_M221_ENABLED": "1", "MEMORY_WRITER_M222_ENABLED": "1"}
WRITER_PROMPT = "writer-extract-v5"           # v4 (LifeBench Batch 2.1 final) + two-person ownership, cancellations by id, third-party events, momentary states
REVIEW_AUTO = "auto_verified;accepted_by:{name}"   # what the store records: deterministic verification + the owner's acceptance, NOT a per-item human review


def fix_candidate(c, msgs, existing):
    """pre-verify hook: a status update whose ``object`` is an existing event id (the model's slip seen in the smoke) becomes a proper
    update: ``refers_to`` = that event, ``object`` = its description. Nothing is invented: both come from the stored event."""
    import dataclasses
    obj = (c.object or "").strip()
    for mid, subj, attr, val in existing:
        if subj.startswith("event:") and obj and (obj == subj or obj == subj.split(":", 1)[1]) and attr == "event.description":
            return dataclasses.replace(c, object=val, refers_to=c.refers_to or subj)
    # an inverted kin relation ("person:小雨 / relationship / 妹妹") is the owner's relation "妹妹 小雨"; the verifier still requires the owner
    # to have said it in a cited message, so nothing is invented here
    if c.predicate == "relationship" and c.subject.startswith("person:") and obj in KIN:
        return dataclasses.replace(c, subject="self", object=f"{obj} {c.subject.split(':', 1)[1].strip()}")
    return c


KIN = {"妹妹", "姐姐", "哥哥", "弟弟", "爸爸", "妈妈", "父亲", "母亲", "爷爷", "奶奶", "外公", "外婆", "姥姥", "姥爷", "叔叔", "阿姨", "舅舅", "姑姑",
       "表哥", "表姐", "表弟", "表妹", "堂哥", "堂姐", "堂弟", "堂妹", "儿子", "女儿", "老公", "老婆", "男朋友", "女朋友", "同事", "室友", "同学", "朋友"}
POLARITY, CERTAINTY = ("positive", "negative"), ("certain", "uncertain")


def normalize_writer_json(text: str) -> str:
    """Bounded, deterministic format repair BEFORE the Writer's schema check (seen on the local model): an event that carries a status
    with an invalid polarity ("neutral") gets "positive" (the status carries the meaning); an invalid certainty becomes "uncertain" (the
    safe side). Anything else invalid is left for the schema check to reject; no fact is ever added."""
    try:
        d = json.loads(text)
    except (TypeError, ValueError):
        return text
    if not isinstance(d, dict) or not isinstance(d.get("candidates"), list):
        return text
    for c in d["candidates"]:
        if not isinstance(c, dict):
            continue
        if c.get("polarity") not in POLARITY and c.get("predicate") == "event" and c.get("event_status"):
            c["polarity"] = "positive"
        if c.get("certainty") not in CERTAINTY and c.get("certainty") is not None:
            c["certainty"] = "uncertain"
    return json.dumps(d, ensure_ascii=False)


def repaired(complete: Callable) -> Callable:
    def call(messages):
        r = dict(complete(messages))
        r["text"] = normalize_writer_json(r.get("text", ""))
        return r
    return call


# ------------------------------------------------------------------ third-party event follow-up (recall; verification unchanged)
EVENT_CUE = re.compile(r"结婚|订婚|离婚|婚礼|办酒|生日|毕业|高考|考研|考试|面试|入职|离职|辞职|升职|住院|出院|手术|生病|怀孕|生孩子|生宝宝|去世|"
                       r"搬家|出国|留学|回国|旅游|旅行|买房|买车|装修|比赛|演出|聚会|相亲|退休")
MESSAGE_LINE = re.compile(r"^\[(m\d+)\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) (.+?) → (.+?)：(.*)$")
FOLLOWUP_MAX = 3
FOLLOWUP_USER = ("只看下面标出的这条消息（前后两条只作背景）。它提到了另一个人（不是说话人自己）的一件事。"
                 "按系统规则只抽取这件事本身：subject 写 \"person:姓名或称呼\"（如 person:小雨、person:妹妹），predicate 写 event，"
                 "activity 写这件事，event_status 写它的状态（计划 PLANNED、已完成 COMPLETED、取消 CANCELLED 等），"
                 "有时间就写 planned_time / event_time，source_message_ids 只写 [\"{label}\"]。没有这样的事件就输出 {{\"candidates\": []}}。\n\n"
                 "【背景】\n{context}\n\n【这条消息】\n{line}")


def event_followup(complete: Callable) -> Callable:
    """Writer-call wrapper (recall only): after the main extraction, a message that mentions another person's event (an event cue
    plus a kin word / a person the model already named) but has no ``event`` candidate gets ONE focused follow-up call with the same
    system rules. Only follow-up ``event`` candidates about a ``person:`` subject that cite exactly that message are merged; every
    merged candidate then goes through the unchanged schema check and verifier (attribution, spans, worthiness). At most
    ``FOLLOWUP_MAX`` follow-ups per session; repair turns (the conversation already has an assistant turn) are passed through."""
    def call(messages):
        r = dict(complete(messages))
        if len(messages) != 2:
            return r
        try:
            d = json.loads(r.get("text", ""))
        except (TypeError, ValueError):
            return r
        if not isinstance(d, dict) or not isinstance(d.get("candidates"), list):
            return r
        lines = [MESSAGE_LINE.match(l) for l in messages[1]["content"].splitlines()]
        lines = [m for m in lines if m]
        cands = [c for c in d["candidates"] if isinstance(c, dict)]
        named = {str(c.get("subject", ""))[7:] for c in cands if str(c.get("subject", "")).startswith("person:")} - {""}
        has_event = {x for c in cands if c.get("predicate") == "event" for x in (c.get("source_message_ids") or [])}
        calls = 0
        for i, m in enumerate(lines):
            label, content = m.group(1), m.group(5)
            if calls >= FOLLOWUP_MAX or label in has_event or not EVENT_CUE.search(content):
                continue
            if not (any(k in content for k in KIN) or any(n and n in content for n in named)):
                continue
            ctx = "\n".join(x.group(0) for x in lines[max(0, i - 2):i] + lines[i + 1:i + 3])
            f = complete([messages[0], {"role": "user", "content": FOLLOWUP_USER.format(label=label, context=ctx or "（无）", line=m.group(0))}])
            calls += 1
            try:
                fd = json.loads(f.get("text", ""))
            except (TypeError, ValueError):
                continue
            for c in (fd.get("candidates") if isinstance(fd, dict) else None) or []:
                if (isinstance(c, dict) and c.get("predicate") == "event" and str(c.get("subject", "")).startswith("person:")
                        and c.get("source_message_ids") == [label]):
                    d["candidates"].append(c)
                    has_event.add(label)
        if calls:
            d["followup_calls"] = calls
            r["text"] = json.dumps(d, ensure_ascii=False)
        return r
    return call


def writer_complete(complete: Callable) -> Callable:
    """The persona Writer's model call: bounded format repair, then the third-party event follow-up (whose calls are repaired too)."""
    return event_followup(repaired(complete))


_CLAUSE_BREAK = re.compile(r"[，,。！？!?；;\n]")


def third_party_actor(c, msgs) -> Optional[str]:
    """The kin word that is the grammatical subject of the cited clause of a "self" fact ("我妹妹小雨下个月结婚" -> "妹妹"), else None.
    The strict speaker rule cannot catch this: the persona did say the line, but about someone else (FAST synthetic gold:
    ``wedding_not_self`` failed). Only a clause that STARTS with the kin word (after an optional 我 / 我的) counts, so "我和妹妹去吃饭"
    or "我去看妹妹" stay the persona's own facts."""
    if c.subject != "self" or c.predicate == "relationship":
        return None
    by_id = {m.message_id: m for m in msgs}
    kin = sorted(KIN, key=len, reverse=True)
    for mid in c.source_message_ids or []:
        m = by_id.get(mid)
        if m is None:
            continue
        text = m.content or ""
        for span in [x for x in (c.source_spans or []) if x] or [c.object or ""]:
            i = text.find(span)
            if i < 0:
                continue
            starts = [x.end() for x in _CLAUSE_BREAK.finditer(text, 0, i)]
            head = re.sub(r"^(我的|我)", "", text[(starts[-1] if starts else 0):i + len(span)].strip())
            for k in kin:
                if head.startswith(k):
                    return k
    return None


def worthy(v, msgs):
    """post-verify hook (memory-worthiness + third-party actor): a momentary chat state - an event / state / routine with no time, no
    status change and only a tiny cited span - stays in the evidence archive and is not made a long-term memory. Short wording alone never
    disqualifies a fact. A "self" fact whose cited clause is about a relative / friend (``third_party_actor``) is AMBIGUOUS: never committed
    as the persona's own fact (no subject is guessed: the archive keeps the line)."""
    from stillhere.writer.contracts import AMBIGUOUS, UNSUPPORTED
    c = v.candidate
    if v.committable:
        actor = third_party_actor(c, msgs)
        if actor:
            v.support = AMBIGUOUS
            v.reasons.append(f"a fact about the owner whose cited clause is about '{actor}' (third-party actor; not committed)")
            return v
    if not v.committable or c.predicate not in ("event", "state", "routine"):
        return v
    timed = any(getattr(c, f) for f in ("planned_time", "event_time", "valid_from", "valid_to"))
    changed = c.event_status in ("PLANNED", "SCHEDULED", "COMPLETED", "CANCELLED") or c.corrects or c.retracts
    span = sum(len((x or "").strip()) for x in c.source_spans)
    if not timed and not changed and span < 6:
        v.support = UNSUPPORTED
        v.reasons.append("not memory-worthy: a momentary chat state without time or change (kept in the evidence archive)")
    return v
ROUND_TYPE = "weflow"
SESSION_GAP_MINUTES = 120
PENDING = "SEMANTIC_EXTRACTION_PENDING"
CHUNK = 2000


class MemoryError_(S.StoreError):
    pass


def mdir(pdir: Path) -> Path:
    return pdir / "memory"


def other_id(pid: str) -> str:
    return f"{pid}.other"


def open_api(pdir: Path, prof: dict):
    from stillhere.memory.api import MemoryAPI
    d = mdir(pdir)
    d.mkdir(parents=True, exist_ok=True)
    name = prof["target"].get("display_name") or "TA"
    return MemoryAPI(str(d / "memory.sqlite"), public_only=False, persona_names={prof["profile_id"]: [name]})


def _mid(platform_id: str) -> str:
    return "w-" + hashlib.sha256(platform_id.encode()).hexdigest()[:20]


def _sessions(msgs) -> Dict[str, str]:
    """platform id -> conversation id (a > SESSION_GAP_MINUTES silence starts a new session; the id is the first message's hash, so a
    later export that only appends messages keeps every earlier session id)."""
    out, cid, last = {}, None, None
    for m in msgs:
        if last is None or m.create_time - last > SESSION_GAP_MINUTES * 60:
            cid = "c-" + hashlib.sha256(m.platform_id.encode()).hexdigest()[:12]
        out[m.platform_id] = cid
        last = m.create_time
    return out


def canonical(exp: W.Export, prof: dict, source_id: str) -> List:
    from stillhere.contracts import CanonicalMessage
    pid, uname = prof["profile_id"], prof["target"]["username"]
    msgs = [m for m in exp.messages if not I.is_system(m)]
    conv = _sessions(msgs)
    out = []
    for m in msgs:
        if m.type in C.TEXT_TYPES and (m.content or "").strip():
            kind, text = "text", m.content.strip()
        elif m.type == "动画表情" and m.emoji_caption:
            kind, text = "sticker", f"[表情：{m.emoji_caption}]"
        else:
            kind, text = "media", C.MEDIA_LABEL.get(m.type, "[其他消息]")
        ts = datetime.fromtimestamp(m.create_time, TZ)
        loc = f"weflow:{m.local_id};kind={kind};type={m.type}" + (f";quote={_mid(m.reply_to)}" if m.reply_to else "")
        out.append(CanonicalMessage(conversation_id=conv[m.platform_id], message_id=_mid(m.platform_id), target_persona_id=pid,
                                    speaker_id=pid if m.sender == uname else other_id(pid), timestamp=ts, content=text, source_type=ROUND_TYPE,
                                    source_id=source_id, locator=loc, recorded_at=ts))
    return out


def _next_manifest(pdir: Path) -> Path:
    d = mdir(pdir) / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    n = 1 + max((int(p.stem.split("_")[1]) for p in d.glob("IMPORT_*.json")), default=0)
    return d / f"IMPORT_{n:04d}.json"


def build_evidence(root, profile_id: str, *, source=None) -> dict:
    """Layer 1. ``source``: a newer export of the SAME chat (incremental); default: the profile's bound file."""
    from stillhere.contracts import DataOrigin, SourceKind, SourceRef
    from stillhere.memory.sqlite_store import _message_sha256
    root = Path(root) if root else S.default_root()
    S.check_private_root(root)
    pdir = S.profile_dir(root, profile_id)
    prof = S.read_json(pdir / "profile.json")
    path = Path(source or prof["source"]["path"])
    exp = W.load(path)
    ident = I.resolve(exp)
    if not any(s.username == prof["target"]["username"] for s in ident.speakers):
        raise MemoryError_("IDENTITY_CHANGED", "the profile's person is not in this export (another chat?)")
    sid = "weflow-" + exp.sha256[:12]
    api = open_api(pdir, prof)
    try:
        api.register_source(SourceRef(source_id=sid, kind=SourceKind.CONVERSATION, origin=DataOrigin.REAL, description="WeFlow private chat export",
                                      locator="private", sha256=exp.sha256))
        msgs = canonical(exp, prof, sid)
        conn = api.store._conn
        import dataclasses
        have = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT message_id, content_sha256, source_id FROM messages WHERE target_persona_id = ?",
                                                          (profile_id,)).fetchall()}
        new, same, edited = [], 0, []
        for m in msgs:
            h = have.get(m.message_id)
            if h is None:
                new.append(m)
            elif h[0] == _message_sha256(dataclasses.replace(m, source_id=h[1])):      # same message as first imported (from another export file)
                same += 1
            else:
                edited.append(m.message_id)
        for i in range(0, len(new), CHUNK):
            api.ingest_messages(new[i:i + CHUNK])
        absent = sorted(set(have) - {m.message_id for m in msgs})
        kinds = {}
        for m in msgs:
            k = m.locator.split("kind=")[1].split(";")[0]
            kinds[k] = kinds.get(k, 0) + 1
        total = conn.execute("SELECT COUNT(*) FROM messages WHERE target_persona_id = ?", (profile_id,)).fetchone()[0]
        convs = conn.execute("SELECT COUNT(DISTINCT conversation_id) FROM messages WHERE target_persona_id = ?", (profile_id,)).fetchone()[0]
    finally:
        api.close()
    man = {"schema": "stillhere.persona.memory_import/1.0", "version": MEMORY_PIPELINE_VERSION, "profile_id": profile_id, "source_sha256": exp.sha256,
           "source_id": sid, "messages_in_export_non_system": len(msgs), "by_kind": kinds, "added": len(new), "unchanged": same,
           "edited_conflicts_kept_original": len(edited), "absent_from_this_export_kept": len(absent), "archive_messages_total": total,
           "archive_sessions": convs, "at": S.now_iso()}
    mp = _next_manifest(pdir)
    S.write_json(mp, man)
    S.append_lineage(pdir, {"artefact": f"memory/manifests/{mp.name}", "kind": "memory_evidence_import", "source_sha256": exp.sha256, "added": len(new)})
    return {**man, "manifest": str(mp), "db": str(mdir(pdir) / "memory.sqlite")}


# ------------------------------------------------------------------ Layer 2: the existing Memory Writer
def deepseek_complete(model: str = "deepseek-flash") -> Callable:  # pragma: no cover - network
    from stillhere.retrieval.deepseek_client import DeepSeekClient
    client = DeepSeekClient(dict(os.environ), None)

    def complete(messages):
        r = client.chat(model, messages, max_tokens=4096, temperature=0.0, json_mode=True)
        return {"text": r.text, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "cost_cny": r.cost_cny, "model": r.model}
    return complete


def _progress(pdir: Path) -> Path:
    return mdir(pdir) / "WRITER_PROGRESS.json"


def _failed(pdir: Path) -> Path:
    return mdir(pdir) / "WRITER_FAILED_QUEUE.json"


CHUNK_MESSAGES = 60        # long-session chunking: a session longer than this is written in windows (labels stay those of the session)
CHUNK_OVERLAP = 5          # neighbouring windows share a few messages (context at the seam); duplicates are resolved by the Writer's slots
MAX_GROUP_ATTEMPTS = 3     # bounded retry of a failing session / chunk


def chunks(n: int):
    """(start, end) windows over a session of ``n`` messages: the whole session when short, else overlapping windows."""
    if n <= CHUNK_MESSAGES:
        return [(0, n)]
    out, a = [], 0
    while a < n:
        b = min(n, a + CHUNK_MESSAGES)
        out.append((a, b))
        if b == n:
            break
        a = b - CHUNK_OVERLAP
    return out


def candidate_status(v) -> str:
    """The audit status of one candidate: AUTO_VERIFIED (deterministic verification passed; NOT a human review), AMBIGUOUS, or REJECTED."""
    from stillhere.writer.contracts import AMBIGUOUS
    if v.committable:
        return "AUTO_VERIFIED"
    return "AMBIGUOUS" if v.support == AMBIGUOUS else "REJECTED"


def log_candidates(pdir: Path, cid: str, key: str, window, res) -> None:
    """One line per written session / chunk in ``memory/writer_candidates.jsonl``: every candidate with its verification result and the
    operations (the provenance of the audit, the Writer judge and the exported MEMORY_CANDIDATES)."""
    import dataclasses
    rows = [{**dataclasses.asdict(v.candidate), "support": v.support, "reasons": list(v.reasons), "status": candidate_status(v)} for v in res.verdicts]
    line = {"conversation_id": cid, "group_key": key, "window": list(window), "schema_ok": (res.extraction or {}).get("schema_ok"),
            "followup_calls": _followups(res.extraction), "candidates": rows, "operations": [o.to_dict() for o in res.operations], "at": S.now_iso()}
    with open(mdir(pdir) / "writer_candidates.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")


def _followups(extraction) -> int:
    for a in (extraction or {}).get("attempts") or []:
        try:
            return int(json.loads(a.get("text") or "{}").get("followup_calls") or 0)
        except (ValueError, AttributeError):
            return 0
    return 0


def build_semantic(root, profile_id: str, *, writer: str = "none", complete: Optional[Callable] = None, base_dir: Optional[str] = None,
                   allow_external: bool = False, reviewer: Optional[str] = None, limit: Optional[int] = None, log=lambda s: None,
                   on_group: Optional[Callable[[int], None]] = None, only: Optional[set] = None) -> dict:
    """Layer 2 over every session of the archive, oldest first (long sessions in overlapping windows). Resumable and idempotent: a
    session / chunk whose content is in ``writer_batches`` is skipped before any model call; answers are cached in
    ``memory/writer_cache.jsonl``; a failing session is retried at most ``MAX_GROUP_ATTEMPTS`` times (``WRITER_FAILED_QUEUE.json``).
    Every candidate and its verification result is logged in ``memory/writer_candidates.jsonl``. ``on_group(n)`` runs after each
    committed session (the Colab run uses it to checkpoint the store)."""
    from stillhere.writer.extract import ExtractionCache
    from stillhere.writer.pipeline import Writer, sha
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    prof = S.read_json(pdir / "profile.json")
    if complete is None:
        if writer == "none":
            st = status(root, profile_id)
            return {**st, "writer": "none", "note": "evidence only; run memory-build --writer hf (CUDA) or deepseek (external, opt-in) to extract facts"}
        if writer == "deepseek":
            if not allow_external:
                raise MemoryError_("EXTERNAL_API_NOT_ALLOWED", "the DeepSeek writer sends chat text to an external API: pass --allow-external-api explicitly")
            complete = deepseek_complete()
        elif writer in ("hf", "local"):
            from . import config as CF
            from .models import local_complete, make_generator
            cfg = CF.load(profile_dir=pdir, overrides={"base_model_dir": base_dir})
            gen = make_generator(CF.resolve_backend(cfg), cfg.writer_dir(), {}, device_index=cfg.device_index)   # the user's local model
            complete = local_complete(gen, cfg.writer_max_new_tokens)
        else:
            raise MemoryError_("BAD_WRITER", "writer must be none, local (alias hf) or deepseek")
    if not (reviewer or "").strip():
        raise MemoryError_("OWNER_REVIEW_REQUIRED", "memories derived from a REAL chat need the owner's decision (the store's write gate): pass "
                                                    "--accept-auto-verified-by <your name> to accept AUTO-VERIFIED memories (recorded as auto-verified, "
                                                    "not as human-reviewed), or review candidates first with memory-build --preview")
    api = open_api(pdir, prof)
    label = writer if complete is not None and writer != "none" else "injected"
    try:
        w = Writer(api, complete=writer_complete(complete), model=label, persona_names={profile_id: prof["target"].get("display_name") or "TA"},
                   speakers={other_id(profile_id): "对方"}, cache=ExtractionCache(mdir(pdir) / "writer_cache.jsonl"), env=WRITER_ENV,
                   round_types=(ROUND_TYPE,), prompt_version=WRITER_PROMPT, reviewer=REVIEW_AUTO.format(name=reviewer.strip()),
                   strict_self_attribution=True, pre_verify=fix_candidate, post_verify=worthy)
        conn = api.store._conn
        convs = [r[0] for r in conn.execute("SELECT conversation_id, MIN(timestamp_utc) t FROM messages WHERE target_persona_id = ? GROUP BY conversation_id ORDER BY t",
                                            (profile_id,)).fetchall()]
        done = skipped = failed = cands = verified = gave_up = 0
        ops: Dict[str, int] = {}
        errors = []
        fails = S.read_json(_failed(pdir)) if _failed(pdir).is_file() else {}
        if only is not None:                         # an audit sample (FAST evaluation): only these sessions, in archive order
            convs = [c for c in convs if c in only]
        for i, cid in enumerate(convs):
            if limit is not None and done >= limit:
                break
            msgs = w.group(profile_id, cid)
            texts = [m for m in msgs if not (m.content.startswith("[") and m.content.endswith("]"))]
            if len(texts) < 2:
                continue
            for chunk, (a, b) in enumerate(chunks(len(msgs))):
                key = cid if b - a == len(msgs) else f"{cid}#c{chunk}"
                part = msgs[a:b]
                content = sha(*[f"{m.message_id}:{sha(m.content)}" for m in part])
                if conn.execute("SELECT 1 FROM writer_batches WHERE group_id = ? AND content_sha256 = ?", (key, content)).fetchone():
                    skipped += 1
                    continue
                if fails.get(key, {}).get("attempts", 0) >= MAX_GROUP_ATTEMPTS:
                    gave_up += 1                     # bounded retry: reported in the failed-session queue, not retried forever
                    continue
                try:
                    res = w.write_group(profile_id, cid, window=None if key == cid else (a, b), group_key=key)
                except Exception as e:  # noqa: BLE001 - one group failed and rolled back; retried on the next run (bounded)
                    failed += 1
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                    fails[key] = {"conversation": cid, "attempts": fails.get(key, {}).get("attempts", 0) + 1, "error": err, "at": S.now_iso()}
                    S.write_json(_failed(pdir), fails)
                    errors.append({"conversation": cid, "group_key": key, "error": err})
                    continue
                if key in fails:
                    fails.pop(key)
                    S.write_json(_failed(pdir), fails)
                done += 1
                cands += len(res.verdicts)
                verified += sum(1 for v in res.verdicts if v.committable)
                for o in res.operations:
                    ops[o.op] = ops.get(o.op, 0) + 1
                log_candidates(pdir, cid, key, (a, b), res)
                if on_group is not None:
                    on_group(done)
                if done % 10 == 0:
                    log(f"writer: {done} sessions this run")
                    S.write_json(_progress(pdir), {"processed_this_run": done, "skipped_done": skipped, "failed": failed, "at": S.now_iso()})
    finally:
        api.close()
    run = {"schema": "stillhere.persona.memory_writer_run/1.1", "writer": label, "prompt_version": WRITER_PROMPT, "sessions_processed": done,
           "sessions_already_done": skipped, "sessions_failed": failed, "sessions_given_up": gave_up, "candidates": cands, "verified": verified,
           "operations": ops, "errors": errors[:20], "chunk_messages": CHUNK_MESSAGES, "at": S.now_iso()}
    S.write_json(_progress(pdir), run)
    S.append_lineage(pdir, {"artefact": "memory/memory.sqlite", "kind": "memory_writer_run", "writer": label, "sessions": done, "verified": verified})
    return {**status(root, profile_id), "run": run}


# ------------------------------------------------------------------ status, query, audit
def status(root, profile_id: str) -> dict:
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    db = mdir(pdir) / "memory.sqlite"
    if not db.is_file():
        return {"memory_status": "NOT_BUILT", "db": str(db)}
    prof = S.read_json(pdir / "profile.json")
    api = open_api(pdir, prof)
    try:
        c = api.store._conn
        q = lambda sql, *a: c.execute(sql, a).fetchone()[0]
        n_msg = q("SELECT COUNT(*) FROM messages WHERE target_persona_id = ?", profile_id)
        n_text = q("SELECT COUNT(*) FROM messages WHERE target_persona_id = ? AND locator NOT LIKE '%kind=media%'", profile_id)
        convs = q("SELECT COUNT(DISTINCT conversation_id) FROM messages WHERE target_persona_id = ?", profile_id)
        has_wb = q("SELECT COUNT(*) FROM sqlite_master WHERE name = 'writer_batches'")
        batches = q("SELECT COUNT(*) FROM writer_batches WHERE persona_id = ?", profile_id) if has_wb else 0
        mem_active = q("SELECT COUNT(*) FROM memories WHERE persona_id = ? AND status = 'active'", profile_id)
        mem_all = q("SELECT COUNT(*) FROM memories WHERE persona_id = ?", profile_id)
        span = c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages WHERE target_persona_id = ?", (profile_id,)).fetchone()
    finally:
        api.close()
    sem = PENDING if batches == 0 else ("PARTIAL" if batches < convs else "DONE")
    return {"memory_status": "EVIDENCE_READY" if sem == PENDING else ("SEMANTIC_PARTIAL" if sem == "PARTIAL" else "READY"), "semantic_extraction": sem,
            "db": str(db), "archive_messages": n_msg, "searchable_text_messages": n_text, "sessions": convs, "writer_batches": batches,
            "memories_active": mem_active, "memories_all_versions": mem_all, "first": span[0], "last": span[1]}


DATE_WEIGHT = 3.0          # persona queries only; EF1's frozen default weights / protocol are not changed


def config_for(question: str, at: datetime, k: int):
    """EF1 config for a persona memory query. When the question carries an explicit date / range (EF1's own temporal resolver finds a
    window), the existing time-window channel gets a higher fusion weight, so 'what did we talk about on 5月3日' is not out-voted by
    keyword matches of '聊了什么'. Configuration of the existing interface only (``EF1Config.weights``)."""
    from stillhere.retrieval.ef1.pipeline import DEFAULT_WEIGHTS, EF1Config
    from stillhere.retrieval.ef1.query import prepare
    from stillhere.retrieval.ef1.temporal import resolve
    windows = resolve(prepare(question, at).body, at).windows
    w = dict(DEFAULT_WEIGHTS, time_window=DATE_WEIGHT) if windows else dict(DEFAULT_WEIGHTS)
    return EF1Config(top_k=k, adaptive_k=False, use_dense=False, weights=w)


def pack(root, profile_id: str, question: str, *, ask_time: Optional[datetime] = None, k: int = 8):
    """The FactPack (EF1 + fact_pack, owner audience) the Fact Reader reads, over the PRODUCTION memory."""
    from stillhere.context import fact_pack as FP
    from stillhere.retrieval.ef1.pipeline import retrieve
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    api = open_api(pdir, S.read_json(pdir / "profile.json"))
    try:
        at = ask_time or datetime.now(TZ)
        cs = retrieve(api, profile_id, question, at, config_for(question, at, k))
        return FP.build(api, cs, names={profile_id: "本人", other_id(profile_id): "对方"})
    finally:
        api.close()


def query(root, profile_id: str, question: str, *, ask_time: Optional[datetime] = None, k: int = 8) -> dict:
    """EF1 over the PRODUCTION memory (messages + memories), rendered by fact_pack exactly as the fact reader sees it."""
    from stillhere.context import fact_pack as FP
    from stillhere.retrieval.ef1.pipeline import EF1Config, retrieve
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    prof = S.read_json(pdir / "profile.json")
    api = open_api(pdir, prof)
    try:
        at = ask_time or datetime.now(TZ)
        cs = retrieve(api, profile_id, question, at, config_for(question, at, k))
        pack = FP.build(api, cs, names={profile_id: "本人", other_id(profile_id): "对方"})
        items = [{"eid": i.eid, "time": i.event_time, "speakers": i.speakers, "text": i.text, "message_ids": i.message_ids,
                  "memories": [m.statement for m in i.memories], "channels": i.channels} for i in pack.items]
        return {"question": question, "ask_time": at.isoformat(), "items": items, "evidence": FP.render_evidence(pack), "intent_windows": cs.trace().get("windows")}
    finally:
        api.close()


def probe(root, profile_id: str) -> dict:
    """Index sanity: the newest text message must come back for its own words through EF1."""
    pdir = S.profile_dir(Path(root) if root else S.default_root(), profile_id)
    api = open_api(pdir, S.read_json(pdir / "profile.json"))
    try:
        row = api.store._conn.execute("SELECT message_id, timestamp, content FROM messages WHERE target_persona_id = ? AND locator LIKE '%kind=text%' "
                                      "AND length(content) >= 4 ORDER BY timestamp_utc DESC LIMIT 1", (profile_id,)).fetchone()
    finally:
        api.close()
    if row is None:
        return {"ok": False, "reason": "no text message"}
    r = query(root, profile_id, row[2][:30], ask_time=datetime.fromisoformat(row[1]) + timedelta(days=1))
    return {"ok": any(row[0] in i["message_ids"] for i in r["items"]), "items": len(r["items"])}


def audit(root, profile_id: str, *, probes: int = 40, seed: str = "memory-audit") -> dict:
    """MEMORY_QUALITY_REPORT.json: counts per layer (imported / searchable / candidates / verified / persisted / retrievable) and
    deterministic retrieval sanity probes from the data itself. Probes are SANITY checks, not a quality benchmark."""
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    st = status(root, profile_id)
    if st["memory_status"] == "NOT_BUILT":
        raise MemoryError_("NOT_BUILT", "run memory-build first")
    prof = S.read_json(pdir / "profile.json")
    api = open_api(pdir, prof)
    try:
        c = api.store._conn
        rows = c.execute("SELECT message_id, timestamp, content, speaker_id FROM messages WHERE target_persona_id = ? AND locator LIKE '%kind=text%' AND length(content) >= 10",
                         (profile_id,)).fetchall()
        mems = c.execute("SELECT memory_id FROM memories WHERE persona_id = ? AND status = 'active'", (profile_id,)).fetchall()
        wb = c.execute("SELECT operations FROM writer_batches WHERE persona_id = ?", (profile_id,)).fetchall() if st["writer_batches"] else []
    finally:
        api.close()
    ops: Dict[str, int] = {}
    cand = 0
    for (o,) in wb:
        for x in json.loads(o):
            cand += 1
            ops[x.get("op", "?")] = ops.get(x.get("op", "?"), 0) + 1
    rng = random.Random(seed)
    sample = rng.sample(rows, min(probes, len(rows)))
    hit = date_hit = 0
    for mid, ts, content, _ in sample:
        t = datetime.fromisoformat(ts)
        r = query(root, profile_id, content[:30], ask_time=t + timedelta(days=1))
        hit += any(mid in i["message_ids"] for i in r["items"])
        r2 = query(root, profile_id, f"{t.year}年{t.month}月{t.day}日聊了什么", ask_time=t + timedelta(days=30))
        date_hit += any((i["time"] or "")[:10] == t.date().isoformat() for i in r2["items"])
    mem_hit = 0
    msample = rng.sample(mems, min(20, len(mems)))
    if msample:
        api = open_api(pdir, prof)
        try:
            stmts = [(m, api.get(profile_id, m)) for (m,) in msample]
        finally:
            api.close()
        for m, rec in stmts:
            if rec is None:
                continue
            r = query(root, profile_id, rec.value or rec.attribute)
            mem_hit += any(rec.value in s for i in r["items"] for s in i["memories"])
    rep = {"schema": "stillhere.persona.memory_quality_report/1.0", "profile_id": profile_id, "label": "FACT (counts) / SANITY PROBES (not a benchmark)",
           "layers": {"imported_messages": st["archive_messages"], "searchable_text_evidence": st["searchable_text_messages"], "sessions": st["sessions"],
                      "writer_sessions_done": st["writer_batches"], "extracted_candidates": cand,
                      "verified_and_committed_operations": sum(v for k, v in ops.items() if k not in ("NOOP",)),
                      "persisted_active_memories": st["memories_active"], "memory_versions_total": st["memories_all_versions"],
                      "retrievable_memories_sampled": f"{mem_hit}/{len(msample)}" if msample else "n/a"},
           "writer_operations": ops, "semantic_extraction": st["semantic_extraction"],
           "retrieval_probes": {"exact_message_recall_at_8": f"{hit}/{len(sample)}", "date_conditioned_hit_at_8": f"{date_hit}/{len(sample)}",
                                "note": "a probe queries with the message's own text (index sanity) or with its date only ('YYYY年M月D日聊了什么': does the time channel bring that day)"},
           "status": st["memory_status"], "at": S.now_iso()}
    out = pdir / "evaluation" / "memory_quality_report.json"
    S.write_json(out, rep)
    return {**rep, "path": str(out)}


# ------------------------------------------------------------------ review-only preview of real sessions (nothing committed)
def preview_sessions(root, profile_id: str, conversation_ids, *, complete: Callable, db_path, out_json=None) -> dict:
    """Run the REAL Writer extraction + its deterministic verification on a few sessions WITHOUT committing anything: memories derived
    from a real chat need the owner's review (store write gate), so this yields PENDING-REVIEW candidates only. The sessions' messages
    are copied into a separate private SQLite file (``db_path``); the production store is only read. ``out_json`` (private) receives
    the candidates for the owner to review; the return value holds counts only."""
    from stillhere.contracts import DataOrigin, SourceKind, SourceRef
    from stillhere.memory.api import MemoryAPI
    from stillhere.writer import m221
    from stillhere.writer.extract import ExtractionCache, extract
    from stillhere.writer.pipeline import Writer
    from stillhere.writer.verify import verify
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    prof = S.read_json(pdir / "profile.json")
    prod = open_api(pdir, prof)
    try:
        c = prod.store._conn
        srcs = {r[0] for r in c.execute("SELECT DISTINCT source_id FROM messages WHERE conversation_id IN (%s)" % ",".join("?" * len(conversation_ids)),
                                        list(conversation_ids)).fetchall()}
        src_refs = [prod.get_source(profile_id, s) for s in srcs]
        msgs = []
        for cid in conversation_ids:
            raw, _ = prod.conversation(profile_id, cid, as_of=datetime(9999, 1, 1, tzinfo=timezone.utc), known_at=datetime(9999, 1, 1, tzinfo=timezone.utc))
            msgs += list(raw)
    finally:
        prod.close()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(db_path).exists():
        Path(db_path).unlink()
    api = MemoryAPI(str(db_path), public_only=False, persona_names={profile_id: [prof["target"].get("display_name") or "TA"]})
    out, totals = [], {"sessions": 0, "candidates": 0, "verified": 0, "ambiguous": 0, "rejected": 0, "schema_ok": 0}
    reasons: Dict[str, int] = {}
    try:
        for s in src_refs:
            api.register_source(s)
        api.ingest_messages(msgs)
        complete = writer_complete(complete)
        w = Writer(api, complete=complete, model="preview", persona_names={profile_id: prof["target"].get("display_name") or "TA"},
                   speakers={other_id(profile_id): "对方"}, cache=ExtractionCache(None), env=WRITER_ENV, round_types=(ROUND_TYPE,), prompt_version=WRITER_PROMPT,
                   strict_self_attribution=True, pre_verify=fix_candidate, post_verify=worthy)
        for cid in conversation_ids:
            group = w.group(profile_id, cid)
            cands, record = extract(complete, "preview", profile_id, prof["target"].get("display_name") or "TA", group, [], ExtractionCache(None), WRITER_PROMPT)
            cands = [m221.prenormalize(x) for x in cands]
            cands = [fix_candidate(x, group, []) for x in cands]
            verdicts = [worthy(verify(x, group, profile_id, [], strict_self=True), group) for x in cands]
            totals["sessions"] += 1
            totals["schema_ok"] += int(bool(record.get("schema_ok")))
            totals["candidates"] += len(cands)
            for v in verdicts:
                key = "verified" if v.committable else ("ambiguous" if v.support == "AMBIGUOUS" else "rejected")
                totals[key] += 1
                if not v.committable:
                    r = (v.reasons[0] if v.reasons else v.support).split(":")[0][:60]
                    reasons[r] = reasons.get(r, 0) + 1
                by = {m.message_id: m for m in group}
                out.append({"conversation": cid, "status": "AUTO_VERIFIED_PENDING_HUMAN_REVIEW" if v.committable else v.support, "subject": v.candidate.subject,
                            "predicate": v.candidate.predicate, "object": v.candidate.object, "event_status": v.candidate.event_status,
                            "certainty": v.candidate.certainty, "event_time": v.candidate.event_time, "planned_time": v.candidate.planned_time,
                            "speakers": sorted({by[m].sender for m in v.candidate.source_message_ids if m in by}),
                            "source_message_ids": v.candidate.source_message_ids, "reasons": v.reasons})
    finally:
        api.close()
    if out_json:
        S.write_json(Path(out_json), {"profile_id": profile_id, "status": "AUTO_VERIFIED candidates await HUMAN review (nothing committed)", "candidates": out})
    return {**totals, "rejection_reasons": reasons, "committed": 0, "db": str(db_path)}
