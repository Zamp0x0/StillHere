"""Disclosure policy (``stillhere.session.disclosure/1.0``): ALLOWED / PROTECTED / POLICY_UNCERTAIN per memory fact, and the redaction applied
BEFORE anything is rendered. Deterministic; no model; the memory schema is not changed.

Categories come from the slot name and from patterns in the value / statement (never from a model):

    HARD (PROTECTED unless an explicit grant):  id_number, bank_card, password, health
    SOFT (POLICY_UNCERTAIN unless granted / denied): phone, email, address_detail, income
    Anything about ANOTHER person in any sensitive category: PROTECTED unless granted for that person.
    Everything else: ALLOWED.

Grants / denials / revocations live in an additive table (``disclosure_policies``), each with a validity interval, read at the asking time:
a grant ALLOWs, a denial or a revocation PROTECTs. Unknown or unconfirmed = fail closed (never ALLOWED by default).

Redaction (``redact``): a non-ALLOWED fact reaches the model only as a statement that such a record exists and may not be disclosed; the raw
value, the original statement, ``replaces`` and ``note`` are removed. Raw values are returned separately as GUARD TERMS (must-not terms for
the output guard) and are never put in prompts, traces or logs (traces keep only the category and a short hash).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

POLICY_VERSION = "stillhere.session.disclosure/1.0"
ALLOWED, PROTECTED, UNCERTAIN = "ALLOWED", "PROTECTED", "POLICY_UNCERTAIN"
HARD = ("id_number", "bank_card", "password", "health")
SOFT = ("phone", "email", "address_detail", "income")
LABEL = {"phone": "联系方式（电话）", "email": "联系方式（邮箱）", "address_detail": "详细住址", "income": "收入", "id_number": "证件号码",
         "bank_card": "银行卡号", "password": "密码", "health": "健康状况"}
PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("id_number", re.compile(r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])")),
    ("bank_card", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)|(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("address_detail", re.compile(r"\d+\s*(号楼|栋|幢|单元|室|号院|弄)")),
    ("income", re.compile(r"(月薪|年薪|工资|收入|税前|税后)\s*[:：]?\s*\d")),
)
KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("id_number", ("身份证", "证件号", "护照号")), ("bank_card", ("银行卡", "卡号")), ("password", ("密码", "口令", "验证码")),
    ("health", ("诊断", "确诊", "住院", "病历", "抑郁", "焦虑症", "怀孕", "化疗", "处方")),
    ("phone", ("手机号", "电话号", "联系电话", "微信号")), ("email", ("邮箱", "email")), ("address_detail", ("门牌", "详细地址", "住址")),
    ("income", ("月薪", "年薪", "工资", "收入")),
)
# phrases that contain a keyword but do not name that category (removed before that category's keyword check only)
KEYWORD_EXCLUSIONS = {"income": ("工资卡",)}
ATTRIBUTE_CATEGORY = {"phone": "phone", "phone_number": "phone", "mobile": "phone", "contact": "phone", "wechat": "phone", "email": "email",
                      "id_number": "id_number", "bank_card": "bank_card", "password": "password", "salary": "income", "income": "income",
                      "health": "health", "diagnosis": "health", "address": "address_detail", "home_address": "address_detail"}

DDL = """
CREATE TABLE IF NOT EXISTS disclosure_policies (
    policy_id   TEXT PRIMARY KEY,
    persona_id  TEXT NOT NULL,
    subject     TEXT NOT NULL,          -- 'persona' (the person) or 'person:<name>' or '*'
    category    TEXT NOT NULL,          -- a category above, or '*'
    requester   TEXT NOT NULL,          -- conversation partner id, or '*'
    decision    TEXT NOT NULL CHECK (decision IN ('ALLOW','DENY','REVOKE')),
    valid_from  TEXT NOT NULL,
    valid_to    TEXT,
    note        TEXT
);
CREATE INDEX IF NOT EXISTS ix_disclosure_persona ON disclosure_policies (persona_id, category)
"""


_ID_FULL = re.compile(r"\d{17}[0-9Xx]")


def _pattern_hits(text: str) -> List[Tuple[str, str]]:
    """(category, matched substring) in ONE text. Each text is matched on its own: joining texts would create matches across the boundary
    (value "3" + statement "室友小王" -> "3 室友…" read as an address). A number that has the shape of an 18-digit ID is an id_number only,
    never also a bank_card."""
    hits = []
    id_spans = [m.span() for m in dict(PATTERNS)["id_number"].finditer(text or "")]
    for cat, pat in PATTERNS:
        for m in pat.finditer(text or ""):
            if cat == "bank_card" and (_ID_FULL.fullmatch(m.group(0)) or any(m.start() < e and s < m.end() for s, e in id_spans)):
                continue                                  # inside an ID number (incl. its first 17 digits before an X): not a bank card
            hits.append((cat, m.group(0)))
    return hits


def categories(*texts: str, attribute: str = "") -> List[str]:
    found = []
    a = ATTRIBUTE_CATEGORY.get((attribute or "").split(".")[-1].lower())
    if a:
        found.append(a)
    for t in texts:                                       # value and statement are matched separately, never concatenated
        for cat, _ in _pattern_hits(t):
            if cat not in found:
                found.append(cat)
    for t in texts:
        for cat, words in KEYWORDS:
            text = t or ""
            for phrase in KEYWORD_EXCLUSIONS.get(cat, ()):  # "工资卡" is a card, not a salary: its card NUMBER is caught by the bank_card pattern
                text = text.replace(phrase, "")
            if any(w in text for w in words) and cat not in found:
                found.append(cat)
    return found


def raw_terms(*texts: str) -> List[str]:
    """The sensitive substrings themselves (numbers, addresses, emails): what the output guard must block."""
    out = [s for t in texts for _cat, s in _pattern_hits(t)]
    return sorted(set(x for x in out if x.strip()), key=lambda s: (-len(s), s))


def short_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


@dataclass
class Decision:
    policy: str                       # ALLOWED | PROTECTED | POLICY_UNCERTAIN
    categories: List[str]
    reason: str
    guard_terms: List[str] = field(default_factory=list)     # restricted: never rendered / logged

    def trace(self) -> dict:
        return {"policy": self.policy, "categories": self.categories, "reason": self.reason, "guard_terms_sha": [short_hash(t) for t in self.guard_terms]}


class PolicyResolver:
    def __init__(self, api, requester: str = "live:user") -> None:
        self.api, self.requester = api, requester
        with api.transaction() as conn:
            for stmt in DDL.strip().split(";\n"):
                if stmt.strip():
                    conn.execute(stmt)

    def set(self, persona_id: str, subject: str, category: str, decision: str, valid_from: datetime, *, requester: str = "*",
            valid_to: Optional[datetime] = None, note: str = "") -> str:
        pid = "pol." + short_hash(f"{persona_id}|{subject}|{category}|{requester}|{decision}|{valid_from.isoformat()}")
        with self.api.transaction() as conn:
            conn.execute("INSERT OR REPLACE INTO disclosure_policies VALUES (?,?,?,?,?,?,?,?,?)",
                         (pid, persona_id, subject, category, requester, decision, valid_from.isoformat(), valid_to.isoformat() if valid_to else None, note))
        return pid

    def _rules(self, persona_id: str, subject: str, category: str, at: datetime) -> List[str]:
        rows = self.api.store._conn.execute(
            "SELECT decision, valid_from, valid_to FROM disclosure_policies WHERE persona_id = ? AND subject IN (?, '*') AND category IN (?, '*') "
            "AND requester IN (?, '*') ORDER BY valid_from", (persona_id, subject, category, self.requester)).fetchall()
        live = [r[0] for r in rows if datetime.fromisoformat(r[1]) <= at and (r[2] is None or at < datetime.fromisoformat(r[2]))]
        return live

    def decide(self, persona_id: str, subject: str, attribute: str, value: str, statement: str, at: datetime) -> Decision:
        cats = categories(value, statement, attribute=attribute)
        if not cats:
            return Decision(ALLOWED, [], "not a sensitive category")
        terms = raw_terms(value, statement) or ([value] if value and value.strip() else [])
        rules = [d for c in cats for d in self._rules(persona_id, subject, c, at)]
        if "DENY" in rules or "REVOKE" in rules:
            return Decision(PROTECTED, cats, "explicit denial / revoked consent", terms)
        if rules and all(d == "ALLOW" for d in rules) and all(self._rules(persona_id, subject, c, at) for c in cats):
            return Decision(ALLOWED, cats, "explicit grant for every category", [])
        if subject not in ("persona", "self", "本人"):
            return Decision(PROTECTED, cats, "another person's sensitive data without a grant", terms)
        if any(c in HARD for c in cats):
            return Decision(PROTECTED, cats, "hard-protected category without a grant", terms)
        return Decision(UNCERTAIN, cats, "sensitive category, permission not established (fail closed)", terms)


def redacted_text(subject_label: str, decision: Decision) -> str:
    what = "、".join(LABEL.get(c, c) for c in decision.categories) or "敏感信息"
    who = subject_label or "本人"
    if decision.policy == PROTECTED:
        return f"{who}有一条{what}的记录，不可透露具体内容"
    return f"{who}有一条{what}的记录，需要先得到本人同意，目前不能确认可以透露"


def leaked(texts: Sequence[str], guard_terms: Sequence[str]) -> List[str]:
    """Which guard terms appear in any of ``texts`` (returns their short hashes, never the terms)."""
    return sorted({short_hash(t) for t in guard_terms if t and any(t in x for x in texts)})


def scrub(obj, guard_terms: Sequence[str]):
    """A copy of ``obj`` (a trace) with every guarded raw value replaced by its short hash: traces never carry protected values."""
    terms = [t for t in guard_terms if t]
    if not terms:
        return obj
    if isinstance(obj, str):
        for t in sorted(terms, key=len, reverse=True):
            obj = obj.replace(t, f"［受保护:{short_hash(t)}］")
        return obj
    if isinstance(obj, dict):
        return {k: scrub(v, terms) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(scrub(v, terms) for v in obj)
    return obj
