"""
magicpin AI Challenge — Vera-beating bot
=========================================

Implements the 5-endpoint contract from challenge-testing-brief.md:
  POST /v1/context   - receive category/merchant/customer/trigger pushes
  POST /v1/tick       - periodic wake-up; bot may initiate messages
  POST /v1/reply      - respond to a merchant/customer reply
  GET  /v1/healthz
  GET  /v1/metadata

Design summary (see README.md for the full writeup):
  - In-memory context + conversation store, keyed exactly as the judge keys it.
  - A single LLM "composer" call per outbound message, built from the 4-context
    framework (category, merchant, trigger, customer), few-shot anchored on the
    brief's own "good message" examples, and a strict output-JSON contract.
  - A cheap deterministic layer HANDLES the things that don't need an LLM and
    that the brief explicitly calls out as scoring gates:
      * auto-reply detection (same/near-same merchant text repeated)
      * explicit intent -> action-mode routing (no re-qualifying)
      * hostile / off-topic handling
      * anti-repetition (never resend an identical body in a conversation)
      * suppression-key dedup and a 3-unanswered-nudge cooldown per merchant
    These are scored heavily in the Phase-4 replay and are easy to get wrong
    with a pure LLM call, so we gate them in code and only hand the LLM the
    cases that need real composition.

Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    uvicorn bot:app --host 0.0.0.0 --port 8080
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")  # anthropic | openai | gemini
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash",
}
LLM_MODEL = os.environ.get("LLM_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, "gemini-3.5-flash"))
TEAM_NAME = os.environ.get("TEAM_NAME", "Ajay")
TEAM_MEMBERS = [os.environ.get("TEAM_MEMBER_1", "Ajay")]
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "you@example.com")

MAX_UNANSWERED_NUDGES = 3  # cooldown after this many sends with no reply

app = FastAPI()
START_TIME = time.time()

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

# (scope, context_id) -> {"version": int, "payload": dict}
contexts: dict[tuple[str, str], dict[str, Any]] = {}

# conversation_id -> conversation state
conversations: dict[str, dict[str, Any]] = {}

# suppression_key -> True (already fired this run) — coarse global dedup
fired_suppression_keys: set[str] = set()

# merchant_id -> count of consecutive sends with no reply (cooldown tracker)
unanswered_nudges: dict[str, int] = {}

# Auto-reply detection is tracked per MERCHANT, not per conversation: the judge
# harness (and real WhatsApp usage) may address the same merchant through a new
# conversation_id on every turn, so conversation-scoped state would never
# accumulate. merchant_recent_texts holds every message seen from that merchant
# across all conversations; merchant_auto_reply_nudged is the one-shot flag.
merchant_recent_texts: dict[str, list[str]] = {}
merchant_auto_reply_nudged: dict[str, bool] = {}


def get_ctx(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


# ---------------------------------------------------------------------------
# LLM client (minimal, dependency-light)
# ---------------------------------------------------------------------------


def llm_complete(system: str, user: str) -> str:
    """Single-shot, temperature=0 completion. Returns raw text."""
    if LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "max_tokens": 500,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []))
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "temperature": 0,
                "max_tokens": 500,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=25,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    elif LLM_PROVIDER == "gemini":
        # Uses Google's OpenAI-compatible endpoint rather than the native
        # generateContent endpoint: Google's Sept-2026 auth-key migration
        # (AI Studio now issues "AQ."-prefixed keys by default) has caused
        # widespread 401 ACCESS_TOKEN_TYPE_UNSUPPORTED errors on the native
        # REST endpoint for many accounts. The OpenAI-compatible endpoint
        # uses a plain Bearer token and works with both old and new key
        # formats, so it's the more reliable integration path right now.
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "temperature": 0,
                "max_tokens": 500,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=25,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER {LLM_PROVIDER}")


def extract_json(text: str) -> dict:
    """LLMs sometimes wrap JSON in prose/fences; pull the first {...} block out."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {text[:200]}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Composer — this is the heart of the submission
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the message-composition engine for "Vera", magicpin's \
merchant-marketing WhatsApp assistant. You write ONE message at a time, given four \
context layers (category, merchant, trigger, optional customer). Your job is to beat \
today's production Vera, whose known weaknesses are: generic discount copy instead of \
service+price offers, losing momentum by re-qualifying merchants who already said yes, \
under-using social proof and "ask the merchant" hooks, and burning turns on WhatsApp \
auto-replies.

HARD RULES:
1. Anchor on a concrete, verifiable fact from the given contexts (a number, a date, a \
   headline, a peer stat). Never write vague lines like "boost your sales" or "grow your \
   business" with no number attached.
2. Match the category's voice exactly: use only vocabulary from vocab_allowed where \
   relevant, and NEVER use a word from vocab_taboo. Clinical/peer categories (dentists, \
   doctors) must sound like a colleague, not an ad.
3. Prefer service+price offers ("Haircut @ ₹99") over generic percentage-off framing, \
   when a matching offer exists in the merchant's or category's offer catalog.
4. Personalize to THIS merchant's real numbers (their CTR, their signals, their \
   conversation history) — don't write something that could apply to any merchant in \
   the category.
5. State plainly why you are messaging now — tie the message to the specific trigger.
6. Use at least one compulsion lever: specificity, loss aversion, social proof, effort \
   externalization ("I've drafted X, just say go"), curiosity, reciprocity, asking the \
   merchant a direct question, or a single binary commitment (reply YES/STOP).
7. Exactly ONE call-to-action, placed in the last sentence. Binary (yes/no-shaped) for \
   action triggers; no CTA at all for pure-information triggers. Never offer 2+ choices \
   in one message unless it's a booking-slot choice (1 for Wed, 2 for Thu is fine).
8. NEVER invent data that isn't in the contexts you were given — no fake research \
   citations, no fake competitor names, no fake numbers.
9. No long preambles ("I hope you're doing well..."). Don't re-introduce yourself if \
   conversation_history shows you've already spoken to this merchant.
10. Match the merchant's/customer's language preference. Hindi-English code-mix is \
    normal and often preferred for Indian merchant audiences — use it when the \
    category's code_mix setting or the customer's language_pref calls for it.
11. If send_as would be "merchant_on_behalf" (a customer-facing message), voice rules \
    are stricter: no medical/outcome claims, no "guaranteed", always identify the \
    merchant by name up front.

You will be told which prior message bodies have already been sent in this \
conversation — never repeat one verbatim or near-verbatim.

Respond with ONLY a JSON object, no prose, no markdown fences:
{
  "body": "<the WhatsApp message text>",
  "cta": "open_ended" | "binary_yes_stop" | "booking_choice" | "none",
  "rationale": "<one sentence: why this message, what it should achieve>"
}
"""


def build_compose_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict],
    prior_bodies: list[str],
) -> str:
    parts = [
        "### CategoryContext",
        json.dumps(category, indent=2, ensure_ascii=False),
        "\n### MerchantContext",
        json.dumps(merchant, indent=2, ensure_ascii=False),
        "\n### TriggerContext",
        json.dumps(trigger, indent=2, ensure_ascii=False),
    ]
    if customer:
        parts += ["\n### CustomerContext (this message goes to the merchant's customer)",
                   json.dumps(customer, indent=2, ensure_ascii=False)]
    if prior_bodies:
        parts += ["\n### Already sent in this conversation (do NOT repeat)",
                   json.dumps(prior_bodies, ensure_ascii=False)]
    parts += [
        "\nCompose the single next best WhatsApp message per the rules in the system prompt.",
    ]
    return "\n".join(parts)


def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    prior_bodies: Optional[list[str]] = None,
) -> dict:
    """Calls the LLM, validates, and retries once on failure. Returns dict with
    body/cta/rationale. Raises on repeated failure (caller should skip the send)."""
    prior_bodies = prior_bodies or []
    prompt = build_compose_prompt(category, merchant, trigger, customer, prior_bodies)

    last_err = None
    for attempt in range(2):
        try:
            raw = llm_complete(SYSTEM_PROMPT, prompt)
            parsed = extract_json(raw)
            body = (parsed.get("body") or "").strip()
            if not body:
                raise ValueError("empty body")
            if body in prior_bodies:
                raise ValueError("repeated a prior body verbatim")
            return {
                "body": body,
                "cta": parsed.get("cta", "open_ended"),
                "rationale": parsed.get("rationale", ""),
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"composer failed after retries: {last_err}")


def fallback_body(merchant: dict) -> str:
    """Used only if the LLM is unreachable — keeps the endpoint contract intact
    (never return malformed/empty) without ever hallucinating specifics."""
    name = merchant.get("identity", {}).get("name", "there")
    return f"Hi {name}, checking in — anything I can help with on your listing today?"


# ---------------------------------------------------------------------------
# Deterministic conversation-safety layer
# ---------------------------------------------------------------------------

AUTO_REPLY_PATTERNS = [
    "thank you for contacting", "thanks for reaching out", "will get back to you",
    "team tak pahuncha", "automated assistant", "aapki jaankari ke liye",
    "shukriya", "we will revert", "busy right now",
]

INTENT_PHRASES = [
    "yes i want to join", "let's do it", "lets do it", "go ahead", "ok let's do it",
    "okay lets do it", "chalo", "haan karo", "start karo", "yes please", "sounds good, do it",
    "proceed", "yes send", "go for it", "mujhe join karna hai", "i'm in",
]

HOSTILE_PATTERNS = [
    "stop messaging", "spam", "useless", "harass", "leave me alone", "f*ck", "fuck",
    "shut up", "annoying", "block you", "report you",
]

NOT_INTERESTED_PATTERNS = [
    "not interested", "no thanks", "nahi chahiye", "stop", "unsubscribe",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def is_auto_reply(text: str, prior_texts: list[str]) -> bool:
    norm = normalize(text)
    if any(p in norm for p in AUTO_REPLY_PATTERNS):
        return True
    # exact-repeat heuristic per the brief: same message verbatim 3+ times
    same_count = sum(1 for t in prior_texts if normalize(t) == norm)
    return same_count >= 2  # this would be the 3rd occurrence


def has_explicit_intent(text: str) -> bool:
    norm = normalize(text)
    return any(p in norm for p in INTENT_PHRASES)


def is_hostile(text: str) -> bool:
    norm = normalize(text)
    return any(p in norm for p in HOSTILE_PATTERNS)


def is_not_interested(text: str) -> bool:
    norm = normalize(text)
    return any(p in norm for p in NOT_INTERESTED_PATTERNS)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class ContextBody(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in contexts.keys():
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": LLM_MODEL,
        "approach": (
            "LLM composer over the 4-context framework, gated by a deterministic "
            "layer for auto-reply detection, intent-to-action routing, hostile/"
            "off-topic handling, anti-repetition and per-merchant nudge cooldown."
        ),
        "contact_email": CONTACT_EMAIL,
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    key = (body.scope, body.context_id)
    current = contexts.get(key)
    if current and current["version"] >= body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": current["version"],
        }
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions: list[dict] = []

    for trigger_id in body.available_triggers[:20]:
        trigger = get_ctx("trigger", trigger_id)
        if not trigger:
            continue

        suppression_key = trigger.get("suppression_key", trigger_id)
        if suppression_key in fired_suppression_keys:
            continue  # already sent this exact trigger family

        merchant_id = trigger.get("merchant_id")
        merchant = get_ctx("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue

        # cooldown: stop nudging a merchant who hasn't replied to N prior sends
        if unanswered_nudges.get(merchant_id, 0) >= MAX_UNANSWERED_NUDGES:
            continue

        category = get_ctx("category", merchant.get("category_slug", ""))
        if not category:
            continue

        customer = None
        customer_id = trigger.get("customer_id")
        if trigger.get("scope") == "customer" and customer_id:
            customer = get_ctx("customer", customer_id)
            if not customer:
                continue  # can't compose a customer-facing message without it

        try:
            composed = compose_message(category, merchant, trigger, customer, prior_bodies=[])
        except Exception:
            continue  # per FAQ: better to skip than to send something broken

        conversation_id = f"conv_{merchant_id}_{trigger_id}"
        send_as = "merchant_on_behalf" if customer else "vera"

        conversations[conversation_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trigger_id,
            "send_as": send_as,
            "history": [{"from": send_as, "msg": composed["body"]}],
            "turns": 1,
            "ended": False,
        }
        fired_suppression_keys.add(suppression_key)
        unanswered_nudges[merchant_id] = unanswered_nudges.get(merchant_id, 0) + 1

        name = merchant.get("identity", {}).get("name", "")
        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trigger_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [name, trigger.get("kind", "")],
            "body": composed["body"],
            "cta": composed["cta"],
            "suppression_key": suppression_key,
            "rationale": composed["rationale"],
        })

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    convo = conversations.get(body.conversation_id)
    if not convo:
        # judge replying to a conversation we don't recognise — start minimal state
        convo = {
            "merchant_id": body.merchant_id,
            "customer_id": body.customer_id,
            "trigger_id": None,
            "send_as": "vera",
            "history": [],
            "turns": 0,
            "ended": False,
        }
        conversations[body.conversation_id] = convo

    convo["history"].append({"from": body.from_role, "msg": body.message})
    if body.merchant_id:
        unanswered_nudges[body.merchant_id] = 0  # they replied — reset cooldown

    text = body.message

    # Track this merchant's message history at the merchant level (not just
    # conversation level) — see merchant_recent_texts comment above for why.
    merchant_texts = merchant_recent_texts.setdefault(body.merchant_id, []) if body.merchant_id else []
    prior_merchant_texts = list(merchant_texts)  # snapshot before appending current
    if body.merchant_id:
        merchant_texts.append(text)

    # 1. Auto-reply detection — try once more, then exit gracefully.
    # Checked against this merchant's full message history across every
    # conversation_id they've ever used, since a new conversation_id per turn
    # (real WhatsApp threads, or this judge harness) must not reset detection.
    if is_auto_reply(text, prior_merchant_texts):
        already_nudged = merchant_auto_reply_nudged.get(body.merchant_id, False)
        if already_nudged:
            convo["ended"] = True
            return {"action": "end", "rationale": "Merchant channel is an auto-reply bot; ending to avoid wasting turns."}
        if body.merchant_id:
            merchant_auto_reply_nudged[body.merchant_id] = True
        reply_body = "Samajh gayi — team tak pahunchane se pehle, kya aap khud 2 min dekh sakte hain? Agar nahi, main directly owner se connect kar lungi."
        convo["history"].append({"from": convo["send_as"], "msg": reply_body})
        return {"action": "send", "body": reply_body, "cta": "open_ended",
                "rationale": "First auto-reply detected; one lightweight nudge before escalating/exiting."}
    elif body.merchant_id:
        merchant_auto_reply_nudged[body.merchant_id] = False  # genuine reply — reset the one-shot flag

    # 2. Hostile — apologize once and stop
    if is_hostile(text):
        convo["ended"] = True
        return {"action": "end", "rationale": "Merchant signaled hostility; exiting immediately, no further contact."}

    # 3. Explicit not-interested / opt-out
    if is_not_interested(text):
        convo["ended"] = True
        return {"action": "end", "rationale": "Merchant declined; respecting opt-out."}

    # 4. Explicit intent -> action mode, skip qualifying entirely
    if has_explicit_intent(text):
        merchant = get_ctx("merchant", convo["merchant_id"]) if convo["merchant_id"] else None
        name = merchant.get("identity", {}).get("name", "") if merchant else ""
        action_body = f"Great, {name} — starting now. I'll confirm here the moment it's done, no further steps needed from you."
        convo["history"].append({"from": convo["send_as"], "msg": action_body})
        return {"action": "send", "body": action_body, "cta": "none",
                "rationale": "Merchant gave explicit go-ahead; routed straight to action instead of re-qualifying."}

    # 5. Off-topic but not hostile — answer briefly, redirect to mission, don't end
    # (heuristic: message doesn't reference merchant/offer/profile keywords at all)
    on_topic_markers = ["profile", "offer", "listing", "review", "customer", "cleaning",
                         "post", "google", "yes", "no", "thanks", "ok", "abstract", "draft"]
    if len(text.split()) > 4 and not any(m in normalize(text) for m in on_topic_markers):
        redirect_body = "That's outside what I can help with directly, but happy to keep helping with your listing/offers — want me to continue where we left off?"
        convo["history"].append({"from": convo["send_as"], "msg": redirect_body})
        return {"action": "send", "body": redirect_body, "cta": "open_ended",
                "rationale": "Off-topic ask; declined briefly without ending the relationship, redirected to core mission."}

    # 6. Normal engaged reply — compose the next message with an LLM, given full context
    merchant = get_ctx("merchant", convo["merchant_id"]) if convo["merchant_id"] else None
    trigger = get_ctx("trigger", convo["trigger_id"]) if convo["trigger_id"] else None
    customer = get_ctx("customer", convo["customer_id"]) if convo["customer_id"] else None
    category = get_ctx("category", merchant.get("category_slug", "")) if merchant else None

    if not (merchant and category):
        convo["ended"] = True
        return {"action": "end", "rationale": "Lost required context for this conversation; ending safely."}

    synthetic_trigger = dict(trigger) if trigger else {
        "id": "reply_followup", "scope": "merchant", "kind": "reply_followup",
        "source": "internal", "payload": {"merchant_reply": text}, "urgency": 2,
        "suppression_key": f"followup:{body.conversation_id}",
    }
    synthetic_trigger["payload"] = {**synthetic_trigger.get("payload", {}), "latest_merchant_message": text}

    prior_bodies = [h["msg"] for h in convo["history"] if h.get("from") == convo["send_as"]]

    try:
        composed = compose_message(category, merchant, synthetic_trigger, customer, prior_bodies)
    except Exception:
        return {"action": "wait", "wait_seconds": 900, "rationale": "Composer temporarily unavailable; backing off rather than sending a broken reply."}

    convo["history"].append({"from": convo["send_as"], "msg": composed["body"]})
    return {"action": "send", "body": composed["body"], "cta": composed["cta"],
            "rationale": composed["rationale"]}


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    fired_suppression_keys.clear()
    unanswered_nudges.clear()
    return {"status": "wiped"}
