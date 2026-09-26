# Vera-beating bot — magicpin AI Challenge

## Approach

One FastAPI service (`bot.py`) implementing all 5 required endpoints
(`/v1/context`, `/v1/tick`, `/v1/reply`, `/v1/healthz`, `/v1/metadata`).

Two layers, deliberately split:

1. **Deterministic safety layer (no LLM call)** — handles the things the
   judge explicitly tests in the Phase-4 replay and penalizes hard if wrong:
   - auto-reply detection (canned-text phrase match + "same message 3rd time"
     rule from the brief) → one soft nudge, then graceful `end`
   - explicit intent ("let's do it", "go ahead") → routes straight to action,
     never re-asks a qualifying question
   - hostile messages → immediate polite `end`, no further contact
   - off-topic-but-not-hostile → brief decline + redirect, conversation stays
     open (doesn't `end`)
   - per-merchant cooldown after 3 sends with no reply
   - anti-repetition: every prior body sent in a conversation is passed to
     the composer and re-checked against the new output
   - suppression-key dedup across ticks

2. **LLM composer** — everything else (the actual message text) goes to a
   single temperature=0 call per message, given the full category + merchant
   + trigger (+ customer) JSON verbatim, plus the hard rules pulled directly
   from `challenge-brief.md` §5, §9, §10, §11 (specificity, category voice,
   service+price over generic %-off, single CTA in the last sentence,
   no fabrication, language matching). Output is forced JSON and validated;
   one retry on a malformed/empty/repeated body, then the tick/reply skips
   that action rather than sending something broken (`{"actions": []}` /
   `{"action":"wait"}`), per the FAQ's own guidance.

## Why split it this way

An LLM *can* be prompted to handle auto-reply/intent/hostility itself, but
those are exactly the cases the brief calls "anti-patterns the judge will
penalize" and the Phase-4 replay scores directly — cheap, deterministic
pattern checks are more reliable there than hoping a single prompt gets it
right under time pressure, and they cost zero latency/tokens.

## Tradeoffs

- The off-topic detector is a simple keyword heuristic, not an LLM call —
  fast and free, but can misfire on a genuinely on-topic message phrased
  unusually. Given the 30s response budget per reply, this felt like the
  right trade.
- `/v1/tick` composes one message per available trigger with a fresh LLM
  call each time; if `available_triggers` is large this is the latency
  bottleneck. Consider batching multiple triggers into one LLM call if you
  see timeouts during a real test run.
- No persistent storage — in-memory only, per the brief's own note that
  this is fine for the test window. Don't restart the process mid-test.

## What additional context would have helped most

A per-merchant "conversation goal" state (e.g. "currently mid-pitch on
offer X") pushed explicitly by the judge, rather than inferred purely from
`conversation_history` text, would make multi-turn cadence planning
(open challenge #3 in the brief) meaningfully more reliable.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...     # or use OpenAI, see below
export TEAM_NAME="Ajay"
export TEAM_MEMBER_1="Ajay"
export CONTACT_EMAIL="you@nsut.ac.in"
uvicorn bot:app --host 0.0.0.0 --port 8080
```

To use OpenAI instead of Anthropic:
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini   # optional override
```

## Local self-test

Use magicpin's own `judge_simulator.py` from the challenge zip against your
running instance:

```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py
```

(Edit its `CONFIGURATION` section at the top with your own LLM key to play
the merchant side — this is a separate key from your bot's own, and can be
the same or a different provider.)

## Deploying to a public URL

Pick whichever is fastest for you (all are free/cheap for a 24h challenge):

- **Render** (easiest): New Web Service → connect a GitHub repo containing
  these 3 files → set `ANTHROPIC_API_KEY` in the dashboard's environment
  variables → start command `uvicorn bot:app --host 0.0.0.0 --port $PORT`.
- **Railway**: similar — push to a repo, `railway up`, set env vars in
  the dashboard.
- **ngrok** (quickest, but only good while your laptop is on and awake):
  ```
  uvicorn bot:app --host 0.0.0.0 --port 8080
  ngrok http 8080
  ```
  Submit the `https://xxxx.ngrok-free.app` URL. Fine if the judge's test
  window is scheduled and short; risky if it can run anytime in the 24h.

Whichever you pick, hit `https://<your-url>/v1/healthz` from your phone's
mobile data (not your own wifi) before submitting, to confirm it's actually
public.

## Pre-submit checklist (from the testing brief §12)

- [ ] `/v1/healthz` reachable from outside your network
- [ ] All 5 endpoints return the exact response shapes above
- [ ] `/v1/context` idempotent — repost the same version, confirm no state change
- [ ] `/v1/tick` returns within 30s even with zero triggers to act on
- [ ] Ran `judge_simulator.py` locally and got non-zero scores
- [ ] Set `TEAM_NAME` / `TEAM_MEMBER_1` / `CONTACT_EMAIL` env vars (shows up
      in `/v1/metadata`, which the judge reads)
- [ ] LLM API key has enough quota/rate limit to survive a ~45-60 min test
