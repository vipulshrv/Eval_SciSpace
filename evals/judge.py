"""
Hardened LLM-as-judge for grounding checks (Steps 3, 4, 5).

Design (from eval_criteria.txt):
  - 3-way verdict: SUPPORTED / UNSUPPORTED / CONTRADICTED — never binary.
  - The judge MUST quote a verbatim span from the source; no span => cannot be
    SUPPORTED (enforced in the prompt and re-checked in code).
  - Adversarial framing: the judge is told to try to REFUTE the claim, so the
    default leans UNSUPPORTED when evidence is weak (fights sycophancy).
  - Structured output via output_config.format (json_schema) so the model is
    forced to return a valid, parseable verdict — no fragile text parsing.
  - `reasoning` comes before `verdict` in the schema so the model reasons first.

Model: defaults to claude-opus-4-8 (strong grader). Override with JUDGE_MODEL.
Reads ANTHROPIC_API_KEY from the environment or the repo .env file.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import anthropic

# ── config ────────────────────────────────────────────────────────────────

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-8")
JUDGE_EFFORT = os.environ.get("JUDGE_EFFORT", "low")  # low is plenty for entailment

# USD per 1M tokens (input, output). Cache reads bill ~0.1x input, writes ~1.25x.
_PRICING = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
}


def _price(model: str) -> tuple[float, float]:
    for key, rates in _PRICING.items():
        if model.startswith(key):
            return rates
    return (5.0, 25.0)  # default to Opus-tier if unknown

VERDICTS = ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief adversarial analysis: what would it take to refute the claim, and does the source actually support it.",
        },
        "evidence_span": {
            "type": "string",
            "description": "A verbatim quote from the SOURCE that supports the claim. Empty string if none exists.",
        },
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["reasoning", "evidence_span", "verdict", "confidence"],
    "additionalProperties": False,
}

_SYSTEM = """You are a rigorous scientific-claim verifier auditing an AI research \
assistant for hallucination. You are ADVERSARIAL: try to REFUTE the claim rather \
than confirm it. Surface plausibility is not enough — the SOURCE must support it.

Judge ONLY the SUBSTANTIVE scientific/factual content of the claim: the entities, \
mechanisms, findings, directions of effect, and quantities. The source must state \
or entail that substantive content (paraphrase and entailment count — verbatim \
wording is not required).

IGNORE the following when judging — they are NOT part of grounding and must never \
by themselves make a claim unsupported:
- Author names or study attributions (e.g. "Zhou et al.", "a study by X") — whether \
  the right study is cited is checked separately.
- Citation markers like [5], [12].
- Framing / hedging / editorializing words ("a compelling study", "notably", \
  "multiple studies", "importantly").

Verdicts:
- SUPPORTED: the source states or entails the substantive claim. You MUST quote a \
verbatim span from the source in `evidence_span` that carries that support. If you \
cannot quote such a span, it is NOT supported.
- CONTRADICTED: the source asserts something that conflicts with the substantive claim.
- UNSUPPORTED: the source neither supports nor contradicts the substantive claim \
(the claim adds scientific content absent from the source). Default here when the \
substantive content is genuinely missing — NOT merely because an author name or \
framing word is absent.

Judge only against the provided source. Do not use outside knowledge."""


@dataclass
class Judgment:
    verdict: str
    confidence: str
    evidence_span: str
    reasoning: str
    error: str | None = None

    @property
    def supported(self) -> bool:
        return self.verdict == "SUPPORTED"

    def to_dict(self) -> dict:
        return asdict(self)


class Judge:
    def __init__(self, model: str = JUDGE_MODEL, effort: str = JUDGE_EFFORT):
        _load_dotenv()
        self.model = model
        self.effort = effort
        self.client = anthropic.Anthropic()
        self.calls = 0
        self._in = 0
        self._out = 0
        self._cache_read = 0
        self._cache_write = 0

    def _track(self, resp) -> None:
        u = resp.usage
        self.calls += 1
        self._in += getattr(u, "input_tokens", 0) or 0
        self._out += getattr(u, "output_tokens", 0) or 0
        self._cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self._cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0

    def cost_report(self) -> dict:
        in_rate, out_rate = _price(self.model)
        input_cost = self._in / 1e6 * in_rate
        output_cost = self._out / 1e6 * out_rate
        cache_read_cost = self._cache_read / 1e6 * in_rate * 0.1
        cache_write_cost = self._cache_write / 1e6 * in_rate * 1.25
        total = input_cost + output_cost + cache_read_cost + cache_write_cost
        return {
            "model": self.model,
            "judge_calls": self.calls,
            "input_tokens": self._in,
            "output_tokens": self._out,
            "cache_read_input_tokens": self._cache_read,
            "cache_creation_input_tokens": self._cache_write,
            "cost_usd": {
                "input": round(input_cost, 4),
                "output": round(output_cost, 4),
                "cache_read": round(cache_read_cost, 4),
                "cache_write": round(cache_write_cost, 4),
                "total": round(total, 4),
            },
        }

    def grade(self, claim: str, source: str, source_label: str = "SOURCE") -> Judgment:
        """Grade whether `claim` is grounded in `source` (3-way, span-quoting)."""
        if not source.strip():
            return Judgment(
                verdict="UNSUPPORTED",
                confidence="high",
                evidence_span="",
                reasoning="No source text available to ground this claim (context-insufficiency).",
            )
        user = (
            f"{source_label}:\n\"\"\"\n{source.strip()}\n\"\"\"\n\n"
            f"CLAIM:\n\"\"\"\n{claim.strip()}\n\"\"\"\n\n"
            "Try to refute the claim against the source, then return your verdict."
        )
        try:
            resp = self._create(user)
        except anthropic.APIError as e:  # pragma: no cover - network dependent
            return Judgment("UNSUPPORTED", "low", "", "", error=f"{type(e).__name__}: {e}")

        self._track(resp)
        data = _extract_json(resp)
        if data is None:
            return Judgment("UNSUPPORTED", "low", "", "", error="unparseable judge output")

        j = Judgment(
            verdict=data.get("verdict", "UNSUPPORTED"),
            confidence=data.get("confidence", "low"),
            evidence_span=data.get("evidence_span", ""),
            reasoning=data.get("reasoning", ""),
        )
        # Enforce the span rule in code: SUPPORTED requires a non-empty span that
        # actually occurs in the source (guards against a hallucinated quote).
        if j.verdict == "SUPPORTED":
            span = j.evidence_span.strip()
            if not span or _normalize(span) not in _normalize(source):
                j.verdict = "UNSUPPORTED"
                j.reasoning = "(demoted: SUPPORTED without a verifiable source span) " + j.reasoning
        return j

    def structured(self, system: str, user: str, schema: dict, max_tokens: int = 1024) -> dict:
        """Generic tracked structured-output call (used by Step 1 / Step 3)."""
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": schema}, "effort": self.effort},
            )
        except anthropic.APIError as e:  # pragma: no cover
            return {"error": f"{type(e).__name__}: {e}"}
        self._track(resp)
        return _extract_json(resp) or {"error": "unparseable"}

    def source_has_topic(self, topic: str, source: str) -> tuple[bool, str]:
        """For false-N/A checks: does the source contain ANY information about `topic`?

        Used when the agent left a criterion cell as 'not reported' — if the source
        actually has relevant data, that's under-extraction (a false absence).
        Returns (has_info, reasoning).
        """
        if not source.strip():
            return (False, "no source text available")
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "has_information": {"type": "boolean"},
                "evidence_span": {"type": "string"},
            },
            "required": ["reasoning", "has_information", "evidence_span"],
            "additionalProperties": False,
        }
        sys = (
            "You check whether a SOURCE contains any concrete information about a TOPIC. "
            "Answer has_information=true only if the source states something substantive "
            "about the topic (quote the span). General mentions with no detail = false."
        )
        user = (f"TOPIC: {topic}\n\nSOURCE:\n\"\"\"\n{source.strip()}\n\"\"\"\n\n"
                "Does the source contain concrete information about the topic?")
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=768, system=sys,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": schema}, "effort": self.effort},
            )
        except anthropic.APIError as e:  # pragma: no cover
            return (False, f"error: {e}")
        self._track(resp)
        data = _extract_json(resp) or {}
        return (bool(data.get("has_information")), data.get("reasoning", ""))

    def _create(self, user: str):
        return self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={
                "format": {"type": "json_schema", "schema": _VERDICT_SCHEMA},
                "effort": self.effort,
            },
        )


# ── helpers ────────────────────────────────────────────────────────────────

def _extract_json(resp) -> dict | None:
    for block in resp.content:
        if block.type == "text":
            try:
                return json.loads(block.text)
            except json.JSONDecodeError:
                continue
    return None


def _normalize(s: str) -> str:
    return " ".join((s or "").lower().split())


def _load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no python-dotenv dependency)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    path = path or (Path(__file__).resolve().parent.parent / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


if __name__ == "__main__":
    judge = Judge()
    print(f"model={judge.model} effort={judge.effort}\n")

    src = ("Approximately 90% of the body's serotonin is synthesized in the "
           "gastrointestinal tract, where gut microbiota regulate its production.")
    for claim in [
        "About 90% of serotonin is produced in the gut.",           # supported
        "Serotonin is produced entirely in the brain.",             # contradicted
        "Gut microbiota cause Parkinson's disease via serotonin.",  # unsupported
    ]:
        j = judge.grade(claim, src)
        print(f"[{j.verdict:12s}] conf={j.confidence:6s} claim={claim!r}")
        print(f"    span: {j.evidence_span!r}")
        if j.error:
            print(f"    error: {j.error}")
        print()
