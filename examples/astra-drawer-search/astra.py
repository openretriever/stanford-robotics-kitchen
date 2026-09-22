"""gpt-6-astra over /v1/responses with forced tool calls and a spend cap. Chat completions
refuses function tools for this model with reasoning_effort; responses is the endpoint robot-sim uses."""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ENDPOINT = "https://api.openai.com/v1/responses"
MODEL = "gpt-6-astra"
# Standard uncached rates, the same figures robot-sim/agent.py charges with.
USD_PER_INPUT_TOKEN = 12.5 / 1_000_000
USD_PER_OUTPUT_TOKEN = 50.0 / 1_000_000
KEY_FILE = Path.home() / ".config/openai/key"


class BudgetExhausted(RuntimeError):
    """Raised instead of quietly spending past the cap the run was given."""


def credential():
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if not value and KEY_FILE.is_file():
        value = KEY_FILE.read_text().strip()
    if not value:
        raise ValueError(f"Set OPENAI_API_KEY or write the key to {KEY_FILE}")
    return value


@dataclass
class Call:
    role: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    usd: float


@dataclass
class Astra:
    """One metered channel to the model; share one instance across flows."""

    model: str = MODEL
    max_calls: int = 60
    max_usd: float = 2.50
    reasoning_effort: str = "low"
    timeout_s: float = 120.0
    calls: list[Call] = field(default_factory=list)
    _key: str | None = None

    # -- accounting ---------------------------------------------------------

    @property
    def spent_usd(self):
        return sum(call.usd for call in self.calls)

    def ledger(self):
        """Per-role totals, for a run record. These are E2's cost metrics."""
        roles = {}
        for call in self.calls:
            row = roles.setdefault(call.role, {"calls": 0, "input_tokens": 0,
                                               "output_tokens": 0, "usd": 0.0, "latency_s": 0.0})
            row["calls"] += 1
            row["input_tokens"] += call.input_tokens
            row["output_tokens"] += call.output_tokens
            row["usd"] += call.usd
            row["latency_s"] += call.latency_s
        return {"by_role": roles, "total_calls": len(self.calls),
                "total_usd": round(self.spent_usd, 4),
                "budget_usd": self.max_usd, "max_calls": self.max_calls}

    def _guard(self, role):
        if len(self.calls) >= self.max_calls:
            raise BudgetExhausted(f"{role}: reached the {self.max_calls}-call ceiling")
        if self.spent_usd >= self.max_usd:
            raise BudgetExhausted(f"{role}: spent ${self.spent_usd:.2f} of ${self.max_usd:.2f}")

    # -- the one request path ----------------------------------------------

    def decide(self, role, system, text, images=(), *, function, parameters, max_tokens=600):
        """Force exactly one call to `function` and return its validated arguments."""
        self._guard(role)
        if self._key is None:
            self._key = credential()

        content = [{"type": "input_text", "text": text}]
        for jpeg in images:
            url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            content.append({"type": "input_image", "detail": "high", "image_url": url})

        body = {
            "model": self.model,
            "instructions": system,
            "input": [{"role": "user", "content": content}],
            # Flat tool shape: on /v1/responses the name sits at the top level,
            # not nested under a "function" key as chat completions expects.
            "tools": [{"type": "function", "name": function, "strict": True,
                       "description": f"Report the {role} result.",
                       "parameters": parameters}],
            "tool_choice": {"type": "function", "name": function},
            "parallel_tool_calls": False,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": self.reasoning_effort},
            "store": False,
        }
        request = urllib.request.Request(
            ENDPOINT, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self._key})
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            # Never surface the upstream body: it can echo request content.
            hint = {401: "the API key was rejected", 403: "account access denied",
                    404: f"{self.model} is unavailable to this key",
                    429: "rate or spending limit reached"}.get(error.code)
            raise RuntimeError(f"{role}: Astra request failed (HTTP {error.code})"
                               + (f" -- {hint}" if hint else "")) from None
        latency = time.time() - started

        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        self.calls.append(Call(
            role=role, input_tokens=input_tokens, output_tokens=output_tokens,
            latency_s=latency,
            usd=input_tokens * USD_PER_INPUT_TOKEN + output_tokens * USD_PER_OUTPUT_TOKEN,
        ))

        if payload.get("status") != "completed":
            raise RuntimeError(f"{role}: response was {payload.get('status')!r}, not completed"
                               f" (an incomplete answer is not a decision)")
        calls = [item for item in payload.get("output", []) if item.get("type") == "function_call"]
        if len(calls) != 1:
            raise RuntimeError(f"{role}: expected exactly one function call, got {len(calls)}")
        arguments = calls[0].get("arguments") or "{}"
        try:
            return json.loads(arguments)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{role}: tool arguments were not JSON ({error})") from None
