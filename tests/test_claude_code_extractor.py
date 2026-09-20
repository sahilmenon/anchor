"""Tests for the Claude Code CLI extractor.

The transport is injected, so none of these spawn a process. What they pin is
that a subscription-routed run inherits every check the API path applies, and
that it never claims the two things it cannot know: what it cost, and which
model answered.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from anchor.extractors.base import Document
from anchor.extractors.claude_code import ClaudeCodeExtractor, _flatten, cli_available
from anchor.schema import LineItem

PAGES = [
    "ACME PTY LTD\nAll amounts in A$'000\n\n"
    "Total revenue                               24,180      21,405\n"
    "Finance costs                                 (842)       (774)\n"
]
ROW = "Total revenue                               24,180      21,405"
NAMES = [item.value for item in LineItem]


@pytest.fixture
def doc() -> Document:
    return Document.from_pages("acme", PAGES)


def reply(value=24180, quote=ROW, name="revenue_ltm", scale=1000.0):
    fields = [{"name": name, "value": value, "unit_scale": scale, "currency": "AUD",
               "page": 1, "quote": quote, "confidence": 0.9}]
    fields += [{"name": n, "value": None, "unit_scale": 1.0, "currency": None,
                "page": None, "quote": None, "confidence": 0.0}
               for n in NAMES if n != name]
    return json.dumps({"fields": fields})


def runner_returning(*responses):
    seq = list(responses)
    calls = []

    def run(argv, stdin, timeout):
        calls.append({"argv": argv, "prompt": argv[2], "timeout": timeout})
        payload = seq.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload

    run.calls = calls
    return run


class TestTransport:
    def test_parses_a_clean_reply(self, doc) -> None:
        ex = ClaudeCodeExtractor(runner=runner_returning(reply()))
        field = ex.extract_document(doc, "acme").get(LineItem.REVENUE_LTM)
        assert field.scaled_value == 24_180_000

    def test_parses_json_wrapped_in_prose(self, doc) -> None:
        """The CLI is conversational; bare-JSON instructions are not enough."""
        ex = ClaudeCodeExtractor(runner=runner_returning(
            f"Sure, here you go:\n\n```json\n{reply()}\n```\n\nLet me know!"))
        assert ex.extract_document(doc, "acme").get(LineItem.REVENUE_LTM).value == 24180

    def test_invokes_the_cli_in_headless_mode(self, doc) -> None:
        run = runner_returning(reply())
        ClaudeCodeExtractor(runner=run).extract_document(doc, "acme")
        assert run.calls[0]["argv"][:2] == ["claude", "-p"]

    def test_the_prompt_carries_the_schema_and_the_page_text(self, doc) -> None:
        run = runner_returning(reply())
        ClaudeCodeExtractor(runner=run).extract_document(doc, "acme")
        prompt = run.calls[0]["prompt"]
        assert "Total revenue" in prompt          # the document
        assert "revenue_ltm" in prompt            # the schema
        assert "principal_repayments" in prompt   # every line item


class TestClaimsItCannotMake:
    def test_reports_no_cost(self, doc) -> None:
        ex = ClaudeCodeExtractor(runner=runner_returning(reply()))
        assert ex.extract_document(doc, "acme").cost_usd == 0.0
        assert ex.priced is False

    def test_does_not_name_a_model(self, doc) -> None:
        """The CLI uses whatever it is configured with; asserting a model id lies."""
        ex = ClaudeCodeExtractor(runner=runner_returning(reply()))
        result = ex.extract_document(doc, "acme")
        assert result.model == "claude-code"
        assert result.extractor == "claude-code"

    def test_measures_latency(self, doc) -> None:
        ex = ClaudeCodeExtractor(runner=runner_returning(reply()))
        assert ex.extract_document(doc, "acme").latency_s > 0


class TestInheritedChecks:
    def test_an_ungrounded_quote_is_retried_then_abstained(self, doc) -> None:
        bad = reply(quote="Total debt   99,999")
        ex = ClaudeCodeExtractor(runner=runner_returning(bad, bad))
        result = ex.extract_document(doc, "acme")
        assert result.get(LineItem.REVENUE_LTM).abstained
        assert result.retries == 1

    def test_the_magnitude_convention_applies(self, doc) -> None:
        row = "Finance costs                                 (842)       (774)"
        ex = ClaudeCodeExtractor(runner=runner_returning(
            reply(value=-842, quote=row, name="interest_expense")))
        assert ex.extract_document(doc, "acme").get(LineItem.INTEREST_EXPENSE).value == 842

    def test_a_reply_with_no_json_is_retried(self, doc) -> None:
        ex = ClaudeCodeExtractor(runner=runner_returning("I can't help with that.", reply()))
        result = ex.extract_document(doc, "acme")
        assert result.retries == 1
        assert result.get(LineItem.REVENUE_LTM).value == 24180


class TestFailureModes:
    def test_a_timeout_abstains(self, doc) -> None:
        ex = ClaudeCodeExtractor(
            runner=runner_returning(subprocess.TimeoutExpired("claude", 1)))
        assert all(f.abstained for f in ex.extract_document(doc, "acme").fields)

    def test_a_missing_cli_abstains(self, doc) -> None:
        ex = ClaudeCodeExtractor(runner=runner_returning(FileNotFoundError()))
        assert all(f.abstained for f in ex.extract_document(doc, "acme").fields)

    def test_a_nonzero_exit_abstains(self, doc) -> None:
        ex = ClaudeCodeExtractor(runner=runner_returning(RuntimeError("claude exited 1")))
        assert all(f.abstained for f in ex.extract_document(doc, "acme").fields)


class TestWiring:
    def test_availability_is_checked_without_running_the_cli(self) -> None:
        assert isinstance(cli_available(), bool)

    def test_it_is_registered_and_marked_unpriced(self) -> None:
        from anchor.cli import EXTRACTORS, UNPRICED_EXTRACTORS

        assert "claude-code" in EXTRACTORS
        assert "claude-code" in UNPRICED_EXTRACTORS

    def test_flatten_states_the_schema_in_the_prompt(self, doc) -> None:
        ex = ClaudeCodeExtractor(runner=runner_returning(reply()))
        prompt = _flatten(ex._request(doc, []))
        assert "json" in prompt.lower()
        assert "no commentary" in prompt.lower()
