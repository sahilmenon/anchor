"""Tests for the Claude extractor.

Every test here runs against a fake client. That is not a compromise — the
behaviour worth testing is what the extractor does with a response, and a live
model would make those cases non-deterministic and unaffordable to assert on.
The one thing a fake cannot cover is whether the prompt elicits good answers,
and no unit test could cover that anyway; `anchor sweep` is where that gets
measured.
"""

from __future__ import annotations

import json
import types

import pytest

from anchor.extractors.base import Document
from anchor.extractors.claude import (
    PRICING,
    ClaimedExtraction,
    ClaudeExtractor,
    estimate_cost,
)
from anchor.schema import LineItem

PAGES = [
    "ACME PTY LTD\nSTATEMENT OF PROFIT OR LOSS\nAll amounts in A$'000\n\n"
    "Total revenue                               24,180      21,405\n"
    "Finance costs                                 (842)       (774)\n",
    "STATEMENT OF FINANCIAL POSITION\nAll amounts in A$'000\n\n"
    "Cash and cash equivalents                    1,842       1,196\n"
    "Borrowings                                   8,750      10,250\n",
]

NAMES = [item.value for item in LineItem]

REVENUE_ROW = "Total revenue                               24,180      21,405"


def claim(name, value, page=None, quote=None, scale=1.0, confidence=0.9, currency="AUD"):
    return {
        "name": name,
        "value": value,
        "unit_scale": scale,
        "currency": currency,
        "page": page,
        "quote": quote,
        "confidence": confidence,
    }


def answer(*claims):
    """A full answer: the given claims, plus an abstention for everything else."""
    named = {c["name"] for c in claims}
    return {"fields": list(claims) + [claim(n, None) for n in NAMES if n not in named]}


class FakeClient:
    """Replays canned responses and records the requests it received."""

    def __init__(self, responses, stop_reason="end_turn", usage=None):
        self.responses = list(responses)
        self.stop_reason = stop_reason
        self.requests: list[dict] = []
        self._usage = usage or types.SimpleNamespace(
            input_tokens=1000,
            output_tokens=200,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("the extractor made more calls than expected")
        payload = self.responses.pop(0)
        if isinstance(payload, Exception):
            raise payload
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=text)],
            usage=self._usage,
            stop_reason=self.stop_reason,
        )


@pytest.fixture
def doc() -> Document:
    return Document.from_pages("acme", PAGES)


def run(doc, *responses, **kwargs):
    client = FakeClient(list(responses))
    extractor = ClaudeExtractor(client=client, **kwargs)
    return extractor.extract_document(doc, doc.doc_id), client


GOOD_REVENUE = claim("revenue_ltm", 24180, 1, REVENUE_ROW, 1000.0)
FABRICATED_DEBT = claim(
    "total_debt", 10250, 2, "Total debt                                  10,250", 1000.0
)
REAL_DEBT = claim(
    "total_debt",
    8750,
    2,
    "Borrowings                                   8,750      10,250",
    1000.0,
)


class TestHappyPath:
    def test_grounded_field_survives(self, doc) -> None:
        result, _ = run(doc, answer(GOOD_REVENUE))
        field = result.get(LineItem.REVENUE_LTM)
        assert field.value == 24180
        assert field.unit_scale == 1000.0
        assert field.scaled_value == 24_180_000
        assert field.evidence.page == 1

    def test_returns_every_line_item(self, doc) -> None:
        result, _ = run(doc, answer(GOOD_REVENUE))
        assert [f.name for f in result.fields] == list(LineItem)

    def test_abstention_passes_through_untouched(self, doc) -> None:
        result, _ = run(doc, answer(GOOD_REVENUE))
        assert result.get(LineItem.CFADS).abstained

    def test_records_cost_and_tokens(self, doc) -> None:
        result, _ = run(doc, answer(GOOD_REVENUE), model="claude-opus-5")
        assert result.input_tokens == 1000
        assert result.output_tokens == 200
        assert result.cost_usd == pytest.approx(estimate_cost("claude-opus-5", 1000, 200))

    def test_is_named_for_its_model(self, doc) -> None:
        result, _ = run(doc, answer(GOOD_REVENUE), model="claude-haiku-4-5")
        assert result.extractor == "claude:claude-haiku-4-5"
        assert result.model == "claude-haiku-4-5"


class TestSignConvention:
    def test_a_parenthesised_expense_is_stored_positive(self, doc) -> None:
        """MAGNITUDE_ITEMS applies to the model's output, not just the regex."""
        result, _ = run(
            doc,
            answer(
                claim(
                    "interest_expense",
                    842,
                    1,
                    "Finance costs                                 (842)       (774)",
                    1000.0,
                )
            ),
        )
        assert result.get(LineItem.INTEREST_EXPENSE).value == 842

    def test_a_negative_claim_is_normalised(self, doc) -> None:
        result, _ = run(
            doc,
            answer(
                claim(
                    "interest_expense",
                    -842,
                    1,
                    "Finance costs                                 (842)       (774)",
                    1000.0,
                )
            ),
        )
        assert result.get(LineItem.INTEREST_EXPENSE).value == 842


class TestRetryAndAbstain:
    def test_a_fabricated_quote_triggers_one_retry(self, doc) -> None:
        result, client = run(doc, answer(FABRICATED_DEBT), answer(REAL_DEBT))
        assert result.retries == 1
        assert len(client.requests) == 2
        assert result.get(LineItem.TOTAL_DEBT).value == 8750

    def test_the_retry_names_the_specific_failure(self, doc) -> None:
        _, client = run(doc, answer(FABRICATED_DEBT), answer(REAL_DEBT))
        blocks = client.requests[1]["messages"][0]["content"]
        feedback = " ".join(b["text"] for b in blocks)
        assert "total_debt" in feedback
        assert "not on page 2" in feedback

    def test_a_field_that_stays_ungrounded_abstains(self, doc) -> None:
        """The whole point: a bad citation that survives correction is not an answer."""
        result, _ = run(doc, answer(FABRICATED_DEBT), answer(FABRICATED_DEBT))
        assert result.get(LineItem.TOTAL_DEBT).abstained
        assert result.retries == 1

    def test_a_good_field_is_not_lost_when_another_is_retried(self, doc) -> None:
        first = answer(GOOD_REVENUE, FABRICATED_DEBT)
        second = answer(REAL_DEBT)  # revenue omitted on the retry
        result, _ = run(doc, first, second)
        assert result.get(LineItem.REVENUE_LTM).value == 24180
        assert result.get(LineItem.TOTAL_DEBT).value == 8750

    def test_malformed_json_is_retried(self, doc) -> None:
        result, client = run(doc, '{"fields": [{"name": "not-a-line-item"}]}',
                             answer(GOOD_REVENUE))
        assert result.retries == 1
        assert result.get(LineItem.REVENUE_LTM).value == 24180
        assert "did not match the schema" in " ".join(
            b["text"] for b in client.requests[1]["messages"][0]["content"]
        )

    def test_max_retries_zero_abstains_immediately(self, doc) -> None:
        result, client = run(doc, answer(FABRICATED_DEBT), max_retries=0)
        assert len(client.requests) == 1
        assert result.retries == 0
        assert result.get(LineItem.TOTAL_DEBT).abstained

    def test_cost_accumulates_across_attempts(self, doc) -> None:
        result, _ = run(doc, answer(FABRICATED_DEBT), answer(REAL_DEBT))
        one_call = estimate_cost("claude-opus-5", 1000, 200)
        assert result.cost_usd == pytest.approx(one_call * 2)


class TestRejectedClaims:
    @pytest.mark.parametrize(
        ("bad", "reason"),
        [
            (claim("cash", 1842, 99, "Cash and cash equivalents                    1,842"), "page"),
            (claim("cash", 1842, None, None), "no page or quote"),
            (claim("cash", 1842, 2, "A row that is not on the page"), "not on page"),
            (
                claim("cash", 9999, 2, "Cash and cash equivalents                    1,842"),
                "does not appear",
            ),
        ],
    )
    def test_unusable_claims_abstain_and_explain(self, doc, bad, reason) -> None:
        result, client = run(doc, answer(bad), answer(bad))
        assert result.get(LineItem.CASH).abstained
        feedback = " ".join(b["text"] for b in client.requests[1]["messages"][0]["content"])
        assert reason in feedback

    def test_confidence_is_clamped_into_range(self, doc) -> None:
        result, _ = run(
            doc, answer(claim("revenue_ltm", 24180, 1, REVENUE_ROW, 1000.0, confidence=7.5))
        )
        assert result.get(LineItem.REVENUE_LTM).confidence == 1.0

    def test_a_non_positive_unit_scale_falls_back_to_one(self, doc) -> None:
        result, _ = run(
            doc, answer(claim("revenue_ltm", 24180, 1, REVENUE_ROW, scale=0.0))
        )
        assert result.get(LineItem.REVENUE_LTM).unit_scale == 1.0


class TestFailureModes:
    def test_a_refusal_abstains_without_raising(self, doc) -> None:
        client = FakeClient([answer(GOOD_REVENUE)], stop_reason="refusal")
        result = ClaudeExtractor(client=client).extract_document(doc, "acme")
        assert all(f.abstained for f in result.fields)

    def test_an_api_error_abstains_without_raising(self, doc) -> None:
        result, _ = run(doc, RuntimeError("connection reset"))
        assert all(f.abstained for f in result.fields)

    def test_an_empty_document_abstains_without_calling_the_api(self) -> None:
        client = FakeClient([])
        result = ClaudeExtractor(client=client).extract_document(
            Document.from_pages("empty", []), "empty"
        )
        assert all(f.abstained for f in result.fields)
        assert client.requests == []

    def test_a_missing_text_block_abstains(self, doc) -> None:
        class NoText(FakeClient):
            def create(self, **kwargs):
                self.requests.append(kwargs)
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(type="thinking", thinking="...")],
                    usage=self._usage,
                    stop_reason="end_turn",
                )

        result = ClaudeExtractor(client=NoText([])).extract_document(doc, "acme")
        assert all(f.abstained for f in result.fields)


class TestRequestShape:
    def test_sends_page_text_not_the_pdf(self, doc) -> None:
        """Both sides must see the same text, or grounding measures the wrong thing."""
        _, client = run(doc, answer(GOOD_REVENUE))
        content = client.requests[0]["messages"][0]["content"]
        assert all(b["type"] == "text" for b in content)
        assert "Total revenue" in content[0]["text"]
        assert '<page number="1">' in content[0]["text"]

    def test_constrains_output_with_the_pydantic_schema(self, doc) -> None:
        _, client = run(doc, answer(GOOD_REVENUE))
        fmt = client.requests[0]["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"] == ClaimedExtraction.model_json_schema()

    def test_caches_the_document_prefix_so_a_retry_is_cheap(self, doc) -> None:
        _, client = run(doc, answer(FABRICATED_DEBT), answer(REAL_DEBT))
        for request in client.requests:
            document_block = request["messages"][0]["content"][0]
            assert document_block["cache_control"] == {"type": "ephemeral"}

    def test_passes_the_effort_setting_through(self, doc) -> None:
        _, client = run(doc, answer(GOOD_REVENUE), effort="low")
        assert client.requests[0]["output_config"]["effort"] == "low"

    def test_does_not_send_sampling_parameters(self, doc) -> None:
        """They are rejected on current models; sending one is a 400."""
        _, client = run(doc, answer(GOOD_REVENUE))
        for banned in ("temperature", "top_p", "top_k"):
            assert banned not in client.requests[0]

    def test_truncates_a_document_beyond_max_pages(self) -> None:
        big = Document.from_pages("big", [f"page {i}" for i in range(1, 11)])
        client = FakeClient([answer(claim("cash", None))])
        ClaudeExtractor(client=client, max_pages=3).extract_document(big, "big")
        text = client.requests[0]["messages"][0]["content"][0]["text"]
        assert '<page number="3">' in text
        assert '<page number="4">' not in text


class TestSchema:
    def test_forbids_extra_properties(self) -> None:
        """Structured outputs require additionalProperties: false on every object."""
        schema = ClaimedExtraction.model_json_schema()
        assert schema["additionalProperties"] is False
        assert schema["$defs"]["ClaimedField"]["additionalProperties"] is False

    def test_enumerates_every_line_item(self) -> None:
        enum = ClaimedExtraction.model_json_schema()["$defs"]["LineItem"]["enum"]
        assert set(enum) == {item.value for item in LineItem}

    def test_carries_no_numeric_constraints(self) -> None:
        """The dialect does not support them; stating them would promise nothing."""
        blob = json.dumps(ClaimedExtraction.model_json_schema())
        for unsupported in ("minimum", "maximum", "exclusiveMinimum", "multipleOf"):
            assert unsupported not in blob


class TestPricing:
    def test_every_priced_model_costs_something(self) -> None:
        for model in PRICING:
            assert estimate_cost(model, 1000, 1000) > 0

    def test_an_unpriced_model_reports_zero_rather_than_guessing(self) -> None:
        assert estimate_cost("some-future-model", 10_000, 10_000) == 0.0
        assert ClaudeExtractor(model="some-future-model").priced is False

    def test_cache_reads_are_cheaper_than_fresh_input(self) -> None:
        fresh = estimate_cost("claude-opus-5", input_tokens=10_000)
        cached = estimate_cost("claude-opus-5", cache_read_tokens=10_000)
        assert cached == pytest.approx(fresh * 0.1)

    def test_ordering_matches_the_published_tiers(self) -> None:
        cost = lambda m: estimate_cost(m, 10_000, 1_000)  # noqa: E731
        assert cost("claude-haiku-4-5") < cost("claude-sonnet-5") < cost("claude-opus-5")
        assert cost("claude-opus-5") < cost("claude-fable-5")


class TestClientConstruction:
    def test_constructing_the_extractor_needs_no_credentials(self) -> None:
        """Listing extractors or running `anchor corpus` must not need a key."""
        extractor = ClaudeExtractor()
        assert extractor._client is None
        assert extractor.name == "claude:claude-opus-5"
