"""An extractor that reaches Claude through the Claude Code CLI.

Why this exists
---------------
The API extractor needs prepaid credits. A Claude subscription cannot pay for
API calls -- Anthropic bills the two separately -- but it does pay for Claude
Code, which ships a documented headless mode. So on a machine with a
subscription and no credit balance, this is the only route to a measured model
row.

It reuses `ClaudeExtractor` wholesale and replaces one thing: the transport.
The prompt, the constrained schema, the coercion, the magnitude convention, the
quote verification and the retry-and-abstain loop are all inherited, so a row
produced here differs from an API row in how the bytes travelled and nothing
else that Anchor scores.

What it cannot tell you
-----------------------
**Cost.** The CLI reports no token usage, so `cost_usd` stays 0.0 and a
frontier row reads "n/a (subscription)" rather than a number. A sweep that puts
this beside a priced model is comparing one measured axis against a blank.

**Whether a document tried to hijack it.** The nested session runs with every
tool stripped (see `NO_TOOLS`), so an injected instruction inside a filing can
spoil one extraction and reach nothing else. Scoring will show that extraction
as ungrounded or abstained, because the quote check still runs.

**Which model answered.** The CLI uses whatever the local Claude Code install
is configured with, and that is not a per-request parameter here. The run
records `claude-code` rather than a model id, because naming a specific model
would assert something this extractor cannot check.

So it yields one honest row, not the three a sweep across `haiku`, `sonnet` and
`opus` would give. Against a baseline at 0.0% that is still worth having, and
it is worth less than $2 of credit.

Latency is real and is recorded.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from anchor.extractors.claude import ClaimedExtraction, ClaudeExtractor, _Attempt

__all__ = ["ClaudeCodeExtractor", "cli_available"]

#: The CLI wraps its answer in prose often enough that asking for bare JSON is
#: not sufficient. This finds the outermost brace-delimited object.
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

#: Strip every tool from the nested session. This is a security control, not a
#: tidiness one.
#:
#: `claude -p` is an agent with filesystem and network access, and the prompt it
#: receives is the text of a third-party PDF. A document containing instructions
#: aimed at a model -- which an attacker can arrange, and which OCR will happily
#: transcribe -- would otherwise be read by something able to act on them.
#: Removing the tools makes the nested call a text completion, so injected
#: instructions can corrupt one extraction and nothing else.
#:
#: Anchor already treats document text as untrusted everywhere else: a quote is
#: checked against the page rather than believed. This extends the same stance
#: to the transport.
NO_TOOLS = ("--disallowedTools", "*")

#: `--disallowedTools` is variadic, so anything after it on the command line is
#: read as another tool name. Passing the prompt there fed the document's own
#: text into a permission flag -- the CLI answered with one parse complaint per
#: word and ran nothing. The prompt goes on stdin for that reason, and the
#: reason is worth keeping: a prompt built from an untrusted PDF must never sit
#: where an argument parser can reach it.


def cli_available() -> bool:
    """Is the Claude Code CLI on PATH?

    Checked by resolving the name, never by running it, so this is safe to call
    while deciding whether the extractor can be built at all.
    """
    return shutil.which("claude") is not None


@dataclass
class ClaudeCodeExtractor(ClaudeExtractor):
    """`ClaudeExtractor` with the SDK call replaced by a headless CLI call."""

    timeout_s: float = 300.0
    executable: str = "claude"
    runner: Any = None
    """Injected for tests. Takes (argv, stdin_text, timeout) and returns the
    CLI's stdout. Defaults to a real subprocess."""

    def __init__(
        self,
        *,
        timeout_s: float = 300.0,
        executable: str = "claude",
        runner: Any = None,
        max_retries: int = 1,
        max_pages: int = 60,
        max_tokens: int = 8000,
    ) -> None:
        super().__init__(
            model="claude-code",
            max_tokens=max_tokens,
            max_retries=max_retries,
            max_pages=max_pages,
        )
        self.timeout_s = timeout_s
        self.executable = executable
        self.runner = runner or _run_cli

    @property
    def name(self) -> str:
        return "claude-code"

    @property
    def priced(self) -> bool:
        """Never. A subscription run has no per-call price to report."""
        return False

    def _attempt(self, doc, feedback):  # type: ignore[override]
        """One headless CLI call, validated exactly as an API response is.

        The prompt is rebuilt from the same `_request` the API path uses, so
        the model sees the same instructions and the same page text. Only the
        envelope differs: no system parameter, no structured-output constraint,
        so the schema is stated in the prompt and enforced on the way back.
        """
        request = self._request(doc, feedback)
        prompt = _flatten(request)

        try:
            stdout = self.runner([self.executable, "-p", *NO_TOOLS], prompt, self.timeout_s)
        except subprocess.TimeoutExpired:
            return _Attempt(fields={}, problems=[], error=f"timed out after {self.timeout_s:g}s")
        except FileNotFoundError:
            return _Attempt(
                fields={},
                problems=[],
                error=(
                    f"{self.executable!r} is not on PATH. The Claude Code CLI is "
                    "required for this extractor."
                ),
            )
        except Exception as exc:
            return _Attempt(fields={}, problems=[], error=f"{type(exc).__name__}: {exc}")

        # No usage is reported, so cost_usd stays 0.0. Wall-clock is measured
        # once per document by extract_document, which is the figure a frontier
        # row should carry.
        result = _Attempt(fields={}, problems=[])

        match = _JSON_BLOCK.search(stdout or "")
        if match is None:
            result.problems = [
                "your previous reply contained no JSON object. Reply with the "
                "JSON object only, no commentary."
            ]
            return result

        try:
            claimed = ClaimedExtraction.model_validate_json(match.group(0))
        except Exception as exc:
            result.problems = [f"your previous reply did not match the schema: {exc}"]
            return result

        for claim in claimed.fields:
            field, problem = self._coerce(claim, doc)
            if problem:
                result.problems.append(problem)
            if field is not None:
                result.fields[claim.name] = field
        return result


def _flatten(request: dict[str, Any]) -> str:
    """Fold the API request into one prompt string for the CLI.

    The CLI takes no system parameter and cannot be given a JSON schema to
    constrain against, so both are stated in the prompt. That is a real
    weakening: the API path rejects a malformed generation at the server, while
    here a malformed reply costs a retry.
    """
    system = "\n".join(
        block["text"] for block in request.get("system", []) if block.get("text")
    )
    user = "\n\n".join(
        block["text"]
        for block in request["messages"][0]["content"]
        if block.get("text")
    )
    schema = json.dumps(ClaimedExtraction.model_json_schema(), indent=2)
    return (
        f"{system}\n\n"
        "Reply with a single JSON object matching this schema and nothing else. "
        "No preamble, no code fence, no commentary.\n\n"
        f"{schema}\n\n{user}"
    )


def _run_cli(argv: list[str], stdin_text: str | None, timeout: float) -> str:
    """Run the CLI and return its stdout.

    Kept separate so tests can replace the transport without a subprocess, and
    so the one place that spawns a process is obvious to a reader auditing what
    this extractor does.
    """
    # Windows ships the CLI as claude.CMD, which CreateProcess will not launch
    # from a bare name. Resolving it here keeps the callers using "claude".
    resolved = shutil.which(argv[0])
    if resolved is None:
        raise FileNotFoundError(argv[0])

    completed = subprocess.run(
        [resolved, *argv[1:]],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"claude exited {completed.returncode}: {(completed.stderr or '').strip()[:200]}"
        )
    return completed.stdout
