"""The regression gate: committed floors that a run has to clear.

What the gate is for
--------------------
An extraction pipeline degrades quietly. A prompt edit, a model version bump, a
new page-parsing rule -- none of them announce that recall on `total_debt` just
fell by a third. The gate turns that into a build failure: the floors live in a
committed JSON file, so moving one is a diff that a human signs off on, with a
reason, in the same review as the change that needed it.

Thresholds file format
----------------------
::

    {
      "corpus": "corpus/synthetic",
      "extractor": "heuristic",
      "allow_undefined": false,
      "metrics": {
        "accuracy":           {"min": 0.45, "note": "measured 0.50 at 6 docs"},
        "hallucination_rate": {"max": 0.10}
      }
    }

Each entry carries its own direction, so a reader never has to remember whether
a bigger `omission_rate` is better. ``note`` is free text and is printed
alongside a failure -- a floor with no recorded provenance is a number nobody
dares move, which is how gates end up disabled.

Undefined metrics are failures by default
-----------------------------------------
`anchor.scoring` returns ``None`` for a rate whose denominator is zero, and the
gate treats that as a breach rather than a pass. A corpus that stopped
containing any unanswerable field would otherwise sail through a
`hallucination_rate` ceiling by never being asked the question. Setting
``allow_undefined`` to true downgrades those to passes, which is occasionally
right for a per-field floor on a small corpus -- and is then visible in the
thresholds diff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from anchor.runner import METRIC_NAMES, RunResult, UnknownMetric, metric_value

__all__ = [
    "Check",
    "GateError",
    "GateResult",
    "Status",
    "Threshold",
    "Thresholds",
    "check_run",
    "load_thresholds",
]


class GateError(Exception):
    """Raised when a thresholds file cannot be read as one."""


class Status(str, Enum):  # noqa: UP042 - matches the str-Enum convention elsewhere
    PASS = "pass"
    FAIL = "fail"
    UNDEFINED = "undefined"
    """The metric has no value: its denominator was zero. A breach unless
    ``allow_undefined`` is set."""


@dataclass(frozen=True)
class Threshold:
    """One committed floor or ceiling."""

    metric: str
    direction: str
    """``"min"`` or ``"max"``."""
    bound: float
    note: str = ""

    def satisfied_by(self, observed: float) -> bool:
        return observed >= self.bound if self.direction == "min" else observed <= self.bound

    def describe(self) -> str:
        return f"{self.metric} {'>=' if self.direction == 'min' else '<='} {self.bound:g}"


@dataclass(frozen=True)
class Thresholds:
    """A parsed thresholds file."""

    entries: list[Threshold]
    allow_undefined: bool = False
    corpus: str = ""
    extractor: str = ""
    path: Path | None = None

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class Check:
    """The verdict on one threshold."""

    threshold: Threshold
    observed: float | None
    status: Status

    @property
    def passed(self) -> bool:
        return self.status is Status.PASS

    def describe(self) -> str:
        required = f"required {self.threshold.describe()}"
        if self.status is Status.UNDEFINED:
            return f"{self.threshold.metric}: undefined (zero denominator), {required}"
        assert self.observed is not None
        verb = "ok" if self.passed else "BREACH"
        return f"{self.threshold.metric}: {self.observed:.4f} ({verb}, {required})"


@dataclass(frozen=True)
class GateResult:
    """Every check, and whether the build should go red."""

    checks: list[Check]
    run: RunResult

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


def _parse_entry(metric: str, spec: Any) -> Threshold:
    """Read one ``"metric": {...}`` entry.

    A bare number is accepted as a minimum, because that is what everyone
    writes first and rejecting it teaches nothing. Anything ambiguous -- both
    bounds at once, or neither -- is an error: a threshold whose direction the
    reader has to infer is a threshold that will be misread.
    """
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return Threshold(metric=metric, direction="min", bound=float(spec))

    if not isinstance(spec, dict):
        raise GateError(f"{metric}: expected a number or an object with min/max, got {spec!r}")

    has_min, has_max = "min" in spec, "max" in spec
    if has_min == has_max:
        raise GateError(
            f"{metric}: give exactly one of 'min' or 'max' so the direction is explicit"
        )

    direction = "min" if has_min else "max"
    bound = spec[direction]
    if not isinstance(bound, (int, float)) or isinstance(bound, bool):
        raise GateError(f"{metric}: '{direction}' must be a number, got {bound!r}")

    return Threshold(
        metric=metric,
        direction=direction,
        bound=float(bound),
        note=str(spec.get("note", "")),
    )


def load_thresholds(path: Path | str) -> Thresholds:
    """Read and validate a thresholds file.

    Metric names are validated here, before any extraction runs. A typo in a
    metric name would otherwise gate on nothing and report a green build, which
    is worse than no gate -- it is a gate that lies.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{path}: not readable as thresholds ({exc})") from exc

    if not isinstance(raw, dict):
        raise GateError(f"{path}: expected a JSON object")
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise GateError(f"{path}: expected a non-empty 'metrics' object")

    entries = [_parse_entry(name, spec) for name, spec in metrics.items()]
    for entry in entries:
        if not _is_known_metric(entry.metric):
            raise GateError(
                f"{path}: unknown metric {entry.metric!r}. "
                f"Known names: {', '.join(METRIC_NAMES)}, or field.<line_item>.<metric>"
            )

    return Thresholds(
        entries=entries,
        allow_undefined=bool(raw.get("allow_undefined", False)),
        corpus=str(raw.get("corpus", "")),
        extractor=str(raw.get("extractor", "")),
        path=path,
    )


def _is_known_metric(name: str) -> bool:
    if name in METRIC_NAMES:
        return True
    if not name.startswith("field.") or name.count(".") != 2:
        return False
    # Validated by probing the same lookup the gate will use, so the two can
    # never disagree about what a valid name is.
    from anchor.runner import _STATS_METRICS, LineItem

    _, item, metric = name.split(".", 2)
    try:
        LineItem(item)
    except ValueError:
        return False
    return metric in _STATS_METRICS


def check_run(run: RunResult, thresholds: Thresholds) -> GateResult:
    """Grade a completed run against committed thresholds."""
    checks: list[Check] = []
    for threshold in thresholds.entries:
        try:
            observed = metric_value(run, threshold.metric)
        except UnknownMetric:  # pragma: no cover - load_thresholds validates first
            observed = None
        if observed is None:
            status = Status.PASS if thresholds.allow_undefined else Status.UNDEFINED
        else:
            status = Status.PASS if threshold.satisfied_by(observed) else Status.FAIL
        checks.append(Check(threshold=threshold, observed=observed, status=status))
    return GateResult(checks=checks, run=run)
