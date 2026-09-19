"""Terminal rendering for runs, gates and corpus status.

Kept apart from `anchor.cli` so the CLI stays about arguments and exit codes,
and apart from `anchor.runner` so the data layer never has to think about
column widths.

Output is deliberately plain ASCII. This runs in CI logs, in PowerShell on a
code page that is not UTF-8, and over SSH; a box-drawing character that renders
as a mojibake smear costs more than the prettier table is worth.

The rendering rule that matters: an undefined rate prints as ``n/a``, never as
``0.00``. `anchor.scoring` goes to some trouble to keep "we never asked" and
"we asked and it always failed" apart, and a formatter that collapses them
throws that away at the last step.
"""

from __future__ import annotations

from anchor.gate import GateResult, Status
from anchor.runner import RatioOutcome, RunResult
from anchor.scoring import Outcome, Stats

__all__ = [
    "render_corpus",
    "render_sweep",
    "render_gate",
    "render_run",
    "render_taxonomy",
]

_NA = "n/a"


def _pct(value: float | None, width: int = 7) -> str:
    """A rate as a percentage, or `n/a` when the denominator was zero."""
    if value is None:
        return _NA.rjust(width)
    return f"{value * 100:.1f}%".rjust(width)


def _num(value: float | None) -> str:
    """A money figure with thousands separators, or a dash when absent."""
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _rule(char: str = "-", width: int = 78) -> str:
    return char * width


def _bar(count: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return ""
    filled = round(width * count / total)
    return "#" * filled + "." * (width - filled)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def render_taxonomy(stats: Stats) -> str:
    """The failure taxonomy as counts, share, and a bar.

    Printed in a fixed order rather than sorted by count, so two runs can be
    read side by side without the rows moving under the reader.
    """
    total = stats.n_fields
    lines = [f"{'outcome':<20} {'n':>4}  {'share':>6}"]
    for outcome in Outcome:
        n = stats.taxonomy[outcome]
        share = f"{n / total * 100:5.1f}%" if total else _NA
        lines.append(f"{outcome.value:<20} {n:>4}  {share:>6}  {_bar(n, total)}")
    lines.append(f"{'total':<20} {total:>4}")
    return "\n".join(lines)


def _headline(stats: Stats, label: str) -> str:
    return (
        f"{label:<14}"
        f"  accuracy {_pct(stats.accuracy)}"
        f"  coverage {_pct(stats.coverage)}"
        f"  grounded {_pct(stats.grounding_rate)}"
        f"  halluc {_pct(stats.hallucination_rate)}"
        f"  abst.prec {_pct(stats.abstention_precision)}"
    )


def render_run(run: RunResult, *, verbose: bool = False) -> str:
    """A full run report for the terminal."""
    r = run.report
    out: list[str] = []

    out.append(_rule("="))
    out.append(f"anchor run {run.run_id}")
    out.append(
        f"extractor={run.extractor}  corpus={run.corpus_root}  "
        f"documents={r.n_documents}  fields={r.n_fields}"
    )
    if not run.verified:
        out.append(
            "NOTE: --no-verify was used. Citations were not checked, so accuracy "
            "here is an upper bound and the ungrounded bucket is empty by construction."
        )
    if run.skipped:
        out.append(
            f"SKIPPED {len(run.skipped)} labelled document(s) with no source file: "
            + ", ".join(run.skipped)
        )
    errored = [d for d in run.documents if d.error]
    if errored:
        out.append(f"ERRORS on {len(errored)} document(s):")
        out.extend(f"  {d.doc_id}: {d.error}" for d in errored)
    out.append(_rule("="))

    out.append("")
    out.append(_headline(r.overall, "overall"))
    if r.on_ambiguous.n_fields:
        out.append(_headline(r.on_ambiguous, "on ambiguous"))
        out.append(
            f"{'':14}  (the abstention test set: {r.on_ambiguous.n_fields} field(s) "
            "where labelling required a judgement call)"
        )

    out.append("")
    out.append("FAILURE TAXONOMY")
    out.append(_rule())
    out.append(render_taxonomy(r.overall))

    out.append("")
    out.append("PER LINE ITEM")
    out.append(_rule())
    out.append(
        f"{'line item':<22} {'n':>3} {'answerable':>10} {'accuracy':>9} "
        f"{'coverage':>9} {'grounded':>9}"
    )
    for name, stats in r.per_field.items():
        out.append(
            f"{name.value:<22} {stats.n_fields:>3} {stats.n_answerable:>10} "
            f"{_pct(stats.accuracy, 9)} {_pct(stats.coverage, 9)} "
            f"{_pct(stats.grounding_rate, 9)}"
        )

    out.append("")
    out.append("DERIVED RATIOS")
    out.append(_rule())
    counts = run.ratio_counts
    out.append(
        f"agreement {_pct(run.ratio_agreement)}   safety {_pct(run.ratio_safety)}   "
        f"(n={run.n_ratios})"
    )
    for outcome in RatioOutcome:
        out.append(f"  {outcome.value:<22} {counts[outcome]:>3}")
    out.append(
        "  safety counts abstention as safe: a ratio the pipeline declined is "
        "visibly missing,"
    )
    out.append("  whereas a ratio truth cannot support is a number nobody should act on.")

    sel = run.selective
    if sel is not None:
        out.append("")
        out.append("ABSTENTION AND CALIBRATION")
        out.append(_rule())
        out.append(
            f"AURC {sel.aurc:.4f} (lower is better; base error {sel.baseline_error:.3f} "
            f"at full coverage)"
        )
        if not sel.aurc_is_informative:
            out.append(
                f"  AURC is not informative here: the extractor reported only "
                f"{sel.n_distinct_confidences} distinct confidence value(s), so the curve "
                "is a few points wide"
            )
            out.append(
                "  and its area mostly measures that width rather than how well the "
                "confidences rank. Do not gate on it until an extractor produces a "
                "confidence that varies."
            )
        out.append(
            f"split conformal at alpha={sel.alpha:g}: "
            f"calibration {sel.n_calibration} field(s), {sel.n_calibration_correct} correct; "
            f"test {sel.n_test}"
        )
        if sel.threshold_supported:
            out.append(
                f"  threshold {sel.threshold:.3f} -> coverage {_pct(sel.coverage).strip()}, "
                f"error on answered {_pct(sel.error_on_answered).strip()}, "
                f"target met: {'yes' if sel.guarantee_held else 'NO'}"
            )
        else:
            out.append(
                f"  no valid threshold at this size: {sel.n_calibration_correct} correct "
                f"calibration field(s) cannot carry a {sel.alpha:g} guarantee "
                f"(needs at least {_min_calibration(sel.alpha)}). The system abstains on "
                "everything rather than pretend to a cutoff it has not earned."
            )
        if sel.threshold_supported and not sel.guarantee_held:
            out.append(
                "  the retention guarantee holds by construction; a failed error target "
                "means the confidences do not separate correct from incorrect."
            )

    if verbose:
        out.append("")
        out.append("PER DOCUMENT")
        out.append(_rule())
        for d in run.documents:
            counts = d.score.counts
            summary = "  ".join(
                f"{o.value}={counts[o]}" for o in Outcome if counts[o]
            )
            out.append(f"{d.doc_id}  ({d.n_pages}p)  {summary}")
            for fs in d.score.fields:
                if fs.outcome in (Outcome.CORRECT, Outcome.CORRECT_ABSTENTION):
                    continue
                out.append(
                    f"    {fs.name.value:<22} {fs.outcome.value:<18} "
                    f"expected={_num(fs.expected):>14}  got={_num(fs.actual):>14}"
                )

    return "\n".join(out)


def _min_calibration(alpha: float) -> int:
    """Smallest correct-calibration count that can carry `alpha`.

    From `conformal.calibrate`: a threshold exists when
    ``ceil((n + 1) * (1 - alpha)) <= n``, which first holds at
    ``n >= (1 - alpha) / alpha``.
    """
    import math

    return max(1, math.ceil((1.0 - alpha) / alpha))


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def render_gate(result: GateResult) -> str:
    """The gate verdict, one line per threshold, failures explained."""
    run = result.run
    out = [
        _rule("="),
        f"anchor gate  extractor={run.extractor}  corpus={run.corpus_root}  "
        f"documents={run.report.n_documents}",
        _rule("="),
    ]

    for check in result.checks:
        marker = {Status.PASS: "  ok  ", Status.FAIL: " FAIL ", Status.UNDEFINED: " ???? "}[
            check.status
        ]
        out.append(f"[{marker}] {check.describe()}")
        if not check.passed and check.threshold.note:
            out.append(f"           note: {check.threshold.note}")

    out.append(_rule())
    if result.passed:
        out.append(f"PASS: {len(result.checks)} threshold(s) met.")
        return "\n".join(out)

    out.append(f"FAIL: {len(result.failures)} of {len(result.checks)} threshold(s) breached.")
    out.append("")
    out.append(
        "If the change that caused this is an intended trade -- more abstention for "
        "fewer hallucinations, say -- move the floor in the thresholds file and say "
        "why in the 'note'. Do not widen it silently: the committed number is the "
        "only record of what this pipeline used to be worth."
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def render_corpus(corpus) -> str:  # noqa: ANN001 - anchor.corpus.Corpus, avoids a cycle
    """Corpus inventory: what is labelled, what is resolvable, what is ambiguous."""
    out = [
        _rule("="),
        f"corpus {corpus.root}  documents={len(corpus)}  "
        f"resolvable={len(corpus.resolvable)}  missing={len(corpus.missing)}",
        _rule("="),
        f"{'doc_id':<26} {'src':>5} {'pages':>5} {'fields':>6} {'stated':>6} "
        f"{'ambig':>5}  scanned",
    ]

    n_fields = n_stated = n_ambiguous = 0
    for item in corpus.items:
        g = item.golden
        stated = sum(1 for f in g.fields if f.value is not None)
        ambiguous = sum(1 for f in g.fields if f.ambiguous)
        n_fields += len(g.fields)
        n_stated += stated
        n_ambiguous += ambiguous
        out.append(
            f"{g.doc_id:<26} {item.kind:>5} {g.pages:>5} {len(g.fields):>6} "
            f"{stated:>6} {ambiguous:>5}  {'yes' if g.scanned else ''}"
        )

    out.append(_rule())
    out.append(
        f"{n_fields} labelled field(s): {n_stated} stated by a document, "
        f"{n_fields - n_stated} genuinely absent, {n_ambiguous} flagged ambiguous."
    )
    out.append(
        "Absent fields are the hallucination test set; ambiguous fields are the "
        "abstention test set."
    )

    if corpus.missing:
        out.append("")
        out.append("MISSING SOURCE DOCUMENTS (labelled but not scoreable):")
        for item in corpus.missing:
            out.append(f"  {item.doc_id}")
        out.append(
            "These are excluded from every rate. The PDFs are git-ignored by design; "
            "see corpus/README.md for provenance."
        )

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def _pareto(rows: list[dict]) -> set[int]:
    """Indices that nothing else beats on both cost and accuracy.

    A row is dominated when some other row is at least as accurate and no more
    expensive, and strictly better on one of the two. Reporting the frontier
    rather than a ranking is the point: "most accurate" and "cheapest" are
    different questions, and the rows in between are where the decision is.
    """
    keep: set[int] = set()
    priced = [
        (i, r) for i, r in enumerate(rows)
        if r.get("accuracy") is not None and r.get("cost_per_document") is not None
    ]
    for i, a in priced:
        dominated = any(
            j != i
            and b["accuracy"] >= a["accuracy"]
            and b["cost_per_document"] <= a["cost_per_document"]
            and (b["accuracy"] > a["accuracy"] or b["cost_per_document"] < a["cost_per_document"])
            for j, b in priced
        )
        if not dominated:
            keep.add(i)
    return keep


def render_sweep(rows: list[dict], corpus_root: str, n_documents: int) -> str:
    """The cost/accuracy frontier, one row per extractor configuration."""
    out = [
        _rule("="),
        f"anchor sweep  corpus={corpus_root}  documents={n_documents}  "
        f"configurations={len(rows)}",
        _rule("="),
        "",
        f"{'extractor':<26} {'accuracy':>9} {'grounded':>9} {'halluc':>8} "
        f"{'abst.prec':>10} {'$/doc':>9} {'p50 s':>7}  frontier",
        _rule(),
    ]

    frontier = _pareto(rows)
    for i, r in enumerate(rows):
        if r.get("error"):
            out.append(f"{r['label']:<26} {'FAILED: ' + r['error'][:60]}")
            continue
        cost = r.get("cost_per_document")
        cost_s = "n/a" if cost is None else (f"${cost:.4f}" if cost else "$0")
        p50 = r.get("latency_p50")
        out.append(
            f"{r['label']:<26} {_pct(r['accuracy'], 9)} {_pct(r['grounding_rate'], 9)} "
            f"{_pct(r['hallucination_rate'], 8)} {_pct(r['abstention_precision'], 10)} "
            f"{cost_s:>9} {(f'{p50:.2f}' if p50 is not None else '-'):>7}"
            f"  {'*' if i in frontier else ''}"
        )

    out.append(_rule())
    out.append(
        "* marks the cost/accuracy frontier: no other configuration here is both "
        "cheaper and at least as accurate."
    )

    best = max(
        (r for r in rows if r.get("accuracy") is not None),
        key=lambda r: r["accuracy"],
        default=None,
    )
    if best is not None and best.get("cost_per_document"):
        for r in rows:
            if r is best or r.get("accuracy") is None or not r.get("cost_per_document"):
                continue
            acc_share = r["accuracy"] / best["accuracy"] if best["accuracy"] else 0.0
            cost_share = r["cost_per_document"] / best["cost_per_document"]
            if acc_share >= 0.9 and cost_share <= 0.5:
                out.append(
                    f"  {r['label']} reaches {acc_share * 100:.0f}% of "
                    f"{best['label']}'s accuracy at {cost_share * 100:.0f}% of its cost."
                )

    out.append("")
    out.append(
        f"Read every figure with its N: {n_documents} document(s). At this size a "
        "single document moves accuracy by several points,"
    )
    out.append("and the ordering between adjacent rows is not established.")
    return "\n".join(out)
