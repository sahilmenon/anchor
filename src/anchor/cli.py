"""The `anchor` command line.

Five verbs, each one step of the loop the project exists to close::

    anchor corpus    what is labelled, and can it be scored
    anchor extract   run an extractor over one document, print the claim
    anchor run       score an extractor over a corpus, write the artifact
    anchor gate      the same run, graded against committed floors, exit 1 on breach
    anchor import    build a corpus from an external labelled dataset
    anchor sweep     score several models, print the cost/accuracy frontier
    anchor serve     browse a run: evidence, ratios, the risk-coverage curve

Exit codes are the contract with CI: 0 success, 1 a gate breach (the run worked,
the numbers are not good enough), 2 a usage or data error (the run never
happened). Conflating the last two is how a broken corpus path gets read as a
quality regression at three in the morning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from anchor import __version__
from anchor.corpus import DEFAULT_CORPUS, CorpusError, load_corpus
from anchor.datasets import DATASETS, import_kleister
from anchor.extractors.base import Extractor
from anchor.extractors.claude import (
    DEFAULT_MODEL,
    PRICING,
    ClaudeExtractor,
    api_key_present,
)
from anchor.extractors.heuristic import HeuristicExtractor
from anchor.extractors.offline import OfflineExtractor
from anchor.gate import GateError, check_run, load_thresholds
from anchor.render import render_corpus, render_gate, render_run, render_sweep
from anchor.runner import DEFAULT_ALPHA, run_extractor, save_run, to_dict

__all__ = ["EXIT_DATA_ERROR", "EXIT_GATE_FAILED", "EXIT_OK", "build_parser", "main"]

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_DATA_ERROR = 2

#: Extractors addressable by name. The LLM extractor registers itself here when
#: the optional `llm` extra is installed; the heuristic always exists so that
#: every command in this CLI is runnable with no API key.
EXTRACTORS: dict[str, type] = {
    "heuristic": HeuristicExtractor,
    "claude": ClaudeExtractor,
    "offline": OfflineExtractor,
}

#: Extractors that replay recorded claims rather than producing them. A run of
#: one carries no cost, no latency and no record of what produced it, so it can
#: be scored but must never appear in a cost/accuracy frontier beside a model.
REPLAY_EXTRACTORS: frozenset[str] = frozenset({"offline"})


def _build_extractor(
    name: str,
    model: str | None = None,
    effort: str = "high",
    transcripts: Path | None = None,
) -> Extractor:
    """Instantiate an extractor by name.

    `--model` is meaningful only for extractors that take one, so passing it to
    the heuristic is a usage error rather than a silently ignored flag: a sweep
    that appeared to vary the model while scoring the same regex every time
    would produce a cost/accuracy chart that is pure noise.
    """
    try:
        cls = EXTRACTORS[name]
    except KeyError:
        raise SystemExit(
            f"unknown extractor {name!r}. Available: {', '.join(sorted(EXTRACTORS))}"
        ) from None

    if cls is OfflineExtractor:
        if transcripts is None:
            raise SystemExit(
                "the offline extractor replays recorded claims, so it needs "
                "--transcripts <dir>"
            )
        root = Path(transcripts)
        if not root.is_dir():
            raise SystemExit(f"{root}: no such transcript directory")
        return OfflineExtractor(root=root, label=root.name)

    if cls is ClaudeExtractor:
        # Checked here, before any document is read. Without it a missing
        # credential fails once per document inside the extractor, every field
        # abstains, and the run reports 0% accuracy -- which reads as "this
        # model is useless" when it means "there is no API key". A sweep row
        # saying that would be worse than no row at all.
        if not api_key_present():
            raise SystemExit(
                "no Anthropic credential found. Set ANTHROPIC_API_KEY, or run "
                "`ant auth login`, before scoring a model-backed extractor."
            )
        return ClaudeExtractor(model=model or DEFAULT_MODEL, effort=effort)
    if model:
        raise SystemExit(f"the {name!r} extractor takes no --model")
    return cls()


# ---------------------------------------------------------------------------
# Shared arguments
# ---------------------------------------------------------------------------


def _add_corpus_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="corpus directory (default: %(default)s)",
    )


def _add_run_args(p: argparse.ArgumentParser) -> None:
    _add_corpus_args(p)
    p.add_argument(
        "--extractor",
        default="heuristic",
        help="extractor to score (default: %(default)s)",
    )
    p.add_argument(
        "--model",
        default=None,
        help=f"model id for a model-backed extractor (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="reasoning effort for a model-backed extractor (default: %(default)s)",
    )
    p.add_argument(
        "--transcripts",
        type=Path,
        default=None,
        help="directory of recorded claims, for --extractor offline",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="conformal target miss rate, in (0, 1) (default: %(default)s)",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "skip quote verification. Diagnostic only: accuracy becomes an upper "
            "bound and the ungrounded bucket empties"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anchor",
        description=(
            "Evaluation harness for evidence-anchored credit-metric extraction. "
            "Scores what an extractor got right, what it invented, and whether it "
            "knew when not to answer."
        ),
    )
    parser.add_argument("--version", action="version", version=f"anchor {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- corpus ----------------------------------------------------------
    p_corpus = sub.add_parser(
        "corpus", help="inventory the labelled corpus and check every label resolves"
    )
    _add_corpus_args(p_corpus)
    p_corpus.add_argument("--json", action="store_true", help="machine-readable output")
    p_corpus.set_defaults(func=cmd_corpus)

    # -- extract ---------------------------------------------------------
    p_extract = sub.add_parser(
        "extract", help="run an extractor over one document and print its claim"
    )
    p_extract.add_argument("document", type=Path, help="a .pdf, or a text/<id>.json page file")
    p_extract.add_argument("--extractor", default="heuristic")
    p_extract.add_argument("--transcripts", type=Path, default=None)
    p_extract.add_argument(
        "--no-verify", action="store_true", help="skip quote verification"
    )
    p_extract.set_defaults(func=cmd_extract)

    # -- run -------------------------------------------------------------
    p_run = sub.add_parser("run", help="score an extractor over the corpus")
    _add_run_args(p_run)
    p_run.add_argument(
        "--out",
        type=Path,
        default=Path("runs"),
        help="directory to write the run artifact to (default: %(default)s)",
    )
    p_run.add_argument("--no-save", action="store_true", help="do not write an artifact")
    p_run.add_argument("--json", action="store_true", help="print the run as JSON")
    p_run.add_argument(
        "-v", "--verbose", action="store_true", help="list every non-correct field"
    )
    p_run.set_defaults(func=cmd_run)

    # -- gate ------------------------------------------------------------
    p_gate = sub.add_parser(
        "gate", help="score the corpus and fail if a committed threshold is breached"
    )
    _add_run_args(p_gate)
    p_gate.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="thresholds file (default: <corpus>/thresholds.json)",
    )
    p_gate.add_argument("--json", action="store_true", help="machine-readable verdict")
    p_gate.set_defaults(func=cmd_gate)

    # -- import ----------------------------------------------------------
    p_import = sub.add_parser(
        "import", help="build a corpus from an external labelled dataset"
    )
    p_import.add_argument("dataset", choices=sorted(DATASETS), help="dataset to import")
    p_import.add_argument(
        "--from",
        dest="source",
        type=Path,
        required=True,
        help="local checkout of the dataset",
    )
    p_import.add_argument(
        "--out",
        type=Path,
        default=None,
        help="corpus directory to write (default: corpus/<dataset>)",
    )
    p_import.add_argument("--split", default="dev-0", help="default: %(default)s")
    p_import.add_argument(
        "--limit", type=int, default=None, help="import at most this many documents"
    )
    p_import.add_argument(
        "--text-column",
        default="text_best",
        help="which OCR pass to use (default: %(default)s)",
    )
    p_import.add_argument(
        "--unambiguous",
        action="store_true",
        help=(
            "do not flag imported values as ambiguous. Off by default: the "
            "mapping embeds a definitional choice, not a reading"
        ),
    )
    p_import.set_defaults(func=cmd_import)

    # -- sweep -----------------------------------------------------------
    p_sweep = sub.add_parser(
        "sweep", help="score several extractors and print the cost/accuracy frontier"
    )
    _add_corpus_args(p_sweep)
    p_sweep.add_argument(
        "--models",
        default=",".join(sorted(PRICING)),
        help="comma-separated model ids to sweep (default: every priced model)",
    )
    p_sweep.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="reasoning effort applied to every model in the sweep (default: %(default)s)",
    )
    p_sweep.add_argument(
        "--no-baseline",
        action="store_true",
        help="omit the heuristic baseline row",
    )
    p_sweep.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help=argparse.SUPPRESS
    )
    p_sweep.add_argument("--no-verify", action="store_true", help=argparse.SUPPRESS)
    p_sweep.add_argument("--out", type=Path, default=Path("runs"))
    p_sweep.add_argument("--no-save", action="store_true")
    p_sweep.add_argument("--json", action="store_true", help="machine-readable frontier")
    p_sweep.set_defaults(func=cmd_sweep)

    # -- serve -----------------------------------------------------------
    p_serve = sub.add_parser("serve", help="browse runs in a browser")
    _add_run_args(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1", help="default: %(default)s")
    p_serve.add_argument("--port", type=int, default=8756, help="default: %(default)s")
    p_serve.add_argument(
        "--runs",
        type=Path,
        default=Path("runs"),
        help="directory of saved run artifacts (default: %(default)s)",
    )
    p_serve.add_argument(
        "--no-open", action="store_true", help="do not open a browser window"
    )
    p_serve.set_defaults(func=cmd_serve)

    return parser


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_corpus(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    if args.json:
        print(
            json.dumps(
                {
                    "root": str(corpus.root),
                    "n_documents": len(corpus),
                    "documents": [
                        {
                            "doc_id": i.doc_id,
                            "kind": i.kind,
                            "source": None if i.source is None else str(i.source),
                            "pages": i.golden.pages,
                            "scanned": i.golden.scanned,
                            "n_fields": len(i.golden.fields),
                            "n_stated": sum(1 for f in i.golden.fields if f.value is not None),
                            "n_ambiguous": sum(1 for f in i.golden.fields if f.ambiguous),
                        }
                        for i in corpus.items
                    ],
                },
                indent=2,
            )
        )
    else:
        print(render_corpus(corpus))

    # A label with no document is a data problem, not a bad score.
    return EXIT_DATA_ERROR if corpus.missing else EXIT_OK


def cmd_extract(args: argparse.Namespace) -> int:
    from anchor.corpus import load_text_document
    from anchor.extractors.base import Document
    from anchor.verify import verify_extraction

    path = Path(args.document)
    if not path.is_file():
        raise CorpusError(f"{path}: no such file")

    doc = (
        Document.from_pdf(path)
        if path.suffix.lower() == ".pdf"
        else load_text_document(path)
    )
    extractor = _build_extractor(
        args.extractor,
        getattr(args, "model", None),
        transcripts=getattr(args, "transcripts", None),
    )
    extraction = extractor.extract_document(doc, doc.doc_id)  # type: ignore[attr-defined]
    if not args.no_verify:
        extraction = verify_extraction(extraction, doc)

    print(extraction.model_dump_json(indent=2))
    return EXIT_OK


def _load_and_run(args: argparse.Namespace):
    corpus = load_corpus(args.corpus)
    if not corpus.resolvable:
        raise CorpusError(
            f"{corpus.root}: {len(corpus)} label(s), none with a source document. "
            "Nothing to score."
        )
    return run_extractor(
        corpus,
        _build_extractor(args.extractor, args.model, args.effort, args.transcripts),
        verify=not args.no_verify,
        alpha=args.alpha,
    )


def cmd_run(args: argparse.Namespace) -> int:
    run = _load_and_run(args)

    if args.json:
        print(json.dumps(to_dict(run), indent=2))
    else:
        print(render_run(run, verbose=args.verbose))

    if not args.no_save:
        path = save_run(run, args.out)
        if not args.json:
            print(f"\nwrote {path}")
    return EXIT_OK


def cmd_gate(args: argparse.Namespace) -> int:
    thresholds_path = args.thresholds or Path(args.corpus) / "thresholds.json"
    # Parsed before extracting anything: a malformed thresholds file should cost
    # a second, not a full run.
    thresholds = load_thresholds(thresholds_path)

    run = _load_and_run(args)
    result = check_run(run, thresholds)

    if args.json:
        print(
            json.dumps(
                {
                    "passed": result.passed,
                    "thresholds": str(thresholds_path),
                    "extractor": run.extractor,
                    "corpus": run.corpus_root,
                    "checks": [
                        {
                            "metric": c.threshold.metric,
                            "direction": c.threshold.direction,
                            "bound": c.threshold.bound,
                            "observed": c.observed,
                            "status": c.status.value,
                            "note": c.threshold.note,
                        }
                        for c in result.checks
                    ],
                },
                indent=2,
            )
        )
    else:
        print(render_gate(result))

    return EXIT_OK if result.passed else EXIT_GATE_FAILED


def cmd_import(args: argparse.Namespace) -> int:
    """Convert an external dataset into a scoreable Anchor corpus.

    The licence check is not paperwork. Kleister declares none, so labels
    derived from it must not be committed, and writing them somewhere
    git-ignored by default is cheaper than remembering not to `git add` them.
    """
    out = args.out or (Path("corpus") / args.dataset)

    if args.dataset != "kleister-charity":  # pragma: no cover - one adapter today
        raise CorpusError(f"no adapter for {args.dataset!r}")

    result = import_kleister(
        args.source,
        out,
        split=args.split,
        limit=args.limit,
        text_column=args.text_column,
        mark_ambiguous=not args.unambiguous,
    )
    print(result.describe())
    if result.n_documents == 0:
        print(
            "\nNothing was imported. Check the split name and that the checkout "
            "contains in.tsv.xz and expected.tsv.",
            file=sys.stderr,
        )
        return EXIT_DATA_ERROR
    print(f"\nScore it:  anchor run --corpus {out}")
    return EXIT_OK


def cmd_sweep(args: argparse.Namespace) -> int:
    """Score several configurations over one corpus and compare them.

    The question a sweep answers is a business one, not a benchmark one: how
    much accuracy does the cheapest model give up, and is that trade worth
    making at your volume. Reporting accuracy alone cannot answer it, and
    reporting cost alone cannot either -- so every row carries both, plus the
    grounding and abstention numbers that say whether the cheap row is cheap
    because it is efficient or because it stopped answering.

    A configuration that fails (no credential, an unknown model, a network
    error) is reported as a failed row rather than aborting the sweep: the
    other rows are still worth having, and a sweep that dies on its third model
    has wasted the first two.
    """
    corpus = load_corpus(args.corpus)
    if not corpus.resolvable:
        raise CorpusError(f"{corpus.root}: nothing to score.")

    configs: list[tuple[str, str | None]] = []
    if not args.no_baseline:
        configs.append(("heuristic", None))

    configs += [("claude", m.strip()) for m in args.models.split(",") if m.strip()]

    # A replay extractor has no cost and no provenance, so a frontier row for it
    # would set a measured model beside an unmeasured transcript. Score one with
    # `anchor run` and report it as a sidebar.
    replay = [name for name, _ in configs if name in REPLAY_EXTRACTORS]
    if replay:
        raise CorpusError(
            f"{replay[0]!r} replays recorded claims and cannot appear in a "
            "cost/accuracy frontier. Score it with `anchor run` instead."
        )

    rows: list[dict] = []
    for name, model in configs:
        label = model or name
        if not args.json:
            print(f"scoring {label} ...", file=sys.stderr)
        try:
            run = run_extractor(
                corpus,
                _build_extractor(name, model, args.effort),
                verify=not args.no_verify,
                alpha=args.alpha,
            )
        except SystemExit as exc:
            rows.append({"label": label, "error": str(exc)})
            continue
        except Exception as exc:
            rows.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
            continue

        overall = run.report.overall
        rows.append(
            {
                "label": label,
                "extractor": run.extractor,
                "accuracy": overall.accuracy,
                "grounding_rate": overall.grounding_rate,
                "hallucination_rate": overall.hallucination_rate,
                "abstention_precision": overall.abstention_precision,
                "cost_per_document": run.cost_per_document,
                "latency_p50": run.latency_p50,
                "total_cost_usd": run.report.total_cost_usd,
                "retries": sum(d.extraction.retries for d in run.documents),
            }
        )
        if not args.no_save:
            save_run(run, args.out)

    if args.json:
        print(json.dumps({"corpus": str(corpus.root), "rows": rows}, indent=2))
    else:
        print(render_sweep(rows, str(corpus.root), len(corpus.resolvable)))
    return EXIT_OK if any("error" not in r for r in rows) else EXIT_DATA_ERROR


def cmd_serve(args: argparse.Namespace) -> int:
    from anchor.web import serve

    serve(
        corpus_root=args.corpus,
        runs_dir=args.runs,
        extractor_name=args.extractor,
        host=args.host,
        port=args.port,
        alpha=args.alpha,
        verify=not args.no_verify,
        open_browser=not args.no_open,
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns an exit code rather than calling sys.exit.

    Data problems (`CorpusError`, `GateError`) are reported as one line on
    stderr with exit code 2, not as a traceback: the person who pointed the gate
    at the wrong directory needs the sentence, not the stack.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (CorpusError, GateError) as exc:
        print(f"anchor: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\ninterrupted", file=sys.stderr)
        return EXIT_DATA_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
