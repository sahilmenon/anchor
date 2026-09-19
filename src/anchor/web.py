"""A local web interface for browsing runs.

Why this exists
---------------
The terminal report gives you rates. It cannot give you the thing that makes
those rates trustworthy: the quote the extractor actually read, next to the
figure it reported, next to what the label says. A credit reviewer reading
"interest_expense: mismatch, expected 842,000, got -842,000" needs one more
click to see the row it came from before they can tell a parsing bug from a
sign convention -- and in a terminal that click does not exist.

So this serves the run artifact with its evidence attached: every field
expands to the page, the quote, the grounding verdict and the labeller's note.
The derived ratios come with their refusal reasons, so a DSCR that was declined
says which input it was declined for.

Design constraints, and why
---------------------------
* **Python standard library only.** The package already has four runtime
  dependencies, all of which earn their place; a web framework to serve one
  page and four JSON endpoints would not. `http.server` is enough.
* **Loopback by default.** A run artifact contains borrower financials. The
  default bind address is 127.0.0.1 and `serve` warns when it is overridden.
* **No filesystem routing.** The only file this server reads on request is its
  own page; run artifacts are addressed by id and matched against the directory
  listing rather than by joining a client-supplied path. That removes path
  traversal as a category rather than defending against it.
* **Read-mostly.** One endpoint has a side effect (`POST /api/run` scores the
  corpus again), and it takes no path or command from the client -- only an
  extractor name checked against the registry, and an alpha.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from pydantic import ValidationError

from anchor import __version__
from anchor.corpus import (
    MAX_PDF_BYTES,
    CorpusError,
    load_corpus,
    load_golden,
    load_text_document,
    safe_doc_id,
    save_golden,
    save_pdf,
    unlabelled_sources,
)
from anchor.extractors.base import Document
from anchor.runner import (
    DEFAULT_ALPHA,
    inspect_document,
    load_run,
    run_extractor,
    save_run,
    to_dict,
)
from anchor.schema import GoldenRecord

__all__ = ["AnchorApp", "make_server", "serve"]

_STATIC = Path(__file__).resolve().parent / "static"
_INDEX = _STATIC / "index.html"


class AnchorApp:
    """Server-side state: where the corpus is, where runs are kept, what to run.

    Holds no per-request state, so the threading server can share one instance.
    The one mutating operation takes a lock: two browser tabs hitting "run"
    together would otherwise write the same artifact twice and race on the
    directory.
    """

    def __init__(
        self,
        corpus_root: Path | str,
        runs_dir: Path | str,
        extractor_name: str = "heuristic",
        alpha: float = DEFAULT_ALPHA,
        verify: bool = True,
    ) -> None:
        self.corpus_root = Path(corpus_root)
        self.runs_dir = Path(runs_dir)
        self.extractor_name = extractor_name
        self.alpha = alpha
        self.verify = verify
        self._lock = threading.Lock()

    # -- data ------------------------------------------------------------

    def config(self) -> dict[str, Any]:
        from anchor.cli import EXTRACTORS

        return {
            "version": __version__,
            "corpus_root": str(self.corpus_root),
            "runs_dir": str(self.runs_dir),
            "extractor": self.extractor_name,
            "extractors": sorted(EXTRACTORS),
            "alpha": self.alpha,
            "verify": self.verify,
        }

    def _artifact_paths(self) -> dict[str, Path]:
        """Saved runs by id. The only mapping from a client id to a path."""
        if not self.runs_dir.is_dir():
            return {}
        return {p.stem: p for p in sorted(self.runs_dir.glob("*.json"))}

    def list_runs(self) -> list[dict[str, Any]]:
        """Saved runs, newest first, summarised for the picker.

        An artifact that will not parse is listed with its error rather than
        omitted: a runs directory that quietly shows four of five files is a
        worse experience than one that says which file is broken.
        """
        out: list[dict[str, Any]] = []
        for run_id, path in self._artifact_paths().items():
            try:
                data = load_run(path)
            except Exception as exc:
                out.append({"run_id": run_id, "error": str(exc)})
                continue
            overall = data.get("report", {}).get("overall", {})
            out.append(
                {
                    "run_id": data.get("run_id", run_id),
                    "extractor": data.get("extractor"),
                    "corpus_root": data.get("corpus_root"),
                    "created_at": data.get("created_at"),
                    "verified": data.get("verified", True),
                    "n_documents": data.get("report", {}).get("n_documents", 0),
                    "accuracy": overall.get("accuracy"),
                    "grounding_rate": overall.get("grounding_rate"),
                    "hallucination_rate": overall.get("hallucination_rate"),
                }
            )
        out.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
        return out

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        path = self._artifact_paths().get(run_id)
        return None if path is None else load_run(path)

    def execute(
        self, extractor_name: str | None = None, alpha: float | None = None, save: bool = True
    ) -> dict[str, Any]:
        """Score the corpus now and return the run as data."""
        from anchor.cli import EXTRACTORS

        name = extractor_name or self.extractor_name
        if name not in EXTRACTORS:
            raise CorpusError(
                f"unknown extractor {name!r}. Available: {', '.join(sorted(EXTRACTORS))}"
            )

        with self._lock:
            corpus = load_corpus(self.corpus_root)
            if not corpus.resolvable:
                raise CorpusError(
                    f"{corpus.root}: {len(corpus)} label(s), none with a source "
                    "document. Nothing to score."
                )
            run = run_extractor(
                corpus,
                EXTRACTORS[name](),
                verify=self.verify,
                alpha=self.alpha if alpha is None else alpha,
            )
            if save:
                save_run(run, self.runs_dir)
        return to_dict(run)

    def corpus_summary(self) -> dict[str, Any]:
        corpus = load_corpus(self.corpus_root)
        return {
            "root": str(corpus.root),
            "n_documents": len(corpus),
            "missing": [i.doc_id for i in corpus.missing],
            # Documents on disk that carry no label yet. They are invisible to
            # `load_corpus` by design, and would be invisible here too -- an
            # upload that vanished with no explanation of what to do next.
            "unlabelled": unlabelled_sources(self.corpus_root),
            "documents": [
                {
                    "doc_id": i.doc_id,
                    "kind": i.kind,
                    "pages": i.golden.pages,
                    "scanned": i.golden.scanned,
                    "source": i.golden.source,
                    "n_fields": len(i.golden.fields),
                    "n_stated": sum(1 for f in i.golden.fields if f.value is not None),
                    "n_ambiguous": sum(1 for f in i.golden.fields if f.ambiguous),
                }
                for i in corpus.items
            ],
        }

    # -- upload, inspect, label -------------------------------------------

    def _document(self, doc_id: str) -> Document:
        """Locate and parse one document by id, labelled or not."""
        root = Path(self.corpus_root)
        pdf = root / "pdfs" / f"{doc_id}.pdf"
        if pdf.is_file():
            return Document.from_pdf(pdf, doc_id)
        text = root / "text" / f"{doc_id}.json"
        if text.is_file():
            return load_text_document(text)
        raise CorpusError(f"no document {doc_id!r} in {root}")

    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        """Store an uploaded PDF and immediately report what Anchor reads in it.

        Inspecting in the same call is the point of the flow: the first useful
        question about a new document is "did it even parse", and answering it
        on a separate round trip invites a silent empty-text-layer failure to
        look like a successful upload.
        """
        doc_id = safe_doc_id(filename)
        with self._lock:
            save_pdf(self.corpus_root, doc_id, data)
        return self.inspect(doc_id)

    def inspect(self, doc_id: str) -> dict[str, Any]:
        """Extract and verify one document, with its label if it has one."""
        from anchor.cli import EXTRACTORS

        document = self._document(doc_id)
        return inspect_document(
            document,
            EXTRACTORS[self.extractor_name](),
            verify=self.verify,
            golden=load_golden(self.corpus_root, doc_id),
        )

    def save_label(self, doc_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate and write a hand label, then re-inspect.

        The record is validated by `GoldenRecord` rather than trusted, so a
        malformed field is a 400 here and never a corpus that will not load
        later. `doc_id` comes from the URL, not the body: a label whose filename
        and contents disagree about which document it describes is a corpus
        that scores the wrong thing.
        """
        body = dict(payload)
        body["doc_id"] = doc_id
        try:
            record = GoldenRecord.model_validate(body)
        except ValidationError as exc:
            raise CorpusError(f"not a valid label: {exc}") from exc

        with self._lock:
            save_golden(self.corpus_root, record)
        return self.inspect(doc_id)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

#: JSON bodies larger than this are refused unread. A label is a handful of
#: numbers and notes; nothing legitimate comes close.
_MAX_BODY = 1024 * 1024

#: Upload bodies are held to the corpus limit instead, and refused before the
#: body is read rather than after.
_MAX_UPLOAD = MAX_PDF_BYTES


def make_handler(app: AnchorApp) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to one `AnchorApp`."""

    class Handler(BaseHTTPRequestHandler):
        server_version = f"anchor/{__version__}"
        protocol_version = "HTTP/1.1"

        # -- plumbing --------------------------------------------------
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
            print(f"  {self.address_string()} {fmt % args}")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # A run artifact is borrower data. Nothing here should sit in a
            # shared cache, and the page is cheap to re-render.
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(
                status,
                json.dumps(payload).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _error(self, status: HTTPStatus, message: str) -> None:
            self._json({"error": message}, status)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            if length > _MAX_BODY:
                raise ValueError("request body too large")
            raw = self.rfile.read(length)
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("expected a JSON object")
            return parsed

        # -- routes ----------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                if path == "/":
                    self._send(HTTPStatus.OK, _index_bytes(), "text/html; charset=utf-8")
                elif path == "/api/config":
                    self._json(app.config())
                elif path == "/api/corpus":
                    self._json(app.corpus_summary())
                elif path == "/api/runs":
                    self._json({"runs": app.list_runs()})
                elif path.startswith("/api/runs/"):
                    run_id = path[len("/api/runs/") :]
                    data = app.get_run(run_id)
                    if data is None:
                        self._error(HTTPStatus.NOT_FOUND, f"no saved run {run_id!r}")
                    else:
                        self._json(data)
                elif path.startswith("/api/inspect/"):
                    self._json(app.inspect(_doc_id(path, "/api/inspect/")))
                elif path.startswith("/api/golden/"):
                    doc_id = _doc_id(path, "/api/golden/")
                    record = load_golden(app.corpus_root, doc_id)
                    if record is None:
                        self._error(HTTPStatus.NOT_FOUND, f"{doc_id!r} has no label yet")
                    else:
                        self._json(record.model_dump())
                else:
                    self._error(HTTPStatus.NOT_FOUND, f"no route {path!r}")
            except CorpusError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:  # pragma: no cover - defensive
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        do_HEAD = do_GET  # noqa: N815 - mirrors the GET routing table

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            try:
                if path == "/api/run":
                    body = self._read_json()
                    alpha = body.get("alpha")
                    self._json(
                        app.execute(
                            extractor_name=body.get("extractor"),
                            alpha=None if alpha is None else float(alpha),
                            save=bool(body.get("save", True)),
                        )
                    )
                elif path == "/api/upload":
                    self._json(app.upload(*self._read_upload(parsed.query)))
                else:
                    self._error(HTTPStatus.NOT_FOUND, f"no route {path!r}")
            except (CorpusError, ValueError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:  # pragma: no cover - defensive
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def do_PUT(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                if path.startswith("/api/golden/"):
                    doc_id = _doc_id(path, "/api/golden/")
                    self._json(app.save_label(doc_id, self._read_json()))
                else:
                    self._error(HTTPStatus.NOT_FOUND, f"no route {path!r}")
            except (CorpusError, ValueError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:  # pragma: no cover - defensive
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def _read_upload(self, query: str) -> tuple[str, bytes]:
            """Read an uploaded PDF: raw bytes in the body, name in the query.

            Raw rather than multipart/form-data on purpose. Both ends of this
            wire are in this repository, so the envelope buys nothing, and the
            stdlib lost its multipart parser when `cgi` was removed in 3.13 --
            leaving the choice between hand-rolling a parser for a format with a
            long history of parser bugs, or not needing one. The browser sends
            `fetch(url, {method: "POST", body: file})`.

            The length check happens before the read, so an oversized upload
            costs a header rather than a body.
            """
            filename = parse_qs(query).get("name", [""])[0]
            if not filename:
                raise ValueError("upload needs a ?name= filename")

            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise ValueError("empty upload")
            if length > _MAX_UPLOAD:
                raise ValueError(
                    f"upload is {length / 1e6:.1f} MB, over the "
                    f"{_MAX_UPLOAD / 1e6:.0f} MB limit"
                )
            return filename, self.rfile.read(length)

    return Handler


def _doc_id(path: str, prefix: str) -> str:
    """Pull a document id out of a URL path and refuse anything path-shaped.

    `safe_doc_id` already builds a clean id on upload, but an id also arrives
    here straight from a URL, and the two routes have to be equally careful.
    Everything downstream joins this to a corpus directory, so a segment
    containing a separator or a dot-dot is rejected outright rather than
    scrubbed -- scrubbing invites an argument about whether the scrub is
    complete.
    """
    doc_id = unquote(path[len(prefix) :])
    if not doc_id or "/" in doc_id or "\\" in doc_id or ".." in doc_id:
        raise CorpusError(f"not a document id: {doc_id!r}")
    return doc_id


def _index_bytes() -> bytes:
    """The single page, read per request so an edit shows up on refresh."""
    try:
        return _INDEX.read_bytes()
    except OSError as exc:  # pragma: no cover - broken install
        return (
            "<!doctype html><title>anchor</title>"
            f"<p>The interface template is missing: {exc}</p>"
        ).encode()


def make_server(app: AnchorApp, host: str = "127.0.0.1", port: int = 8756) -> ThreadingHTTPServer:
    """Build (but do not start) the HTTP server."""
    return ThreadingHTTPServer((host, port), make_handler(app))


def serve(
    corpus_root: Path | str,
    runs_dir: Path | str = Path("runs"),
    extractor_name: str = "heuristic",
    host: str = "127.0.0.1",
    port: int = 8756,
    alpha: float = DEFAULT_ALPHA,
    verify: bool = True,
    open_browser: bool = True,
) -> None:
    """Serve the interface until interrupted."""
    app = AnchorApp(
        corpus_root=corpus_root,
        runs_dir=runs_dir,
        extractor_name=extractor_name,
        alpha=alpha,
        verify=verify,
    )
    httpd = make_server(app, host=host, port=port)
    url = f"http://{host}:{port}/"

    print(f"anchor {__version__}  serving {url}")
    print(f"  corpus  {app.corpus_root}")
    print(f"  runs    {app.runs_dir}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"  WARNING: bound to {host}, not loopback. Run artifacts contain "
            "borrower financials and this server has no authentication."
        )
    print("  ctrl-c to stop")

    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
