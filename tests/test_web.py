"""Tests for the web interface.

The server is exercised over a real socket rather than by calling the handler
directly. Most of what could break here -- routing, status codes, the fact that
a run id from a client never becomes a filesystem path -- only exists at the
HTTP boundary.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http import HTTPStatus
from pathlib import Path

import pytest

from anchor.web import AnchorApp, make_server


@pytest.fixture
def server(corpus_dir: Path, tmp_path: Path) -> Iterator[str]:
    app = AnchorApp(corpus_root=corpus_dir, runs_dir=tmp_path / "runs")
    # Port 0 lets the OS pick a free one, so tests never collide with a
    # developer's own `anchor serve`.
    httpd = make_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def get(base: str, path: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(base + path) as response:
        return response.status, response.read()


def get_json(base: str, path: str):
    return json.loads(get(base, path)[1])


def post_json(base: str, path: str, payload: dict):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


class TestPage:
    def test_serves_the_interface(self, server: str) -> None:
        status, body = get(server, "/")
        assert status == HTTPStatus.OK
        assert b"<title>Anchor</title>" in body

    def test_the_page_loads_nothing_from_the_network(self, server: str) -> None:
        """Offline by construction, and no request that could leak a run elsewhere.

        Checked as "no element fetches a remote resource" rather than "the
        string http:// never appears": the SVG namespace URI is spelled like a
        URL but is an identifier and is never fetched.
        """
        body = get(server, "/")[1].decode("utf-8")
        for pattern in ('src="http', "src='http", 'href="http', "href='http",
                        "@import", "cdn.", "googleapis"):
            assert pattern not in body, f"page references {pattern}"

    def test_artifacts_are_not_cached(self, server: str) -> None:
        with urllib.request.urlopen(server + "/api/config") as response:
            assert response.headers["Cache-Control"] == "no-store"


class TestApi:
    def test_config(self, server: str) -> None:
        config = get_json(server, "/api/config")
        assert "heuristic" in config["extractors"]
        assert config["verify"] is True

    def test_corpus_summary(self, server: str) -> None:
        summary = get_json(server, "/api/corpus")
        assert summary["n_documents"] == 2
        assert summary["missing"] == []

    def test_runs_is_empty_before_anything_has_run(self, server: str) -> None:
        assert get_json(server, "/api/runs") == {"runs": []}

    def test_post_run_scores_and_saves(self, server: str) -> None:
        run = post_json(server, "/api/run", {})
        assert run["report"]["n_documents"] == 2
        assert run["extractor"] == "heuristic"

        listed = get_json(server, "/api/runs")["runs"]
        assert [r["run_id"] for r in listed] == [run["run_id"]]

    def test_a_saved_run_can_be_fetched_by_id(self, server: str) -> None:
        run = post_json(server, "/api/run", {})
        assert get_json(server, "/api/runs/" + run["run_id"]) == run

    def test_run_carries_evidence_for_the_audit_view(self, server: str) -> None:
        run = post_json(server, "/api/run", {})
        fields = run["documents"][0]["fields"]
        revenue = next(f for f in fields if f["name"] == "revenue_ltm")
        assert revenue["quote"] and revenue["page"] == 1

    def test_run_carries_the_risk_coverage_curve(self, server: str) -> None:
        run = post_json(server, "/api/run", {})
        curve = run["selective"]["curve"]
        assert len(curve["coverage"]) == len(curve["risk"])
        assert curve["coverage"] == sorted(curve["coverage"])

    def test_save_false_leaves_the_runs_directory_alone(self, server: str) -> None:
        post_json(server, "/api/run", {"save": False})
        assert get_json(server, "/api/runs") == {"runs": []}

    def test_alpha_is_honoured(self, server: str) -> None:
        loose = post_json(server, "/api/run", {"alpha": 0.5, "save": False})
        assert loose["selective"]["alpha"] == 0.5


class TestErrors:
    def test_unknown_route_is_404(self, server: str) -> None:
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(server, "/api/nope")
        assert exc.value.code == HTTPStatus.NOT_FOUND

    def test_unknown_run_id_is_404(self, server: str) -> None:
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(server, "/api/runs/does-not-exist")
        assert exc.value.code == HTTPStatus.NOT_FOUND

    @pytest.mark.parametrize(
        "attempt",
        [
            "/api/runs/..%2f..%2fpyproject",
            "/api/runs/%2Fetc%2Fpasswd",
            "/api/runs/....//....//setup",
            "/pyproject.toml",
        ],
    )
    def test_a_run_id_never_becomes_a_filesystem_path(self, server: str, attempt: str) -> None:
        """Ids are matched against the directory listing, never joined to it."""
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(server, attempt)
        assert exc.value.code == HTTPStatus.NOT_FOUND

    def test_an_unknown_extractor_is_a_400_not_a_500(self, server: str) -> None:
        with pytest.raises(urllib.error.HTTPError) as exc:
            post_json(server, "/api/run", {"extractor": "gpt-9"})
        assert exc.value.code == HTTPStatus.BAD_REQUEST
        assert "unknown extractor" in json.loads(exc.value.read())["error"]

    def test_post_to_the_wrong_route_is_404(self, server: str) -> None:
        with pytest.raises(urllib.error.HTTPError) as exc:
            post_json(server, "/api/config", {})
        assert exc.value.code == HTTPStatus.NOT_FOUND

    def test_a_corpus_that_cannot_be_read_is_a_400(
        self, corpus_dir: Path, tmp_path: Path, server: str
    ) -> None:
        for path in (corpus_dir / "text").glob("*.json"):
            path.unlink()
        with pytest.raises(urllib.error.HTTPError) as exc:
            post_json(server, "/api/run", {})
        assert exc.value.code == HTTPStatus.BAD_REQUEST
        assert "Nothing to score" in json.loads(exc.value.read())["error"]


class TestAnchorApp:
    def test_an_unreadable_artifact_is_listed_with_its_error(
        self, corpus_dir: Path, tmp_path: Path
    ) -> None:
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "broken.json").write_text("{ not json", encoding="utf-8")

        app = AnchorApp(corpus_root=corpus_dir, runs_dir=runs)
        listed = app.list_runs()
        assert len(listed) == 1
        assert listed[0]["run_id"] == "broken"
        assert "error" in listed[0]

    def test_runs_are_listed_newest_first(self, corpus_dir: Path, tmp_path: Path) -> None:
        app = AnchorApp(corpus_root=corpus_dir, runs_dir=tmp_path / "runs")
        app.execute()
        app.execute()
        listed = app.list_runs()
        created = [r["created_at"] for r in listed]
        assert created == sorted(created, reverse=True)
