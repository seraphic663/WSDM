#!/usr/bin/env python3
"""Server-only integration smoke for the paper pair builder.

Uses the real M00 tokenizer and builder CLI, but a deterministic localhost
OpenAI-compatible judge and tiny synthetic corpus/trajectories. No GPU,
external data, model download, or persistent artifact is used.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class JudgeHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        prompt = request["messages"][0]["content"]
        verdict = "NOT_RELEVANT" if "clearly irrelevant marker" in prompt else "RELEVANT"
        payload = json.dumps(
            {"choices": [{"message": {"content": verdict}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args) -> None:
        return


def search(query: str, *doc_ids: str) -> dict:
    return {
        "type": "tool_call",
        "tool_name": "search",
        "arguments": json.dumps({"query": [query]}),
        "output": "\n".join(f"DocID: {doc_id}" for doc_id in doc_ids),
    }


def browse(doc_id: str) -> dict:
    return {
        "type": "tool_call",
        "tool_name": "get_document",
        "arguments": json.dumps({"docid": doc_id}),
    }


def main() -> None:
    repo = Path("/root/data/LRAT")
    model = repo / "ccir/models/Qwen3-Embedding-0.6B"
    with tempfile.TemporaryDirectory(prefix="paper-flywheel-pairs-smoke-") as temporary:
        root = Path(temporary)
        corpus = root / "corpus.jsonl"
        corpus.write_text(
            "".join(
                json.dumps({"docid": f"d{i}", "text": f"document {i}"}) + "\n"
                for i in range(1, 7)
            )
        )
        trajectories = root / "trajectories"
        trajectories.mkdir()
        values = [
            {
                "query_id": "0",
                "result": [
                    search("query zero", "d1", "d2", "d3"),
                    browse("d2"),
                    {"type": "reasoning", "output": "useful evidence"},
                    browse("d3"),
                    {
                        "type": "reasoning",
                        "output": "clearly irrelevant marker",
                    },
                ],
            },
            {
                "query_id": "1",
                "result": [
                    search("query one", "d4", "d5", "d6"),
                    browse("d4"),
                    {
                        "type": "reasoning",
                        "output": "a longer useful reasoning trace with evidence",
                    },
                ],
            },
        ]
        for index, value in enumerate(values):
            (trajectories / f"run_{index}.json").write_text(
                json.dumps(value)
            )

        server = ThreadingHTTPServer(("127.0.0.1", 0), JudgeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            output = root / "pairs.jsonl"
            summary = root / "summary.json"
            command = [
                str(repo / ".venv/bin/python"),
                str(repo / "solution/src/build_paper_flywheel_pairs.py"),
                "--corpus-path",
                str(corpus),
                "--traj-dir",
                str(trajectories),
                "--output-path",
                str(output),
                "--summary-path",
                str(summary),
                "--tokenizer-path",
                str(model),
                "--judge-api-url",
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                "--judge-model",
                "local-smoke-judge",
                "--max-workers",
                "2",
            ]
            completed = subprocess.run(
                command,
                cwd=repo,
                check=True,
                text=True,
                capture_output=True,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        rows = [
            json.loads(line)
            for line in output.read_text().splitlines()
            if line.strip()
        ]
        value = json.loads(summary.read_text())
        assert len(rows) == 2, rows
        assert value["browse_candidates"] == 3, value
        assert value["judge_relevant"] == 2, value
        assert value["judge_irrelevant"] == 1, value
        assert [row["pos_id"] for row in rows] == [["d2"], ["d4"]]
        assert rows[0]["neg_id"] == ["d1", "d3"]
        assert rows[1]["neg_id"] == ["d5", "d6"]
        assert all(row["satisfied"] is True for row in rows)
        assert abs(
            statistics.fmean(row["reweight_rate"] for row in rows) - 1.0
        ) < 1e-12
        print(
            json.dumps(
                {
                    "passed": True,
                    "builder_stdout": completed.stdout,
                    "retained_pairs": len(rows),
                    "summary": value,
                    "temporary_artifacts_removed_on_exit": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
