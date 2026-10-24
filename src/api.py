"""Bonus 5: Simple Local HTTP API to query index and answer questions."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from src.answerer import generate_answer
from src.indexer import load_index
from src.retriever import build_bm25, search


class RAGRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler exposing /search and /answer."""

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        """Helper to send JSON response.

        Args:
            data: Dictionary payload to serialize.
            status: HTTP status code.
        """
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Handle incoming GET requests."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        query = params.get("q", [""])[0]
        k = int(params.get("k", [10])[0])

        if parsed.path == "/search":
            try:
                data = load_index()
                bm25 = build_bm25(data["tokenized"])
                results = search(query, data["chunks"], bm25, k=k)
                self._send_json(
                    {"results": [r.model_dump() for r in results], "k": k}
                )
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif parsed.path == "/answer":
            try:
                data = load_index()
                bm25 = build_bm25(data["tokenized"])
                sources = search(query, data["chunks"], bm25, k=k)
                ans = generate_answer(query, sources)
                self._send_json(
                    {"answer": ans, "sources": [s.model_dump() for s in sources]}
                )
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
        else:
            self._send_json(
                {"message": "RAG API running. Available endpoints: /search, /answer"}
            )

    def log_message(self, format: str, *args: object) -> None:
        """Silence standard console logging."""
        pass


def run_server(port: int = 8000) -> None:
    """Start local HTTP server.

    Args:
        port: TCP port to listen on.
    """
    server = HTTPServer(("127.0.0.1", port), RAGRequestHandler)
    print(f"RAG HTTP API server running on http://127.0.0.1:{port}")
    print(f"Test query: curl 'http://127.0.0.1:{port}/search?q=lora&k=3'")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()
