"""OpenAI-compatible DeepSeek stub used only for local integration tests."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages", [])
        question = messages[-1].get("content", "") if messages else ""
        data = {"choices": [{"message": {"role": "assistant", "content": f"[Mock DeepSeek] 已收到问题：{question[:300]}"}}]}
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_):
        pass


ThreadingHTTPServer(("127.0.0.1", 9100), Handler).serve_forever()
