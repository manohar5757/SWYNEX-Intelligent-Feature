"""Simple web demo (standard library only).  python app.py  ->  http://localhost:8000
Set ANTHROPIC_API_KEY and run with --llm to use the real model; otherwise uses rules."""
import html, sys, os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
from triage import triage, make_anthropic_llm

LLM = None
if "--llm" in sys.argv and os.getenv("ANTHROPIC_API_KEY"):
    LLM = make_anthropic_llm()

PAGE = """<!doctype html><meta charset=utf-8><title>Ticket Triage</title>
<style>body{{font-family:system-ui;max-width:640px;margin:2rem auto;padding:0 1rem}}
textarea{{width:100%;height:120px}}.card{{border:1px solid #ccc;border-radius:8px;padding:1rem;margin-top:1rem}}
.warn{{color:#b45309}}.err{{color:#b91c1c}}</style>
<h2>Smart Ticket Triage</h2>
<form method=post><textarea name=t placeholder="Paste a support ticket...">{t}</textarea>
<br><button>Triage</button></form>{out}"""

def render(r):
    if r.source == "rejected":
        return f'<div class=card><b class=err>Rejected:</b> {html.escape(r.error)}</div>'
    w = "".join(f"<li class=warn>{html.escape(x)}</li>" for x in r.warnings)
    flag = " &#9888; needs human review" if r.needs_human else ""
    return (f"<div class=card><b>Category:</b> {r.category} &nbsp; <b>Priority:</b> {r.priority}<br>"
            f"<b>Summary:</b> {html.escape(r.summary)}<br><b>Confidence:</b> {r.confidence} "
            f"<small>(source: {r.source}){flag}</small><ul>{w}</ul></div>")

class H(BaseHTTPRequestHandler):
    def _send(self, body):
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
        self.wfile.write(body.encode())
    def do_GET(self): self._send(PAGE.format(t="", out=""))
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        t = parse_qs(self.rfile.read(n).decode()).get("t", [""])[0]
        self._send(PAGE.format(t=html.escape(t), out=render(triage(t, llm=LLM))))

if __name__ == "__main__":
    print("Serving on http://localhost:8000"); HTTPServer(("", 8000), H).serve_forever()
