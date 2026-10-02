#!/bin/bash
# Pre-flight for the demo. Run it before the call; it fixes what it can.
cd "$(dirname "$0")" || exit 1
URL="https://green-poster-desert-herein.trycloudflare.com"
ok=1

say() { printf "  %-26s %s\n" "$1" "$2"; }

echo "── demo pre-flight ──"

if ! pgrep -f "uvicorn app.server" >/dev/null; then
  say "server" "down — restarting"
  nohup .venv/bin/python -m uvicorn app.server:app --port 8000 >/tmp/demo-server.log 2>&1 &
  for _ in $(seq 1 30); do curl -s -m 2 localhost:8000/health >/dev/null 2>&1 && break; sleep 1; done
fi
curl -s -m 5 localhost:8000/health >/dev/null 2>&1 && say "server" "up" || { say "server" "FAILED — see /tmp/demo-server.log"; ok=0; }

if pgrep -f "cloudflared tunnel" >/dev/null; then
  say "tunnel process" "up"
else
  say "tunnel process" "DOWN — the emailed link is dead"
  echo "     restart with: cloudflared tunnel --url http://localhost:8000"
  echo "     the URL WILL CHANGE, so email him the new one"
  ok=0
fi

code=$(curl -s -m 15 -o /dev/null -w "%{http_code}" "$URL/health")
[ "$code" = "200" ] && say "public link" "200 — he can reach it" || { say "public link" "$code — BROKEN"; ok=0; }

pgrep -f caffeinate >/dev/null && say "sleep guard" "on" || {
  say "sleep guard" "off — turning on"; nohup caffeinate -i -s >/dev/null 2>&1 &
}

ans=$(curl -s -m 30 -X POST localhost:8000/text -H 'Content-Type: application/json' \
  -d '{"call_id":"preflight","message":"Tell me about one of his projects"}' \
  | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['answer'][:60])" 2>/dev/null)
[ -n "$ans" ] && say "end-to-end answer" "\"$ans...\"" || { say "end-to-end answer" "FAILED"; ok=0; }

echo
[ "$ok" = "1" ] && echo "  READY. Open $URL" || echo "  NOT READY — fix the lines above."
