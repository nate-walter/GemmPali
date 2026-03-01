#!/usr/bin/env bash
set -euo pipefail

cd /Users/jerrry/clawd/GemmPali/colqwen-dashboard

if [ ! -d node_modules ]; then
  npm install
fi

PORT=${PORT:-3473}
nohup env PORT="$PORT" npm start > /tmp/colqwen_dashboard.log 2>&1 &
echo $! > /tmp/colqwen_dashboard.pid

echo "ColQwen dashboard started"
echo "PID: $(cat /tmp/colqwen_dashboard.pid)"
echo "URL: http://127.0.0.1:${PORT}"
echo "LAN: http://10.46.150.108:${PORT}"
