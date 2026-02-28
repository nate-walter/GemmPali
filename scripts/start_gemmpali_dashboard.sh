#!/usr/bin/env bash
set -euo pipefail

cd /Users/jerrry/clawd/GemmPali/dashboard

if [ ! -d node_modules ]; then
  npm install
fi

PORT=${PORT:-3472}
nohup env PORT="$PORT" npm start > /tmp/gemmpali_dashboard.log 2>&1 &
echo $! > /tmp/gemmpali_dashboard.pid

echo "GemmPali dashboard started"
echo "PID: $(cat /tmp/gemmpali_dashboard.pid)"
echo "URL: http://127.0.0.1:${PORT}"
echo "LAN: http://10.46.150.108:${PORT}"
