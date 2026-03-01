#!/usr/bin/env bash
set -euo pipefail

if [ -f /tmp/colqwen_dashboard.pid ]; then
  PID=$(cat /tmp/colqwen_dashboard.pid)
  kill "$PID" 2>/dev/null || true
  rm -f /tmp/colqwen_dashboard.pid
fi

pkill -f "/Users/jerrry/clawd/GemmPali/colqwen-dashboard/server.js" 2>/dev/null || true
pkill -f "npm start" 2>/dev/null || true

echo "ColQwen dashboard stopped"
