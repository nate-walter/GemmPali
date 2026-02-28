#!/usr/bin/env bash
set -euo pipefail

if [ -f /tmp/gemmpali_dashboard.pid ]; then
  PID=$(cat /tmp/gemmpali_dashboard.pid)
  kill "$PID" 2>/dev/null || true
  rm -f /tmp/gemmpali_dashboard.pid
fi

pkill -f "node server.js" 2>/dev/null || true

echo "GemmPali dashboard stopped"
