#!/usr/bin/env sh
set -e

PORT="${PLAYWRIGHT_SERVER_PORT:-3000}"

# start Playwright server in background
npx -y playwright@1.55.0 run-server --port "$PORT" --host 0.0.0.0 &
PW_PID=$!

# wait until healthy (max ~60s)
i=0
until curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; do
  i=$((i+1))
  [ "$i" -ge 60 ] && echo "Playwright server failed to start" && exit 1
  sleep 1
done

# run your app (connects via ws://127.0.0.1:$PORT)
exec sh -lc "$APP_CMD"
