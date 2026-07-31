#!/usr/bin/env bash
# Rebuild the site and (re)start the production servers on ports 3000 (site) and
# 3001 (auth API). Build runs in the foreground so errors surface; servers are
# launched in a new session (setsid) so they keep running after this script exits.
set -euo pipefail
cd "$(dirname "$0")"

# Group-writable so any team member can publish over another member's build.
umask 002
mkdir -p .run

# The workspace starts as sources only (the coming-soon placeholder serves from
# the image's pre-built copy), so the first publish installs deps here. No-op
# once node_modules is current.
bun install
bun run build

# Start the auth API server on port 3001
setsid nohup bun run src/auth-server.ts > .run/auth-server.log 2>&1 < /dev/null &

# Start the main site server on port 3000
setsid nohup bun run start > .run/server.log 2>&1 < /dev/null &

# Wait for the main server to actually answer before reporting success, so a
# startup crash surfaces here instead of silently leaving the old page live.
for _ in $(seq 1 50); do
  if curl -sf -o /dev/null http://localhost:3000; then
    echo "site published; serving on port 3000 (auth on 3001)"
    exit 0
  fi
  sleep 0.2
done
echo "warning: published, but the server isn't responding — check .run/server.log" >&2
exit 1
