#!/usr/bin/env bash
# Build the React web UI and place the bundle where the wheel and the server
# expect it (scilink/server/static/). Run before `python -m build` for a
# release, or once after cloning to use `scilink-web` from a checkout.
#
# Requires Node 20+ and npm. Deterministic in CI: `npm ci` installs exactly
# webui/package-lock.json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/webui"

if ! command -v npm >/dev/null 2>&1; then
  echo "build_webui: npm not found — install Node.js 20+ (https://nodejs.org)" >&2
  exit 1
fi

if [ -f package-lock.json ]; then
  npm ci --silent
else
  npm install --silent
fi
npm run build:package

echo "build_webui: bundle written to $ROOT/scilink/server/static/"
