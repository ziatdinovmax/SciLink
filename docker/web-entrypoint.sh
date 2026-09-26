#!/bin/sh
# Entrypoint of the `web` image: one container per workspace.
#
# /workspace is a mounted volume (an EFS access point, a bind mount), and a
# fresh one is EMPTY: whatever the image created under it is hidden by the
# mount. So the layout the server expects is made here, at start, and
# scilink-web then gets the session root that now exists. Extra arguments
# are passed through (e.g. --auth-header behind a proxy).
set -e
mkdir -p /workspace/sessions /workspace/home /workspace/data
exec scilink-web --host 0.0.0.0 --port 8422 --session-root /workspace/sessions --no-open "$@"
