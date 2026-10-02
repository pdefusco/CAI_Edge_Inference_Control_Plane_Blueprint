# Shared shell plumbing for the dev scripts. Sourced, never executed.
#
# One decision worth stating: these scripts *mint* the admin token and write it to
# .dev/admin-token rather than scraping the ephemeral one the control plane prints
# at startup. The printed token exists so a bare `lighthouse` is never silently
# open; a harness that has to parse stdout to talk to its own API is a harness
# that breaks the first time a log line moves.

# shellcheck shell=bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV_DIR="${LIGHTHOUSE_DEV_DIR:-$REPO_ROOT/.dev}"
TOKEN_DIR="$DEV_DIR/tokens"
ADMIN_TOKEN_FILE="$DEV_DIR/admin-token"

LIGHTHOUSE_PORT="${LIGHTHOUSE_PORT:-8000}"
BASE_URL="${LIGHTHOUSE_URL:-http://127.0.0.1:$LIGHTHOUSE_PORT}"
API="$BASE_URL/api/v1"

DEVICE_ID="${DEVICE_ID:-jetson-sim-01}"
MODEL_NAME="${MODEL_NAME:-fashion-cnn}"

die() {
  printf '\n  error: %s\n\n' "$*" >&2
  exit 1
}

say() { printf '  %s\n' "$*"; }

# Prefer the repo virtualenv over whatever is on PATH. An editable install in
# .venv is the documented setup, and silently running a different interpreter's
# `lighthouse` is a confusing way to spend an afternoon.
resolve_bin() {
  local name="$1"
  if [[ -x "$REPO_ROOT/.venv/bin/$name" ]]; then
    printf '%s' "$REPO_ROOT/.venv/bin/$name"
  elif command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
  else
    die "'$name' not found. Create the venv and install both packages:
    python3 -m venv .venv && .venv/bin/pip install -e contracts -e control-plane -e edge-agent"
  fi
}

PYTHON="$(resolve_bin python)"

# Extract one value from a JSON document on stdin. Python rather than jq: jq is
# not installable on every box this demo might land on, and the interpreter is
# already a hard dependency.
json_get() {
  "$PYTHON" -c '
import json, sys
doc = json.load(sys.stdin)
for key in sys.argv[1:]:
    doc = doc[int(key)] if isinstance(doc, list) else doc[key]
print("" if doc is None else doc)
' "$@"
}

ensure_dev_dir() {
  mkdir -p "$TOKEN_DIR"
  chmod 700 "$DEV_DIR" "$TOKEN_DIR"
}

# The admin token: honour the environment first (so `make dev` can be pointed at
# a fixed token), else reuse the one from a previous run, else mint one. Reuse
# matters -- regenerating it on every run would invalidate the browser session
# the operator just signed in with.
admin_token() {
  if [[ -n "${LIGHTHOUSE_ADMIN_TOKEN:-}" ]]; then
    printf '%s' "$LIGHTHOUSE_ADMIN_TOKEN"
    return
  fi
  ensure_dev_dir
  if [[ ! -s "$ADMIN_TOKEN_FILE" ]]; then
    "$PYTHON" -c 'import secrets; print("lha_" + secrets.token_urlsafe(24))' >"$ADMIN_TOKEN_FILE"
    chmod 600 "$ADMIN_TOKEN_FILE"
  fi
  tr -d '\n' <"$ADMIN_TOKEN_FILE"
}

device_token_file() { printf '%s/%s.token' "$TOKEN_DIR" "$1"; }

# curl as an operator, returning "<status>\n<body>".
#
# The status rides in the output rather than in a global, because every caller
# wraps this in `$(...)` -- and a global assigned inside a command substitution is
# assigned in a subshell and silently lost. Branching on 409 (already enrolled)
# only works if the status survives the call.
api_call() {
  local method="$1" path="$2" body="${3:-}"
  local args=(-sS -X "$method" "$API$path"
    -H "X-Lighthouse-Admin-Token: $(admin_token)"
    -w '\n%{http_code}')
  if [[ -n "$body" ]]; then
    args+=(-H 'Content-Type: application/json' -d "$body")
  fi
  local response
  response="$(curl "${args[@]}")" || die "cannot reach $API$path -- is the control plane running? (scripts/dev.sh)"
  # curl appended the status on its own line; move it to the front so the body,
  # which may itself be multi-line JSON, stays intact behind it.
  printf '%s\n%s' "${response##*$'\n'}" "${response%$'\n'*}"
}

status_of() { printf '%s' "${1%%$'\n'*}"; }
body_of() { printf '%s' "${1#*$'\n'}"; }

require_ok() {
  local status="$1" what="$2"
  case "$status" in
    2*) return 0 ;;
    401 | 403) die "operator credential rejected (HTTP $status). Stale $ADMIN_TOKEN_FILE? Delete it, then restart the control plane so it loads the new one." ;;
    *) die "HTTP $status from the control plane: $what" ;;
  esac
}

wait_for_health() {
  local pid="${1:-}" tries="${2:-80}"
  for ((i = 0; i < tries; i++)); do
    if curl -fsS "$API/health" >/dev/null 2>&1; then
      return 0
    fi
    # Fail fast if the process is already gone: waiting out the full timeout on a
    # port conflict teaches the operator nothing.
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    sleep 0.25
  done
  return 1
}
