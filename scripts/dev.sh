#!/usr/bin/env bash
#
# The whole M1 demo in one process group: control plane + fake registry +
# dashboard + one simulated Jetson running the real agent.
#
#   scripts/dev.sh              # start everything, follow both logs
#   DEVICE_ID=bench-02 scripts/dev.sh
#
# Ctrl-C stops both processes. State lives under .dev/ and survives restarts, so
# a device stays enrolled and a deployment stays deployed across runs -- which is
# also how the reboot-convergence behaviour gets exercised by accident.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

LIGHTHOUSE="$(resolve_bin lighthouse)"
ensure_dev_dir

plane_log="$DEV_DIR/lighthouse.log"
agent_log="$DEV_DIR/keeper-$DEVICE_ID.log"
plane_pid=""
agent_pid=""
tail_pid=""

cleanup() {
  trap - INT TERM EXIT
  printf '\n'
  for pid in "$tail_pid" "$agent_pid" "$plane_pid"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  # Give uvicorn a moment to close its socket; a half-second here prevents the
  # "address already in use" that makes the next `make dev` look broken.
  wait "$agent_pid" "$plane_pid" 2>/dev/null || true
  say "stopped. logs kept in $DEV_DIR"
}
trap cleanup INT TERM EXIT

if curl -fsS "$API/health" >/dev/null 2>&1; then
  die "something is already serving $BASE_URL -- stop it, or set LIGHTHOUSE_PORT"
fi

# -- control plane ----------------------------------------------------------

token="$(admin_token)"

say "starting the control plane on $BASE_URL"
LIGHTHOUSE_ENV=local \
  LIGHTHOUSE_DATA_DIR="$DEV_DIR/lighthouse" \
  LIGHTHOUSE_REGISTRY="${LIGHTHOUSE_REGISTRY:-fake}" \
  LIGHTHOUSE_ADMIN_TOKEN="$token" \
  LIGHTHOUSE_HEARTBEAT_INTERVAL="${LIGHTHOUSE_HEARTBEAT_INTERVAL:-3}" \
  PORT="$LIGHTHOUSE_PORT" \
  "$LIGHTHOUSE" >"$plane_log" 2>&1 &
plane_pid=$!

if ! wait_for_health "$plane_pid"; then
  printf '\n'
  tail -n 30 "$plane_log" >&2
  die "the control plane did not come up -- see $plane_log"
fi

# -- device -----------------------------------------------------------------

"$REPO_ROOT/scripts/register_device.sh" "$DEVICE_ID" >/dev/null
say "enrolled $DEVICE_ID"

"$REPO_ROOT/scripts/simulate_device.sh" "$DEVICE_ID" >"$agent_log" 2>&1 &
agent_pid=$!

# -- the recipe -------------------------------------------------------------
#
# Printed rather than executed. The point of the demo is watching desired state
# and actual state converge, and that only lands if a human drives it.

versions="$(body_of "$(api_call GET "/models/$MODEL_NAME/versions")")"
version="$(printf '%s' "$versions" | json_get 0 version 2>/dev/null || printf '1')"

cat <<EOF

  ──────────────────────────────────────────────────────────────────────────
  Lighthouse is up.

    dashboard     $BASE_URL
    admin token   $token
    device        $DEVICE_ID   (real keeper, mock runtime, polling every ${KEEPER_POLL_INTERVAL:-3}s)
    registry      ${LIGHTHOUSE_REGISTRY:-fake}  ($MODEL_NAME, version $version)
    logs          $plane_log
                  $agent_log

  Sign in to the dashboard with the admin token above, then drive the loop from
  another terminal and watch the Desired and Actual columns disagree, then agree:

    make deploy            # → Desired RUNNING, Actual DOWNLOADING → RUNNING
    make stop              # → STOP_PENDING while Actual is still RUNNING
    make deploy            # → RUNNING again, no download (artifacts were kept)
    make revoke            # → REVOKE_PENDING → REVOKED, local bytes deleted
    make events            # the audit trail behind all of it

  Or with curl, which is all the dashboard itself does:

    curl -X PUT $API/devices/$DEVICE_ID/deployment \\
      -H "X-Lighthouse-Admin-Token: \$(cat $ADMIN_TOKEN_FILE)" \\
      -H 'Content-Type: application/json' \\
      -d '{"model_name":"$MODEL_NAME","model_version":"$version","desired_state":"RUNNING"}'

  Kill this script and the device goes STALE, then OFFLINE, from heartbeat age
  alone. Start it again and it converges with no manual step.
  ──────────────────────────────────────────────────────────────────────────

EOF

# Both logs, prefixed, until Ctrl-C.
tail -n 0 -f "$plane_log" "$agent_log" 2>/dev/null |
  awk '/^==> .*keeper-/ {tag="[keeper] "; next}
       /^==> / {tag="[plane ] "; next}
       NF {print tag $0; fflush()}' &
tail_pid=$!

wait "$plane_pid" "$agent_pid" 2>/dev/null || true
