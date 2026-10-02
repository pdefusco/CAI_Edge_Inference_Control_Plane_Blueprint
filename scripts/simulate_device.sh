#!/usr/bin/env bash
#
# Run the real `keeper` agent against the local control plane, with the mock
# runtime standing in for onnxruntime.
#
#   scripts/simulate_device.sh [device_id]
#
# This is not a mock client: it is the same binary, the same reconciler and the
# same resumable download that will run on the Jetson. Only ModelRuntime is
# swapped (KEEPER_RUNTIME=mock), which is the point -- a simulator that
# reimplemented the agent would prove nothing about the agent.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

device_id="${1:-$DEVICE_ID}"
# Consume the positional so any remaining arguments (e.g. --once) pass through to
# keeper itself rather than being mistaken for one.
(($# > 0)) && shift
token_file="$(device_token_file "$device_id")"

if [[ ! -s "$token_file" ]]; then
  say "no token for $device_id yet -- enrolling it"
  "$REPO_ROOT/scripts/register_device.sh" "$device_id" >/dev/null
  [[ -s "$token_file" ]] || die "enrolment produced no token file at $token_file"
fi

KEEPER="$(resolve_bin keeper)"
data_dir="${KEEPER_DATA_DIR:-$DEV_DIR/keeper/$device_id}"
mkdir -p "$data_dir"

export KEEPER_DEVICE_ID="$device_id"
export KEEPER_CONTROL_PLANE_URL="$BASE_URL"
export KEEPER_TOKEN_FILE="$token_file"
export KEEPER_DATA_DIR="$data_dir"
export KEEPER_RUNTIME="${KEEPER_RUNTIME:-mock}"
# Faster than the Jetson default so a demo converges while someone is watching.
export KEEPER_POLL_INTERVAL="${KEEPER_POLL_INTERVAL:-3}"
export KEEPER_LOG_LEVEL="${KEEPER_LOG_LEVEL:-INFO}"

say "keeper: device=$device_id runtime=$KEEPER_RUNTIME data=$data_dir"
exec "$KEEPER" "$@"
