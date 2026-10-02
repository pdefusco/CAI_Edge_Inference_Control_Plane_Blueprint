#!/usr/bin/env bash
#
# Enrol a device and store its bearer token.
#
#   scripts/register_device.sh [device_id] [display_name] [platform]
#
# Idempotent on purpose: re-running it against an already-enrolled device mints an
# *additional* token rather than failing, because that is also the rotation
# procedure (issue, install on the device, revoke the old one) and the demo should
# not need a second script for it.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

device_id="${1:-$DEVICE_ID}"
display_name="${2:-Simulated Jetson}"
platform="${3:-jetson-orin}"

ensure_dev_dir

payload="$("$PYTHON" -c '
import json, sys
print(json.dumps({
    "device_id": sys.argv[1],
    "display_name": sys.argv[2] or None,
    "platform": sys.argv[3] or None,
}))
' "$device_id" "$display_name" "$platform")"

response="$(api_call POST /devices "$payload")"

if [[ "$(status_of "$response")" == "409" ]]; then
  say "$device_id is already enrolled -- issuing an additional token"
  response="$(api_call POST "/devices/$device_id/tokens" '{"reason":"scripts/register_device.sh"}')"
  require_ok "$(status_of "$response")" "could not issue a token for $device_id"
  body="$(body_of "$response")"
  token="$(printf '%s' "$body" | json_get token)"
  token_id="$(printf '%s' "$body" | json_get token_id)"
else
  require_ok "$(status_of "$response")" "could not enrol $device_id"
  body="$(body_of "$response")"
  token="$(printf '%s' "$body" | json_get credentials token)"
  token_id="$(printf '%s' "$body" | json_get credentials token_id)"
fi

[[ -n "$token" ]] || die "the control plane returned no token"

token_file="$(device_token_file "$device_id")"
printf '%s' "$token" >"$token_file"
chmod 600 "$token_file"

cat <<EOF

  device    $device_id ($platform)
  token_id  $token_id
  token     $token
  saved to  $token_file

  The server stored only sha256(secret); this is the one time the token is
  printed. On the real Jetson, install it as KEEPER_TOKEN_FILE rather than
  KEEPER_TOKEN so it never appears in the unit file or in \`ps\`.

EOF
