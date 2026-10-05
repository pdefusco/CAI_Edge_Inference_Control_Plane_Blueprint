#!/usr/bin/env bash
#
# The Phase 6 acceptance check, as one command. Run on the device; reads only.
#
#   sudo scripts/accept.sh               # masks the control-plane host
#   sudo scripts/accept.sh --show-host   # prints it, for your own notes
#
# docs/jetson-setup.md §7 lists five things that must all be true and, until
# this, left confirming them to a human reading `journalctl` by eye. Four of the
# five are one grep each. The fifth -- "the reported execution provider is not
# CPU-only" -- is the one most easily read wrong, because the journal carries one
# provider list that is evidence and the installed build advertises another that
# is not.
#
# Why this runs here rather than against the API: `make fleet`'s ACCEL column is
# derived from what a *heartbeat* carried, so it reports what the device said.
# This reads what the device's own session did. The two agreeing is the point; if
# they disagree, the journal wins and the heartbeat path is the bug.
#
# It deliberately does NOT source scripts/_common.sh, for the reason
# install_keeper.sh gives at the same place: that file is the laptop harness and
# it mints admin tokens, and an admin token must never exist on a device. This
# script reads no credential at all -- not even the device token, which it has no
# use for.
#
# Three verdicts, not two:
#
#   PASS      the evidence is here.
#   FAIL      the evidence is here and it says no.
#   UNPROVEN  there is no evidence either way -- usually a journal that rotated
#             past the deployment, or a check that was skipped.
#
# UNPROVEN is separate for the same reason `Acceleration.UNKNOWN` is separate
# from `CPU_ONLY` in the contracts: "nothing was measured" and "the measurement
# says CPU" send you to different places. Neither is a pass, and the exit status
# does not distinguish them.

set -euo pipefail

UNIT=keeper
# Overridable only so this can be exercised against a fixture directory off the
# device, which is how its parsing was checked before it ever ran on a Jetson.
# On a device the defaults are the installer's paths and there is no reason to
# set any of these.
VENV="${VENV:-/opt/keeper/venv}"
ENV_FILE="${ENV_FILE:-/etc/keeper/keeper.env}"
STATE_FILE="${STATE_FILE:-/var/lib/keeper/state.json}"

die() {
  printf '\n  error: %s\n\n' "$*" >&2
  exit 1
}
say() { printf '  %s\n' "$*"; }

show_host=false
case "${1:-}" in
"") ;;
--show-host) show_host=true ;;
-h | --help)
  printf 'usage: sudo %s [--show-host]\n' "$0"
  exit 0
  ;;
*) die "unknown argument: $1 (the only option is --show-host)" ;;
esac

# The control-plane URL names a tenant exactly as the registry domain does, so it
# is masked by default -- the convention scripts/probe_app.py follows, and for
# the same reason: this output is meant to be pasteable.
mask() {
  if $show_host; then
    printf '%s' "$1"
  else
    printf '%s' "$1" | sed -E 's#^(https?://)[^/]+#\1<app-host>#'
  fi
}

passes=0
failures=0
verdict() { # verdict <PASS|FAIL|UNPROVEN> <label> [evidence lines...]
  local state=$1 label=$2 line
  shift 2
  printf '  %-9s %s\n' "$state" "$label"
  for line in "$@"; do
    [[ -n $line ]] && printf '            %s\n' "$line"
  done
  if [[ $state == PASS ]]; then
    passes=$((passes + 1))
  else
    failures=$((failures + 1))
  fi
  return 0
}

# == preflight ===============================================================
#
# None of this is scored. It is the context that makes a FAIL readable: a check
# that fails because the runtime is `mock` is a different conversation from one
# that fails on the wheel.

(($(id -u) == 0)) || die "run this with sudo -- the journal and $ENV_FILE are not world-readable"
command -v journalctl >/dev/null 2>&1 || die "no journalctl; this check reads the unit's journal"
systemctl cat "$UNIT" >/dev/null 2>&1 || die "no $UNIT unit installed -- docs/jetson-setup.md §3"
[[ -r $ENV_FILE ]] || die "$ENV_FILE is missing -- docs/jetson-setup.md §5"
[[ -x $VENV/bin/python ]] || die "$VENV/bin/python is missing; re-run scripts/install_keeper.sh"

# systemd's EnvironmentFile is not shell, so this reads values rather than
# sourcing them: no expansion, no substitution, last assignment wins.
env_value() {
  sed -nE "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*(.*)\$/\1/p" "$ENV_FILE" |
    tail -n 1 | sed -E "s/^\"(.*)\"\$/\1/; s/^'(.*)'\$/\1/"
}

device_id="$(env_value KEEPER_DEVICE_ID)"
cp_url="$(env_value KEEPER_CONTROL_PLANE_URL)"
configured_runtime="$(env_value KEEPER_RUNTIME)"

journal="$(journalctl -u "$UNIT" --no-pager -o cat 2>/dev/null || true)"
[[ -n $journal ]] || die "the journal for $UNIT is empty -- start the unit and let it converge"

last_match() { printf '%s\n' "$journal" | { grep -F -- "$1" || true; } | tail -n 1; }

# What the agent said about itself at its last start. Preferred over the env file
# for the runtime, because this is the value that actually took effect.
start_line="$(last_match 'keeper starting:')"
running_runtime="$(printf '%s' "$start_line" | sed -nE 's/.*runtime=([^ ]+).*/\1/p')"

# The agent's own record of what it has converged to. Read through the venv's
# python so this script needs nothing the device does not already have.
#
# Fields are separated by \x1f and not by a tab. A tab is an IFS *whitespace*
# character, so bash collapses runs of them: one empty field -- `message`, on
# every healthy device -- would shift every field after it by one, which is how
# the first run of this script reported generation 7 as a message. A
# non-whitespace separator keeps empty fields empty.
state="$("$VENV/bin/python" - "$STATE_FILE" <<'PY'
import json, sys

fields = ["", "", "UNREADABLE", "", "", ""]
try:
    data = json.load(open(sys.argv[1]))
    model = data.get("model") or {}
    fields = [
        str(model.get("name") or ""),
        str(model.get("version") or ""),
        str(data.get("actual_state") or "UNKNOWN"),
        str(data.get("message") or ""),
        # Not `or ""`: generation 0 is a real value -- the agent has converged to
        # nothing yet -- and `or` would turn it into "unknown".
        str(data.get("observed_generation", "?")),
        "yes" if data.get("inference_running") else "no",
    ]
except FileNotFoundError:
    fields[3] = "no state file yet -- the agent has never completed a pass"
except Exception as exc:  # a truncated file reads as "I know nothing", as in state.py
    fields[3] = str(exc)
print("\x1f".join(fields))
PY
)"
IFS=$'\x1f' read -r model_name model_version actual_state state_message generation inference_running <<<"$state"

printf '\n'
say "device    ${device_id:-<unset>}"
say "control   $(mask "${cp_url:-<unset>}")"
say "unit      $(systemctl is-active "$UNIT" || true), $(systemctl is-enabled "$UNIT" || true)"
say "runtime   ${running_runtime:-<no start line in the journal>} (configured: ${configured_runtime:-<unset, which defaults to mock>})"
say "state     ${actual_state:-UNKNOWN}, generation ${generation:-?}, inference_running=${inference_running:-?}"
[[ -n $state_message ]] && say "message   $state_message"
say "model     ${model_name:-<none>}${model_version:+/$model_version}"
printf '\n'

# Told apart deliberately. "No model" is an operator who has not deployed one
# yet; an unreadable state file is a device that cannot say what it is serving,
# and the acceptance check has no label to anchor its evidence to in either case
# -- but only one of them is fixed with `make deploy`.
[[ $actual_state != UNREADABLE ]] ||
  die "$STATE_FILE could not be read, so this device cannot say what it is serving. state.py treats a corrupt file as \"I know nothing\" and recovers on the next pass: sudo systemctl restart $UNIT, let it reconverge, then run this again."
[[ -n $model_name && -n $model_version ]] ||
  die "the agent has no deployed model, so there is nothing to accept. Deploy one from the control plane -- make deploy MODEL=<model> DEVICE=${device_id:-<device-id>} -- and let the device converge."

label="$model_name/$model_version"

# Every line below is matched against THIS label. A journal proving that a
# previous deployment loaded on the GPU proves nothing about the one being served
# now, and right-name-right-version-different-bytes is exactly the shape of
# mistake the digest discipline elsewhere in this repo exists to catch.
say "evidence for $label, from the journal of unit $UNIT:"
printf '\n'

if [[ $running_runtime == mock ]]; then
  # Stop here rather than scoring the five items. A mock runtime never writes a
  # provider or smoke line, so the only way those lines can be in this journal is
  # an earlier onnx run -- and reading them now would credit the current, fake
  # session with a previous, real one's evidence. That is the same stale-evidence
  # mistake as reading another version's lines, and it is the one a device left
  # on the default runtime would hit.
  verdict FAIL "the runtime is mock, so none of §7 can be answered here" \
    "the agent last started with runtime=mock; set KEEPER_RUNTIME=onnx in $ENV_FILE" \
    "a mock device reports a plausible heartbeat and never runs a model -- §5" \
    "any provider or smoke line in this journal is from an earlier onnx run"
  printf '\n'
  say "NOT ACCEPTED -- the device is not running the real runtime."
  printf '\n'
  exit 1
fi

# == 1. the artifact came from the Application ===============================
#
# The agent fetches only from KEEPER_CONTROL_PLANE_URL, so "not from a laptop" is
# two questions: did a verified download happen, and is that URL the Application
# rather than something on the bench next to the device.

downloaded="$(last_match "downloaded and verified $label")"
cached="$(last_match "archive for $label already downloaded and verified")"

# When nothing matches, the two reasons are worth telling apart: a journal that
# rotated past the deployment, and a journal whose evidence belongs to a version
# this device is no longer serving. The second reads as the first unless it is
# said out loud, and "there are provider lines right there" is how someone talks
# themselves into accepting a stale one. Labels are collected from the log line
# and compared as strings, so a model name is never interpolated into a regex.
stale=""
while read -r seen; do
  [[ -n $seen && $seen != "$label" && $seen == "$model_name/"* ]] && stale="${stale:+$stale, }$seen"
done < <(printf '%s\n' "$journal" | sed -nE 's#.*loaded ([^ ]+) with providers.*#\1#p' | sort -u)
if [[ -n $stale ]]; then
  stale="the journal does have evidence for $stale -- a previous deployment, and not evidence for $label"
fi
host="$(printf '%s' "$cp_url" | sed -E 's#^https?://##; s#[:/].*$##')"
case "$host" in
"") origin="KEEPER_CONTROL_PLANE_URL is unset" ;;
localhost | 127.* | ::1 | 10.* | 192.168.* | 172.1[6-9].* | 172.2[0-9].* | 172.3[01].* | *.local)
  origin="$(mask "$cp_url") is a loopback or LAN address, not an Application"
  ;;
*) origin="" ;;
esac

if [[ -n $origin ]]; then
  verdict FAIL "1. the artifact downloaded from the Application" "$origin"
elif [[ -n $downloaded ]]; then
  verdict PASS "1. the artifact downloaded from the Application" \
    "$downloaded" "over $(mask "$cp_url")"
elif [[ -n $cached ]]; then
  # Not a weaker pass. artifact_manager.py:165 re-digests the file on disk before
  # taking this branch, so the bytes were checked again just now; only the
  # download itself is older than this journal.
  verdict PASS "1. the artifact downloaded from the Application" \
    "$cached" "an earlier download, re-verified on disk against the desired digest"
else
  verdict UNPROVEN "1. the artifact downloaded from the Application" \
    "no download line for $label in the journal -- it may have rotated away" \
    "$stale" \
    "sudo systemctl restart $UNIT, let it reconverge, then run this again"
fi

# == 2. its SHA-256 validated ================================================
#
# Both lines above say "verified", and the word is load-bearing: the digest is
# computed over the assembled file and compared before anything is unpacked, and
# a mismatch deletes the bytes and raises rather than activating. So item 1's
# evidence carries item 2 -- with one exception worth printing.

mismatch="$(last_match 'expected sha256')"
if [[ -n $mismatch ]]; then
  verdict FAIL "2. its SHA-256 validated" "$mismatch" \
    "the bytes were deleted rather than activated, which is the design"
elif [[ -n $downloaded$cached ]]; then
  verdict PASS "2. its SHA-256 validated" \
    "carried by the line above: nothing is unpacked before the digest matches"
else
  verdict UNPROVEN "2. its SHA-256 validated" "no verified-download line to carry it"
fi

# == 3. the ONNX graph loaded ================================================

loaded="$(last_match "loaded $label with providers")"
if [[ -n $loaded ]]; then
  verdict PASS "3. the ONNX graph loaded" "$loaded"
else
  verdict UNPROVEN "3. the ONNX graph loaded" \
    "no \`loaded $label with providers\` line in the journal" \
    "$stale"
fi

# == 4. it executed once =====================================================
#
# The distinguishing item of this milestone: RUNNING is meant to say the graph
# executed, not that a session object was constructed.

smoke="$(last_match "serving $label -- smoke inference")"
smoke_skipped="$(last_match "serving $label but could not prove it executes")"
smoke_disabled="$(last_match "serving $label (smoke check disabled)")"
if [[ -n $smoke ]]; then
  verdict PASS "4. it executed once" "$smoke"
elif [[ -n $smoke_skipped ]]; then
  # A skip, never a refusal to serve. The model is being served and nothing has
  # proven it can execute; the heartbeat carries the same reason as
  # `smoke_check: skipped: ...`.
  verdict UNPROVEN "4. it executed once" "$smoke_skipped" \
    "served unproven -- the check could not synthesize an input, see §6"
elif [[ -n $smoke_disabled ]]; then
  verdict UNPROVEN "4. it executed once" "$smoke_disabled" \
    "the check was turned off, so RUNNING means only that a session was built"
else
  verdict UNPROVEN "4. it executed once" "no smoke-check line for $label"
fi

# == 5. the execution provider is not CPU-only ===============================
#
# The list in item 3's line is `session.get_providers()` -- what the loaded
# session is actually using. The build's `get_available_providers()` is NOT this
# list and is not evidence: a provider that cannot handle a node falls back to
# the CPU silently and per-node, so a device can advertise CUDA and serve on its
# CPU.
#
# The judgement itself comes from `GPU_PROVIDERS` in the installed contracts
# rather than from a pattern written here. Two readers of that set already exist
# -- the agent's hardware_info() and the control plane's derive_acceleration() --
# and a third copy, in shell, would drift in the direction that makes a CPU-only
# Jetson look accepted.

# The log line carries python's repr of the list, so the names arrive quoted.
# Quotes are stripped for display only; the python below re-strips them anyway,
# because the shape of that repr is onnxruntime's business and not a contract.
active="$(printf '%s' "$loaded" | sed -nE 's/.*with providers \[([^]]*)\].*/\1/p' | tr -d "\"'")"
if [[ -z $loaded ]]; then
  verdict UNPROVEN "5. the execution provider is not CPU-only" \
    "nothing loaded, so no session reported a provider list"
elif [[ -z $active ]]; then
  verdict UNPROVEN "5. the execution provider is not CPU-only" \
    "could not parse a provider list out of: $loaded"
else
  accel="$("$VENV/bin/python" - "$active" <<'PY'
import sys

try:
    from lighthouse_contracts import GPU_PROVIDERS
except Exception as exc:
    print("IMPORT-FAILED\x1f%s" % exc)
    raise SystemExit(0)

active = [p.strip().strip("'\"") for p in sys.argv[1].split(",")]
active = [p for p in active if p]
hits = [p for p in active if p in GPU_PROVIDERS]
print("%s\x1f%s" % ("ACCELERATED" if hits else "CPU_ONLY", ", ".join(hits or active)))
PY
  )"
  IFS=$'\x1f' read -r accel_state accel_detail <<<"$accel"
  case "$accel_state" in
  ACCELERATED)
    verdict PASS "5. the execution provider is not CPU-only" \
      "active: $active" "accelerated by: $accel_detail"
    ;;
  CPU_ONLY)
    verdict FAIL "5. the execution provider is not CPU-only" \
      "active: $active" \
      "no provider in GPU_PROVIDERS -- a device bought for its GPU, on its CPU" \
      "work through docs/jetson-setup.md §9 in order; step 1 is the wheel"
    ;;
  *)
    # Deliberately not guessed from a substring of the provider names: a verdict
    # this script invented would be a third copy of the judgement.
    verdict UNPROVEN "5. the execution provider is not CPU-only" \
      "active: $active" \
      "could not import GPU_PROVIDERS from $VENV: $accel_detail" \
      "re-run scripts/install_keeper.sh so the venv has the current contracts"
    ;;
  esac
fi

# == the verdict =============================================================

printf '\n'
last_reconcile="$(last_match 'reconcile: ')"
[[ -n $last_reconcile ]] && say "last reconcile: $last_reconcile"

if ((failures == 0 && passes == 5)); then
  say "ACCEPTED -- all five of §7's items are evidenced for $label."
  printf '\n'
  say "Two things left, and both are the control plane's side rather than this one:"
  say "confirm the fleet view agrees (GOVERNANCE HEALTHY, ACCEL ACCELERATED), then"
  say "prove it can be taken away again -- STOP_PENDING until this device confirms:"
  say "  make fleet"
  say "  make stop DEVICE=${device_id:-<device-id>}"
  printf '\n'
  exit 0
fi

say "NOT ACCEPTED -- $passes of 5 items evidenced, $failures not."
say "Any one item failing is the whole check failing (docs/jetson-setup.md §7)."
printf '\n'
exit 1
