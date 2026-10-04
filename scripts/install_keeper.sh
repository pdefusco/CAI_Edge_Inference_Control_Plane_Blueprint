#!/usr/bin/env bash
#
# Install the Lighthouse edge agent as a systemd service. Run on the device.
#
#   sudo scripts/install_keeper.sh [device_id]
#
# Idempotent, and non-destructive where it matters: it never overwrites
# /etc/keeper/keeper.env or /etc/keeper/token once they exist, because those are
# the two files that hold what an operator typed and a credential that cannot be
# re-read from the control plane. Re-running it upgrades the code and the unit.
#
# It deliberately does NOT source scripts/_common.sh. That file is the laptop dev
# harness: it mints admin tokens into .dev/, defaults the control plane to
# 127.0.0.1:8000, and prefers the repo's own .venv. None of that belongs on a
# device, and an admin token least of all -- a device is given a bearer token for
# itself and nothing more.
#
# What this does not do: enrol the device. A device has no admin credential, so it
# cannot. Run scripts/register_device.sh against the control plane and bring the
# token here over a channel you trust; this script makes a place for it with the
# right mode and tells you what is still missing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER=keeper
PREFIX=/opt/keeper
VENV="$PREFIX/venv"
CONF_DIR=/etc/keeper
ENV_FILE="$CONF_DIR/keeper.env"
TOKEN_FILE="$CONF_DIR/token"
UNIT=/etc/systemd/system/keeper.service

die() {
  printf '\n  error: %s\n\n' "$*" >&2
  exit 1
}
say() { printf '  %s\n' "$*"; }

device_id="${1:-$(hostname -s 2>/dev/null || hostname)}"

(($(id -u) == 0)) || die "run this with sudo -- it creates a system user and writes to /etc"
command -v systemctl >/dev/null 2>&1 || die "no systemctl; this installer is systemd-only"
[[ -f "$REPO_ROOT/edge-agent/pyproject.toml" ]] ||
  die "run this from a checkout: $REPO_ROOT/edge-agent/pyproject.toml is missing"

# The agent targets python >= 3.10 and uses `X | None` annotations at runtime in
# dataclasses, so an older interpreter fails at import rather than at type-check
# time. Checked here because the next failure would be a pip resolution error
# three screens long.
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found; set PYTHON=/path/to/python3"
"$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' ||
  die "$PYTHON is $("$PYTHON" -V 2>&1), and keeper needs 3.10 or newer"

say "python    $("$PYTHON" -V 2>&1) at $(command -v "$PYTHON")"

# == the service account =====================================================

if id "$SERVICE_USER" >/dev/null 2>&1; then
  say "user      $SERVICE_USER exists"
else
  # --home-dir is the state directory on purpose. CUDA writes its JIT cache to
  # $HOME/.nv/ComputeCache, and $HOME comes from the passwd entry: a home of
  # /nonexistent means the cache silently cannot be written and every start pays
  # the compile again. StateDirectory= in the unit creates and owns this path.
  useradd --system --no-create-home --home-dir /var/lib/keeper \
    --shell /usr/sbin/nologin "$SERVICE_USER"
  say "user      created $SERVICE_USER (home /var/lib/keeper for CUDA's JIT cache)"
fi

# Device access for a non-root user comes from here, not from the unit file. The
# Tegra nodes are root:video on some L4T releases and world-accessible on others,
# so this is either necessary or harmless and there is no way to tell which from a
# laptop. `render` exists on newer kernels only, hence the per-group check.
for group in video render; do
  if getent group "$group" >/dev/null 2>&1; then
    usermod -aG "$group" "$SERVICE_USER"
    say "group     $SERVICE_USER is in $group"
  else
    say "group     no $group group on this system -- skipped"
  fi
done

# == the code ================================================================

mkdir -p "$PREFIX"
if [[ ! -x "$VENV/bin/python" ]]; then
  # --system-site-packages is the Jetson-specific decision in this script. The
  # aarch64 onnxruntime build with CUDA and TensorRT comes from NVIDIA -- an index
  # or a .whl they publish -- and is usually already installed against the system
  # python. An isolated venv would hide it, `KEEPER_RUNTIME=onnx` would refuse to
  # start, and the obvious next move (pip install onnxruntime) pulls the CPU-only
  # PyPI wheel, which starts fine and runs the model on the CPU forever.
  "$PYTHON" -m venv --system-site-packages "$VENV"
  say "venv      created $VENV with --system-site-packages"
else
  say "venv      reusing $VENV"
fi

# No [onnx] extra. Installing it would resolve onnxruntime from PyPI and shadow
# whatever NVIDIA put in the system site-packages -- see above. The agent checks
# for an importable onnxruntime at startup and says what to do when there is none.
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet --upgrade "$REPO_ROOT/edge-agent"
say "keeper    $("$VENV/bin/keeper" --help >/dev/null 2>&1 && echo installed || echo 'installed but not runnable')"

if "$VENV/bin/python" -c 'import onnxruntime' >/dev/null 2>&1; then
  version="$("$VENV/bin/python" -c 'import onnxruntime; print(onnxruntime.__version__)')"
  providers="$("$VENV/bin/python" -c 'import onnxruntime; print(" ".join(onnxruntime.get_available_providers()))')"
  say "onnx      onnxruntime $version"
  say "          providers: $providers"
  case "$providers" in
    *CUDAExecutionProvider* | *TensorrtExecutionProvider*) ;;
    *)
      # Not fatal here. The agent will start and serve on the CPU, and that is a
      # decision for whoever is standing in front of the device -- but it is the
      # failure this whole milestone is about, so it does not get to be quiet.
      say ""
      say "          WARNING: no CUDA or TensorRT provider. The agent will run this"
      say "          device's models on its CPU. docs/jetson-setup.md covers the"
      say "          wheel situation; a GPU device serving on the CPU is a failed"
      say "          acceptance check, not a working install."
      say ""
      ;;
  esac
else
  say "onnx      onnxruntime is NOT importable in $VENV"
  say "          KEEPER_RUNTIME=onnx will exit 2 until it is. See docs/jetson-setup.md."
fi

# == the configuration =======================================================

install -d -m 0750 -o root -g "$SERVICE_USER" "$CONF_DIR"

if [[ -e "$ENV_FILE" ]]; then
  say "env       keeping the existing $ENV_FILE"
else
  # Only ever written on a first install, so the device id substitution cannot
  # clobber a value someone edited by hand.
  sed "s|^KEEPER_DEVICE_ID=.*|KEEPER_DEVICE_ID=$device_id|" \
    "$REPO_ROOT/deploy/keeper.env.example" >"$ENV_FILE"
  chown "root:$SERVICE_USER" "$ENV_FILE"
  chmod 0640 "$ENV_FILE"
  say "env       wrote $ENV_FILE (0640 root:$SERVICE_USER) with device_id=$device_id"
fi

if [[ -e "$TOKEN_FILE" ]]; then
  # Fix the mode even on an existing file: the common mistake is `scp` as root
  # followed by a default 0600 root:root, which the service user cannot read.
  chown "root:$SERVICE_USER" "$TOKEN_FILE"
  chmod 0640 "$TOKEN_FILE"
  say "token     $TOKEN_FILE exists (mode reset to 0640 root:$SERVICE_USER)"
else
  install -m 0640 -o root -g "$SERVICE_USER" /dev/null "$TOKEN_FILE"
  say "token     created an empty $TOKEN_FILE (0640 root:$SERVICE_USER)"
fi

# == the unit ================================================================

install -m 0644 -o root -g root "$REPO_ROOT/deploy/keeper.service" "$UNIT"
systemctl daemon-reload
systemctl enable keeper >/dev/null
say "unit      installed $UNIT and enabled it"

# == what is still missing ===================================================

missing=()
[[ -s "$TOKEN_FILE" ]] || missing+=("the device token in $TOKEN_FILE")
grep -q '^KEEPER_CONTROL_PLANE_URL=.\+' "$ENV_FILE" ||
  missing+=("KEEPER_CONTROL_PLANE_URL in $ENV_FILE")

if ((${#missing[@]} == 0)); then
  cat <<EOF

  Ready. Start it and watch the first reconcile:

    sudo systemctl start keeper
    journalctl -u keeper -f

EOF
else
  printf '\n  Not started -- it would exit 2. Still needed:\n\n'
  for item in "${missing[@]}"; do printf '    * %s\n' "$item"; done
  cat <<EOF

  The token comes from the control plane, not from here:

    # on a machine with the admin token
    scripts/register_device.sh $device_id
    # then, on this device, with the value it printed
    sudo install -m 0640 -o root -g $SERVICE_USER /dev/stdin $TOKEN_FILE

  Then: sudo systemctl start keeper && journalctl -u keeper -f

EOF
fi
