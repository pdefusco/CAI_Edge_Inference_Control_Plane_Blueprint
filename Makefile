# Lighthouse — edge model governance control plane.
#
# The operator verbs (deploy/stop/revoke/events) are plain curl against the same
# API the dashboard uses. They are here because a demo you can drive from a second
# terminal is far more convincing than one that only works through a browser --
# and because anything the Makefile can do, a device fleet script can do too.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PY       := .venv/bin/python
PIP      := .venv/bin/pip
PYTEST   := ../.venv/bin/python -m pytest

DEV_DIR    ?= .dev
# LIGHTHOUSE_URL is the variable scripts/_common.sh reads, so one export drives
# the shell harness and this Makefile both, rather than two names for the same
# fact -- which is how `scripts/register_device.sh` and `make fleet` ended up
# needing separately-spelled configuration to talk to the same deployment.
# `:/=` strips a trailing slash, for the 404 reason _common.sh explains.
API        ?= $(if $(LIGHTHOUSE_URL),$(LIGHTHOUSE_URL:/=)/api/v1,http://127.0.0.1:$(or $(LIGHTHOUSE_PORT),8000)/api/v1)
DEVICE     ?= jetson-sim-01
MODEL      ?= fashion-cnn
VERSION    ?= 1

# Read at recipe time, not parse time: the token file does not exist until the
# first `make dev`, and a Makefile that fails to parse before setup is hostile.
#
# LIGHTHOUSE_ADMIN_TOKEN wins when set, for the same one-name reason as API, and
# because against a deployed control plane $(DEV_DIR)/admin-token holds a token
# minted on this laptop that the deployment has never heard of -- so the fallback
# is not a fallback but a different credential, and it fails as 401 rather than
# as "you did not say which token".
TOKEN   = $(or $(LIGHTHOUSE_ADMIN_TOKEN),$$(cat $(DEV_DIR)/admin-token 2>/dev/null))
AUTH    = -H "X-Lighthouse-Admin-Token: $(TOKEN)"
JSON    = -H 'Content-Type: application/json'
CURL    = curl -sS --fail-with-body

.PHONY: help venv dev test test-fast test-plane test-agent test-agent-onnx \
        deploy rollback stop revoke device events models enroll fleet clean

help:
	@printf '\n  Lighthouse\n\n'
	@printf '  setup     make venv          create .venv and install all three packages editable\n'
	@printf '  run       make dev           control plane + dashboard + simulated Jetson\n'
	@printf '  test      make test          both suites        make test-fast  (no random order)\n'
	@printf '            make test-agent-onnx  tier 2: the real onnxruntime wheel, opt-in\n'
	@printf '\n  drive the loop (needs `make dev` running in another terminal)\n\n'
	@printf '  make deploy  [VERSION=2]     PUT desired state RUNNING\n'
	@printf '  make stop                    reversible: artifacts stay on the device\n'
	@printf '  make revoke                  irreversible: the device deletes them\n'
	@printf '  make fleet                   desired vs actual, one line per device\n'
	@printf '  make device                  full view of DEVICE=%s\n' '$(DEVICE)'
	@printf '  make events                  the audit trail\n'
	@printf '  make enroll  DEVICE=bench-02 mint a token for another device\n'
	@printf '\n'

# -- setup ------------------------------------------------------------------

venv:
	python3 -m venv .venv
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -e contracts -e control-plane -e edge-agent
	@printf '\n  installed. next: make dev\n\n'

# -- run --------------------------------------------------------------------

dev:
	@scripts/dev.sh

# -- test -------------------------------------------------------------------

# Both suites, in separate processes on purpose: the edge-agent conftest refuses
# to import `lighthouse` so that a layering violation fails loudly instead of
# working by accident. Running them together would hide that.
test: test-plane test-agent

test-plane:
	@cd control-plane && $(PYTEST) tests -q

test-agent:
	@cd edge-agent && $(PYTEST) tests -q

# Tier 2: the same agent code against the real onnxruntime wheel and the real
# fixture graph. Not part of `make test`, which must stay runnable on a laptop
# with no ML stack -- the `onnx` marker is deselected by `addopts` in
# edge-agent/pyproject.toml and this `-m onnx` overrides it, because pytest keeps
# only the last `-m` on the command line.
#
# Needs `pip install -e 'edge-agent[onnx]'` (on a Jetson, onnxruntime comes from
# NVIDIA's index instead). Without the wheel every test here skips rather than
# fails, so this target is safe to run anywhere; it just proves nothing.
test-agent-onnx:
	@cd edge-agent && $(PYTEST) tests -q -m onnx

# Deterministic order, for bisecting a failure that only appears under one seed.
test-fast:
	@cd control-plane && $(PYTEST) tests -q -p no:randomly -x
	@cd edge-agent && $(PYTEST) tests -q -p no:randomly -x

# -- drive the loop ---------------------------------------------------------

deploy:
	@$(CURL) -X PUT $(API)/devices/$(DEVICE)/deployment $(AUTH) $(JSON) \
	  -d '{"model_name":"$(MODEL)","model_version":"$(VERSION)","desired_state":"RUNNING"}' \
	  | $(PY) -m json.tool
	@printf '\n  desired state set. The device is still RUNNING the old version (or nothing)\n'
	@printf '  until it polls -- that gap is what the dashboard shows.\n\n'

# Rollback is not its own endpoint: it is a deployment naming an earlier version,
# which the server classifies as DEPLOYMENT_ROLLED_BACK by comparing history.
rollback:
	@$(MAKE) --no-print-directory deploy VERSION=$(VERSION)

stop:
	@$(CURL) -X POST $(API)/devices/$(DEVICE)/stop $(AUTH) | $(PY) -m json.tool
	@printf '\n  STOP_PENDING until the device confirms. Artifacts are kept.\n\n'

revoke:
	@$(CURL) -X POST $(API)/devices/$(DEVICE)/revoke $(AUTH) | $(PY) -m json.tool
	@printf '\n  REVOKE_PENDING until the device confirms it deleted the bytes.\n\n'

enroll:
	@scripts/register_device.sh $(DEVICE)

# -- look at it -------------------------------------------------------------

device:
	@$(CURL) $(API)/devices/$(DEVICE) $(AUTH) | $(PY) -m json.tool

fleet:
	@$(CURL) $(API)/devices $(AUTH) | $(PY) scripts/show.py fleet

models:
	@$(CURL) $(API)/models $(AUTH) | $(PY) -m json.tool

events:
	@$(CURL) "$(API)/devices/$(DEVICE)/events?limit=25" $(AUTH) | $(PY) scripts/show.py events

# -- housekeeping -----------------------------------------------------------

# Deletes the admin token, the device tokens, the SQLite database, the artifact
# cache and the agent's local state -- i.e. a genuinely fresh demo.
clean:
	@rm -rf $(DEV_DIR)
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@printf '  removed %s and bytecode caches\n' '$(DEV_DIR)'
