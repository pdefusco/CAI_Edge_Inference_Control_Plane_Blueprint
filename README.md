# Lighthouse — an edge model governance control plane

Models are trained and registered in **Cloudera AI**. A web app running as a CAI
Application is the governance control plane. A small agent (`keeper`) on each
edge device polls it, reconciles, and reports back. The control plane can
deploy, upgrade, roll back, stop and revoke a model on a fleet of devices, and
shows desired state against actual state for each one.

The architectural rule, from which most of this design follows:

> **CAI owns desired state. The edge agent owns reconciliation and reports
> actual state.**

Nothing SSHes into a device and nothing opens an inbound connection to the edge
network. Devices make outbound HTTPS calls to the control plane and are
authenticated by a per-device bearer token the control plane issued.

## What is actually proven

A blueprint that has only ever run on a laptop is a design document with tests.
Measured end to end on 2026-10-04, on an **NVIDIA Jetson Orin Nano** (JetPack
7.2 / L4T R39.2.1, Ubuntu 24.04, CUDA 13.2, Python 3.12) against the control
plane deployed as a CAI Application:

| | |
|---|---|
| device enrolled, token issued, heartbeating | `ONLINE` / `HEALTHY` |
| model pulled from the CAI registry through the Application | `downloaded and verified smoke-test/1 (876 bytes)` |
| SHA-256 verified before activation | nothing is unpacked before the digest matches |
| ONNX graph loaded | `providers ['CUDAExecutionProvider', 'CPUExecutionProvider']` |
| an inference actually executed | `smoke inference on a zero [1, 4] input returned 1 output(s)` |
| the GPU is genuinely in use | `ACCEL ACCELERATED`, derived from the session's own providers |
| the control plane can take it away | `make stop` → `STOP_PENDING` → `STOPPED`, artifacts kept |

`scripts/accept.sh` is that table as a script: it reads the device's journal and
reports PASS/FAIL per item, because "it looks like it's working" is not an
acceptance check. Run it on the device.

## Layout

```
contracts/        the API schemas and enums both sides import -- one definition,
                  so the control plane and the agent cannot disagree about what
                  a field means
control-plane/    FastAPI app + dashboard. Devices, models, deployments,
                  heartbeats, tokens, audit events, and the CAI registry adapter
edge-agent/       `keeper` -- the device agent: poll, reconcile, download,
                  verify, load, smoke-test, serve, report
scripts/          the dev harness and the operator tools (see `make help`)
deploy/           the systemd unit and its env file, for a real device
docs/             how to deploy it for real; read these before the device arrives
app.py            the entry point a CAI Application's script field points at
```

## Run it on a laptop, with no device and no CAI

```bash
make venv          # .venv + all three packages, editable
make dev           # control plane, dashboard, and a simulated Jetson
```

`make dev` is self-contained: a fake registry, a SQLite database and a simulated
device, all under `.dev/`. It prints the dashboard URL and an admin token. Then,
from a second terminal, drive the loop:

```bash
make fleet                      # desired vs actual, one line per device
make deploy  VERSION=2          # PUT desired state RUNNING
make device                     # the whole view of one device
make stop                       # reversible: the artifacts stay on the device
make revoke                     # irreversible: the device deletes the bytes
make events                     # the audit trail
```

`make help` lists the rest. Every one of those is plain `curl` against the same
API the dashboard uses — deliberately, because anything the Makefile can do a
fleet script can do too.

## Run it for real

Two documents, in this order. Both are written to be followed by someone who has
the hardware in front of them and does not yet trust the repo.

* **[docs/cai-deployment.md](docs/cai-deployment.md)** — the control plane as a
  CAI Application: the registry adapter, the auth chain, the environment
  variables, and what the Application's URL is.
* **[docs/jetson-setup.md](docs/jetson-setup.md)** — the device: measuring what
  you have, the onnxruntime wheel situation (**§2, and it is the hard part**),
  enrolment, the systemd unit, the acceptance check, and hardening it one
  directive at a time.

```bash
# on the laptop, against a deployed control plane
export LIGHTHOUSE_URL=https://<your-application-host>
export LIGHTHOUSE_ADMIN_TOKEN=<the value set in the Application's env vars>
scripts/register_device.sh <device-id>        # mints that device's token

# on the device
sudo scripts/install_keeper.sh <device-id>    # user, venv, unit, config
sudo scripts/accept.sh                        # prove all five items
```

## Tests

```bash
make test              # 449 control-plane + 227 edge-agent, no ML stack needed
make test-agent-onnx   # tier 2: the real onnxruntime wheel, opt-in
```

Two suites in two processes on purpose: the edge-agent conftest refuses to
import `lighthouse`, so a layering violation fails loudly instead of working by
accident. Tier 1 runs anywhere, including a laptop with no ML libraries at all.
Tier 2 runs the same agent code against a real wheel and a real graph, and
skips rather than fails when the wheel is absent — so it is safe to run
anywhere and only proves something where it is installed.

## The thing this repo is actually about

Every layer of an edge inference stack will tell you it is working. Most of them
can be wrong about that, and the failure is always in the reassuring direction:

* `pip install onnxruntime` gives you a CPU-only build that imports, loads,
  infers and never errors. The only symptom is a provider list.
* `get_available_providers()` reported CUDA for a wheel with no kernels for this
  board, and reports TensorRT on a device with no TensorRT installed at all.
* `session.get_providers()` — one layer deeper, and it lied too: CUDA on a
  session whose first `Gemm` then died with `cudaErrorNoKernelImageForDevice`.
* Asking onnxruntime for `[TensorRT, CUDA, CPU]` with no TensorRT present makes
  it discard **CUDA as well**, raise nothing, and serve every inference on the
  CPU. A GPU device, `RUNNING`, healthy, green, on its CPU.

So the agent runs an inference before it reports `RUNNING`, the control plane
derives acceleration from the session's own active providers rather than from
anything a build advertises, and `hardware.provider_fallbacks` says which
accelerator went missing and why. A device that is serving-but-unproven,
healthy-but-CPU-only, or online-but-stale is visible as exactly that.

`HEALTHY` + `CPU_ONLY` is the trap worth internalising: the device downloaded the
right bytes, loaded them and is serving them, so it is genuinely in sync with its
desired state — and it is failing the check you actually cared about. Governance
status and acceleration are different claims and are reported separately.

## What this is not

A proof of concept, and honest about which parts are load-bearing:

* **The admin token is the only gate on the control plane's API.** With CAI
  platform authentication disabled (which is what lets a device reach it with a
  bearer token), that token is all there is. Treat it accordingly.
* **No TLS pinning, no mTLS, no token expiry.** Device tokens are revocable by
  `token_id` and that is the whole credential story; `docs/jetson-setup.md` §10
  has the rotation order.
* **SQLite.** One Application, one writer. Fine for a fleet you can name.
* **This repo is public, and carries no tenant identifiers of any kind** — no
  hostname, no CRN, no private address, no workload id. That is a deliberate
  convention, not an oversight: a registry hostname identifies a tenant.
  `.env.example` holds placeholders only, `scripts/probe_app.py` and
  `scripts/accept.sh` mask the host unless asked not to, and `config.py` gives
  `registry_domain` no default and never will. Keep it that way.

`CAI_EDGE_MODEL_HUB_PROJECT_SPEC.md` is the original brief, kept as written so
the design can be read against what was asked for.
