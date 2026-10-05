# Bringing up a Jetson as a Lighthouse device

Spec Phase 6, device side. The control plane half is `docs/cai-deployment.md`; the
unit and the installer are `deploy/keeper.service` and
`scripts/install_keeper.sh`, and every landmine in them is commented in place
rather than repeated here.

**This document asserts no version facts.** Not the JetPack release, not the L4T
version, not the Python the device ships, and above all not where a working
`onnxruntime` comes from. Those were not measurable from the machine this repo was
written on, and a setup guide that guesses them is worse than one that asks: a
wrong version number sends an operator looking for a wheel that does not exist
while the one they have sits unused. §2 is a table for you to fill in; what you
write there is the thing commit 14 of this milestone puts in
`edge-agent/pyproject.toml`.

The acceptance criterion is therefore behavioural, not a version match:

> **`hardware_info()` must report a non-CPU execution provider.** A device
> reporting `RUNNING` on `CPUExecutionProvider` alone is a **failed** check, not a
> working install -- it is a device bought for its GPU, serving on its CPU, and
> looking healthy on the dashboard while doing it.

---

## 1. What has to be true before you touch the device

| | why, and where it is settled |
|---|---|
| A control-plane URL a bearer-token client can reach | **Measured 2026-10-05: row 1.** `scripts/probe_app.py` off VPN reported `REACHABLE, and bearer auth is intact` against a deployed Application, so `KEEPER_CONTROL_PLANE_URL` is the app URL. `docs/cai-deployment.md` §6 has the outcome table and what the other rows would have cost — re-run the probe for your own deployment rather than inheriting this |
| A model `READY` and `deployable: true` | `scripts/probe_registry.py`. Nothing on the device can fix a model that is not deployable |
| A `device_id` you have chosen | it must match what you enrol in §4; a mismatch is a 403 on every poll, which on the dashboard is indistinguishable from a device that never came online |
| This checkout, on the device | `scripts/install_keeper.sh` installs from it |

If the probe said the app is SSO-gated or private-ingress-only, **stop here and
read that row first.** The device is purely outbound and has no browser; no amount
of device-side configuration gets a bearer token past a login page. That outcome
is a finding about the deployment, not a problem with this document.

---

## 2. Write down what you actually have

Fill this in on the device, before installing anything. Each command is read-only.

| | command | yours |
|---|---|---|
| JetPack / L4T | `cat /etc/nv_tegra_release` | |
| what the board calls itself | `cat /proc/device-tree/model` | |
| OS | `lsb_release -d` or `cat /etc/os-release` | |
| kernel / arch | `uname -mr` | |
| system python | `python3 -V` | |
| CUDA, if present | `nvcc --version` or `ls /usr/local/cuda*` | |
| is onnxruntime already installed? | `python3 -c 'import onnxruntime as o; print(o.__version__, o.get_available_providers())'` | |
| where it came from | `python3 -m pip show onnxruntime onnxruntime-gpu 2>/dev/null \| grep -E 'Name\|Version\|Location'` | |

The last two rows are the ones the whole milestone turns on, and the trap is
specific: **a plain `pip install onnxruntime` gives you a build with no CUDA.** It
imports, it loads the model, it runs inference, it never errors -- the only
symptom is the provider list. Read that list rather than the version string, and
read it sceptically: on the laptop this repo was written on, the same check
reports `AzureExecutionProvider`, which is a *remote* inference endpoint and not
local acceleration at all. The accelerated aarch64 build comes from NVIDIA.
If your `providers` row has no `CUDAExecutionProvider` or
`TensorrtExecutionProvider` in it, resolve that **before** §3; the agent cannot
create a provider that the wheel does not have.

The repo's own floor is `requires-python = ">=3.10"`
(`edge-agent/pyproject.toml`), and it is a floor rather than a pin for exactly
this reason: the Python that NVIDIA's wheels are built against is the Python this
agent has to run on, so the agent bends and the wheel does not.

---

## 3. Install the agent

```
sudo scripts/install_keeper.sh <device-id>
```

Idempotent: re-running it upgrades the code and the unit, and never overwrites
`/etc/keeper/keeper.env` or `/etc/keeper/token`. It creates the `keeper` system
user, adds it to `video` (and `render` where that exists), builds
`/opt/keeper/venv` **with `--system-site-packages`**, installs the agent, and
enables the unit without starting it.

Two of those deserve a sentence, because both are silent when wrong:

* **`--system-site-packages`** is how the venv can see a system-installed
  onnxruntime. An isolated venv hides it, `KEEPER_RUNTIME=onnx` then refuses to
  start, and the natural next move -- `pip install onnxruntime` inside the venv --
  gets you the CPU-only wheel from §2 and a device that works and is wrong.
* **Group membership, not `PrivateDevices`**, is what grants access to
  `/dev/nvhost-*`. The installer prints the groups it set; `id keeper` and
  `ls -l /dev/nvhost-ctrl` are the two things to compare if §7 fails.

The installer reports the onnxruntime it can see from inside the venv and warns
loudly when there is no CUDA or TensorRT provider. It does not refuse to install
in that case -- that call is yours -- but the warning is the §7 failure, arriving
early.

---

## 4. Enrol the device, and move the token by hand

A device cannot enrol itself. Enrolment needs the **admin** credential, and the
whole point of a device token is that the device holds nothing else: it can report
its own state and fetch its own artifacts, and it cannot deploy, stop or revoke
anything, including itself.

On a machine that has the admin token, against the control plane:

```
scripts/register_device.sh <device-id>
```

It prints the token **once** -- the server stored only `sha256(secret)` -- in the
form `lhd_<token_id>.<secret>`. Move it to the device over a channel you trust and
install it as a file, not as an environment variable:

```
# on the device, pasting the value
sudo install -m 0640 -o root -g keeper /dev/stdin /etc/keeper/token
```

* **A file, because `KEEPER_TOKEN` is public on the device.** An environment
  variable set in the unit is readable in `/proc/<pid>/environ` and printed by
  `systemctl show keeper`, which makes a permanent deployment credential visible
  to every account on the box.
* **`0640 root:keeper`, not `0600 root:root`.** The unit runs as `User=keeper`,
  which cannot read a root-only file. The property that matters is "not
  world-readable", and 0640 with the service's group has it. Get this wrong and
  the agent exits 2 with a message naming the mode -- `config.py` checks the read
  rather than letting it surface as a `PermissionError` traceback -- which is a
  deliberate fix and still a wasted ten minutes.
* Re-running `register_device.sh` on an enrolled device mints an **additional**
  token rather than failing, because that is also the rotation procedure: issue,
  install, confirm, then retire the old token by its `token_id`. §10 has the exact
  call, and the warning about which revoke it is not.

---

## 5. Point it at the control plane

```
sudo -e /etc/keeper/keeper.env
```

Every variable is documented in that file (it is `deploy/keeper.env.example`).
Three decide whether this works:

```
KEEPER_DEVICE_ID=<exactly what you enrolled>
KEEPER_CONTROL_PLANE_URL=<the app base URL, no /api/v1 suffix>
KEEPER_RUNTIME=onnx
```

`KEEPER_RUNTIME` defaults to `mock` in the agent's own code, and the example file
sets it to `onnx` for exactly that reason: a device left on `mock` reports a
plausible heartbeat, shows green on the dashboard, and never runs a model. That is
the one default worth overriding before the first start.

Leave `KEEPER_DATA_DIR` alone unless you have a reason. The unit's
`StateDirectory=keeper` creates and owns `/var/lib/keeper`, and
`ProtectSystem=strict` makes everything else read-only -- pointing the data
directory elsewhere needs a `ReadWritePaths=` to match, and the failure if you
forget is a read-only filesystem error on the first download.

---

## 6. Start it, and read the first reconcile

```
sudo systemctl start keeper
journalctl -u keeper -f
```

The first poll happens immediately. The lines to look for, in the order they
appear -- quoted from the agent's own log calls, so you can grep for them:

1. `keeper starting: device=... control_plane=... runtime=...` -- check all three.
   `runtime=mock` here means §5 did not take, and nothing after this point is
   real;
2. `could not fetch desired state: ...` is the one you do **not** want. A 401/403
   is the token or the `device_id`; HTML or a redirect is §1's SSO row;
3. `reconcile: <note>` once per pass -- the summary of what it decided, including
   "nothing to do" when you have not deployed anything yet
   (`make deploy MODEL=... DEVICE=...` from the control-plane side);
4. `downloaded and verified <model>/<version> (N bytes)` -- the SHA-256 passed. A
   mismatch refuses to activate, by design, and says so instead;
5. `activated <model>/<version> at <path>`;
6. `loaded <model>/<version> with providers [...] (input=...)` -- **this list is
   the gate**, see §7;
7. `serving <model>/<version> -- smoke inference on a zero [...] input returned N
   output(s)`.

That last line is the one that distinguishes this milestone from the one before
it. `RUNNING` now means the graph **executed once** on this device, not that a
session object was constructed: `OnnxRuntime.start()` runs one inference on a
synthesized zero input before reporting the model as serving, because a model can
load cleanly and still have no kernel for a node or a CUDA library that resolves
at load and dies at the first launch.

If instead you see `serving ... but could not prove it executes: ...`, the graph
is being served **unproven** -- the check could not invent an input for it (several
declared inputs, a non-float input type, no declared shape, or a tensor over its
element cap). That is a skip, never a refusal to serve, and the reason travels in
every heartbeat as `smoke_check`.

A failure exits nothing: `start()` raising becomes `FAILED` with the message and a
backoff, the agent keeps heartbeating, and the dashboard shows a governance
failure instead of a device that went quiet. The only things that exit are
misconfigurations, and they exit **2**, which `RestartPreventExitStatus=2` turns
into a stopped unit and one readable journal line rather than a restart loop.

---

## 7. The acceptance check

Spec Phase 6's Definition of Done, as one sequence. Deploy a model to the device
from the control-plane side:

```
make deploy MODEL=<model> DEVICE=<device-id>
```

Then confirm all five of these, and treat any one of them failing as the whole
check failing:

1. the artifact downloaded from the **Application**, not from a laptop;
2. its SHA-256 validated;
3. the ONNX graph loaded;
4. it **executed once** -- the `smoke inference ... returned N output(s)` line, or
   `smoke_check: passed` in the heartbeat;
5. the reported execution provider is **not CPU-only**.

### Where to read the provider list

On the device, which is the direct evidence:

```
journalctl -u keeper | grep -E 'with providers|smoke inference'
```

From the control plane, which is what the fleet actually received. **Note the gap
here:** the heartbeat carries `hardware` and the control plane stores it, but
`DeviceView` does not expose it, so neither the dashboard nor `make device` shows
the provider list. Until something surfaces it, read it out of SQLite:

```
python - <<'PY'
import json, sqlite3
db = sqlite3.connect("<data_dir>/lighthouse.db")
db.row_factory = sqlite3.Row
for r in db.execute("SELECT device_id, actual_state, hardware_json FROM actual_deployment"):
    print(r["device_id"], r["actual_state"], json.dumps(json.loads(r["hardware_json"] or "{}"), indent=2))
PY
```

The two fields that matter in that JSON:

* **`active_providers`** -- what the loaded session is *actually* using. This is
  the gate. `providers` next to it only says what the installed build *could* do,
  and the difference is the whole problem: a provider that cannot handle a node
  falls back to the CPU silently and per-node, so a device can report CUDA as
  available while running the model on its CPU.
* **`smoke_check`** -- `passed`, `not run`, `disabled`, `skipped: <why>` or
  `failed: <why>`. A `RUNNING` device whose check was skipped has not been proven
  to execute anything, and that is a different claim from a proven one.

Finally, prove the control plane can take it away again:

```
make stop DEVICE=<device-id>
```

`STOP_PENDING` until the device confirms, then `STOPPED`. Artifacts are kept --
`make revoke` is the one that deletes bytes.

---

## 8. Harden one directive at a time

`deploy/keeper.service` ships a conservative sandbox enabled and a block of
stronger directives **commented out, in the order to add them**. That order is not
caution for its own sake: the sandbox on this device can remove GPU access without
removing the service, and the result passes every check except §7's provider list.

The loop, for each one:

```
sudo -e /etc/systemd/system/keeper.service     # uncomment exactly one
sudo systemctl daemon-reload && sudo systemctl restart keeper
journalctl -u keeper | grep 'with providers'   # still non-CPU?
```

If the provider list changed, you have found the cost of that directive. Re-comment
it and write down why in the unit file, where the next person will read it.

Two are already decided and should not be revisited casually, both documented in
place: `PrivateDevices=yes` hides `/dev/nvhost-*` and is the single worst mistake
available in that file, and `ProcSubset=pid` would hide
`/proc/device-tree/model`, costing the heartbeat the only field that identifies an
Orin as an Orin.

---

## 9. When the provider list is CPU-only

In order, because each step rules out the one below it:

1. **Is it the wheel?** `/opt/keeper/venv/bin/python -c 'import onnxruntime as o;
   print(o.__version__, o.__file__, o.get_available_providers())'`. If the
   providers are CPU-only here, nothing about the unit is involved -- you are in
   §2's trap, and the fix is the wheel. The `__file__` says whether the venv is
   seeing the system installation or a pip copy that shadowed it.
2. **Is it device access?** `id keeper` against `ls -l /dev/nvhost-ctrl
   /dev/nvmap`. A missing group is the installer's job and it prints what it did;
   a missing device node is a driver or boot problem and not an agent problem.
3. **Is it the sandbox?** Comment out the hardening block wholesale, restart, and
   re-read the list. If it comes back, bisect with §8.
4. **Is it the graph?** `active_providers` is per-session. A provider that cannot
   handle a node falls back silently, so a graph of unsupported ops yields a CUDA
   build serving on the CPU. The repo's fixture graph keeps a `Conv` specifically
   because `Conv` is the op that exercises CUDA and TensorRT -- if the fixture gets
   a GPU provider and your model does not, the answer is in your model's ops.
5. **Does the library load at all?** A wheel built against a CUDA the device does
   not have imports and then dies on `libcublas.so`. That failure is caught around
   the runtime construction in `main.py` and exits 2 with the message, so check
   `systemctl status keeper` before assuming a silent fallback.

---

## 10. Rotating a token, and removing the agent

Rotation, in this order, so the device is never without a working credential:

```
# 1. on the admin machine -- issues an ADDITIONAL token and prints its token_id
scripts/register_device.sh <device-id>

# 2. on the device
sudo install -m 0640 -o root -g keeper /dev/stdin /etc/keeper/token
sudo systemctl restart keeper
journalctl -u keeper -n 20          # a clean poll, not "could not fetch desired state"

# 3. only after that, retire the old one by its token_id
curl -sS --fail-with-body -X DELETE \
  "<app-url>/api/v1/devices/<device-id>/tokens/<old-token_id>" \
  -H "X-Lighthouse-Admin-Token: <admin-token>"
```

**That last call is not `make revoke`.** The two revokes are different
operations on the same device and the Makefile only wraps the other one:
`DELETE .../tokens/<token_id>` retires a *credential*, while `make revoke`
(`POST .../revoke`) is the governance action that makes the device **delete the
model bytes**. Reaching for the wrong one during a token rotation takes the
deployment down.

Removal:

```
sudo systemctl disable --now keeper
sudo rm /etc/systemd/system/keeper.service && sudo systemctl daemon-reload
sudo rm -rf /opt/keeper /etc/keeper /var/lib/keeper   # the token and the cache
```

Leaving the enrolment in place is deliberate if you plan to re-flash. Otherwise
delete the device's tokens as in step 3 above, so a credential that still exists
on a disk somewhere stops being able to heartbeat.

---

## What this document does not assert

Every version in §2 is **yours**, measured on your device. Nothing here claims a
JetPack release, a Python version, or a wheel source, because none of those were
observed from the machine this was written on, and the one that is remembered
rather than measured -- the Python floor in `edge-agent/pyproject.toml` -- is
corrected in the next commit with what the device actually reported.

If you find yourself about to add a version number to this file, add the command
that produced it instead.
