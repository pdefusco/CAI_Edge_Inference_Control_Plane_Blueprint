# Bringing up a Jetson as a Lighthouse device

Spec Phase 6, device side. The control plane half is `docs/cai-deployment.md`; the
unit and the installer are `deploy/keeper.service` and
`scripts/install_keeper.sh`, and every landmine in them is commented in place
rather than repeated here.

**This document asserts no version facts about your device.** Not the JetPack
release, not the L4T version, not the Python it ships, and above all not where a
working `onnxruntime` comes from. A setup guide that guesses them is worse than
one that asks: a wrong version number sends an operator looking for a wheel that
does not exist while the one they have sits unused. §2 is a table for you to
fill in, and what you write there is what belongs in
`edge-agent/pyproject.toml`'s comment.

One device has now been measured, and §2 carries it as a clearly-labelled
**worked example** -- because "the accelerated build comes from NVIDIA" turned
out to be true and not actionable. It is one Jetson on one day, not a
requirement, and reading it as a specification is the mistake this paragraph
exists to prevent.

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
| the libraries a provider needs | `ls /usr/lib/aarch64-linux-gnu/libcudnn.so* /usr/local/cuda/lib64/libcublas.so* /usr/lib/aarch64-linux-gnu/libnvinfer.so*` | |
| can this python be installed into at all? | `python3 -m pip --version; ls /usr/lib/python3*/EXTERNALLY-MANAGED` | |

The onnxruntime rows are the ones the whole milestone turns on, and the trap is
specific: **a plain `pip install onnxruntime` gives you a build with no CUDA.** It
imports, it loads the model, it runs inference, it never errors -- the only
symptom is the provider list. Read that list rather than the version string, and
read it sceptically: on the laptop this repo was written on, the same check
reports `AzureExecutionProvider`, which is a *remote* inference endpoint and not
local acceleration at all. The accelerated aarch64 build comes from NVIDIA.
If your `providers` row has no `CUDAExecutionProvider` or
`TensorrtExecutionProvider` in it, resolve that **before** §3; the agent cannot
create a provider that the wheel does not have.

The last two rows each explain a missing provider you would otherwise blame on
the wheel. A provider needs its libraries at *runtime* --
`CUDAExecutionProvider` wants cuDNN and cuBLAS, `TensorrtExecutionProvider`
wants `libnvinfer` -- and a provider with no library behind it either drops out
of the list or imports and dies at the first session (§9 step 5). A system
python that ships `EXTERNALLY-MANAGED` and no `pip` cannot be installed into at
all, which decides *where* the wheel goes rather than whether you can have one;
§3 is where that lands.

The repo's own floor is `requires-python = ">=3.10"`
(`edge-agent/pyproject.toml`), and it is a floor rather than a pin for exactly
this reason: the Python that NVIDIA's wheels are built against is the Python this
agent has to run on, so the agent bends and the wheel does not.

### One device, as a worked example

**Not a specification.** The point of this section is that your device answers
for itself, and the next Jetson will answer differently. This is here because
"the accelerated build comes from NVIDIA" was too vague to act on, and because
the numbers in `edge-agent/pyproject.toml`'s comment should be traceable to the
commands that produced them.

Measured 2026-10-05 on a Jetson Orin Nano Developer Kit Super: L4T **R39.2.1**
(the JetPack 7.2 line), Ubuntu **24.04.4**, CUDA **13.2**, system Python
**3.12.3** with no `pip` and an `EXTERNALLY-MANAGED` marker, cuDNN 9 and cuBLAS
present, TensorRT absent, and **no onnxruntime of any kind**. What fit:

```
pip install --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130 onnxruntime-gpu
```

`onnxruntime-gpu 1.30.0`, cp312, `linux_aarch64`. Two things about that line.
The distribution is **`onnxruntime-gpu`** -- a different name from the CPU one
although both import as `onnxruntime`, which is why `edge-agent[onnx]` must not
be installed on a device. And the path is `sbsa/cu130`, not a `jp7/` path: this
JetPack is near enough to generic aarch64 plus CUDA 13 that the server-ARM index
is the one that resolves, so if `jp<n>/cu<m>` gives you nothing, look there
before concluding that no wheel exists.

Then the check from the table:

```
python -c "import onnxruntime as o; print(o.get_available_providers())"
→ ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

**which is still not evidence.** That call is what the build *offers*; a
provider that cannot handle a node falls back to the CPU silently and per-node,
so the list that settles it is `session.get_providers()` on a real graph.
Creating a session on a one-`Conv` graph -- `Conv` for the reason §9 step 4
gives -- reported `['CUDAExecutionProvider', 'CPUExecutionProvider']` and ran one
inference on a zero input. That is §7's items 3, 4 and 5 proven at the wheel,
before the agent exists, and it is worth doing in that order: a wheel that fails
here fails the same way under systemd, with six more moving parts in the way.

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
  gets you the CPU-only wheel from §2 and a device that works and is wrong. But
  on a device with no system installation to see, **the venv turns out to be the
  right home for the wheel after all**: see below, because what makes that move
  wrong is the missing index, not the venv.
* **Group membership, not `PrivateDevices`**, is what grants access to
  `/dev/nvhost-*`. The installer prints the groups it set; `id keeper` and
  `ls -l /dev/nvhost-ctrl` are the two things to compare if §7 fails.

The installer reports the onnxruntime it can see from inside the venv and warns
loudly when there is no CUDA or TensorRT provider. It does not refuse to install
in that case -- that call is yours -- but the warning is the §7 failure, arriving
early.

### When there is no system onnxruntime to see

§2's last row decides this, and on a JetPack 7-era device it decides it against
you: the system python is marked `EXTERNALLY-MANAGED` and ships no `pip`, so
there is no system installation for `--system-site-packages` to expose and no
supported way to create one. The wheel then goes into the venv the installer
just built:

```
sudo /opt/keeper/venv/bin/pip install --index-url <the index §2 found> onnxruntime-gpu
sudo /opt/keeper/venv/bin/python -c 'import onnxruntime as o; print(o.__version__, o.get_available_providers())'
```

This is the move §2 calls a trap, with the single difference that makes it
correct: the index. A bare `pip install onnxruntime` resolves the CPU-only
distribution from PyPI and nothing downstream will mention it. Install
`onnxruntime-gpu` from the index your device answered with, then read the
providers back out of **the venv's own python** -- that is §9 step 1's command,
and running it once here beats meeting it for the first time while debugging.

`--system-site-packages` stays either way. It costs nothing when there is
nothing to see, and a device that does have a system installation needs it.

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
5. the reported execution provider is **not CPU-only** -- `ACCEL: ACCELERATED`,
   and `UNKNOWN` is not a pass: it means nothing was measured, which is a
   different problem from a measured CPU fallback but not a better one.

### Run it rather than reading it by eye

```
sudo scripts/accept.sh            # on the device; add --show-host to unmask the URL
```

It checks all five from the journal and the agent's own state file, prints the
line it is relying on for each one, and exits non-zero unless all five hold.
Read its header for the two judgements it makes that a grep cannot:

* **every line is matched against the model the agent is serving right now**, so
  a journal full of provider lines from the version before this one reads as no
  evidence rather than as a pass -- and it says which version it found instead;
* **three verdicts, not two.** `UNPROVEN` is "no evidence either way" -- a journal
  that rotated past the deployment, a smoke check that was skipped -- and it is
  kept apart from `FAIL` for the same reason `UNKNOWN` is kept apart from
  `CPU_ONLY`. Neither one is a pass.

It reads only: no unit is started or stopped, nothing is written, and it never
reads the device token, which it has no use for. `sudo` is for the journal and
for `/etc/keeper/keeper.env`, which is `0640 root:keeper`. The control-plane URL
is masked unless you ask for it, so the output is safe to paste.

### Where to read the provider list

On the device, which is the direct evidence the script is quoting:

```
journalctl -u keeper | grep -E 'with providers|smoke inference'
```

From the control plane, which is what the fleet actually received -- three views of
the same heartbeat, in increasing detail:

```
make fleet                      # one ACCEL column per device
make device DEVICE=<device-id>  # the whole `hardware` dict as the device sent it
```

and the dashboard's **Hardware** card in the device detail panel, which shows the
acceleration verdict, both provider lists and the smoke check side by side.

`ACCEL` and the card's Acceleration row are the same derived field,
`DeviceView.acceleration`: `ACCELERATED`, `CPU_ONLY`, or `UNKNOWN` when nothing
was reported at all. It is derived server-side from `active_providers` so that
this judgement has one home rather than being re-made in the dashboard, in
`make fleet` and in an operator's head -- `Acceleration` in
`contracts/src/lighthouse_contracts/enums.py` carries the reasoning, including
why `CoreMLExecutionProvider` and `AzureExecutionProvider` do **not** count.

Read `GOVERNANCE` and `ACCEL` as a pair. `HEALTHY` + `CPU_ONLY` is the trap: the
device downloaded the right bytes, loaded them and is serving them, so it is
genuinely in sync with its desired state -- and it is failing this check. A
correctly green governance badge is not evidence for item 5.

Earlier revisions of this section said `DeviceView` did not expose `hardware` and
gave a `sqlite3` snippet to read `actual_deployment.hardware_json` by hand. That
gap is closed; the snippet is gone because it was reading a column that the API
now serves verbatim.

The two fields that matter inside `hardware`:

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

Every version in §2's **table** is yours, measured on your device. §2's worked
example is the one exception and says so: it names a JetPack release, a Python
version and a wheel index because a device was finally in front of this repo, and
a single measurement is an example of an answer rather than a claim about your
hardware. Treat it as the shape of one, and expect the numbers to be stale.

The Python floor in `edge-agent/pyproject.toml` is no longer remembered. Its
comment carries what that device reported and what the previous comment got
wrong -- which was JetPack 6, Python 3.10 and cp310 wheels, against a device
that answered JetPack 7.2, Python 3.12 and CUDA 13. The floor itself needed no
change, and that is the argument for keeping it a floor.

If you find yourself about to add a version number to this file, add the command
that produced it instead. If you must add the number, say which device and which
day it came from.
