# Deploying the control plane as a Cloudera AI Application

Spec Phase 7. The goal is one URL that both a browser and a Jetson can reach: the
dashboard for a human, and the three outbound device routes for the agent.

Everything here is either read out of the code (with the file named, so you can
check it) or marked as something you have to **measure**. Two things in this
milestone could not be verified from a laptop, and they are called out as open
questions rather than written down as facts. Do not let this file grow an
assertion that nobody observed.

---

## What has to be true before you start

| | why |
|---|---|
| A model is `READY` and `deployable: true` in the AI Registry | M3's acceptance gate; `scripts/probe_registry.py` says so |
| You can run a CAI **Session** in the project | that is where `cdp` works and where you debug this by hand |
| The project filesystem has room for the artifact cache | default ceiling is 8 GiB, see below |

---

## 1. Decide what the Application is allowed to serve

The process serves three things on one port:

* the **dashboard** (`/`, `/static/...`), behind a sign-in gate backed by
  `SessionStore`;
* the **admin API** (`/api/v1/devices`, `/deployment`, `/stop`, `/revoke`, …),
  which needs the admin credential;
* the **device API** — desired state GET, heartbeat POST, artifact GET — which
  needs a device bearer token.

`GET /api/v1/health` is the only unauthenticated route (`api/meta.py:17`). That is
deliberate and it is what the probe script leans on.

**The device never listens.** Everything is outbound from the Jetson, so the
Application needs no inbound route to the device and no port forwarding anywhere.

---

## 2. Set the environment

Copy `.env.example`. Every variable is documented there; the ones that decide
whether this works at all:

```
LIGHTHOUSE_ENV=cai
LIGHTHOUSE_ADMIN_TOKEN=<mint one, see below>
LIGHTHOUSE_REGISTRY_DOMAIN=<your registry host>     # or _ENVIRONMENT, not both
```

Mint the admin token once and keep it:

```
python -c 'import secrets; print("lha_" + secrets.token_urlsafe(24))'
```

Three defaults flip under `LIGHTHOUSE_ENV=cai` (`config.py:255-325`), and each one
exists to close a hole:

* **`LIGHTHOUSE_ADMIN_TOKEN` becomes fatal-if-missing.** Under `local` the process
  mints an ephemeral one and prints it. Under `cai` it refuses to boot. A control
  plane that can revoke models, reachable from the internet with authentication
  silently disabled, is worse than one that fails to start.
* **`LIGHTHOUSE_REGISTRY` defaults to `cai`** rather than `fake`, so a deployment
  cannot quietly serve generated fixtures while looking like production.
* **`LIGHTHOUSE_DATA_DIR` defaults to `/home/cdsw/.lighthouse`** — the project
  filesystem, which survives a restart. See §4.

There is **no per-run token minting to worry about**: `config.py:301-316` only
mints under `local`. Nothing to fix here, which is why this section documents
rather than changes it.

---

## 3. Create the Application

**Script: `app.py`**, at the repo root. It is committed, so there is nothing to
paste — point the Application's script field at it. A launcher in the project is
easier to point an Application at than the `lighthouse` console script
(`control-plane/pyproject.toml:52` → `lighthouse.main:run`) on `PATH`.

It does one thing before calling `run()`: it puts `contracts/src` and
`control-plane/src` on `sys.path`, ahead of anything installed. That makes
`lighthouse` and `lighthouse_contracts` importable with **no install at all**, and
means the Application serves the project's current code rather than a copy
installed weeks ago. Observed 2026-10-04: in an interpreter with neither
`pydantic` nor `fastapi` present, all three of `lighthouse`,
`lighthouse_contracts` and `lighthouse.main` resolve to the repo's own `src`
trees.

What the bootstrap cannot do is supply the third-party half — see
**Dependencies** below. It is also a deliberate departure from how every other
entry point in this repo finds the packages (`Makefile:49-53`,
`scripts/_common.sh:32-43` both rely on an editable install), and `app.py` says
so in a comment so nobody 'fixes' it back.

**Runtime.** `control-plane/pyproject.toml:20` is `requires-python = ">=3.11"`,
and that floor is load-bearing rather than aspirational: the comment above it
records 3.10 being measured — 3 failures, one root cause, "Do not lower this
floor." CAI runtimes ship a range of Pythons, so check the one you select *before*
creating the Application, with `python3 -V` in a Session on that runtime. A 3.10
runtime fails in ways that do not look like a version problem.

**Dependencies.** The bootstrap covers the first-party packages and nothing else.
These still have to be present in the Application container: `fastapi`,
`uvicorn[standard]`, `jinja2`, `pydantic`, `pyyaml`, `python-multipart`
(`control-plane/pyproject.toml:23-29`), plus `httpx` from the `cai` extra
(`:42-44`) — required here, not optional, because `LIGHTHOUSE_REGISTRY` defaults
to `cai` under `LIGHTHOUSE_ENV=cai`. The install that gets all of them:

```
pip install --user -e contracts -e 'control-plane[cai]'
```

`contracts` is a local path package pip cannot resolve from an index, which is why
it is named explicitly. `cdp` also has to be on `PATH` for the default
`registry_token_source=cli` (`registry/cai.py:190` names `pip install cdpcli`);
that is §5's open question rather than a separate one.

**Open question. Measure it; do not assume it.** Whether a `--user` install run in
a Session reaches an *Application* container was not verified in this milestone.
An Application gets a fresh container from the runtime image, so this turns on
whether `--user` wrote into the project filesystem or into the image. The
`sys.path` bootstrap removes the first-party half of that risk on purpose; it
cannot remove the third-party half. If the Application dies on a
`ModuleNotFoundError` for `fastapi` or `httpx`, this is why — check from inside
the Application's own environment:

```
python3 -c 'import fastapi, httpx; print("ok")'
```

**Port.** CAI sets `CDSW_APP_PORT` and routes the Application's public URL to it.
`main.py:297` reads `CDSW_APP_PORT` first, then `PORT`, then falls back to 8000 —
so in an Application you set **neither**, and binding anything else yields an app
that starts cleanly and is unreachable. Leave `LIGHTHOUSE_HOST` unset too:
`main.py:298` already binds `0.0.0.0` whenever the env is not `local`.

**That 8000 fallback is a laptop-only convenience, and it bites in a Session.**
Port 8000 inside a CAI Session is held by something outside your namespace: the
bind fails `EADDRINUSE` while `ss -ltn` and `netstat -ltn` both show nothing, and
uvicorn logs `Application startup complete` *before* reporting the failure, so the
log reads like a successful boot right up to the error. Observed 2026-10-04. When
you run it by hand in a Session:

```
env -u CDSW_APP_PORT PORT=8900 python app.py
```

Prefer `app.py` over the `lighthouse` console script here: the script only exists
on `PATH` if the editable install happened, and `app.py` needs nothing more than
the third-party dependencies.

**Unauthenticated access.** If the Application is created with platform
authentication *enabled*, CML puts Cloudera SSO in front of it and a bearer-token
client gets a login page. See §6 — that is the measurement, not an assumption.

**Write down which way you set that toggle.** §6's result cannot be read back
without it: "SSO-gated" means one thing if you opted in and something entirely
different if the platform imposed it on an Application you created with
authentication off.

### Where the URL comes from

There is no URL until the Application exists. §6, §7 and
`docs/jetson-setup.md:349` all write `<app-url>` as though you already had it;
this is where you get it.

The address is the **subdomain** — a field you fill in when creating the
Application — joined to the **workbench domain**. The workbench domain is in
every Session's environment:

```
echo $CDSW_DOMAIN
```

`scripts/probe_registry.py:630` already prints it among the Session variables it
reports, so you may have seen it there.

**Copy the URL from the project's Applications list rather than assembling it by
hand.** That list is the authoritative value; the two-part shape above is what to
confirm against it, not a fact to lean on. And do not reuse the registry host
here — `probe_registry.py:645-646` notes the registry lives on a *different* host
from `$CDSW_DOMAIN`, which is correct for the registry and wrong for the
Application. Conflating the two is the easy mistake.

A hostname identifies a tenant and this repo is public, so keep it out of
anything you commit. `probe_app.py` masks the app host to `<app-host>` unless you
pass `--show-host` (`probe_app.py:106-111`), which is what makes its output safe
to paste into a note or an issue.

---

## 4. What survives a restart, and what does not

An Application restarts: on a project change, on a resource bump, when someone
stops and starts it. Two kinds of state behave differently, and the difference is
worth knowing before someone reports it as a bug.

**Device enrollments survive.** A device token is `lhd_<token_id>.<secret>` and
only `sha256(secret)` is stored (`services/device_service.py:76,110,211`), in
SQLite at `<data_dir>/lighthouse.db`. With `data_dir` on the project filesystem,
a restart is invisible to the fleet: every Jetson keeps heartbeating with the
token it already has.

**Dashboard logins do not.** `SessionStore` keeps sessions in an in-process dict
(`services/sessions.py:80`, `self._expiry: dict[str, float]`), so a restart logs
every browser out. That is the correct trade for a blueprint — a session table
would be state to migrate — but it means "I got logged out" after a redeploy is
expected behaviour and not a symptom.

**If you leave `LIGHTHOUSE_DATA_DIR` pointing at a container-local path, every
enrollment is lost on restart** and every device starts 401ing with a token the
control plane has never heard of. The `cai` default avoids this; an explicit
override can reintroduce it.

**Artifact cache.** `<data_dir>/artifacts`, ceiling
`LIGHTHOUSE_ARTIFACT_CACHE_MAX_BYTES`, default **8 GiB**. Eviction is LRU and
never touches a version a live desired deployment still references. On the project
filesystem that default can quietly eat a quota — set it to something matching
what the project actually has.

---

## 5. Does the Application have the registry's credential?

**Open question. Measure it; do not assume it.**

`registry_token_source=cli` shells out to `cdp iam generate-workload-auth-token
--workload-name DE` (`registry/cai.py:161-194`). That chain is **verified in a CAI
Session**. Whether a CAI *Application* container has the `cdp` CLI on `PATH` and a
workload identity configured was not verified in this milestone —
`registry/cai.py:163` says "a CAI Session/Application", and only the Session half
was observed.

Check it from inside the Application's own environment before relying on it:

```
which cdp && cdp iam generate-workload-auth-token --workload-name DE >/dev/null && echo ok
```

If that fails, `LIGHTHOUSE_REGISTRY_TOKEN_SOURCE=env` or `=file` exists for
exactly this case — they hand the refresh problem to whatever supplies the value,
which is a real cost and the reason `cli` is the default. A short-lived JWT pasted
into an Application's environment variables expires and takes the registry
integration down with it.

Startup failures here are loud by design: `main.py:278-284` builds the whole
service graph — registry construction included, which is what resolves a domain
and a credential — inside the `ConfigError` handler, so a missing `cai` extra, an
unnamed domain or a rejected token exits **2** with the actionable message rather
than a raw traceback.

---

## 6. Measure whether a device can reach it

**This is the one thing Phase 7 cannot be designed without**: can a Jetson reach
the app with a bearer token, or does something in front of it want a browser? The
device is purely outbound, so if the ingress gates those routes with Cloudera SSO
the agent receives an HTML login page where it expects JSON.

```
python scripts/probe_app.py --url https://<app-url>
```

If you do not have `<app-url>` yet, §3's *Where the URL comes from* says where it
comes from and why you should copy it rather than assemble it.

Run it **twice**: once from a CAI Session inside the workbench (the baseline — if
it fails there, the app is broken, not the ingress), and once from a laptop with
the **VPN off**, which is the Jetson's vantage point. The second run is the one
that decides. The script is read-only, masks the app host unless you pass
`--show-host`, and distinguishes three failures that look identical in a browser:
SSO in front, a stripped `Authorization` header, and simply not reachable.

| what you see | what it means | what it changes |
|---|---|---|
| 401/403 **JSON** from our own app | reachable, bearer auth intact | nothing; the device points here |
| 302 to an identity provider, or HTML | SSO-gated | the device cannot enroll against this URL |
| 200 on health, 302 on device routes | route-level gating | needs answering before the device half |
| nothing off-VPN | private ingress only | this is a dashboard deployment, not a device endpoint |

The last three are **findings, not failures of this document.** Record what you
saw; the device-side setup (`docs/jetson-setup.md`) depends on which row you are
in.

`probe_app.py` also settles the question recorded at `api/auth.py:22-27` — whether
CML's ingress forwards custom request headers to an Application — by sending the
operator credential both as `X-Lighthouse-Admin-Token` and as
`Authorization: Bearer`, and reporting which survived. The admin surface accepts
three credentials precisely because that was unverified; the probe says whether it
needed to.

---

## 7. Confirm it from outside

```
curl -s https://<app-url>/api/v1/health
```

Expect JSON naming the version, the selected registry and the env — `registry`
should read `cai`, not `fake`. Then, with the admin token:

```
curl -s https://<app-url>/api/v1/models \
  -H 'Authorization: Bearer <LIGHTHOUSE_ADMIN_TOKEN>'
```

A model list means the Application's registry credential works end to end: §5's
open question is closed for your deployment, and the answer belongs in your own
notes rather than in this file.

The `Makefile` verbs work against a deployed app too — they are plain curl:

```
make fleet  API=https://<app-url>/api/v1
```

`TOKEN` is read from `.dev/admin-token`, so either put the deployed token there or
pass `AUTH` yourself.

---

## What this does not cover

Device-side setup — the systemd unit, out-of-band token transport, and bringing up
the ONNX runtime on the Orin — is `deploy/keeper.service` and
`docs/jetson-setup.md`. TensorRT is spec Phase 8 and off this milestone's critical
path.
