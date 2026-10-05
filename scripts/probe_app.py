#!/usr/bin/env python3
"""Probe a Lighthouse control plane deployed as a Cloudera AI Application.

The question this answers is the one thing Phase 7 cannot be designed without:
**can a Jetson reach this app with a bearer token, or does something in front of
it want a browser?** The device is purely outbound -- desired-state GET,
heartbeat POST, artifact GET -- so if CML's ingress gates those with Cloudera
SSO, the agent receives an HTML login page where it expects JSON and the whole
device story has to point somewhere else. Nothing in the repo can answer that by
reading; it has to be measured against a real deployment.

It distinguishes **three failures that look identical from a browser**:

  1. *SSO in front.* A 302 to an identity provider, or HTML where JSON belongs.
  2. *The `Authorization` header is stripped or rewritten.* Subtle and fatal: the
     app answers with its own JSON envelope, so everything looks healthy, but the
     device route reports "device token required" (the no-credential message)
     rather than "invalid device token". The bearer never arrived. `api/auth.py`
     distinguishes those two messages, which is the only reason this is
     detectable from outside.
  3. *Not reachable at all off-VPN.* A private ingress. The app is fine; the
     device just cannot see it.

It also settles the open question recorded at `api/auth.py:21-27` -- "whether
CML's ingress forwards custom request headers to a CAI Application is
unverified" -- by sending an operator credential both ways, in the custom
`X-Lighthouse-Admin-Token` header and as `Authorization: Bearer lha_...`, and
reporting which survived. That is why the admin surface accepts three
credentials; this says whether it needed to.

Survival is read from the *message*, not the status code. A 401 happens both
when the ingress strips the header and when the header arrives carrying the
wrong token, and those need opposite fixes -- no token will solve the first, and
no ingress change will solve the second. `api/auth.py` answers "operator
credential required" only when nothing was presented and "invalid operator
credential" only when it had something to compare, so the two are separable from
outside. An earlier version of this script judged these on `status == 200` alone
and printed "did NOT work" for both cases, which reads as an ingress failure
when a mistyped token is far likelier.

Usage, from wherever you want to measure *from* (that choice is the experiment):

    python scripts/probe_app.py --url https://<app>.<domain>
    python scripts/probe_app.py --url https://<app>.<domain> --admin-token-env LIGHTHOUSE_ADMIN_TOKEN
    python scripts/probe_app.py --url http://127.0.0.1:8900 --show-host   # local baseline

Run it **twice**: once from a CAI Session inside the workbench (the baseline --
if it fails there, the app itself is broken, not the ingress), and once from a
laptop with the VPN **off**, which is the Jetson's vantage point. The second run
is the one that decides.

Three rules this script keeps:

  * **Read-only.** Every request is a GET except one heartbeat POST, and that
    POST carries a deliberately invalid credential so it cannot write. No
    enrollment, no deployment, no state.
  * **It does not guess at credentials.** The device token is the literal
    `lhd_probe.invalid`, which is not a credential and is not meant to work --
    the *rejection* is the measurement. An operator token is used only if you
    name an env var holding one.
  * **The output is safe to paste back.** The app host is a tenant identifier,
    so it is masked to `<app-host>` unless you pass `--show-host`. Redirect
    targets are reported as same-host / different-host plus whether they look
    like an identity provider -- never as a hostname, and never with the query
    string, which on an SSO bounce carries state and client ids.

There is **no default URL**, by the same rule that keeps a default out of
`LIGHTHOUSE_REGISTRY_DOMAIN`: this repo is public and a hostname identifies a
tenant.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

API_PREFIX = "/api/v1"

# Not a credential. It is the right *shape* (`lhd_<id>.<secret>`, per
# `api/auth.py:50`) so it reaches the device authenticator and is rejected on its
# merits, rather than being turned away for malformed syntax. The rejection is
# the whole measurement.
FAKE_DEVICE_TOKEN = "lhd_probe.invalid"

# A device id no enrollment would produce. If this ever matches a real device the
# probe still writes nothing -- the credential above cannot authenticate -- but
# the 401/403 distinction in `api/auth.py:147` would get muddier to read.
PROBE_DEVICE_ID = "probe-not-a-real-device"

_HTML_RE = re.compile(rb"<\s*(!doctype\s+html|html|head|title)\b", re.I)

# Hosts and paths that mean "an identity provider answered". Matched against the
# redirect target only to *classify* it; the target itself is never printed.
_SSO_HINTS = (
    "oauth2", "openid", "saml", "/sso", "sso.", "login", "signin", "sign-in",
    "auth0", "okta", "keycloak", "adfs", "accounts.google", "microsoftonline",
    "knox", "cdp-sso", "altus",
)

_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]+")


def say(*parts: object) -> None:
    print("  " + " ".join(str(p) for p in parts))


# -- output hygiene ----------------------------------------------------------


def mask(text: str, host: str, show_host: bool) -> str:
    """Replace the app host and anything JWT-shaped before printing."""
    out = _JWT_RE.sub("<redacted jwt>", text)
    if not show_host and host:
        out = out.replace(host, "<app-host>")
    return out


def classify_location(location: str, app_host: str) -> str:
    """Describe a redirect target without naming it.

    Where it points is the finding; *what* it points at is tenant data. "a
    different host that looks like an identity provider" is the whole answer, and
    it carries none of the hostname.
    """
    if not location:
        return "3xx with no Location header (unusual -- note it verbatim)"
    try:
        target = urllib.parse.urlsplit(location)
    except ValueError:
        return "an unparsable Location header"
    host = (target.netloc or "").lower()
    lowered = location.lower()
    looks_sso = any(hint in lowered for hint in _SSO_HINTS)
    if not host or host == app_host.lower():
        where = "the same host"
    else:
        where = "a DIFFERENT host"
    return f"{where}{', which looks like an identity provider' if looks_sso else ''}"


# -- one request -------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turn a 3xx into an HTTPError instead of transparently following it.

    Following it is exactly what must not happen. An SSO bounce followed to
    completion returns 200 and an HTML login page, which reads as success; and a
    redirect to an object store would carry the Authorization header somewhere
    it was never meant to go. `registry/cai.py` follows redirects on a separate
    unauthenticated request for that second reason.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 20.0,
) -> dict:
    """Issue one request and report what came back, without raising.

    Returns status (0 on a network-level failure), the response headers, the
    first 2 KiB of the body, and on a failure the exception text. 2 KiB is
    plenty: an error envelope is a few hundred bytes and an HTML login page
    announces itself in its first tag.
    """
    req = urllib.request.Request(url, method=method, data=body)
    req.add_header("Accept", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return {
                "status": resp.status,
                "headers": dict(resp.headers.items()),
                "body": resp.read(2048),
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": exc.code,
            "headers": dict(exc.headers.items()) if exc.headers else {},
            "body": exc.read(2048),
            "error": None,
        }
    except Exception as exc:  # network-level: DNS, TLS, refused, timeout
        return {
            "status": 0,
            "headers": {},
            "body": b"",
            "error": f"{type(exc).__name__}: {exc}",
        }


# -- what answered -----------------------------------------------------------


def who_answered(result: dict) -> tuple[str, dict | None]:
    """Decide whether *our* app produced this response.

    The fingerprint is the error envelope `api/errors.py:131-141` is the sole
    producer of -- `{"code", "message", "detail"}` -- or the health schema. This
    checks for the *presence* of those keys rather than mirroring the slug table
    from `errors.py:109-123`, so the probe cannot drift out of step with it.
    """
    if 300 <= result["status"] < 400:
        return ("a redirect -- see where it points, below", None)
    body = result["body"]
    if not body:
        return ("an empty body", None)
    if _HTML_RE.search(body):
        return ("HTML -- a browser surface, not this API", None)
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ("a non-JSON body", None)
    if isinstance(parsed, list):
        # The collection routes answer with a bare array, so a 2xx array is this
        # app behaving normally. On any other status it is a shape nothing here
        # serves, which is worth saying out loud rather than calling it ours.
        if 200 <= result["status"] < 300:
            return (f"lighthouse (a {len(parsed)}-item JSON array)", None)
        return ("a JSON array, which no error path here produces", None)
    if not isinstance(parsed, dict):
        return ("JSON, but neither an object nor an array", None)
    if {"code", "message"} <= parsed.keys():
        return ("lighthouse (its error envelope)", parsed)
    if {"status", "registry_reachable"} <= parsed.keys():
        return ("lighthouse (its health schema)", parsed)
    return ("JSON from something else", parsed)


def report(label: str, url: str, result: dict, app_host: str, show_host: bool) -> dict:
    """Print one probe step and return it for the verdict pass."""
    print(f"\n--- {label} " + "-" * max(0, 60 - len(label)))
    if result["error"]:
        say("no response:", mask(result["error"], app_host, show_host))
        return result
    source, parsed = who_answered(result)
    say(f"HTTP {result['status']}   answered by {source}")

    ctype = result["headers"].get("Content-Type", "")
    if ctype:
        say("content-type:", ctype.split(";")[0])
    if 300 <= result["status"] < 400:
        say("redirects to", classify_location(result["headers"].get("Location", ""), app_host))
    if result["headers"].get("WWW-Authenticate"):
        say("www-authenticate:", result["headers"]["WWW-Authenticate"])
    if "Set-Cookie" in result["headers"]:
        # Cookie *names* are a strong ingress fingerprint; values never print.
        names = sorted({c.split("=")[0].strip() for c in result["headers"]["Set-Cookie"].split(",")})
        say("sets cookies:", ", ".join(n for n in names if n)[:120])
    if parsed is not None:
        trimmed = {k: parsed[k] for k in list(parsed)[:8]}
        say("body:", mask(json.dumps(trimmed, default=str)[:300], app_host, show_host))
    elif result["body"]:
        say("body:", mask(result["body"][:160].decode(errors="replace"), app_host, show_host))

    result["source"] = source
    result["parsed"] = parsed
    return result


# -- the probe ---------------------------------------------------------------


def probe_health(base: str, app_host: str, show_host: bool) -> dict:
    """The one unauthenticated route (`api/meta.py:17-33`), with no headers.

    If this is gated, nothing else needs testing: the app deliberately leaves
    exactly one route open so a CAI Application has something to probe, and an
    ingress that gates it gates everything.
    """
    return report(
        "GET /health -- unauthenticated, no credential sent",
        f"{base}{API_PREFIX}/health",
        request(f"{base}{API_PREFIX}/health"),
        app_host,
        show_host,
    )


def probe_device_routes(base: str, app_host: str, show_host: bool) -> dict[str, dict]:
    """The three routes the Jetson actually calls, with an invalid device token.

    All three are probed rather than one, because an ingress can gate them
    differently: methods (a POST may be treated as a write), paths, and
    streaming responses are all things a gateway rules on separately. The
    artifact route is the one that streams, and the heartbeat is the only POST
    the device makes.
    """
    auth = {"Authorization": f"Bearer {FAKE_DEVICE_TOKEN}"}
    dev = f"{base}{API_PREFIX}/devices/{PROBE_DEVICE_ID}"
    steps = {}
    steps["desired_state"] = report(
        "GET desired-state -- invalid device token",
        f"{dev}/desired-state",
        request(f"{dev}/desired-state", headers=auth),
        app_host,
        show_host,
    )
    steps["heartbeat"] = report(
        "POST heartbeat -- invalid device token",
        f"{dev}/heartbeat",
        request(
            f"{dev}/heartbeat",
            method="POST",
            headers={**auth, "Content-Type": "application/json"},
            # An empty object. If auth is reached first this is a 401; if body
            # validation wins it is a 422. Either proves the app answered, and
            # neither writes anything.
            body=b"{}",
        ),
        app_host,
        show_host,
    )
    steps["artifact"] = report(
        "GET artifact -- invalid device token",
        f"{dev}/artifact",
        request(f"{dev}/artifact", headers=auth),
        app_host,
        show_host,
    )
    return steps


def probe_header_stripping(base: str, app_host: str, show_host: bool) -> dict:
    """Did the `Authorization` header arrive at all?

    `api/auth.py:135-146` emits two different messages: **"device token
    required"** when no bearer was presented, and **"invalid device token"** when
    one was presented and rejected. So sending a bearer and getting the *former*
    back proves the header did not survive the hop -- an ingress consumed it.

    This is the failure mode worth the most here, because every other signal
    looks healthy while it happens: the app is reachable, it answers in JSON,
    its health is fine, and no device can ever authenticate.
    """
    dev = f"{base}{API_PREFIX}/devices/{PROBE_DEVICE_ID}"
    return report(
        "GET desired-state -- NO Authorization header (control)",
        f"{dev}/desired-state",
        request(f"{dev}/desired-state"),
        app_host,
        show_host,
    )


def probe_operator_headers(
    base: str, token: str, app_host: str, show_host: bool
) -> dict[str, dict]:
    """Which operator credential survives the ingress (settles `api/auth.py:21-27`).

    `GET /devices` is read-only. Two requests, same credential, two transports:
    the custom header and the standard bearer. The admin surface accepts three
    forms *because* this was unverified; this is the measurement that was
    missing.
    """
    url = f"{base}{API_PREFIX}/devices"
    steps = {}
    steps["custom_header"] = report(
        "GET /devices -- operator via X-Lighthouse-Admin-Token",
        url,
        request(url, headers={"X-Lighthouse-Admin-Token": token}),
        app_host,
        show_host,
    )
    steps["bearer"] = report(
        "GET /devices -- operator via Authorization: Bearer lha_...",
        url,
        request(url, headers={"Authorization": f"Bearer {token}"}),
        app_host,
        show_host,
    )
    return steps


def probe_dashboard(base: str, app_host: str, show_host: bool) -> dict:
    """The browser surface. Gated differently from the API on many ingresses, and
    an SSO bounce shows up here first.

    **HTML here is the correct answer** -- this route serves the dashboard page,
    so it is excluded from the SSO verdict. What matters is a *3xx*: a redirect
    on the dashboard while the API routes answer normally means the ingress gates
    the browser surface only, which is the one SSO arrangement the device can
    live with.
    """
    return report(
        "GET / -- the dashboard (HTML expected)", base, request(base), app_host, show_host
    )


# -- the verdict -------------------------------------------------------------


def _ours(step: dict | None) -> bool:
    return bool(step) and str(step.get("source", "")).startswith("lighthouse")


# What `require_operator` says, and what each answer proves about the ingress.
# These strings are the measurement: `api/auth.py` raises "operator credential
# required" only when `presented` is falsy, and "invalid operator credential"
# only when it had something to compare. So the second one means the header
# *arrived* -- a 401 is not evidence of a stripped header, and treating it as
# one is how "did NOT work" gets read as "the ingress dropped it".
_ARRIVED_BUT_REFUSED = "invalid operator credential"
_NEVER_ARRIVED = "operator credential required"

_TRANSPORT_LABEL = {
    "accepted": "FORWARDED and accepted",
    "forwarded": "FORWARDED (arrived; token refused)",
    "stripped": "NOT FORWARDED (never reached the app)",
    "unconfigured": "unknown -- the app has no admin token configured",
    "unknown": "unclear; read the body above",
}


def _transport(step: dict) -> str:
    """Whether an operator credential reached `api/auth.py`, from the outside.

    Status alone cannot tell you: a 401 is returned both when the header was
    stripped in transit and when it arrived carrying the wrong value, and those
    call for opposite fixes -- one is an ingress problem that no token can
    solve, the other is a typo. The message separates them, which is the same
    trick the device verdict above turns on and the reason `api/auth.py` keeps
    the two phrasings distinct.
    """
    if step.get("error") or not _ours(step):
        return "unknown"
    status = step.get("status", 0)
    if status == 200:
        return "accepted"
    if status == 503:
        # "operator authentication is not configured" -- the app refused before
        # looking at the credential, so this says nothing about transport.
        return "unconfigured"
    message = str((step.get("parsed") or {}).get("message", ""))
    if message == _ARRIVED_BUT_REFUSED:
        return "forwarded"
    if message == _NEVER_ARRIVED:
        return "stripped"
    return "unknown"


def verdict(
    health: dict,
    device: dict[str, dict],
    control: dict,
    dashboard: dict,
    operator: dict | None,
) -> int:
    """Name the outcome and what it costs, in the plan's own terms.

    Returns a shell exit code: 0 only when a bearer-token device can actually
    work against this deployment.
    """
    print("\n=== verdict " + "=" * 56)
    desired = device.get("desired_state") or {}

    if health.get("error") and all(s.get("error") for s in device.values()):
        say("UNREACHABLE from here. Nothing answered at all.")
        say("If this also fails from a CAI Session, the app is down. If it")
        say("only fails from outside, the ingress is private: Phase 7 is a")
        say("dashboard deployment and the device must point elsewhere.")
        return 3

    # The dashboard is deliberately not in this list: `GET /` serves HTML
    # because it is a browser page, and a redirect there affects only operators
    # signing in. The device never calls it.
    sso = [
        name
        for name, step in (("health", health), *device.items())
        if 300 <= step.get("status", 0) < 400 or "HTML" in str(step.get("source", ""))
    ]
    if sso:
        say("SSO-GATED (or fronted by something that is not this app).")
        say("Affected:", ", ".join(sorted(set(sso))))
        say("A bearer-token client receives a login page, not JSON. The device")
        say("cannot enroll or heartbeat through this URL. This is the finding,")
        say("not a cut: point the Jetson at a reachable control plane instead.")
        return 4

    if not _ours(desired):
        say("SOMETHING ELSE ANSWERED the device routes. Not SSO, not us.")
        say("Capture the bodies above verbatim before designing around it.")
        return 5

    # Both are 401s from our app. The *message* is what separates them, and it
    # is the only way to see a stripped header from outside.
    with_bearer = str((desired.get("parsed") or {}).get("message", ""))
    without = str((control.get("parsed") or {}).get("message", ""))
    if with_bearer and with_bearer == without:
        say("THE AUTHORIZATION HEADER IS NOT ARRIVING.")
        say(f"Sending a bearer and sending none both answer {with_bearer!r}.")
        say("The app is healthy and no device will ever authenticate. Fix the")
        say("ingress, or carry the device credential in a header the ingress")
        say("forwards -- and re-run this before writing any device code.")
        return 6

    say("REACHABLE, and bearer auth is intact.")
    say("A rejected device token is rejected on its merits, which means a real")
    say("one would be accepted. Proceed with the device commits.")
    if 300 <= dashboard.get("status", 0) < 400:
        say("")
        say("The dashboard redirects while the API does not: the ingress gates")
        say("the browser surface only. Operators sign in through it; the device")
        say("is unaffected. This is the one SSO arrangement Phase 7 survives.")
    if operator:
        custom = _transport(operator["custom_header"])
        bearer = _transport(operator["bearer"])
        say("")
        say("operator credential transport (api/auth.py:21-27):")
        say(f"  X-Lighthouse-Admin-Token  -> {_TRANSPORT_LABEL[custom]}")
        say(f"  Authorization: Bearer     -> {_TRANSPORT_LABEL[bearer]}")
        arrived = {"accepted", "forwarded"}
        if custom in arrived and bearer in arrived:
            say("  Both transports reach api/auth.py. The ingress forwards the")
            say("  custom header as well as the standard one, which is the")
            say("  question api/auth.py:22-27 was left open on -- answered.")
        elif bearer in arrived and custom not in arrived:
            say("  The custom header does not arrive. The bearer fallback is why")
            say("  the admin surface is reachable at all -- keep all three forms.")
        elif custom in arrived and bearer not in arrived:
            say("  The bearer form does not arrive while the custom header does.")
            say("  Note it; the dashboard's cookie exchange depends on neither.")
        if "forwarded" in (custom, bearer) and "accepted" not in (custom, bearer):
            say("")
            say("  Note the credential ARRIVED and was refused on its merits:")
            say("  'invalid operator credential' is what api/auth.py answers when")
            say("  it has a credential to judge, and 'operator credential")
            say("  required' is what it answers when the header never came. So")
            say("  this is the token value, not the ingress -- compare what you")
            say("  passed against the Application's LIGHTHOUSE_ADMIN_TOKEN.")
        if "unconfigured" in (custom, bearer):
            say("")
            say("  The app answered 503: it has NO admin token configured, so its")
            say("  whole operator surface is closed. Under LIGHTHOUSE_ENV=cai that")
            say("  should be fatal at startup -- check the Application's env.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # No default, deliberately. See the module docstring.
    parser.add_argument(
        "--url",
        required=True,
        help="base URL of the deployed control plane, e.g. https://host (no /api/v1)",
    )
    parser.add_argument(
        "--admin-token-env",
        metavar="VAR",
        help="env var holding an operator token (lha_...), to test which header "
        "transport the ingress forwards. Read-only. The value is never printed.",
    )
    parser.add_argument(
        "--show-host",
        action="store_true",
        help="print the real app host instead of <app-host>. Off by default "
        "because the output is meant to be safe to paste; a hostname is a "
        "tenant identifier and this repo is public.",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    split = urllib.parse.urlsplit(base)
    if split.scheme not in ("http", "https") or not split.netloc:
        return _fail(f"--url must be an absolute http(s) URL, got {args.url!r}")
    if base.endswith(API_PREFIX):
        return _fail(f"pass the base URL without {API_PREFIX}; this script appends it")
    app_host = split.netloc

    token = None
    if args.admin_token_env:
        token = (os.environ.get(args.admin_token_env) or "").strip()
        if not token:
            return _fail(f"${args.admin_token_env} is empty or unset")

    print("=== probing a Lighthouse CAI Application ===")
    say("scheme:", split.scheme)
    say("host:  ", app_host if args.show_host else "<app-host> (pass --show-host to reveal)")
    if split.scheme == "http":
        say("^ plain HTTP. A device token on the wire in clear is not a")
        say("  deployment, it is a demo. Fine for a local baseline only.")
    say("run this from a CAI Session AND from a laptop with the VPN off;")
    say("the second run is the one that decides.")

    health = probe_health(base, app_host, args.show_host)
    device = probe_device_routes(base, app_host, args.show_host)
    control = probe_header_stripping(base, app_host, args.show_host)
    dashboard = probe_dashboard(base, app_host, args.show_host)
    operator = (
        probe_operator_headers(base, token, app_host, args.show_host) if token else None
    )

    return verdict(health, device, control, dashboard, operator)


def _fail(message: str) -> int:
    print(f"\n  error: {message}\n", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
