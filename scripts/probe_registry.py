#!/usr/bin/env python3
"""Probe a CAI model registry to spec the M2 adapter.

A workbench with `endpointPublicAccess: false` resolves its registry to a private
address, so this runs *in a CAI Session inside that workbench*. It reports the
exact wire shapes `registry/cai.py` has to parse:

  * `GET /api/v2/models`            -> names + `model_id` lineage
  * `GET .../versions`              -> version labels, status, flavors
  * `GET .../artifact`              -> status, Content-Type, gzip magic, and on a
                                       3xx the redirect *host* -- never followed
  * the real `artifact_uri`         -> the prefix shape to parse and resolve
  * what objects actually exist at that prefix (the thing docs don't tell us)
  * the served API spec, including every write route's **request** body, which
    is what a registration has to be written against

Auth follows the documented CAI chain, not guesswork. The registry sits behind
CDP's external-authz gateway (`/gateway/cdpauth/auth/api/v1/extauthz/...`), which
wants a **UMS workload JWT** from the CDP control plane:

    cdp iam generate-workload-auth-token --workload-name DE   ->  ["token"]
    Authorization: Bearer <that token>

`--workload-name DE` is not a typo: the flag accepts only DE/DF/OPDB, and any of
them mints the same general-purpose UMS JWT. There is no `ML` value, which is
what makes this look like a dead end if you read the flag as naming the service
you are calling. A workbench key such as `$CDSW_APIV2_KEY` is scoped to the
workbench API and this gateway returns 401 for it.

The domain is discovered the same way, rather than hardcoded -- a registry lives
on its own `ml-<id>` host, which is *not* the session's `$CDSW_DOMAIN`:

    cdp ml list-model-registries   ->  filter environmentName  ->  ["domain"]

Usage, in a CAI Session terminal (needs `pip install cdpcli` and `cdp configure
set` with your workload user access keys):

    python probe_registry.py --environment my-cdp-env
    python probe_registry.py --environment my-cdp-env --model smoke-test
    python probe_registry.py --environment my-cdp-env --spec-only
    python probe_registry.py --domain https://... --token-env SOME_VAR

Two rules this script keeps:

  * **The token is never printed.** It is minted, passed straight into the
    Authorization header, and never logged, echoed, or written to disk.
  * **One credential, named.** It does not search the session for tokens to try,
    and it does not iterate the registry list looking for one that answers --
    that list is tenant-wide and most entries belong to other people. You name
    the environment; it uses that one and stops.

Output is deliberately secret-free: anything resembling a token, key, or signed
URL is redacted before printing, so the output is safe to paste back.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

DOMAIN_ENV = "LIGHTHOUSE_REGISTRY_DOMAIN"

# Anything matching these is replaced before printing. The probe's whole value is
# that its output can be pasted into a chat, which is only true if it cannot leak.
_SECRET_KEY_RE = re.compile(
    r"(token|secret|password|passwd|credential|api_?key|authorization|session|"
    r"signature|x-amz-|access_?key|private_?key)",
    re.I,
)
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]+")


def redact(obj, _depth=0):
    """Recursively redact secret-ish values. Applied to everything printed."""
    if _depth > 12:
        return "<...>"
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _SECRET_KEY_RE.search(str(k)):
                out[k] = f"<redacted {type(v).__name__}>"
            else:
                out[k] = redact(v, _depth + 1)
        return out
    if isinstance(obj, list):
        return [redact(v, _depth + 1) for v in obj[:8]] + (
            [f"<+{len(obj) - 8} more>"] if len(obj) > 8 else []
        )
    if isinstance(obj, str):
        s = _JWT_RE.sub("<redacted jwt>", obj)
        # Pre-signed URLs carry credentials in the query string.
        if "?" in s and _SECRET_KEY_RE.search(s):
            s = s.split("?", 1)[0] + "?<redacted query>"
        return s if len(s) <= 300 else s[:300] + f"...<+{len(s) - 300} chars>"
    return obj


def show(label, obj):
    print(f"\n--- {label} " + "-" * max(0, 64 - len(label)))
    print(json.dumps(redact(obj), indent=2, default=str)[:4000])


# -- the cdp CLI -------------------------------------------------------------


def _cdp(*args, what: str):
    """Run a `cdp` subcommand and parse its JSON.

    Returns the parsed object, or exits with a readable message. On failure only
    *stderr* is shown: stdout can contain a freshly minted JWT, and this script
    does not print credentials even in error paths.
    """
    if shutil.which("cdp") is None:
        sys.exit(
            "the `cdp` CLI is not on PATH.\n"
            "  In a CAI Session: pip install cdpcli, then\n"
            "  cdp configure set cdp_access_key_id ... / cdp_private_key ..."
        )
    proc = subprocess.run(["cdp", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip()[:400] or f"exit {proc.returncode}"
        sys.exit(f"{what} failed:\n  {detail}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit(f"{what} returned output that is not JSON.")


def mint_workload_jwt(workload_name: str) -> str:
    """Mint a UMS workload JWT. The value is returned, never printed."""
    payload = _cdp(
        "iam", "generate-workload-auth-token",
        "--workload-name", workload_name,
        what="cdp iam generate-workload-auth-token",
    )
    token = payload.get("token")
    if not token:
        sys.exit(
            "the token response had no 'token' field (keys: "
            f"{sorted(payload)}). Check your cdp configuration."
        )
    return str(token).strip()


def discover_domain(environment: str) -> str:
    """Resolve one named environment's registry domain.

    Filters on an exact environmentName the operator supplied. The listing is
    tenant-wide and mostly other people's registries, so there is deliberately
    no "just pick the first one" fallback.
    """
    payload = _cdp("ml", "list-model-registries", what="cdp ml list-model-registries")
    registries = payload.get("modelRegistries") or []
    for reg in registries:
        if reg.get("environmentName") == environment:
            domain = reg.get("domain")
            if not domain:
                sys.exit(f"{environment!r} has a registry but no domain field yet.")
            status = reg.get("status")
            public = reg.get("endpointPublicAccess")
            print(f"  resolved via cdp ml list-model-registries")
            print(f"  status={status}  endpointPublicAccess={public}")
            if status and not str(status).endswith(":finished"):
                print(f"  ^ not finished provisioning; calls below may fail")
            return str(domain)
    names = sorted(r.get("environmentName", "?") for r in registries)
    sys.exit(
        f"no registry for environment {environment!r}.\n"
        f"  {len(names)} registries are visible to you; re-run with one of their\n"
        f"  environment names if {environment!r} is a typo."
    )


# -- credentials -------------------------------------------------------------
#
# ONE credential. By default the documented CDP chain above; otherwise exactly
# the env var or file the operator named. This script deliberately does NOT
# search the environment for usable tokens: enumerating every credential on a
# host and trying each against an API is credential scanning regardless of
# intent, and it is not a thing a repo should carry.


def load_token(args) -> tuple[str, str]:
    """Return (label, token) from the single source the operator chose."""
    if args.token_env:
        val = os.environ.get(args.token_env)
        if not val:
            sys.exit(f"${args.token_env} is empty or unset in this session.")
        return f"env:{args.token_env}", val.strip()
    if args.token_file:
        try:
            with open(args.token_file) as fh:
                raw = fh.read().strip()
        except OSError as exc:
            sys.exit(f"cannot read {args.token_file}: {exc}")
        if not raw:
            sys.exit(f"{args.token_file} is empty.")
        # Some CAI token files are JSON wrapping the bearer value.
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return f"file:{args.token_file}", raw
        if isinstance(parsed, dict):
            if args.token_json_key:
                val = parsed.get(args.token_json_key)
                if not val:
                    sys.exit(f"key {args.token_json_key!r} not in {args.token_file}")
                return f"file:{args.token_file}[{args.token_json_key}]", str(val).strip()
            sys.exit(
                f"{args.token_file} holds JSON with keys: {sorted(parsed)}\n"
                f"Re-run naming one, e.g. --token-json-key access_token"
            )
        return f"file:{args.token_file}", raw
    return (
        f"cdp UMS workload JWT (--workload-name {args.workload_name})",
        mint_workload_jwt(args.workload_name),
    )


def get_json(url, token, timeout=20):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, {"<non-json body>": body[:300].decode(errors="replace")}
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:400].decode(errors="replace")
        return exc.code, {"<error body>": detail}
    except Exception as exc:  # network-level
        return 0, {"<exception>": f"{type(exc).__name__}: {exc}"}


def check_auth(domain, label, token):
    """Report how the chosen credential fares. One credential, one request."""
    url = f"{domain.rstrip('/')}/api/v2/models"
    print("\n=== auth check ===")
    status, body = get_json(url, token)
    print(f"  {label}  -> {status}   (value never printed)")
    if status in (401, 403):
        err = json.dumps(redact(body))
        if "extauthz" in err or "cdpauth" in err:
            print("  The CDP authz gateway rejected it. A workbench-scoped key")
            print("  (CDSW_APIV2_KEY) always 401s here -- it needs the UMS")
            print("  workload JWT, which is this script's default. Drop")
            print("  --token-env/--token-file and re-run.")
        else:
            print("  Rejected. Report the code rather than trying other")
            print("  credentials: the adapter needs the *right* one, which is a")
            print("  docs/admin question, not a guessing game.")
    return status


# -- artifact prefix ---------------------------------------------------------


def probe_artifact_location(uri):
    """List what actually exists at an `s3a://bucket/prefix` artifact URI.

    The earlier investigation found `artifact_uri` is a *prefix*, not always a
    concrete `model.tar.gz`, so the adapter has to resolve prefix -> object. This
    reports the real object layout so that resolution can be written correctly
    instead of guessed.
    """
    print("\n--- objects at artifact_uri " + "-" * 37)
    if not uri:
        print("  (no artifact_uri to probe)")
        return
    m = re.match(r"^(s3a?|abfss?|gs)://([^/]+)/(.*)$", uri)
    if not m:
        print(f"  unrecognized scheme: {redact(uri)}")
        return
    scheme, bucket, prefix = m.group(1), m.group(2), m.group(3)
    print(f"  scheme={scheme}  bucket={bucket}")
    print(f"  prefix={prefix}")
    if scheme not in ("s3", "s3a"):
        print("  non-S3 scheme; adapter needs the matching SDK, not boto3")
        return
    try:
        import boto3  # noqa: PLC0415
    except ImportError:
        print("  boto3 not installed in this session: pip install boto3")
        return
    try:
        client = boto3.client("s3")
        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=50)
    except Exception as exc:
        print(f"  list_objects_v2 FAILED: {type(exc).__name__}: {exc}")
        print("  ^ if this is AccessDenied, the control plane needs its own")
        print("    object-store identity -- note it, it shapes M5.")
        return
    contents = resp.get("Contents") or []
    if not contents:
        print("  prefix exists but is EMPTY (or no permission to list)")
    for obj in contents:
        print(f"  {obj['Size']:>12,}  {obj['Key']}")
    if resp.get("IsTruncated"):
        print("  (truncated)")


# -- the artifact response ---------------------------------------------------


def _first_version(payload):
    """Pull one concrete version label out of whatever the versions call returned.

    `model_versions` is what the spec declares -- versions arrive nested on
    `GET /models/{id}` and there is no list route -- but the alternatives cost
    nothing, and this script exists precisely because a declaration can be wrong.
    """
    entries = payload if isinstance(payload, list) else None
    if isinstance(payload, dict):
        for key in ("model_versions", "versions", "items"):
            val = payload.get(key)
            if isinstance(val, list) and val:
                entries = val
                break
        else:
            inner = payload.get("model")
            if isinstance(inner, dict):
                return _first_version(inner)
    if not entries:
        return None
    first = entries[0]
    if isinstance(first, dict):
        for key in ("version", "model_version", "version_number"):
            if first.get(key) is not None:
                return str(first[key])
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turn a 3xx into an HTTPError instead of transparently following it.

    urllib follows redirects by default, and here that would be actively
    harmful: the redirect target is an object store, and following it would
    carry the workload JWT there. `registry/cai.py` follows it on a *separate*
    request with no Authorization header for exactly that reason.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def probe_artifact_response(domain, model_id, version, token):
    """Report the *shape* of the artifact response, without downloading it.

    This is the largest of the three shapes `registry/cai.py` had to guess at.
    `open_artifact` branches at runtime on the observed status and Content-Type
    -- 200 octet-stream/gzip/tar, 200 multipart, a 3xx to object storage, 400
    for an HF/NGC version -- because the spec declares
    `produces: multipart/form-data`, which is very likely a mislabel for a
    binary download. One real response settles which branch is live.

    Only the first couple of KB are read: enough for the gzip magic number and
    a multipart boundary, nowhere near the whole artifact.
    """
    print("\n=== GET .../artifact -- response shape only ===")
    url = (
        f"{domain.rstrip('/')}/api/v2/models/{model_id}"
        f"/versions/{version}/artifact"
    )
    # `Accept-Encoding: identity` asks for the body undecoded. urllib does not
    # transparently decompress the way httpx does, so this probe is the one
    # place that can see what the registry actually put on the wire -- and the
    # difference matters: if the registry serves the tarball with
    # `Content-Encoding: gzip`, httpx hands the control plane a *bare* tar
    # while `Packaging` still says tar.gz. The SHA-256 then matches end to end,
    # so every checksum passes and the only thing that fails is the unpack, on
    # the device.
    req = urllib.request.Request(url, headers={"Accept": "*/*", "Accept-Encoding": "identity"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=30) as resp:
            status, headers, head = resp.status, resp.headers, resp.read(2048)
    except urllib.error.HTTPError as exc:
        # Both a real error and -- thanks to _NoRedirect -- every 3xx land here.
        status, headers, head = exc.code, exc.headers, exc.read(2048)
    except Exception as exc:  # network-level
        print(f"  request failed: {type(exc).__name__}: {exc}")
        return

    ctype = (headers.get("Content-Type") or "").strip()
    cenc = (headers.get("Content-Encoding") or "").strip()
    print(f"  status         = {status}")
    print(f"  Content-Type   = {ctype!r}")
    print(f"  Content-Length = {headers.get('Content-Length')!r}")
    print(f"  Content-Encoding = {cenc!r}")
    if cenc and cenc.lower() not in ("identity", ""):
        print("  [shape] *** the body is content-encoded. httpx decodes this")
        print("          transparently, so cai.py would cache a decoded body")
        print("          while Packaging still claims tar.gz -- and the digest")
        print("          would match end to end, leaving the unpack on the")
        print("          device as the only thing that fails. Report this line.")

    base = ctype.split(";")[0].strip().lower()
    if 300 <= status < 400:
        # The host, never the URL: a pre-signed URL's query string is a
        # credential, and `redact()` would blank it anyway.
        host = urllib.parse.urlsplit(headers.get("Location") or "").netloc
        print(f"  Location host  = {host or '<none>'}")
        print("  [shape] the registry redirects to object storage. cai.py")
        print("          follows this on a second request carrying no")
        print("          Authorization header. If that leg 403s, the control")
        print("          plane needs its own object-store identity -- the one")
        print("          finding that would pull boto3 into the design.")
    elif status == 400:
        print("  [shape] 400 is the spec's HF/NGC case: no brokered bytes for a")
        print("          version the edge could not run anyway. cai.py already")
        print("          maps this to UnsupportedFlavor.")
    elif status in (401, 403):
        print("  [shape] the artifact route rejected the same token the listing")
        print("          accepted -- report this, it changes the byte path.")
    elif 200 <= status < 300:
        if base.startswith("multipart/"):
            print("  [shape] multipart, so the spec's `produces` was literal and")
            print("          cai.py's multipart branch is the live one.")
        elif base in ("application/octet-stream", "application/gzip", "application/x-tar"):
            print("  [shape] a plain binary body -- cai.py streams it through.")
        else:
            print(f"  [shape] unexpected content-type {base!r}; cai.py falls back")
            print("          to streaming it as raw bytes and warns.")
        if head[:2] == b"\x1f\x8b":
            print("  [shape] body starts with the gzip magic (1f 8b), which is")
            print("          what Packaging.MLFLOW_TAR_GZ assumes.")
        else:
            print(f"  [shape] body does NOT start with gzip magic: {head[:8]!r}")
            print("          -> Packaging.MLFLOW_TAR_GZ would be wrong.")


def probe_api_spec(domain, token):
    """Ask the registry to describe itself.

    This is the one probe that works against an *empty* registry, which matters
    a lot: the version and artifact_uri shapes are the whole reason this script
    exists, and with no models registered there is no payload to read them off.
    A served OpenAPI document gives the adapter its field names and types
    without needing a single model to exist first.
    """
    print("\n=== looking for a served API spec ===")
    for path in (
        "/api/v2/openapi.json",
        "/openapi.json",
        "/api/v2/swagger.json",
        "/swagger.json",
        "/api/v2/api-docs",
        "/v2/api-docs",
        "/api/v2/docs",
    ):
        status, body = get_json(f"{domain}{path}", token)
        marker = isinstance(body, dict) and (
            "openapi" in body or "swagger" in body or "paths" in body
        )
        print(f"  {path} -> {status}{'  [spec]' if marker else ''}")
        if not (200 <= status < 300 and marker):
            continue

        # The prefix matters more than it looks: the spec declares paths like
        # `/models` while `/api/v2/models` is what actually answers, so the
        # adapter's URL construction comes from basePath, not from the paths.
        print("\n[spec] version/prefix:")
        for key in ("swagger", "openapi", "basePath"):
            if body.get(key):
                print(f"  {key} = {body[key]}")
        for srv in (body.get("servers") or [])[:3]:
            print(f"  server.url = {srv.get('url')}")
        if not body.get("basePath") and not body.get("servers"):
            print("  (no basePath/servers declared -- prefix is the gateway's)")

        paths = body.get("paths") or {}
        print(f"\n[spec] all {len(paths)} paths:")
        for p in sorted(paths):
            verbs = ",".join(sorted(
                v.upper() for v in paths[p] if v.lower() in
                ("get", "post", "put", "patch", "delete")
            ))
            print(f"  {verbs:<18} {p}")

        # The artifact route decides whether the control plane can broker bytes
        # through the registry API or needs its own object-store identity, which
        # is a load-bearing difference for the byte proxy. So print its full
        # response contract rather than just its existence.
        for p in sorted(paths):
            if p.endswith("/artifact"):
                print(f"\n[spec] {p} -- response contract:")
                spec_get = (paths[p].get("get") or {})
                for code, resp in sorted((spec_get.get("responses") or {}).items()):
                    desc = (resp.get("description") or "").strip()[:60]
                    schema = resp.get("schema") or (
                        next(iter((resp.get("content") or {}).values()), {}) or {}
                    ).get("schema") or {}
                    ref = schema.get("$ref", "").split("/")[-1]
                    typ = schema.get("type") or ref or ""
                    ctypes = ",".join((resp.get("content") or {}).keys())
                    print(f"  {code}  {desc}  {typ} {ctypes}".rstrip())
                produces = spec_get.get("produces")
                if produces:
                    print(f"  produces: {', '.join(produces)}")
                for prm in (spec_get.get("parameters") or []):
                    print(f"  param: {prm.get('name')} in={prm.get('in')}")

        # The *request* contracts, which this probe never printed -- it only ever
        # showed responses. M3 has to create a model and a version, so which body
        # each write route wants is the thing that has to be known before any
        # registration code is written. Handles both Swagger 2.0 (a `body`
        # parameter carrying `schema.$ref`) and OpenAPI 3 (`requestBody`).
        print("\n[spec] write routes -- request contracts:")
        for p in sorted(paths):
            for verb in ("post", "put", "patch"):
                op = paths[p].get(verb)
                if not op:
                    continue
                print(f"\n  {verb.upper()} {p}")
                for prm in (op.get("parameters") or []):
                    ref = (prm.get("schema") or {}).get("$ref", "").split("/")[-1]
                    where = prm.get("in")
                    req_mark = "*" if prm.get("required") else " "
                    print(
                        f"   {req_mark}param {prm.get('name')} in={where}"
                        + (f" schema={ref}" if ref else "")
                    )
                for ctype, node in ((op.get("requestBody") or {}).get("content") or {}).items():
                    schema = node.get("schema") or {}
                    ref = schema.get("$ref", "").split("/")[-1]
                    print(f"    body {ctype} schema={ref or schema.get('type') or '?'}")

        # The *read* routes' query parameters, which this probe also never
        # printed. This block is the cheapest thing in M3: the pagination
        # request parameter name is one of the three open M2 decisions, it is
        # currently a guess in `cai.py` (`page_token`, degrading to
        # first-page-only), and it is declared right here in a document the
        # probe was already fetching and throwing away. Confirming it needs no
        # registration, no second model and no write of any kind.
        #
        # Printed for every GET with query parameters rather than just the
        # listings, because `page_size` being ignored would matter too and
        # costs nothing extra to see.
        print("\n[spec] read routes -- query parameters:")
        for p in sorted(paths):
            op = paths[p].get("get")
            if not op:
                continue
            query = [
                prm for prm in (op.get("parameters") or [])
                if prm.get("in") == "query"
            ]
            if not query:
                continue
            print(f"\n  GET {p}")
            for prm in query:
                schema = prm.get("schema") or {}
                typ = prm.get("type") or schema.get("type") or "?"
                req_mark = "*" if prm.get("required") else " "
                default = prm.get("default", schema.get("default"))
                extra = f" default={default!r}" if default is not None else ""
                print(f"   {req_mark}{prm.get('name')}: {typ}{extra}")

        schemas = (body.get("components") or {}).get("schemas") or body.get("definitions") or {}

        def fmt(meta):
            """Render a property's type, following one $ref and showing enums."""
            ref = meta.get("$ref", "").split("/")[-1]
            typ = meta.get("type") or ref or "?"
            if typ == "array":
                inner = meta.get("items") or {}
                iref = inner.get("$ref", "").split("/")[-1]
                typ = f"array<{inner.get('type') or iref or '?'}>"
            if meta.get("enum"):
                typ += "  enum=" + ",".join(str(e) for e in meta["enum"][:12])
            if meta.get("format"):
                typ += f"  ({meta['format']})"
            return typ

        # Every schema, not a filtered subset. The earlier pass filtered on
        # name and so hid Status's enum and the MLFlow metadata -- i.e. exactly
        # the fields the adapter needs for readiness and for the entrypoint.
        print(f"\n[spec] all {len(schemas)} schemas:")
        for name in sorted(schemas):
            node = schemas[name]
            props = node.get("properties") or {}
            required = set(node.get("required") or [])
            if not props:
                # Bare enums (Status is likely one) have no properties.
                line = fmt(node)
                print(f"\n  {name}: {line}")
                continue
            print(f"\n  {name}:")
            for field, meta in sorted(props.items()):
                mark = "*" if field in required else " "
                print(f"   {mark}{field:<30} {fmt(meta)}")
        return True
    print("  no spec served at the usual paths")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--environment", metavar="NAME",
                    help="CDP environment whose registry to probe (resolves --domain)")
    ap.add_argument("--domain", default=os.environ.get(DOMAIN_ENV),
                    help=f"registry base URL; else ${DOMAIN_ENV}, else --environment")
    ap.add_argument("--model", default=None, help="focus a single model name")
    ap.add_argument("--spec-only", action="store_true",
                    help="dump the served API spec and stop (no model reads)")
    ap.add_argument("--workload-name", default="DE", choices=("DE", "DF", "OPDB"),
                    help="workload name for the UMS token mint (any mints the same JWT)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--token-env", metavar="VAR", help="use this env var instead of minting")
    src.add_argument("--token-file", metavar="PATH", help="use this file instead of minting")
    ap.add_argument("--token-json-key", metavar="KEY", help="if --token-file is JSON, the key")
    args = ap.parse_args()

    print("=" * 72)
    print("CAI model registry probe")
    print("=" * 72)
    for var in ("CDSW_PROJECT", "CDSW_DOMAIN", "CDSW_ENGINE_ID"):
        if os.environ.get(var):
            print(f"  {var}={os.environ[var]}")

    if not args.domain:
        if not args.environment:
            ap.error(
                f"need a registry. Pass --environment NAME (recommended; resolves\n"
                f"  the domain via cdp ml list-model-registries), or --domain URL,\n"
                f"  or set ${DOMAIN_ENV}."
            )
        print(f"\n=== resolving the registry for {args.environment!r} ===")
        args.domain = discover_domain(args.environment)
    domain = args.domain.rstrip("/")
    print(f"\ndomain: {domain}")
    if os.environ.get("CDSW_DOMAIN") and os.environ["CDSW_DOMAIN"] not in domain:
        print("  (note: a different host from $CDSW_DOMAIN -- expected; the")
        print("   registry is its own workspace)")

    label, token = load_token(args)
    status = check_auth(domain, label, token)
    if not (200 <= status < 300):
        print("\nAuth did not succeed, so the shapes below will be empty.")
        print("The status code above is itself the finding -- report it.")

    # The spec dump runs whether or not the registry has models. It used to live
    # inside the empty-registry branch below, as the consolation prize for having
    # nothing to read shapes off -- which meant it silently stopped happening the
    # moment the first model was registered, i.e. exactly when the write-route
    # request schemas are wanted.
    found_spec = probe_api_spec(domain, token)
    if args.spec_only:
        print("\n--spec-only: stopping before any model reads.")
        return

    status, models = get_json(f"{domain}/api/v2/models", token)
    print(f"\n=== GET /api/v2/models -> {status} ===")
    show("models (redacted)", models)

    # The listing's own shape is unknown; try the likely container keys. Note the
    # `key in models` test rather than a truthiness test: an empty registry
    # answers `{"models": null}`, so the key being *present and null* tells us
    # the container name even with no data -- and warns the adapter that this
    # field is nullable, which would otherwise be found the hard way by
    # iterating None in production.
    entries = None
    if isinstance(models, dict):
        for key in ("models", "items", "data", "results", "content"):
            if key in models:
                val = models[key]
                if isinstance(val, list):
                    print(f"\n[shape] model list lives under key: {key!r}")
                    entries = val
                elif val is None:
                    print(f"\n[shape] key {key!r} is present but NULL, not []")
                    print("        -> the registry is empty, and the adapter must")
                    print("           coerce null to an empty list rather than")
                    print("           iterating it.")
                break
    elif isinstance(models, list):
        print("\n[shape] response is a bare list")
        entries = models

    if not entries:
        print("\n" + "=" * 72)
        if found_spec:
            print("Registry is empty, but its API spec is above -- that is enough")
            print("to write the adapter's parsing against. Paste it back.")
        else:
            print("Registry is empty and serves no spec, so the version and")
            print("artifact_uri shapes cannot be learned yet: register one model")
            print("(M3) and re-run to capture them.")
        print("=" * 72)
        return

    print(f"\n[shape] {len(entries)} model(s); keys on first entry:")
    if isinstance(entries[0], dict):
        print("  " + ", ".join(sorted(entries[0].keys())))

    target = None
    if args.model:
        for e in entries:
            if isinstance(e, dict) and args.model in (e.get("name"), e.get("model_name")):
                target = e
                break
        if target is None:
            print(f"\n{args.model!r} not found; falling back to the first model.")
    target = target or entries[0]
    show("first/target model", target)

    model_id = None
    for key in ("id", "model_id", "modelId", "uuid", "crn"):
        if target.get(key):
            model_id = target[key]
            print(f"\n[shape] model identifier key: {key!r}")
            break
    if not model_id:
        print("\nNo id-like key on the model entry; paste the block above.")
        return

    # Versions: the path is one of the open questions, so try the plausible ones.
    for path in (
        f"/api/v2/models/{model_id}/versions",
        f"/api/v2/models/{model_id}",
        f"/api/v2/registry/models/{model_id}/versions",
    ):
        status, versions = get_json(f"{domain}{path}", token)
        print(f"\n=== GET {path} -> {status} ===")
        if 200 <= status < 300:
            show("versions (redacted)", versions)
            blob = json.dumps(versions)
            uris = re.findall(r"(?:s3a?|abfss?|gs)://[^\"\\s]+", blob)
            if uris:
                print(f"\n[shape] artifact_uri values found: {len(uris)}")
                for u in uris[:3]:
                    print(f"  {u}")
                probe_artifact_location(uris[0])
            else:
                print("\n[shape] no object-store URI in the versions payload --")
                print("  the adapter may need a separate artifact-resolution call.")

            # The artifact route is the one shape the adapter had to guess at
            # outright, and it needs a concrete version to ask about.
            version = _first_version(versions)
            if version is None:
                print("\n[shape] no version label found in the payload above, so")
                print("        the artifact route cannot be probed. Paste the")
                print("        block and the adapter's guess stands for now.")
            else:
                probe_artifact_response(domain, model_id, version, token)
            break
    else:
        print("\nNone of the version paths returned 2xx; paste the codes above.")

    print("\n" + "=" * 72)
    print("Done. This output is redacted and safe to paste back.")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
