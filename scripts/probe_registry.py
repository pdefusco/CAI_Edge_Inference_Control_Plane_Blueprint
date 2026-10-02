#!/usr/bin/env python3
"""Probe the CAI model registry from inside a CAI Session, to spec the M2 adapter.

A CAI workbench with `endpointPublicAccess: false` resolves its registry to a
private address, so it cannot be called from a laptop. This script runs *in a CAI
Session inside the workbench* and reports the exact wire shapes
`registry/cai.py` has to parse:

  * `GET /api/v2/models`            -> names + `model_id` lineage
  * `GET .../versions`              -> version labels, status, flavors
  * the real `artifact_uri`         -> the prefix shape to parse and resolve
  * what objects actually exist at that prefix (the thing docs don't tell us)

Usage, in a CAI Session terminal. The registry domain is yours, so it is required
rather than defaulted -- this repo is a blueprint and does not carry anyone's
hostnames:

    export LIGHTHOUSE_REGISTRY_DOMAIN=https://modelregistry.<your-workbench>
    python probe_registry.py --token-env CDSW_APIV2_KEY
    python probe_registry.py --domain https://... --token-env CDSW_APIV2_KEY
    python probe_registry.py --token-file /tmp/jwt --token-json-key access_token

The credential is named explicitly, and exactly one is used. This script does
not search the session for usable tokens: enumerating a host's credentials and
trying each against an API is credential scanning whatever the motive, so if the
first choice is rejected the script says so and stops rather than hunting for
another. Which credential the registry wants is a docs/admin question.

Output is deliberately secret-free: every value that looks like a token, key, or
signed URL is redacted before printing, so the output is safe to paste back.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

# No default domain on purpose. A real registry hostname identifies a specific
# tenant's private infrastructure, and this file is in a public repo.
DOMAIN_ENV = "LIGHTHOUSE_REGISTRY_DOMAIN"

# Anything matching these is replaced before printing. The probe's whole value is
# that its output can be pasted into a chat, which is only true if it cannot leak.
_SECRET_KEY_RE = re.compile(
    r"(token|secret|password|passwd|credential|api_?key|authorization|session|"
    r"signature|x-amz-|access_?key)",
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


# -- credentials -------------------------------------------------------------
#
# ONE credential, named explicitly by the operator via --token-env or
# --token-file. This script deliberately does NOT search the environment for
# usable tokens: enumerating every credential on a host and trying each against
# an API is credential scanning, regardless of intent, and it is not a thing a
# repo should carry. If you do not know which credential the registry wants,
# check the CAI docs or ask your admin -- then name it here.


def load_token(args):
    """Return (label, token) from the single source the operator named."""
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
    return None, None


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
        detail = exc.read()[:300].decode(errors="replace")
        return exc.code, {"<error body>": detail}
    except Exception as exc:  # network-level
        return 0, {"<exception>": f"{type(exc).__name__}: {exc}"}


def check_auth(domain, label, token):
    """Report how the named credential fares. One credential, one request."""
    url = f"{domain.rstrip('/')}/api/v2/models"
    print("\n=== auth check ===")
    if token is None:
        status, _ = get_json(url, None)
        print(f"  (no credential given)  -> {status}")
        print("  If this is 401/403, re-run with --token-env VAR or --token-file PATH")
        print("  naming the credential the registry expects.")
        return status
    status, _ = get_json(url, token)
    print(f"  {label}  -> {status}   (value never printed)")
    if status in (401, 403):
        print("  Rejected. That is a useful finding: report it rather than")
        print("  trying other credentials -- the adapter needs the *right* one,")
        print("  which is a docs/admin question, not a guessing game.")
    return status


# -- artifact prefix ---------------------------------------------------------


def probe_artifact_location(uri):
    """List what actually exists at an `s3a://bucket/prefix` artifact URI.

    The earlier investigation found `artifact_uri` is a *prefix*, not always a
    concrete `model.tar.gz`, so the adapter has to resolve prefix -> object. This
    reports the real object layout so that resolution can be written correctly
    instead of guessed.
    """
    print(f"\n--- objects at artifact_uri " + "-" * 37)
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
        print(f"  non-S3 scheme; adapter needs the matching SDK, not boto3")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domain",
        default=os.environ.get(DOMAIN_ENV),
        help=f"registry base URL; defaults to ${DOMAIN_ENV}",
    )
    ap.add_argument("--model", default=None, help="focus a single model name")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--token-env", metavar="VAR", help="env var holding the bearer token")
    src.add_argument("--token-file", metavar="PATH", help="file holding the bearer token")
    ap.add_argument("--token-json-key", metavar="KEY", help="if --token-file is JSON, the key to use")
    args = ap.parse_args()

    if not args.domain:
        ap.error(
            f"no registry domain. Pass --domain or set ${DOMAIN_ENV}.\n"
            "  In a CAI Session it is usually https://modelregistry.<the CDSW_DOMAIN\n"
            "  of this workbench>; check the Model Registry page in the CAI UI."
        )
    domain = args.domain.rstrip("/")
    print("=" * 72)
    print("CAI model registry probe")
    print("=" * 72)
    print(f"domain: {domain}")
    for var in ("CDSW_PROJECT", "CDSW_DOMAIN", "CDSW_ENGINE_ID", "CDSW_APP_PORT"):
        if os.environ.get(var):
            print(f"  {var}={os.environ[var]}")

    label, token = load_token(args)
    status = check_auth(domain, label, token)
    if not (200 <= status < 300):
        print("\nAuth did not succeed, so the shapes below will be empty.")
        print("The status code above is itself the finding -- report it.")

    status, models = get_json(f"{domain}/api/v2/models", token)
    print(f"\n=== GET /api/v2/models -> {status} ===")
    show("models (redacted)", models)

    # The listing's own shape is unknown; try the likely container keys.
    entries = None
    if isinstance(models, dict):
        for key in ("models", "items", "data", "results", "content"):
            if isinstance(models.get(key), list):
                print(f"\n[shape] model list lives under key: {key!r}")
                entries = models[key]
                break
    elif isinstance(models, list):
        print("\n[shape] response is a bare list")
        entries = models
    if not entries:
        print("\nCould not find a model array; paste the block above and stop here.")
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
            print(f"\n[shape] model identifier key: {key!r} = {model_id}")
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
            break
    else:
        print("\nNone of the version paths returned 2xx; paste the codes above.")

    print("\n" + "=" * 72)
    print("Done. This output is redacted and safe to paste back.")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
