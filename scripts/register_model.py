#!/usr/bin/env python3
"""Register the smoke-test model, from inside a Cloudera AI Session.

M3's goal is one real model in the AI Registry, so three fail-safe guesses in
`registry/cai.py` can be replaced with observed behaviour. This script puts it
there and prints exactly what those three decisions need.

## Run it in a Session, not on a laptop

The registry's DNS resolves to a private address with `endpointPublicAccess:
false`. There is no route to it from a laptop, VPN or not. Everything here that
touches the registry has to run from a Session inside the workbench.

    pip install onnx mlflow cdpcli        # none of these is a repo dependency
    python scripts/register_model.py --environment <YOUR-ENV> --dry-run
    python scripts/register_model.py --environment <YOUR-ENV>

`--dry-run` builds and validates the graph, resolves the domain, reports which
registration mechanism this workbench offers, and writes nothing. Run it first.

## Why MLflow, and not the REST API

This is forced, not preferred. Two independent constraints:

  * `cai.py:539-563` derives `ArtifactFormat` from registry metadata alone. Only
    an MLflow-sourced version yields a deployable `ONNX`; Hugging Face and NGC
    both yield `UNKNOWN`, which `cai.py:910-913` refuses with
    `UnsupportedFlavor`.
  * The registry REST API has no byte-upload endpoint at all. `POST
    /models/{id}/versions` takes a `downloadModelRepoRequest` whose `source` is
    one of NGC, HF or REMOTE -- none of which is "here are the bytes".

So the only path that produces a deployable version is MLflow from inside a
Session. `ModelRegistry` (`registry/base.py`) is read-only by design and will
not grow a `register()`, which is why this lives in `scripts/` and not behind
the protocol seam.

## The registration entry point is discovered, not assumed

Whether CAI's AI Registry is fed by `cmlapi` or by MLflow's own model registry
differs by workbench version, and guessing wrong wastes an irreversible write.
So `--dry-run` reports what this workbench actually offers and the real run uses
it, rather than this file hard-coding a mechanism it cannot verify from a
laptop. If neither mechanism is present the script stops and says so: that would
mean M3's premise does not hold for this workbench, which is a scope question
and not something to paper over.

## Credential discipline, inherited from probe_registry.py

The helpers are imported from `probe_registry` rather than copied, so there is
exactly one redaction regex in the repo and one definition of the auth chain. A
second copy of a redaction regex is a leak waiting to happen.

  * One credential, the one the operator named. No scanning the Session for
    usable tokens.
  * The domain is resolved from a named environment with no "pick the first
    one" fallback -- that listing is tenant-wide.
  * The JWT is returned, never printed, and never interpolated into an
    exception message.
  * On a non-zero `cdp` exit, only stderr is shown and only truncated: stdout
    can carry a freshly minted token.
  * Everything printed goes through `redact()` first.

## What it prints, and what you must strip before pasting anywhere public

The findings block is the deliverable. It prints the version's `metadata`, its
`status`, and -- the crux -- the `ArtifactFormat` that `_format_and_packaging`
*would* derive from that metadata. A model can be `READY` in the registry and
still undeployable in Lighthouse; that combination is exactly the empty-metadata
failure, and this is the line that reveals it.

It cannot redact identifiers it has to print to be useful. The **registry
domain, model_id, and the s3a:// bucket name are tenant identifiers**. Strip
them before the output leaves a private channel, and never commit them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# One redaction regex, one auth chain, one `cdp` failure discipline for both
# scripts. Imported rather than duplicated; `probe_registry` guards its own
# entry point with `if __name__ == "__main__"`, so importing it runs nothing.
from probe_registry import (  # noqa: E402
    DOMAIN_ENV,
    check_auth,
    discover_domain,
    get_json,
    load_token,
    redact,
    show,
)

from build_model import (  # noqa: E402
    INPUT_SHAPE,
    OPSET,
    OUTPUT_SHAPE,
    build_minimal_onnx,
)

DEFAULT_MODEL_NAME = "smoke-test"
DEFAULT_EXPERIMENT = "lighthouse-smoke"

# Terminal registry states, from the status mapping at `cai.py:893-909`. Only
# READY is a success; the rest are reported verbatim and exit non-zero.
_READY = {"READY"}
_TERMINAL_BAD = {"UPLOAD_FAILED", "DELETE_FAILED", "FAILED", "REGISTRATION_FAILED"}


# --------------------------------------------------------------------------
# the one rule this script has to replicate rather than import
# --------------------------------------------------------------------------


def would_be_deployable(metadata) -> tuple[str, str]:
    """Replicate `cai.py:_format_and_packaging` (lines 539-563) exactly.

    Deliberately a copy and not an import. This runs in a CAI Session where the
    `lighthouse` package is very likely not installed, and a script importing
    the server package would invert the layering `registry/base.py:5` sets up.
    The cost is that this function can drift from the original -- so it is kept
    to a literal transcription, with the line reference above, and the findings
    block prints the raw metadata next to the verdict so a human can check the
    derivation rather than trusting this copy.
    """
    if not isinstance(metadata, dict):
        return "UNKNOWN", "raw_file"
    repo_type = str(metadata.get("model_repo_type") or "").strip().lower()
    has_mlflow = "mlflowMetadata" in metadata or "mlflow_metadata" in metadata
    has_hf = "huggingface_metadata" in metadata
    has_ngc = "ngc_metadata" in metadata

    if repo_type == "mlflow" or (has_mlflow and not has_hf and not has_ngc):
        return "ONNX", "mlflow_tar_gz"
    if repo_type in {"hf", "huggingface", "ngc"} or has_hf or has_ngc:
        return "UNKNOWN", "raw_file"
    return "UNKNOWN", "raw_file"


# --------------------------------------------------------------------------
# preflight: what does this workbench actually offer?
# --------------------------------------------------------------------------


def _probe_import(name: str):
    try:
        return __import__(name)
    except Exception:  # ImportError, but also a broken install raising elsewhere
        return None


def discover_mechanism() -> dict:
    """Report which registration mechanisms exist, without using any of them.

    Read-only: imports modules and inspects attribute names. Nothing here
    contacts the registry, mints a credential or writes.
    """
    print("\n=== preflight: what this workbench offers ===")
    found: dict = {"mlflow": None, "cmlapi": None, "cmlapi_methods": [], "onnx": None}

    onnx_mod = _probe_import("onnx")
    found["onnx"] = getattr(onnx_mod, "__version__", None) if onnx_mod else None
    print(f"  onnx           {found['onnx'] or '<not installed>'}")

    mlflow_mod = _probe_import("mlflow")
    found["mlflow"] = getattr(mlflow_mod, "__version__", None) if mlflow_mod else None
    print(f"  mlflow         {found['mlflow'] or '<not installed>'}")

    # onnxruntime matters for a reason that is not obvious:
    # `mlflow.onnx.get_default_pip_requirements()` pins onnxruntime, which means
    # it *imports* it. In a Session without onnxruntime, `log_model` can raise
    # before writing anything. This script passes `pip_requirements` explicitly
    # to avoid that path, so a missing onnxruntime here is informational.
    ort = _probe_import("onnxruntime")
    print(
        f"  onnxruntime    {getattr(ort, '__version__', None) or '<not installed>'}"
        "   (not required: pip_requirements is passed explicitly)"
    )

    cmlapi_mod = _probe_import("cmlapi")
    found["cmlapi"] = getattr(cmlapi_mod, "__version__", "present") if cmlapi_mod else None
    if cmlapi_mod is None:
        print("  cmlapi         <not installed>")
    else:
        try:
            client = cmlapi_mod.default_client()
            found["cmlapi_methods"] = sorted(
                m
                for m in dir(client)
                if "registered_model" in m.lower() or "model_version" in m.lower()
            )
        except Exception as exc:
            print(f"  cmlapi         present, but default_client() failed: {type(exc).__name__}")
            found["cmlapi_methods"] = []
        else:
            print("  cmlapi         present; registry-shaped methods:")
            for m in found["cmlapi_methods"] or ["<none>"]:
                print(f"                   {m}")

    if mlflow_mod is not None:
        # Booleans, not values: a CAI tracking URI embeds the workbench
        # hostname, which is a tenant identifier this script must not print.
        try:
            print(
                f"  mlflow tracking URI set: {bool(mlflow_mod.get_tracking_uri())}"
                f"   registry URI set: {bool(mlflow_mod.get_registry_uri())}"
            )
        except Exception as exc:
            print(f"  mlflow URIs unreadable: {type(exc).__name__}")

    if found["cmlapi_methods"]:
        found["mechanism"] = "cmlapi"
    elif found["mlflow"]:
        found["mechanism"] = "mlflow"
    else:
        found["mechanism"] = None
    print(f"\n  -> mechanism: {found['mechanism'] or 'NONE FOUND'}")
    return found


# --------------------------------------------------------------------------
# build and log
# --------------------------------------------------------------------------


def log_to_mlflow(model_name: str, experiment: str) -> tuple[str, str, str]:
    """Log the graph as an MLflow run artifact.

    Returns `(run_id, artifact_path, experiment_id)`. The experiment id is
    returned because `cmlapi.CreateRegisteredModelRequest` asks for it
    alongside the run id -- confirmed from the generated client's
    `attribute_map`, not assumed.

    `pip_requirements` is passed explicitly so MLflow does not call
    `get_default_pip_requirements()`, which imports onnxruntime and can fail a
    Session that does not have it. `input_example` is deliberately omitted:
    mlflow >= 2.9 validates an example by predicting with it, which needs a
    runtime and turns a metadata convenience into a hard dependency.
    """
    import mlflow
    import mlflow.onnx
    import onnx

    model_proto = onnx.load_model_from_string(build_minimal_onnx())

    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        experiment_id = str(run.info.experiment_id)
        kwargs = dict(
            onnx_model=model_proto,
            pip_requirements=["onnx"],
        )
        # mlflow 2.x takes `artifact_path`; 3.x renamed it to `name` and warns
        # or errors on the old one. Try the new name first, fall back.
        try:
            mlflow.onnx.log_model(name="model", **kwargs)
        except TypeError:
            mlflow.onnx.log_model(artifact_path="model", **kwargs)
        mlflow.set_tag("lighthouse.purpose", "m3-plumbing-proof")
        mlflow.log_param("opset", OPSET)
        mlflow.log_param("input_shape", str(list(INPUT_SHAPE)))
        mlflow.log_param("output_shape", str(list(OUTPUT_SHAPE)))
    print(f"  logged run {run_id} (experiment {experiment_id}) with artifact_path 'model'")
    return run_id, "model", experiment_id


PROJECT_ID_ENVS = ("CDSW_PROJECT_ID", "CML_PROJECT_ID")


def resolve_project_id(explicit: str | None) -> str:
    """Find the Session's own project id, for `CreateRegisteredModelRequest`.

    Not a tenant secret the way a hostname is, but still read from the
    environment rather than guessed, and never defaulted: registering into the
    wrong project would put a version somewhere the operator did not ask for.
    On failure this lists the *names* of the Session's CDSW/CML variables --
    names only, because their values are this workbench's business.
    """
    if explicit:
        return explicit
    for var in PROJECT_ID_ENVS:
        val = os.environ.get(var)
        if val:
            print(f"  project_id from ${var}")
            return val.strip()
    candidates = sorted(
        k for k in os.environ if k.startswith(("CDSW_", "CML_")) and "PROJECT" in k
    )
    sys.exit(
        "cmlapi needs a project_id and none of "
        f"{', '.join('$' + v for v in PROJECT_ID_ENVS)} is set.\n"
        f"  project-ish variables present here: {candidates or ['<none>']}\n"
        "  Pass --project-id, or use --mechanism mlflow."
    )


def register(
    mechanism: str,
    model_name: str,
    run_id: str,
    artifact_path: str,
    *,
    experiment_id: str = "",
    project_id: str = "",
) -> str | None:
    """Promote the logged run artifact into the AI Registry.

    Returns the registry's version label if it reported one. Both paths print
    what they called, so a failure can be read against the workbench's own API
    rather than against this script's assumptions.
    """
    if mechanism == "mlflow":
        import mlflow

        uri = f"runs:/{run_id}/{artifact_path}"
        print(f"  mlflow.register_model({uri!r}, {model_name!r})")
        result = mlflow.register_model(uri, model_name)
        return str(getattr(result, "version", "") or "") or None

    import cmlapi

    client = cmlapi.default_client()
    # Signature varies by workbench version, so build the call from whatever
    # the client exposes rather than from a remembered signature.
    fn = getattr(client, "create_registered_model", None)
    if fn is None:
        sys.exit(
            "cmlapi has registry-shaped methods but not `create_registered_model`.\n"
            "Re-run with --dry-run and paste the method list: the call has to be\n"
            "written against this workbench's real signature, not a guess."
        )
    # Field names come from `CreateRegisteredModelRequest.attribute_map` as
    # read off a real workbench (cmlapi, mlflow 3.16):
    #   project_id, experiment_id, run_id, model_path, model_name, tags,
    #   description, notes, visibility
    # Note `model_name`, not `name` -- an earlier version of this script
    # guessed `name` and would have been rejected or silently dropped.
    body = {
        "project_id": project_id,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "model_path": artifact_path,
        "model_name": model_name,
        "visibility": "private",
    }
    # Build the typed request when the class is there, so the generated
    # client validates the field names instead of posting a dict the server
    # may quietly ignore.
    req_cls = getattr(cmlapi, "CreateRegisteredModelRequest", None)
    payload: Any = body
    if req_cls is not None:
        try:
            payload = req_cls(**body)
        except TypeError as exc:
            sys.exit(
                f"CreateRegisteredModelRequest rejected these fields: {exc}\n"
                "  This workbench's cmlapi differs from the one this body was\n"
                "  written against. Re-run the inspect snippet in the module\n"
                "  docstring and paste the attribute_map, or use\n"
                "  --mechanism mlflow."
            )
    print(f"  cmlapi create_registered_model({redact(body)})")
    result = fn(payload)
    # `RegisteredModel.model_versions` is a list; the version number lives on
    # `RegisteredModelVersion.number`. Both shapes are tried because this is
    # the one return value that has not been observed yet.
    versions = getattr(result, "model_versions", None) or []
    if versions:
        first = versions[0]
        for attr in ("number", "version_name", "model_version_id"):
            val = getattr(first, attr, None)
            if val:
                return str(val)
    version = getattr(result, "model_version", None)
    return str(getattr(version, "version", "") or "") or None


# --------------------------------------------------------------------------
# read back over REST, with the same credential the control plane will use
# --------------------------------------------------------------------------


def find_model_id(domain: str, token: str, model_name: str) -> str | None:
    """Locate the registered model by name over the v2 REST API.

    Uses the UMS workload JWT rather than the Session's own identity on
    purpose: this is the exact credential and the exact route the control plane
    will use, so a mismatch here is a finding rather than a surprise later.
    """
    status, body = get_json(f"{domain.rstrip('/')}/api/v2/models?page_size=200", token)
    if status != 200 or not isinstance(body, dict):
        print(f"  listing returned {status}")
        show("listing response", body)
        return None
    for entry in body.get("models") or []:
        if entry.get("name") == model_name:
            # `id`, not `model_id` -- confirmed against the registry's swagger.
            return str(entry.get("id") or "") or None
    return None


def poll_until_terminal(
    domain: str, token: str, model_id: str, version: str, wait_seconds: int
) -> dict | None:
    """Poll one version until its status is terminal or the budget runs out.

    Never claims success on a non-READY status. A script that printed
    "registered" over a `REGISTERING` version would send the operator to the
    deploy gate for a 409 with no visible cause.
    """
    url = f"{domain.rstrip('/')}/api/v2/models/{model_id}/versions/{version}"
    deadline = time.monotonic() + wait_seconds
    last = None
    delay = 2.0
    while True:
        status, body = get_json(url, token)
        if status != 200 or not isinstance(body, dict):
            print(f"  version read returned {status}")
            show("version response", body)
            return None
        payload = body.get("model_version") or body
        last = payload
        state = str(payload.get("status") or "").upper()
        print(f"  status={state or '<none>'}")
        if state in _READY or state in _TERMINAL_BAD:
            return last
        if time.monotonic() >= deadline:
            print(f"  gave up after {wait_seconds}s; last status was {state or '<none>'}")
            return last
        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
        delay = min(delay * 1.5, 15.0)


# --------------------------------------------------------------------------


def report(version_payload: dict) -> bool:
    """The findings block. Returns True if the version is actually deployable."""
    metadata = version_payload.get("metadata")
    status = str(version_payload.get("status") or "").upper()
    fmt, packaging = would_be_deployable(metadata)

    print("\n=== M3 findings ===")
    print(f"  status                   {status or '<none>'}")
    print(f"  metadata                 {json.dumps(redact(metadata), default=str)[:600]}")
    if isinstance(metadata, dict):
        print(f"  metadata.model_repo_type {metadata.get('model_repo_type')!r}")
        print(f"  metadata keys            {sorted(metadata)}")
    print(f"  _format_and_packaging -> {fmt} / {packaging}")
    show("the whole version object", version_payload)

    deployable = status in _READY and fmt == "ONNX"
    print("\n=== what this settles ===")
    if status not in _READY:
        print(f"  status is {status!r}, not READY. cai.py:893-909 refuses this at the")
        print("  deploy gate. Nothing downstream is settled until it is READY.")
    if fmt != "ONNX":
        print("  *** metadata did NOT yield ONNX. This is the empty-metadata finding:")
        print("      cai.py:539-563 makes every real model undeployable, and that")
        print("      line is the one to revisit. The model will appear in the")
        print("      catalog with deployable=false and reason")
        print('      "format unknown is not runnable at the edge" (catalog.py:73).')
    else:
        print("  metadata yields ONNX / mlflow_tar_gz, so cai.py:539-563 is correct")
        print("  as written and needs no change.")
    print("\n  Still to observe, and this script cannot: the /artifact response")
    print("  shape. Run probe_registry.py next -- it reports status,")
    print("  Content-Type and Content-Encoding without following the redirect.")
    return deployable


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Register the M3 smoke-test model from a CAI Session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--environment",
        help="CDP environment name whose registry to use; resolved via "
        "`cdp ml list-model-registries` with no first-one fallback",
    )
    src.add_argument(
        "--domain",
        help=f"registry domain directly, else ${DOMAIN_ENV}. Never defaulted: "
        "a hostname is a tenant identifier and this repo is public",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--project-id",
        help="project for the cmlapi registration; else $CDSW_PROJECT_ID / "
        "$CML_PROJECT_ID. Never defaulted",
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--mechanism",
        default="auto",
        choices=("auto", "cmlapi", "mlflow"),
        help="override the preflight's choice. `auto` prefers cmlapi when it "
        "exposes registry methods; pass `mlflow` to force the documented "
        "MLflow path if the cmlapi call signature does not match",
    )
    parser.add_argument(
        "--wait-seconds", type=int, default=300, help="budget for reaching READY"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="preflight, build and validate; resolve the domain; write nothing",
    )
    # The credential flags mirror probe_registry.py exactly, because
    # `load_token` is the same function.
    parser.add_argument(
        "--workload-name",
        default="DE",
        choices=("DE", "DF", "OPDB"),
        help="workload name for the UMS token mint (any mints the same JWT)",
    )
    cred = parser.add_mutually_exclusive_group()
    cred.add_argument("--token-env", metavar="VAR", help="use this env var instead of minting")
    cred.add_argument("--token-file", metavar="PATH", help="use this file instead of minting")
    parser.add_argument("--token-json-key", help="if the token file is JSON, its key")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    found = discover_mechanism()
    if args.mechanism != "auto":
        # The operator overrides the preference order. Still report what the
        # preflight would have picked, so the transcript records the override
        # rather than hiding it.
        if found["mechanism"] != args.mechanism:
            print(
                f"  -> overridden: using {args.mechanism!r} "
                f"(preflight preferred {found['mechanism'] or 'NONE'})"
            )
        found["mechanism"] = args.mechanism
    if found["onnx"] is None:
        print("\nonnx is required to build the graph:\n    pip install onnx", file=sys.stderr)
        return 2

    no_mechanism = (
        "Neither mlflow nor a registry-shaped cmlapi is available here. Those are\n"
        "the only two ways to put a deployable version in the AI Registry -- the\n"
        "REST API has no byte-upload route at all. Try `pip install mlflow`; if\n"
        "that is not possible, M3's premise does not hold for this workbench, and\n"
        "that is a scope question rather than a bug."
    )
    if found["mechanism"] is None and not args.dry_run:
        print(f"\n{no_mechanism}", file=sys.stderr)
        return 2

    # Build first. It needs no credential and no network, so a broken graph
    # fails before anything is minted or written.
    print("\n=== build ===")
    onnx_bytes = build_minimal_onnx()
    print(f"  {len(onnx_bytes)} bytes, opset {OPSET}, {list(INPUT_SHAPE)} -> {list(OUTPUT_SHAPE)}")
    print("  (run `python scripts/build_model.py --self-check` to validate it)")

    # A dry run on a laptop is the plan's laptop-side check: it has no tenant,
    # no `cdp`, and nothing to read back. Resolving a domain and minting a JWT it
    # will not use would be the one thing a dry run must not do. So the
    # credential path is entered only when the operator actually named a
    # registry -- which is also what makes `--dry-run --environment X` a useful
    # pre-flight inside a Session.
    named = bool(args.domain or args.environment or os.environ.get(DOMAIN_ENV))
    if args.dry_run and not named:
        print("\n=== dry run: nothing was written, no credential minted ===")
        print(f"  would log to experiment {args.experiment!r}")
        print(f"  would register {args.model_name!r} via {found['mechanism'] or '<none>'}")
        if found["mechanism"] is None:
            print("\n" + textwrap.indent(no_mechanism, "  "))
        print("\n  Add --environment NAME to also resolve the domain and check the")
        print("  credential; drop --dry-run to register for real.")
        return 0

    domain = args.domain or os.environ.get(DOMAIN_ENV)
    if not domain:
        if not args.environment:
            print(
                "\nname a registry: --environment NAME, or --domain, or "
                f"${DOMAIN_ENV}.\nThere is deliberately no default -- a hostname "
                "is a tenant identifier.",
                file=sys.stderr,
            )
            return 2
        print("\n=== resolving the registry domain ===")
        domain = discover_domain(args.environment)

    label, token = load_token(args)
    print(f"\n  credential: {label}  (value never printed)")

    # Read before writing. A registration that lands and then cannot be read
    # back leaves tenant state behind with no findings to show for it, and a
    # rejected credential is the likeliest cause -- so prove the credential
    # first, while nothing has been written.
    if check_auth(domain, label, token) != 200:
        print(
            "\nthe listing the control plane will use is not readable with this\n"
            "credential. Fix that before registering: a write now would leave a\n"
            "version behind that this script cannot report on.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print("\n=== dry run: nothing was written ===")
        print(f"  would log to experiment {args.experiment!r}")
        print(f"  would register {args.model_name!r} via {found['mechanism'] or '<none>'}")
        if found["mechanism"] is None:
            print("\n" + textwrap.indent(no_mechanism, "  "))
        existing = find_model_id(domain, token, args.model_name)
        if existing:
            print(f"\n  NOTE {args.model_name!r} already exists (model_id {existing}).")
            print("  Registering again adds a version rather than replacing one.")
        print("\n  re-run without --dry-run to register for real")
        return 0

    print("\n=== logging to MLflow ===")
    run_id, artifact_path, experiment_id = log_to_mlflow(args.model_name, args.experiment)

    print("\n=== registering ===")
    project_id = (
        resolve_project_id(args.project_id) if found["mechanism"] == "cmlapi" else ""
    )
    version = register(
        found["mechanism"],
        args.model_name,
        run_id,
        artifact_path,
        experiment_id=experiment_id,
        project_id=project_id,
    )
    print(f"  registry reported version {version!r}")

    print("\n=== reading it back over REST, as the control plane will ===")
    model_id = find_model_id(domain, token, args.model_name)
    if not model_id:
        print(
            f"  {args.model_name!r} is not in the v2 listing yet. The registration\n"
            "  call returned, so this is a propagation delay or a mechanism that\n"
            "  writes somewhere the v2 API does not read. Re-run probe_registry.py."
        )
        return 1
    print(f"  model_id = {model_id}   <- tenant identifier, strip before pasting")

    payload = poll_until_terminal(
        domain, token, model_id, version or "1", args.wait_seconds
    )
    if payload is None:
        return 1

    return 0 if report(payload) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
