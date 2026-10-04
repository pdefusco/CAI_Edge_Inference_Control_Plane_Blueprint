"""Application factory and entry point.

In CAI this runs as an Application bound to `CDSW_APP_PORT`, which is why the port
is read from the environment rather than hard-coded: binding the wrong port makes
the app unreachable with no error anywhere.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from lighthouse_contracts import ErrorResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .api import artifacts, devices, meta, models
from .api.deps import AppContext, build_context
from .api.errors import code_for_status, error_body, registry_error_body
from .config import ConfigError, Settings, load_settings
from .registry import RegistryError
from .repositories import Store
from .services import ServiceError

log = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _PACKAGE_DIR.parent.parent / "templates"
_STATIC_DIR = _PACKAGE_DIR.parent.parent / "static"

API_PREFIX = "/api/v1"


class RedactingFilter(logging.Filter):
    """Strip credential material out of log records.

    Access logs and tracebacks are the classic place a bearer token leaks, and a
    device token in a log file is a permanent deployment credential. Cheap
    insurance applied at the root logger rather than trusted to every call site.
    """

    _PATTERNS = [
        re.compile(r"(lhd_[A-Za-z0-9]+)\.[A-Za-z0-9_\-]+"),
        re.compile(r"(lha_)[A-Za-z0-9_\-]{8,}"),
        re.compile(r"(?i)(authorization|x-lighthouse-admin-token|cookie)([=:]\s*)\S+"),
        re.compile(r"(?i)(x-amz-signature=)[0-9a-f]+"),
        # A bare UMS workload JWT. The patterns above only catch a token that
        # arrives labelled -- in an `Authorization:` header or as our own
        # `lhd_`/`lha_` shape -- but the registry adapter's token is a raw
        # three-segment JWT that can reach a log through a traceback or a
        # repr with no header around it. `eyJ` is the base64 of `{"`, so every
        # JWT header segment starts with it.
        re.compile(r"(eyJ)[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]+"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - never break logging
            return True
        redacted = message
        for pattern in self._PATTERNS:
            redacted = pattern.sub(lambda m: f"{m.group(1)}{'' if m.lastindex == 1 else m.group(2)}***", redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    handler.addFilter(RedactingFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))
    # uvicorn installs its own handlers; route them through ours so access lines
    # get redacted too.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv = logging.getLogger(name)
        uv.handlers = [handler]
        uv.propagate = False


def create_app(
    settings: Settings | None = None,
    *,
    store: Store | None = None,
    registry=None,
    context: AppContext | None = None,
) -> FastAPI:
    """Build the ASGI app.

    The injection points exist so tests can run the real routes against an
    in-memory store and the fake registry, with no network and no temp-file
    bookkeeping.
    """
    settings = settings or load_settings()
    context = context or build_context(settings, store=store, registry=registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info(
            "lighthouse %s starting (env=%s registry=%s data_dir=%s)",
            __version__,
            settings.env,
            context.registry.name,
            settings.data_dir,
        )
        if settings.admin_token_ephemeral:
            # Printed, not logged through the redactor: in local dev the operator
            # needs to see it, and this is the only place it is ever shown.
            print(
                f"\n  Lighthouse admin token (ephemeral, this run only):\n"
                f"    {settings.admin_token}\n",
                flush=True,
            )
        if settings.dev_corrupt_artifacts:
            log.warning(
                "LIGHTHOUSE_DEV_CORRUPT_ARTIFACTS is on -- served artifacts are "
                "deliberately corrupted to exercise checksum rejection"
            )
        try:
            yield
        finally:
            context.close()

    app = FastAPI(
        title="Lighthouse",
        version=__version__,
        description=(
            "Edge model governance control plane for Cloudera AI. "
            "CAI owns desired state; the edge agent owns reconciliation and "
            "reports actual state."
        ),
        lifespan=lifespan,
        # Declared once for every route rather than per-route: the handlers
        # below guarantee it globally, so documenting it per-route would be 27
        # chances to forget one.
        responses={
            "4XX": {"model": ErrorResponse, "description": "Error"},
            "5XX": {"model": ErrorResponse, "description": "Error"},
        },
    )
    app.state.ctx = context

    app.include_router(meta.router, prefix=API_PREFIX)
    app.include_router(devices.router, prefix=API_PREFIX)
    app.include_router(models.router, prefix=API_PREFIX)
    app.include_router(artifacts.router, prefix=API_PREFIX)

    if settings.cors_allow_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # -- one error shape, whichever path the error took --------------------
    #
    # These four handlers exist so that `ErrorResponse` is what a client
    # actually receives. The two interesting ones are the last two: FastAPI's
    # own defaults for `HTTPException` and request validation serve
    # `{"detail": ...}`, so without them every `raise HTTPException(...)` in
    # the route layer -- which is most of the error surface -- would keep
    # emitting a second shape alongside this one. Overriding the defaults is
    # what lets the ~27 existing raise sites stay exactly as they are.

    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError) -> JSONResponse:
        """Backstop. Routes translate these explicitly; this keeps a missed one
        from surfacing as an opaque 500."""
        log.warning("unmapped service error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=409, content=error_body(type(exc).__name__, str(exc))
        )

    @app.exception_handler(RegistryError)
    async def _registry_error(request: Request, exc: RegistryError) -> JSONResponse:
        status_code, body, headers = registry_error_body(exc)
        return JSONResponse(status_code=status_code, content=body, headers=headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Re-envelope FastAPI's `{"detail": ...}`.

        A route that already knows its code may raise with a dict detail; the
        rest hand us prose and get a status-derived code. `headers` is carried
        through because `WWW-Authenticate` on a 401 and `Retry-After` on a 503
        are load-bearing -- dropping them would break the agent's backoff and
        the dashboard's sign-in gate.
        """
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail and "message" in detail:
            body = error_body(
                str(detail["code"]), str(detail["message"]), detail.get("detail")
            )
        else:
            body = error_body(code_for_status(exc.status_code), str(detail))
        return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """422s carry a list, which the envelope's `detail` cannot hold, so the
        per-field errors go under a key. They are kept rather than summarized:
        for a contract this strict, which field was rejected is the whole
        message."""
        return JSONResponse(
            status_code=422,
            content=error_body(
                code_for_status(422),
                "request does not match the API contract",
                {"errors": jsonable_encoder(exc.errors())},
            ),
        )

    _mount_dashboard(app, context)
    return app


def _mount_dashboard(app: FastAPI, context: AppContext) -> None:
    """Serve the dashboard if its files are present.

    Absence is not an error: the API has to be usable before the UI exists, and a
    missing template directory should not stop a device from reconciling.
    """
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    if not (_TEMPLATES_DIR / "index.html").is_file():
        log.info("no dashboard templates at %s; serving API only", _TEMPLATES_DIR)
        return

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    app.state.templates = templates

    @app.get("/", include_in_schema=False)
    async def dashboard(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "version": __version__,
                "registry": context.registry.name,
                "env": context.settings.env,
                "poll_interval": context.settings.heartbeat_interval_seconds,
            },
        )


def run() -> None:
    """Console-script entry point (`lighthouse`)."""
    import uvicorn

    # `create_app` is inside the handler on purpose. It builds the service graph
    # eagerly -- including the registry, whose construction resolves a domain and
    # a credential and so can raise ConfigError long after `load_settings()`
    # returned cleanly. Building it outside meant a missing `cai` extra, an
    # unnamed registry domain or a rejected token exited on a raw traceback,
    # throwing away the actionable message config.py had already written.
    try:
        settings = load_settings()
        configure_logging(settings)
        app = create_app(settings)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    # CDSW_APP_PORT is what CAI routes the Application's public URL to. Binding
    # anything else produces an app that starts cleanly and is unreachable.
    #
    # The 8000 fallback is for a laptop only. Inside a CAI *Session* -- which is
    # where you run this by hand to check it against the real registry -- 8000 is
    # already held by something outside the user's namespace: the bind fails
    # EADDRINUSE while `ss -ltn` and `netstat -ltn` both show nothing, so the
    # cause is invisible from inside. Observed 2026-10-04. Pass `PORT=8900` (or
    # any free high port) for that, and note uvicorn logs "Application startup
    # complete" *before* it reports the bind failure, so the log reads like a
    # successful boot right up to the error.
    port = int(os.environ.get("CDSW_APP_PORT") or os.environ.get("PORT") or 8000)
    host = os.environ.get("LIGHTHOUSE_HOST", "127.0.0.1" if settings.env == "local" else "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_config=None)


if __name__ == "__main__":  # pragma: no cover
    run()
