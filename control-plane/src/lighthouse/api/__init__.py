"""HTTP layer. Thin by design -- routes validate, delegate, and map errors.

Nothing here contains governance logic: the status an operator sees is derived in
`services.governance`, so it is testable without an HTTP client and identical
whether it reaches the dashboard, the API, or a test assertion.
"""

from . import artifacts, devices, meta, models
from .deps import AppContext, build_context, ctx

__all__ = ["AppContext", "artifacts", "build_context", "ctx", "devices", "meta", "models"]
