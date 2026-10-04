"""Entry script for the control plane running as a Cloudera AI Application.

Point the Application's script field at this file. It exists because a one-line
launcher in the project is easier to point an Application at than the
`lighthouse` console script (`control-plane/pyproject.toml:52`) on `PATH`.

Everything that decides whether the process is reachable -- the port, the bind
address, the exit-2-on-ConfigError path -- is already in `main.py:268-299`. This
file adds no behaviour and should not grow any.
"""

from __future__ import annotations

import sys
from pathlib import Path

# A deliberate departure from the repo's install precedent. Everywhere else --
# `Makefile:49-53`, `scripts/_common.sh:32-43` -- the packages are pip-installed
# editable, and `scripts/build_model.py:737` assumes that outright. Here the
# repo's own `src` trees go on `sys.path` ahead of anything installed, for two
# reasons: an Application container is not the container a Session's install ran
# in (see `docs/cai-deployment.md` §3, which marks that an open question), and an
# Application should serve the project's current code rather than a copy
# installed weeks ago.
#
# This makes the *first-party* half of the import work with no install at all.
# The third-party half -- fastapi, uvicorn, jinja2, pydantic, pyyaml,
# python-multipart, and httpx for the `cai` extra -- it cannot help with.
_REPO = Path(__file__).resolve().parent
for _pkg in ("contracts", "control-plane"):
    _src = _REPO / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from lighthouse.main import run

# Called at module scope, not under `if __name__ == "__main__"`: CAI executes
# this script rather than importing it, and a guard here would make an
# Application that starts cleanly and serves nothing.
run()
