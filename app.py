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

# The two packages whose `src` trees have to be importable, which is also what
# identifies the project root below.
_PACKAGES = ("contracts", "control-plane")


def _project_root() -> Path:
    """The project root, found WITHOUT depending on `__file__`.

    `python app.py` defines `__file__`. A CAI Application does not necessarily:
    with the Workbench editor the script is handed to an IPython-style kernel
    that executes it as numbered cells, echoes this docstring as output, and
    leaves `__file__` undefined. A bare `Path(__file__)` then dies on

        NameError: name '__file__' is not defined

    before `run()` is ever reached, and the Application exits 1 with no hint
    that the cause is the entry script rather than the config. Observed
    2026-10-04 in a deployed Application.

    Every candidate is *verified* to contain both `src` trees rather than
    trusted, so a wrong guess fails here with a readable message instead of
    resurfacing later as a `ModuleNotFoundError` for `lighthouse`.
    """
    starts: list[Path] = []
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass  # Expected under the kernel; the candidates below cover it.
    starts.append(Path.cwd().resolve())
    # The project filesystem's conventional mount, and the same path the `cai`
    # default for `data_dir` is built on (`docs/cai-deployment.md` §4).
    starts.append(Path("/home/cdsw"))

    for start in starts:
        for directory in (start, *start.parents):
            if all((directory / pkg / "src").is_dir() for pkg in _PACKAGES):
                return directory

    print(
        "could not locate the project root. Looked at these and every parent:\n"
        + "".join(f"  {s}\n" for s in starts)
        + "None contains both "
        + " and ".join(f"{pkg}/src" for pkg in _PACKAGES)
        + ".\nPoint the Application's script field at app.py in the project"
        " root, beside those two directories.",
        file=sys.stderr,
    )
    raise SystemExit(2)


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
_REPO = _project_root()
for _pkg in _PACKAGES:
    _src = _REPO / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from lighthouse.main import run

# Called at module scope, not under `if __name__ == "__main__"`: CAI executes
# this script rather than importing it, and a guard here would make an
# Application that starts cleanly and serves nothing.
run()
