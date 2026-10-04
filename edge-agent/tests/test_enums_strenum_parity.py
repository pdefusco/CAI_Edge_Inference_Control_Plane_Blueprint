"""The 3.10 StrEnum fallback must behave exactly like `enum.StrEnum`.

`lighthouse_contracts.enums` gets `StrEnum` from the stdlib on 3.11+ and from a
`(str, Enum)` subclass on 3.10. That fallback exists for the Jetson: JetPack 6
ships Python 3.10 and NVIDIA's accelerated aarch64 onnxruntime wheels are built
against it, so `keeper` must import on 3.10.

These names are **on the wire** (spec SS4), so the fallback is only safe if it is
indistinguishable from the real thing. The dangerous difference is `__str__`: a
plain `(str, Enum)` stringifies as `"DesiredState.RUNNING"` rather than
`"RUNNING"`, which silently rewrites every log line and f-string that
interpolates a state -- and `enums.py` defeats that with `__str__ = str.__str__`.

The point of building the fallback locally rather than importing it is that this
file then tests the 3.10 path **on every interpreter**. Testing only the branch
the current interpreter happens to take would leave the fallback unexercised on
the 3.11 laptops and CI runners where the suite actually runs, which is exactly
how it would rot until a Jetson found out.
"""

from __future__ import annotations

import json
from enum import Enum

import pytest
from lighthouse_contracts import DesiredState
from lighthouse_contracts.enums import StrEnum as ContractStrEnum


class _Fallback(str, Enum):
    """The 3.10 branch of `enums.py`, rebuilt here so it is always tested."""

    __str__ = str.__str__

    RUNNING = "RUNNING"


class _Naive(str, Enum):
    """The same thing *without* the `__str__` line -- the bug being guarded."""

    RUNNING = "RUNNING"


class _Real(ContractStrEnum):
    """Whatever `enums.py` actually resolved `StrEnum` to on this interpreter."""

    RUNNING = "RUNNING"


def _observations(member: object) -> dict[str, object]:
    """Every way a value escapes into a log line, a payload or a comparison."""
    return {
        "str": str(member),
        "fstring": f"{member}",
        "format": format(member),
        "percent": "%s" % member,  # noqa: UP031 - the old spelling is the risk
        "value": member.value,  # type: ignore[attr-defined]
        "eq_str": member == "RUNNING",
        "json": json.dumps({"state": member}),
        "join": ",".join([member]),  # type: ignore[list-item]
    }


def test_the_fallback_is_indistinguishable_from_whatever_this_python_uses() -> None:
    """Parity, field by field, rather than one `str()` spot check.

    On 3.11+ this compares the fallback against the genuine `enum.StrEnum`; on
    3.10 both sides are the fallback and the assertion degenerates to a tautology
    -- which is fine, because the 3.11 runs are the ones that catch drift.
    """
    assert _observations(_Fallback.RUNNING) == _observations(_Real.RUNNING)


def test_dropping_the_dunder_str_line_is_what_would_break_the_wire() -> None:
    """Pin the failure mode, so the `__str__` line is never tidied away.

    `.value`, `==` and `json.dumps` all still look correct without it -- which is
    precisely why this is worth a test: the naive version passes every obvious
    check and corrupts only the stringified forms.

    The f-string case is deliberately not asserted, and the reason is a trap.
    Measured 2026-10-04 on a bare `(str, Enum)` with no `__str__`:

        3.10   str() 'N.RUNNING'   f-string 'RUNNING'
        3.13   str() 'N.RUNNING'   f-string 'N.RUNNING'

    On 3.10 the mixin's own `__format__` wins, so f-strings look right while
    `str()` is already wrong; 3.12 changed `Enum.__format__` to defer to
    `__str__`, so both break. An earlier version of this test asserted the
    f-string diverged and passed on 3.13 while failing on 3.10 -- the one
    interpreter it was written to protect. `str()` and `"%s"` are the forms that
    break on every version, so those are what get asserted.

    That divergence is a second, independent argument for the shim: it makes
    these enums behave identically across interpreters, which a bare mixin does
    not.
    """
    naive = _observations(_Naive.RUNNING)
    assert naive["value"] == "RUNNING"
    assert naive["eq_str"] is True
    assert naive["json"] == '{"state": "RUNNING"}'

    assert naive["str"] == "_Naive.RUNNING"
    assert naive["percent"] == "_Naive.RUNNING"
    assert naive["str"] != str(_Real.RUNNING)
    assert naive["percent"] != "%s" % _Real.RUNNING  # noqa: UP031


@pytest.mark.parametrize("member", list(DesiredState))
def test_a_real_contract_enum_stringifies_to_its_bare_wire_value(member: DesiredState) -> None:
    """The property the rest of the codebase relies on, asserted on a real enum.

    If this ever fails, a payload or a log line somewhere is carrying
    `DesiredState.RUNNING` where the protocol says `RUNNING`.
    """
    assert str(member) == member.value
    assert f"{member}" == member.value
    assert json.dumps(member) == f'"{member.value}"'
