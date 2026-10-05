"""The dashboard's visibility toggles are an attribute, and CSS can cancel them.

`show()` in `static/app.js` sets nothing but `node.hidden`, so every gate on the
page -- the operator sign-in panel, the console body, the device detail pane, the
three error banners -- is one `hidden` attribute away from being wrong. The trap
is that `hidden` is `display: none` in the **user-agent** stylesheet only, and an
author rule that sets `display` on the same element beats it no matter how
unspecific it is, because author origin outranks UA origin. Nothing warns you:
the attribute is set correctly, the DOM looks right in the inspector, and the
element is simply still on screen.

Observed 2026-10-04 in a deployed Application. `main { display: grid }` and
`.gate { display: grid }` cancelled `hidden` on `#main` and `#gate`, which are
exactly the two elements `gate()`/`enter()` toggle, so the console rendered for
signed-out visitors *and* the sign-in form stayed up after a successful sign-in.
No data leaked -- every data route takes `require_operator`, and the fleet tables
stayed empty because the fetches 401'd -- but the page asserted the opposite of
the truth in both directions at once, which is worse than a blank screen for the
one reader it has.

These two tests are deliberately a pair. The first asserts the reset exists; the
second asserts the hazard it defends against is still live, so the first cannot
degrade into a test that guards nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent.parent
_CSS = (_ASSETS / "static" / "app.css").read_text()
_HTML = (_ASSETS / "templates" / "index.html").read_text()

_COMMENTS = re.compile(r"/\*.*?\*/", re.DOTALL)
_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}", re.DOTALL)
_DISPLAY = re.compile(r"(?:^|;)\s*display\s*:([^;]*)", re.IGNORECASE)
_HIDDEN_TAG = re.compile(r"<(\w+)((?:[^>\"']|\"[^\"]*\"|'[^']*')*?)\bhidden\b", re.DOTALL)


def _rules() -> list[tuple[list[str], str]]:
    """Every author rule as (selectors, body), comments stripped.

    Good enough for this file and not a CSS parser: the stylesheet is flat, with
    no nesting and one `@media` block whose inner rules this still reaches.
    """
    return [
        ([s.strip() for s in sel.split(",") if s.strip()], body)
        for sel, body in _RULE.findall(_COMMENTS.sub("", _CSS))
    ]


def _hidden_elements() -> list[set[str]]:
    """For each element carrying `hidden`, the simple selectors that match it."""
    found = []
    for tag, attrs in _HIDDEN_TAG.findall(_HTML):
        selectors = {tag}
        if match := re.search(r"id=\"([^\"]+)\"", attrs):
            selectors.add(f"#{match.group(1)}")
        if match := re.search(r"class=\"([^\"]+)\"", attrs):
            selectors.update(f".{c}" for c in match.group(1).split())
        found.append(selectors)
    return found


def test_the_hidden_attribute_is_reset_with_important():
    """Without this rule, `show(node, false)` is a suggestion."""
    resets = [
        body
        for selectors, body in _rules()
        if "[hidden]" in selectors
        for declaration in _DISPLAY.findall(body)
        if "none" in declaration and "!important" in declaration
    ]

    assert resets, (
        "static/app.css must contain `[hidden] { display: none !important }`. "
        "Every toggle in app.js sets only `node.hidden`, and an author `display` "
        "rule silently overrides the UA stylesheet's `[hidden]`."
    )


def test_the_reset_is_load_bearing_not_decorative():
    """Proves the test above is guarding something.

    If this ever fails it is good news -- no author rule sets `display` on a
    gated element any more -- but read it before deleting the reset, because the
    next `display` rule someone adds brings the bug straight back, and the
    symptom appears nowhere near the cause.
    """
    gated = _hidden_elements()
    assert gated, "no `hidden` attributes found in index.html -- did the regex rot?"

    overrides = {
        selector
        for selectors, body in _rules()
        for selector in selectors
        if selector != "[hidden]"
        for declaration in _DISPLAY.findall(body)
        if "!important" not in declaration
        for element in gated
        if selector in element
    }

    assert overrides, (
        "expected at least one author `display` rule aimed at an element that "
        "app.js gates with `hidden`; that collision is why the reset exists"
    )
