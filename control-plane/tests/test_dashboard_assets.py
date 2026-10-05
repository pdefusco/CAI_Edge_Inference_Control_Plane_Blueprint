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

The first two tests are deliberately a pair. The first asserts the reset exists;
the second asserts the hazard it defends against is still live, so the first
cannot degrade into a test that guards nothing.

The rest are about the session gate specifically, which is what the CSS bug made
visible: a landing page must show the sign-in panel and nothing else, and that
depends on two things no type checker sees -- that every element the gate
controls starts `hidden` in the template, and that signing out puts back exactly
what signing in revealed.
"""

from __future__ import annotations

import re
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent.parent
_CSS = (_ASSETS / "static" / "app.css").read_text()
_HTML = (_ASSETS / "templates" / "index.html").read_text()
_JS = (_ASSETS / "static" / "app.js").read_text()

_COMMENTS = re.compile(r"/\*.*?\*/", re.DOTALL)
_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}", re.DOTALL)
_DISPLAY = re.compile(r"(?:^|;)\s*display\s*:([^;]*)", re.IGNORECASE)
_HIDDEN_TAG = re.compile(r"<(\w+)((?:[^>\"']|\"[^\"]*\"|'[^']*')*?)\bhidden\b", re.DOTALL)
_SHOW = re.compile(r"""show\(\$\(["'](\w[\w-]*)["']\)""")


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


def _gate_toggles(name: str) -> set[str]:
    """The element ids `gate()` or `enter()` passes to `show()`.

    Reading the source rather than running it: there is no DOM here and no build
    step, and a headless browser for two functions would be a heavier dependency
    than the whole dashboard. The bodies are matched up to a closing brace in
    column 1, which holds because nothing inside them is unindented.
    """
    body = re.search(rf"\n(?:async )?function {name}\(\) \{{\n(.*?)\n\}}", _JS, re.DOTALL)
    assert body, f"could not find `{name}()` in static/app.js"
    return set(_SHOW.findall(body.group(1)))


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


def test_signing_out_puts_back_everything_signing_in_revealed():
    """`gate()` and `enter()` must name the same elements.

    An id revealed by `enter()` but not re-hidden by `gate()` survives a sign-out
    -- which is how the header's fleet facts came to sit above a sign-in form in
    the first place. Asserting set equality rather than a hardcoded list so the
    next element added to the gate is covered without anyone remembering to.
    """
    assert _gate_toggles("gate") == _gate_toggles("enter")


def test_nothing_the_gate_controls_is_visible_before_it_runs():
    """The landing page is the sign-in panel and nothing else.

    `gate()` runs only after `main()`'s boot probe answers, so the server's HTML
    is what a visitor sees for one network round trip -- longer, if the control
    plane is slow or unreachable. Any gated element without `hidden` in the
    template is on screen for that whole window regardless of what app.js later
    decides, and `#facts` carried the fleet size.
    """
    gated_in_html = {
        selector.removeprefix("#")
        for element in _hidden_elements()
        for selector in element
        if selector.startswith("#")
    }

    missing = _gate_toggles("enter") - gated_in_html
    assert not missing, (
        f"{sorted(missing)} are toggled by the session gate but do not carry "
        "`hidden` in templates/index.html, so they render before sign-in"
    )
