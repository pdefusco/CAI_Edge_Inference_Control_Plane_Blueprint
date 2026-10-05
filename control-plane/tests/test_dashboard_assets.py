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

The next group is about the session gate specifically, which is what the CSS bug
made visible: a landing page must show the sign-in panel and nothing else, and
that depends on two things no type checker sees -- that every element the gate
controls starts `hidden` in the template, and that signing out puts back exactly
what signing in revealed.

The last group exists because everything above it was green while the deployed
dashboard was still broken, and the reason is worth stating plainly: **every
test above reads a file off disk, so none of them assert anything about what a
browser receives.** On 2026-10-04 the repo was correct, the container served the
correct bytes, and the operator's browser still ran the pre-`a340e82` CSS and JS
against post-`7841774` HTML -- the sign-in panel and the console on screen
together, because `StaticFiles` sends no `Cache-Control` and the asset URLs
carried no cache key. Reading the files could not have caught that. Rendering
the page can, so those tests go through the app.
"""

from __future__ import annotations

import re
from pathlib import Path

from lighthouse import main

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


# -- what the browser actually receives -------------------------------------
#
# See the last paragraph of the module docstring. These render the page.

_ASSET_REF = re.compile(r"""(?:href|src)=["'](/static/[^"']+)["']""")


def test_every_asset_the_page_requests_carries_a_content_digest(client):
    """No bare `/static/...` may reach a browser.

    A URL without a cache key is one a browser is entitled to satisfy from disk
    without asking, and `StaticFiles` sends no `Cache-Control` to argue
    otherwise. That is how a cached stylesheet with no `[hidden]` reset outlived
    two deploys that fixed it.
    """
    refs = _ASSET_REF.findall(client.get("/").text)
    assert refs, "no /static/ references in the rendered page -- did the template change?"

    unstamped = [ref for ref in refs if not re.search(r"\?v=[0-9a-f]{12}$", ref)]
    assert not unstamped, (
        f"{unstamped} are served without a content digest. Reference assets as "
        "`{{ assets['app.css'] }}`, not as a literal path -- see "
        "`lighthouse.main.asset_urls`."
    )


def test_the_digest_stamped_url_is_one_the_app_will_serve(client):
    """A cache key that 404s is worse than none: the page loses its CSS entirely.

    `StaticFiles` ignores the query string, so this should hold -- which is the
    point of asserting it rather than assuming it.
    """
    for ref in _ASSET_REF.findall(client.get("/").text):
        response = client.get(ref)
        assert response.status_code == 200, f"{ref} -> {response.status_code}"
        assert response.content, f"{ref} served empty"


def test_the_page_requests_the_digest_of_the_bytes_it_will_get(client):
    """The stamp must be of the served content, not of anything else.

    Guards the failure that would make the whole mechanism theatre: a digest
    that never matches what `/static` returns changes on nothing, or changes on
    everything, and either way stops being a signal.
    """
    import hashlib

    for ref in _ASSET_REF.findall(client.get("/").text):
        path, _, query = ref.partition("?")
        served = client.get(path).content
        assert query == f"v={hashlib.sha256(served).hexdigest()[:12]}", (
            f"the stamp on {ref} is not the digest of what {path} serves"
        )


def test_the_url_changes_when_the_asset_changes(tmp_path, monkeypatch):
    """The one property that fixes the bug. Everything else is bookkeeping."""
    monkeypatch.setattr(main, "_STATIC_DIR", tmp_path)
    asset = tmp_path / "app.css"

    asset.write_text("main { display: grid }")
    before = main.asset_urls()["app.css"]

    asset.write_text("[hidden] { display: none !important }")
    assert main.asset_urls()["app.css"] != before


def test_the_url_is_keyed_by_content_and_not_by_mtime(tmp_path, monkeypatch):
    """A fresh checkout or a rebuilt container rewrites mtimes, changing nothing.

    Busting the cache then is not harmlessly conservative: a digest that moves
    when the bytes did not is noise, and a number that is usually noise is one
    nobody reads when it finally means something.
    """
    monkeypatch.setattr(main, "_STATIC_DIR", tmp_path)
    asset = tmp_path / "app.js"
    asset.write_text("function gate() {}")

    before = main.asset_urls()["app.js"]
    asset.touch()
    assert main.asset_urls()["app.js"] == before


def test_a_missing_asset_does_not_break_the_page(tmp_path, monkeypatch):
    """`_mount_dashboard` treats an absent dashboard as legitimate, so stamping
    must degrade to the bare path rather than raise on a half-present one."""
    monkeypatch.setattr(main, "_STATIC_DIR", tmp_path)
    assert main.asset_urls()["app.css"] == "/static/app.css"
