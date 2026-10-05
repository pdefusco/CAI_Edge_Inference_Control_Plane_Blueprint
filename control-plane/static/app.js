/* Lighthouse dashboard.

   Vanilla, deliberately. The whole page is ~400 lines against the same JSON API
   an operator can curl, which keeps one property true: there is no state the
   dashboard can show that the API would not also report. A framework here would
   buy reactivity for a page whose entire data model is "re-read the fleet every
   two seconds".

   Two rules it follows that are easy to break:

   1. **Desired and actual are never merged.** Not in the table, not in the detail
      panel. The gap between them *is* the product -- a UI that showed STOPPED the
      moment Stop was clicked would be reporting compliance that no device had
      confirmed.
   2. **Nothing is built with innerHTML from API data.** Device IDs, display names
      and agent messages are attacker-influenced in the general case (an enrolled
      device chooses its own heartbeat message), so every value goes in through
      textContent.
*/

"use strict";

const API = "/api/v1";
// Faster than the 10s heartbeat on purpose: at heartbeat cadence the page would
// routinely skip straight past STOP_PENDING, which is the one transition the demo
// exists to show.
const REFRESH_MS = 2000;

const state = {
  devices: [],
  selected: null,
  // Which device the detail panel last pre-filled its pickers for, so a poll does
  // not overwrite a selection the operator is in the middle of making.
  detailFor: null,
  models: [],
  timer: null,
  signedIn: false,
};

// -- tiny DOM helpers -------------------------------------------------------

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child);
  }
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function show(node, visible) {
  node.hidden = !visible;
}

// -- API --------------------------------------------------------------------

class Unauthorized extends Error {}

async function api(path, options = {}) {
  const response = await fetch(API + path, {
    credentials: "same-origin",
    headers: options.body ? { "Content-Type": "application/json" } : {},
    ...options,
  });
  if (response.status === 401) {
    gate();
    throw new Unauthorized("not signed in");
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      // `message` is the ErrorResponse envelope the API now always serves.
      // `detail` is only a fallback for a body from outside the app's own
      // handlers, and must be a string: in the envelope it holds an object,
      // which would render as "[object Object]" in the control-error banner.
      if (typeof body.message === "string") detail = body.message;
      else if (typeof body.detail === "string") detail = body.detail;
    } catch (_) {
      /* non-JSON error body; the status text will have to do */
    }
    throw new Error(`${response.status} ${detail}`);
  }
  if (response.status === 204) return null;
  return response.json();
}

// -- formatting -------------------------------------------------------------

const GOVERNANCE = {
  HEALTHY: "ok",
  OUT_OF_SYNC: "pending",
  STOP_PENDING: "pending",
  REVOKE_PENDING: "pending",
  REVOKED: "info",
  FAILED: "bad",
  UNKNOWN: "idle",
};

const CONNECTIVITY = {
  ONLINE: "ok",
  STALE: "pending",
  OFFLINE: "bad",
  NEVER_SEEN: "idle",
};

const ACTUAL = {
  RUNNING: "ok",
  FAILED: "bad",
  DOWNLOADING: "pending",
  DEPLOYING: "pending",
  STOPPING: "pending",
  REVOKING: "pending",
  STOPPED: "idle",
  IDLE: "idle",
  REVOKED: "info",
  UNKNOWN: "idle",
};

const DESIRED = { RUNNING: "info", STOPPED: "idle", REVOKED: "bad" };

// `CPU_ONLY` is deliberately `bad` and not `pending` or `info`. A device bought
// for its GPU and serving on its CPU is a failed Phase 6 acceptance check, and
// it will sit next to a green HEALTHY badge -- correctly, because governance
// asks whether desired and actual agree and they do. If this badge is not the
// loud one on the row, nothing on the page says the check failed.
const ACCELERATION = { ACCELERATED: "ok", CPU_ONLY: "bad", UNKNOWN: "idle" };

function badge(value, palette) {
  const text = value || "—";
  return el("span", { class: `badge ${palette[text] || "idle"}`, text });
}

function ago(iso) {
  if (!iso) return "never";
  const seconds = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  if (seconds < 60) return `${Math.floor(seconds)}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function clock(iso) {
  return iso ? new Date(iso).toLocaleTimeString() : "";
}

function modelLabel(name, version) {
  return name ? `${name}:${version || "?"}` : "—";
}

// -- fleet table ------------------------------------------------------------

function renderFleet() {
  const body = $("fleet");
  clear(body);
  show($("fleet-empty"), state.devices.length === 0);

  for (const device of state.devices) {
    const behind =
      device.observed_generation !== null &&
      device.observed_generation !== undefined &&
      device.observed_generation < device.generation;

    const row = el(
      "tr",
      {
        class: device.device_id === state.selected ? "selected" : "",
        onclick: () => select(device.device_id),
      },
      el(
        "td",
        {},
        el("div", { class: "device-id", text: device.device_id }),
        el("div", { class: "device-name", text: device.display_name || device.platform || "" })
      ),
      el("td", {}, badge(device.connectivity, CONNECTIVITY)),
      el("td", {}, badge(device.governance_status, GOVERNANCE)),
      el(
        "td",
        {},
        el(
          "div",
          { class: "state" },
          badge(device.desired_state, DESIRED),
          el("span", {
            class: "what",
            text: modelLabel(device.desired_model_name, device.desired_model_version),
          }),
          el("span", { class: "gen", text: `gen ${device.generation}` })
        )
      ),
      el(
        "td",
        {},
        el(
          "div",
          { class: "state" },
          badge(device.actual_state, ACTUAL),
          el("span", {
            class: "what",
            text: modelLabel(device.actual_model_name, device.actual_model_version),
          }),
          el("span", {
            class: `gen${behind ? " behind" : ""}`,
            // "observed" rather than "gen": this is the generation the device has
            // converged to, which is a different claim from the one it has seen.
            text: `observed ${device.observed_generation ?? "—"}`,
          })
        )
      ),
      el("td", { class: "gen", text: ago(device.last_seen) })
    );
    body.append(row);
  }
}

// -- detail panel -----------------------------------------------------------

function pairs(dl, entries) {
  clear(dl);
  for (const [term, value] of entries) {
    // A Node value is appended as the <dd>'s child so a row can carry a badge;
    // anything else goes in as text, which keeps rule 2 of the module docstring
    // (nothing from the API reaches innerHTML) true for every existing caller.
    const dd = value instanceof Node ? el("dd", {}, value) : el("dd", { text: value ?? "—" });
    dl.append(el("dt", { text: term }), dd);
  }
}

// `smoke_check` arrives as a sentence rather than an enum: "passed", "not run",
// "disabled", "skipped: <why>", "failed: <why>" -- see `hardware_info()` in
// `keeper/runtime/onnx.py`. The leading word colours the badge; the reason after
// the colon is kept verbatim beside it, because the reason is the only part that
// says what to do next. A badge that swallowed "skipped: the graph declares 2
// inputs" would turn a diagnosis into a shrug.
const SMOKE = { passed: "ok", failed: "bad", skipped: "pending", disabled: "info" };

function smokeCell(value) {
  if (!value) return null;
  const [head, ...rest] = String(value).split(":");
  const verdict = head.trim();
  const reason = rest.join(":").trim();
  const node = el("span", { class: `badge ${SMOKE[verdict] || "idle"}`, text: verdict });
  if (!reason) return node;
  return el("span", {}, node, el("span", { class: "muted", text: ` ${reason}` }));
}

function providerList(value) {
  return Array.isArray(value) && value.length ? value.join(", ") : null;
}

function renderHardware(device) {
  const hw = device.hardware || {};
  // The device's own name for itself when it has one -- `/proc/device-tree/model`
  // on a Jetson -- falling back to the platform triple.
  $("hardware-sub").textContent = hw.device_model || hw.platform || "";

  const accel = ["Acceleration", badge(device.acceleration, ACCELERATION)];
  if (!Object.keys(hw).length) {
    pairs($("hardware-dl"), [
      accel,
      ["Reported", "nothing yet — no heartbeat has carried hardware"],
    ]);
    return;
  }

  pairs($("hardware-dl"), [
    accel,
    // Both provider lists, never one. The gap between them is the trap this card
    // exists for: a build can report CUDA as available while the loaded session
    // runs every node on the CPU, because a provider that cannot handle a node
    // falls back silently and per-node. Either list alone hides that.
    ["Active", providerList(hw.active_providers) ?? "not reported"],
    ["Build offers", providerList(hw.providers) ?? "not reported"],
    ["Smoke check", smokeCell(hw.smoke_check) ?? "not reported"],
    ["Runtime", [hw.runtime, hw.onnxruntime_version].filter(Boolean).join(" ") || null],
    ["CPUs", hw.cpu_count ?? null],
  ]);
}

function renderDetail() {
  const device = state.devices.find((d) => d.device_id === state.selected);
  show($("detail"), Boolean(device));
  if (!device) return;

  $("detail-id").textContent = device.device_id;
  $("detail-sub").textContent =
    [device.display_name, device.platform].filter(Boolean).join(" · ") ||
    "no display name recorded";

  $("desired-gen").textContent = `generation ${device.generation}`;
  $("actual-gen").textContent = `observed ${device.observed_generation ?? "—"}`;

  pairs($("desired-dl"), [
    ["State", device.desired_state],
    ["Model", modelLabel(device.desired_model_name, device.desired_model_version)],
    ["Artifact", device.artifact_ready ? "ready" : "materializing…"],
  ]);

  pairs($("actual-dl"), [
    ["State", device.actual_state],
    ["Model", modelLabel(device.actual_model_name, device.actual_model_version)],
    ["Inference", device.inference_running ? "running" : "stopped"],
    ["Last seen", `${ago(device.last_seen)} (${device.connectivity})`],
    ["Message", device.message],
  ]);

  renderHardware(device);

  // Pre-select what is deployed, so the picker opens on the truth rather than on
  // whatever happened to be first in the registry -- but only when the panel
  // first opens on this device. Doing it on every poll would reset the operator's
  // choice of version every two seconds, mid-click.
  if (state.detailFor !== device.device_id) {
    state.detailFor = device.device_id;
    if (device.desired_model_name) {
      selectModel(device.desired_model_name, device.desired_model_version);
    }
  }
}

function selectModel(name, version) {
  const models = $("model");
  if (models.value !== name && [...models.options].some((o) => o.value === name)) {
    models.value = name;
    renderVersions();
  }
  const versions = $("version");
  if (version && [...versions.options].some((o) => o.value === version)) {
    versions.value = version;
  }
}

function renderModels() {
  const select = $("model");
  const previous = select.value;
  clear(select);
  for (const model of state.models) {
    select.append(el("option", { value: model.name, text: model.name }));
  }
  if (previous && [...select.options].some((o) => o.value === previous)) {
    select.value = previous;
  }
  renderVersions();
}

function renderVersions() {
  const model = state.models.find((m) => m.name === $("model").value);
  const select = $("version");
  const previous = select.value;
  clear(select);
  for (const version of model ? model.versions : []) {
    // An undeployable version is shown and disabled rather than filtered out. The
    // server already decided -- it would refuse the PUT with `reason` -- and a
    // version that silently vanishes from the picker sends an operator to the
    // registry UI to work out why.
    select.append(
      el("option", {
        value: version.version,
        text: version.deployable
          ? version.version
          : `${version.version} — ${version.reason || version.status || "not deployable"}`,
        disabled: version.deployable ? null : "disabled",
        title: version.reason || null,
      })
    );
  }
  if (previous && [...select.options].some((o) => o.value === previous)) {
    select.value = previous;
  }
}

/* The rollback target is derived from what the device actually reported running,
   not from deployment history: "go back to the last version that worked here" is
   the operator's real intent after a bad upgrade, and a version that was deployed
   but never reached RUNNING is not that. */
function rollbackTarget(events, current) {
  for (const event of events) {
    if (event.event_type !== "DEVICE_STATE_CHANGED") continue;
    const details = event.details || {};
    if (details.actual_state !== "RUNNING") continue;
    if (!details.model_version || details.model_version === current) continue;
    return { name: details.model_name, version: details.model_version };
  }
  return null;
}

function renderEvents(list, events) {
  clear(list);
  for (const event of events) {
    const details = event.details || {};
    const summary = [
      details.model_name ? modelLabel(details.model_name, details.model_version) : null,
      details.previous_state && details.actual_state
        ? `${details.previous_state} → ${details.actual_state}`
        : details.actual_state || details.desired_state || null,
      details.message,
    ]
      .filter(Boolean)
      .join(" · ");

    list.append(
      el(
        "li",
        {},
        el("time", { datetime: event.timestamp, text: clock(event.timestamp) }),
        badge(event.event_type, {
          RECONCILE_FAILED: "bad",
          CHECKSUM_FAILED: "bad",
          ARTIFACT_MATERIALIZE_FAILED: "bad",
          DEPLOYMENT_REQUESTED: "info",
          MODEL_VERSION_CHANGED: "info",
          DEPLOYMENT_ROLLED_BACK: "pending",
          REVOKE_REQUESTED: "bad",
          STOP_REQUESTED: "pending",
          DEVICE_STATE_CHANGED: "idle",
          ARTIFACT_MATERIALIZED: "ok",
          ARTIFACT_DOWNLOADED: "ok",
          DEVICE_REGISTERED: "ok",
        }),
        el(
          "span",
          { class: "what" },
          event.device_id && list.id === "events"
            ? el("b", { text: `${event.device_id} ` })
            : null,
          document.createTextNode(summary || "—"),
          event.generation !== null && event.generation !== undefined
            ? el("span", { class: "gen", text: `  gen ${event.generation}` })
            : null
        )
      )
    );
  }
  if (!events.length) list.append(el("li", { class: "what", text: "nothing yet" }));
}

// -- data loading -----------------------------------------------------------

async function refresh() {
  try {
    const [devices, events] = await Promise.all([
      api("/devices"),
      api("/events?limit=40"),
    ]);
    state.devices = devices;
    if (state.selected && !devices.some((d) => d.device_id === state.selected)) {
      state.selected = null;
    }
    renderFleet();
    renderDetail();
    renderEvents($("events"), events);
    if (state.selected) await refreshDevice();
  } catch (error) {
    if (!(error instanceof Unauthorized)) console.error(error);
  }
}

async function refreshDevice() {
  const current = state.selected;
  const events = await api(`/devices/${encodeURIComponent(current)}/events?limit=30`);
  if (state.selected !== current) return; // the operator moved on mid-request
  renderEvents($("device-events"), events);

  const device = state.devices.find((d) => d.device_id === current);
  const target = rollbackTarget(events, device?.desired_model_version);
  const button = $("rollback");
  button.disabled = !target;
  button.textContent = target ? `Roll back to ${target.version}` : "Roll back";
  button.dataset.name = target ? target.name : "";
  button.dataset.version = target ? target.version : "";
}

async function loadModels() {
  try {
    state.models = await api("/models");
    renderModels();
  } catch (error) {
    if (!(error instanceof Unauthorized)) console.error(error);
  }
}

async function health() {
  try {
    const body = await api("/health");
    const reachable = body.registry_reachable !== false;
    $("health-dot").className = `dot ${reachable ? "up" : "down"}`;
    // `device_count` is null for an anonymous caller, which is not the same as
    // zero devices -- so say nothing about the fleet rather than report "0
    // devices", which would be a confident lie about someone else's fleet.
    const count = body.device_count;
    $("health-text").textContent = !reachable
      ? "registry unreachable"
      : count == null
        ? "ok"
        : `${count} device${count === 1 ? "" : "s"}`;
    $("fact-registry").textContent = body.registry;
  } catch (_) {
    $("health-dot").className = "dot down";
    $("health-text").textContent = "unreachable";
  }
}

function select(deviceId) {
  state.selected = state.selected === deviceId ? null : deviceId;
  state.detailFor = null; // re-prefill the pickers for whatever opens next
  renderFleet();
  renderDetail();
  if (state.selected) refreshDevice().catch(() => {});
  else show($("detail"), false);
}

// -- control actions --------------------------------------------------------

async function act(label, request) {
  const error = $("control-error");
  show(error, false);
  try {
    await request();
    // Refresh immediately so the *pending* state is visible at once: the operator
    // should see STOP_PENDING the moment they click, not two seconds later.
    await refresh();
  } catch (problem) {
    if (problem instanceof Unauthorized) return;
    error.textContent = `${label} failed: ${problem.message}`;
    show(error, true);
  }
}

function deviceUrl(suffix = "") {
  return `/devices/${encodeURIComponent(state.selected)}${suffix}`;
}

function wireControls() {
  $("model").addEventListener("change", renderVersions);

  $("deploy").addEventListener("click", () =>
    act("Deploy", () =>
      api(deviceUrl("/deployment"), {
        method: "PUT",
        body: JSON.stringify({
          model_name: $("model").value,
          model_version: $("version").value,
          desired_state: "RUNNING",
        }),
      })
    )
  );

  $("rollback").addEventListener("click", (event) => {
    const { name, version } = event.currentTarget.dataset;
    if (!version) return;
    // A rollback is an ordinary deployment naming an earlier version; the server
    // classifies it by comparing against history, so the operator cannot
    // mislabel -- or forget to label -- what this is.
    act("Rollback", () =>
      api(deviceUrl("/deployment"), {
        method: "PUT",
        body: JSON.stringify({
          model_name: name,
          model_version: version,
          desired_state: "RUNNING",
        }),
      })
    );
  });

  $("stop").addEventListener("click", () =>
    act("Stop", () => api(deviceUrl("/stop"), { method: "POST" }))
  );

  $("revoke").addEventListener("click", () => {
    const device = state.selected;
    if (!window.confirm(`Revoke the model on ${device}? The device deletes its artifacts.`)) {
      return;
    }
    act("Revoke", () => api(deviceUrl("/revoke"), { method: "POST" }));
  });

  $("detail-close").addEventListener("click", () => select(state.selected));
  $("refresh").addEventListener("click", () => refresh());
  $("auto").addEventListener("change", (event) => setLive(event.currentTarget.checked));
}

// -- enrollment -------------------------------------------------------------

function wireEnrollment() {
  const dialog = $("enroll");
  const form = $("enroll-form");
  const result = $("enroll-result");

  $("show-enroll").addEventListener("click", () => {
    show(form, true);
    show(result, false);
    show($("enroll-error"), false);
    form.reset();
    dialog.showModal();
  });

  $("enroll-cancel").addEventListener("click", () => dialog.close());
  $("enroll-done").addEventListener("click", () => {
    dialog.close();
    refresh();
  });

  $("copy-token").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("new-token").textContent);
      $("copy-token").textContent = "Copied";
    } catch (_) {
      // Clipboard access is blocked in plenty of contexts; the token is on
      // screen and selectable, so this is not worth an error dialog.
      $("copy-token").textContent = "Select it manually";
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const error = $("enroll-error");
    show(error, false);
    try {
      const body = await api("/devices", {
        method: "POST",
        body: JSON.stringify({
          device_id: $("new-id").value.trim(),
          display_name: $("new-name").value.trim() || null,
          platform: $("new-platform").value.trim() || null,
        }),
      });
      $("new-token").textContent = body.credentials.token;
      $("copy-token").textContent = "Copy";
      show(form, false);
      show(result, true);
      await refresh();
    } catch (problem) {
      if (problem instanceof Unauthorized) return;
      error.textContent = problem.message;
      show(error, true);
    }
  });
}

// -- session ----------------------------------------------------------------

// `#facts` is gated alongside the console: the header carries the fleet size, so
// leaving it up for a signed-out visitor defeats the gate it sits above.
function gate() {
  state.signedIn = false;
  setLive(false);
  show($("gate"), true);
  show($("main"), false);
  show($("logout"), false);
  show($("facts"), false);
}

async function enter() {
  state.signedIn = true;
  show($("gate"), false);
  show($("main"), true);
  show($("logout"), true);
  show($("facts"), true);
  // Re-read health immediately rather than waiting out the 15s interval: the
  // reading taken while signed out withheld `device_count`, so the bar would
  // otherwise show "ok" to a signed-in operator for up to fifteen seconds.
  health();
  await loadModels();
  await refresh();
  setLive($("auto").checked);
}

function setLive(on) {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  if (on && state.signedIn) {
    state.timer = setInterval(refresh, REFRESH_MS);
  }
}

function wireSession() {
  $("login").addEventListener("submit", async (event) => {
    event.preventDefault();
    const error = $("login-error");
    show(error, false);
    try {
      await api("/session", {
        method: "POST",
        body: JSON.stringify({ token: $("token").value.trim() }),
      });
      $("token").value = "";
      await enter();
    } catch (problem) {
      // A 401 here is a wrong token, not a lost session, so it is reported in
      // place rather than bouncing through gate() again.
      error.textContent =
        problem instanceof Unauthorized ? "That token was not accepted." : problem.message;
      show(error, true);
    }
  });

  $("logout").addEventListener("click", async () => {
    try {
      await api("/session", { method: "DELETE" });
    } finally {
      gate();
    }
  });
}

// -- boot -------------------------------------------------------------------

async function main() {
  wireSession();
  wireControls();
  wireEnrollment();

  health();
  setInterval(health, 15000);

  // Probe with a real authenticated request rather than looking for a cookie:
  // the session cookie is HttpOnly and therefore invisible here, which is the
  // point of it.
  //
  // Every path out of here must end at the gate or at the console, never at
  // neither. Both start hidden, so a `catch` that only logged left the page
  // permanently blank below the header whenever this request failed for any
  // reason other than 401 -- a 502 from the CAI ingress, a 503 while the
  // registry was unreachable, or a dropped connection. The operator saw a dead
  // page with no sign-in form and nothing to read.
  try {
    state.devices = await api("/devices");
    await enter();
  } catch (error) {
    // `api()` has already called `gate()` on a 401; calling it again is both
    // harmless and what makes the non-401 path land somewhere.
    gate();
    if (!(error instanceof Unauthorized)) {
      const banner = $("login-error");
      banner.textContent = `Could not reach the control plane: ${error.message}`;
      show(banner, true);
      console.error(error);
    }
  }
}

main();
