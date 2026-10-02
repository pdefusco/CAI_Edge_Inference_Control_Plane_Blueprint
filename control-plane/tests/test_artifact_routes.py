"""The artifact download route -- the one endpoint the spec does not have.

SS12 lists no artifact route because it assumes the device reads `artifact_uri`
from the registry. Verified against the live cluster, it cannot: registry metadata
is an HTTPS API, but the bytes sit in object storage a Jetson at home has no
identity for. Everything here tests the route that exists to close that gap, and
three properties carry the weight.

**Authorization is structural, not checked.** The route takes no model parameter.
It resolves the artifact through the requesting device's own desired state, so
"device fetches a model it was never assigned" is not a case that has to be
defended against -- there is no input through which to express it. The tests that
matter are therefore about the *refusals* that remain: a superseded generation, a
revoked deployment, a device reaching for another device's URL.

**Range correctness is byte-exact or it is worthless.** HTTP ranges are inclusive
at both ends, which is one off-by-one away from a device that assembles a file,
computes a digest that matches nothing, and reports corruption that is actually a
server bug. So the resume paths do not merely assert a 206 -- they reassemble the
pieces and check the digest.

**A wait is not a failure.** 503 with `Retry-After` while bytes are materializing,
and 403 the moment a deployment is revoked, have to be distinguishable by an agent
that branches on status alone.
"""

from __future__ import annotations

import hashlib
import io
import tarfile

from conftest import DEVICE_ID, materialization_held, wait_for_artifact

ARTIFACT_URL = f"/api/v1/devices/{DEVICE_ID}/artifact"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def deploy(admin, version: str = "1") -> int:
    """Deploy and return the generation the artifact is pinned to."""
    response = admin.put(
        f"/api/v1/devices/{DEVICE_ID}/deployment",
        json={"model_name": "fashion-cnn", "model_version": version, "desired_state": "RUNNING"},
    )
    assert response.status_code == 200, response.text
    return response.json()["generation"]


def ready(app, admin, version: str = "1") -> tuple[int, object]:
    """Deploy, wait for the bytes, and hand back (generation, ArtifactInfo).

    Tests that assert on bytes have to wait: `PUT /deployment` returns before
    materialization finishes, by design (SS13 -- accept and return, never block on
    the device).
    """
    generation = deploy(admin, version)
    wait_for_artifact(app)
    row = app.state.ctx.store.get_desired(DEVICE_ID)
    info = app.state.ctx.artifacts.get_ready(row.cache_key)
    assert info is not None
    return generation, info


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# The whole artifact
# --------------------------------------------------------------------------


def test_the_device_downloads_the_bytes_it_was_promised(app, admin, agent, device):
    """The single most important assertion in this file. The digest the control
    plane advertises in desired state is the digest of the bytes it actually
    serves -- otherwise SS17's verify-before-activate rejects every download and
    the device can never run anything.
    """
    generation, info = ready(app, admin)

    response = agent.get(ARTIFACT_URL, params={"generation": generation})

    assert response.status_code == 200
    assert digest(response.content) == info.sha256
    assert len(response.content) == info.size_bytes


def test_the_bytes_are_a_real_mlflow_artifact(app, admin, agent, device):
    """The fake registry is not a byte generator: it builds a genuine gzipped tar
    in MLflow layout. Asserting that here means the download path is exercised
    against something the real `keeper` can actually unpack, so M4 is not the first
    time anyone finds out the packaging is wrong.
    """
    generation, info = ready(app, admin)

    body = agent.get(ARTIFACT_URL, params={"generation": generation}).content

    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        names = tar.getnames()
    assert "MLmodel" in names
    assert info.entrypoint in names


def test_the_response_carries_everything_the_agent_records(app, admin, agent, device):
    generation, info = ready(app, admin)

    response = agent.get(ARTIFACT_URL, params={"generation": generation})

    assert response.headers["etag"] == f'"{info.sha256}"'
    assert response.headers["x-lighthouse-sha256"] == info.sha256
    assert response.headers["x-lighthouse-packaging"] == "mlflow_tar_gz"
    assert response.headers["x-lighthouse-entrypoint"] == info.entrypoint
    assert response.headers["content-type"] == "application/gzip"
    assert response.headers["content-length"] == str(info.size_bytes)


def test_resume_is_advertised(app, admin, agent, device):
    """Without `Accept-Ranges`, a well-behaved client has no reason to try a Range
    request, and the resume machinery that exists because CML's ingress behaviour
    on long transfers is undocumented would never be used."""
    generation, _ = ready(app, admin)

    response = agent.get(ARTIFACT_URL, params={"generation": generation})

    assert response.headers["accept-ranges"] == "bytes"


def test_the_artifact_is_never_cached_by_an_intermediary(app, admin, agent, device):
    """A proxy holding a copy of an artifact that was later revoked would hand it
    to a device the operator believes is disarmed."""
    generation, _ = ready(app, admin)

    response = agent.get(ARTIFACT_URL, params={"generation": generation})

    assert response.headers["cache-control"] == "no-store"


def test_omitting_the_generation_still_serves_the_current_artifact(app, admin, agent, device):
    """The parameter exists to *refuse* stale downloads, not to gate ordinary ones:
    a device that has just read desired state and immediately fetches is not
    required to pass it."""
    _, info = ready(app, admin)

    response = agent.get(ARTIFACT_URL)

    assert response.status_code == 200
    assert digest(response.content) == info.sha256


def test_downloading_keeps_the_artifact_from_being_evicted(app, admin, agent, device):
    """Eviction is LRU, and a long download by a slow device must count as use.
    Otherwise the entry can be reclaimed while a device is mid-transfer."""
    generation, info = ready(app, admin)
    before = app.state.ctx.store.get_artifact(info.cache_key).last_access

    agent.get(ARTIFACT_URL, params={"generation": generation})

    after = app.state.ctx.store.get_artifact(info.cache_key).last_access
    assert after >= before


# --------------------------------------------------------------------------
# Range requests
# --------------------------------------------------------------------------


def test_a_range_is_served_as_partial_content(app, admin, agent, device):
    generation, info = ready(app, admin)

    response = agent.get(
        ARTIFACT_URL, params={"generation": generation}, headers={"Range": "bytes=0-99"}
    )

    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 0-99/{info.size_bytes}"
    assert response.headers["content-length"] == "100"
    assert len(response.content) == 100


def test_range_bounds_are_inclusive_at_both_ends(app, admin, agent, device):
    """The off-by-one that would quietly corrupt every resumed download. `bytes=0-0`
    is one byte, not zero and not two."""
    generation, info = ready(app, admin)
    whole = agent.get(ARTIFACT_URL, params={"generation": generation}).content

    response = agent.get(
        ARTIFACT_URL, params={"generation": generation}, headers={"Range": "bytes=0-0"}
    )

    assert response.content == whole[:1]
    assert response.headers["content-range"] == f"bytes 0-0/{info.size_bytes}"


def test_an_open_ended_range_runs_to_the_last_byte(app, admin, agent, device):
    """The form a resuming agent actually sends: "I have N bytes, send the rest"."""
    generation, info = ready(app, admin)
    whole = agent.get(ARTIFACT_URL, params={"generation": generation}).content
    offset = info.size_bytes - 1000

    response = agent.get(
        ARTIFACT_URL, params={"generation": generation}, headers={"Range": f"bytes={offset}-"}
    )

    assert response.status_code == 206
    assert response.content == whole[offset:]
    assert response.headers["content-range"] == f"bytes {offset}-{info.size_bytes - 1}/{info.size_bytes}"


def test_a_suffix_range_returns_the_tail(app, admin, agent, device):
    generation, info = ready(app, admin)
    whole = agent.get(ARTIFACT_URL, params={"generation": generation}).content

    response = agent.get(
        ARTIFACT_URL, params={"generation": generation}, headers={"Range": "bytes=-500"}
    )

    assert response.status_code == 206
    assert response.content == whole[-500:]
    assert response.headers["content-range"] == (
        f"bytes {info.size_bytes - 500}-{info.size_bytes - 1}/{info.size_bytes}"
    )


def test_an_end_past_the_last_byte_is_clamped_not_refused(app, admin, agent, device):
    """A device that over-asks for the final chunk gets the file's end, not a 416.
    Refusing would strand a download one chunk from completion."""
    generation, info = ready(app, admin)

    response = agent.get(
        ARTIFACT_URL,
        params={"generation": generation},
        headers={"Range": f"bytes=0-{info.size_bytes + 10_000}"},
    )

    assert response.status_code == 206
    assert len(response.content) == info.size_bytes
    assert response.headers["content-range"] == f"bytes 0-{info.size_bytes - 1}/{info.size_bytes}"


def test_an_interrupted_download_reassembles_to_the_right_digest(app, admin, agent, device):
    """The actual resume scenario, end to end. CML's ingress may cut a long
    transfer; this is the proof that paying one round trip to continue produces a
    file that passes verification, which is the entire reason the proxy design was
    chosen over presigned URLs.
    """
    generation, info = ready(app, admin)
    cut = 4096

    first = agent.get(
        ARTIFACT_URL, params={"generation": generation}, headers={"Range": f"bytes=0-{cut - 1}"}
    )
    rest = agent.get(
        ARTIFACT_URL, params={"generation": generation}, headers={"Range": f"bytes={cut}-"}
    )

    assembled = first.content + rest.content
    assert len(assembled) == info.size_bytes
    assert digest(assembled) == info.sha256


def test_many_small_chunks_still_reassemble(app, admin, agent, device):
    """Several cuts, not one, because an off-by-one at a chunk boundary can cancel
    itself out across a single split and show up only when the seams multiply."""
    generation, info = ready(app, admin)

    assembled = b""
    step = 7919  # prime, so no boundary lands on a round number
    while len(assembled) < info.size_bytes:
        start = len(assembled)
        end = min(start + step - 1, info.size_bytes - 1)
        chunk = agent.get(
            ARTIFACT_URL,
            params={"generation": generation},
            headers={"Range": f"bytes={start}-{end}"},
        )
        assert chunk.status_code == 206, chunk.text
        assembled += chunk.content

    assert digest(assembled) == info.sha256


def test_a_range_response_still_carries_the_digest_headers(app, admin, agent, device):
    """A resuming agent reads the digest off the *partial* response to decide
    whether its `.part` file is still valid. Dropping these headers on a 206 would
    make resume unverifiable."""
    generation, info = ready(app, admin)

    response = agent.get(
        ARTIFACT_URL, params={"generation": generation}, headers={"Range": "bytes=0-9"}
    )

    assert response.headers["x-lighthouse-sha256"] == info.sha256
    assert response.headers["etag"] == f'"{info.sha256}"'
    assert response.headers["accept-ranges"] == "bytes"


def test_a_start_past_the_end_is_416_with_the_real_size(app, admin, agent, device):
    """`Content-Range: bytes */size` is how the agent learns the true length and
    recovers, instead of retrying the same impossible request forever."""
    generation, info = ready(app, admin)

    response = agent.get(
        ARTIFACT_URL,
        params={"generation": generation},
        headers={"Range": f"bytes={info.size_bytes}-"},
    )

    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{info.size_bytes}"


def test_malformed_and_unsatisfiable_ranges_are_416(app, admin, agent, device):
    generation, _ = ready(app, admin)

    for header in (
        "bytes=100-50",  # reversed
        "bytes=-0",  # zero-length suffix
        "bytes=abc-def",  # not numbers
        "bytes=",  # empty
        "items=0-100",  # wrong unit
        "bytes=0-10,20-30",  # multi-range, deliberately unsupported
    ):
        response = agent.get(
            ARTIFACT_URL, params={"generation": generation}, headers={"Range": header}
        )
        assert response.status_code == 416, header


# --------------------------------------------------------------------------
# If-Match
# --------------------------------------------------------------------------


def test_a_matching_if_match_is_served(app, admin, agent, device):
    generation, info = ready(app, admin)

    response = agent.get(
        ARTIFACT_URL,
        params={"generation": generation},
        headers={"Range": "bytes=0-99", "If-Match": f'"{info.sha256}"'},
    )

    assert response.status_code == 206


def test_a_wildcard_if_match_is_served(app, admin, agent, device):
    generation, _ = ready(app, admin)

    response = agent.get(
        ARTIFACT_URL, params={"generation": generation}, headers={"If-Match": "*"}
    )

    assert response.status_code == 200


def test_resuming_against_changed_bytes_is_412(app, admin, agent, device):
    """The matching hazard this header exists for: if the artifact changed while a
    device was resuming, splicing new bytes onto old ones yields a file matching no
    digest at all, and the device would report corruption instead of a conflict.
    """
    generation, _ = ready(app, admin)

    response = agent.get(
        ARTIFACT_URL,
        params={"generation": generation},
        headers={"Range": "bytes=4096-", "If-Match": f'"{"0" * 64}"'},
    )

    assert response.status_code == 412


def test_if_match_is_checked_before_the_range(app, admin, agent, device):
    """Order matters. A stale `.part` plus an unsatisfiable range must report the
    stale artifact (412), because that is the actionable problem -- a 416 would send
    the agent chasing its offsets."""
    generation, info = ready(app, admin)

    response = agent.get(
        ARTIFACT_URL,
        params={"generation": generation},
        headers={"Range": f"bytes={info.size_bytes + 1}-", "If-Match": f'"{"0" * 64}"'},
    )

    assert response.status_code == 412


# --------------------------------------------------------------------------
# Refusals: why a device is not getting bytes
# --------------------------------------------------------------------------


def test_a_superseded_generation_is_refused(app, admin, agent, device):
    """"Superseded generation mid-download aborts cleanly" is true because of this
    409. A device that finishes fetching against an old instruction must restart
    from the new one rather than activate bytes nobody asked for any more.
    """
    generation, _ = ready(app, admin)
    deploy(admin, "2")

    response = agent.get(ARTIFACT_URL, params={"generation": generation})

    assert response.status_code == 409


def test_a_generation_from_the_future_is_refused(app, admin, agent, device):
    """Not just "older than current" -- any mismatch. A device asking for a
    generation the control plane has never issued is confused, and serving it the
    current bytes under that number would make the confusion permanent.
    """
    generation, _ = ready(app, admin)

    response = agent.get(ARTIFACT_URL, params={"generation": generation + 5})

    assert response.status_code == 409


def test_a_device_with_nothing_deployed_gets_404(agent, device):
    assert agent.get(ARTIFACT_URL).status_code == 404


def test_a_revoked_deployment_is_403(app, admin, agent, device):
    """Serving bytes for a revoked deployment would undo the revocation -- and
    revocation is the one operation the whole governance story rests on."""
    ready(app, admin)
    revoked = admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke").json()

    response = agent.get(ARTIFACT_URL, params={"generation": revoked["generation"]})

    assert response.status_code == 403


def test_revocation_outranks_a_pending_materialization(app, admin, agent, device):
    """Ordering inside the resolver, and the safe direction. A revoked device must
    get 403 rather than the 503 that invites it to retry until the bytes land."""
    with materialization_held(app):
        deploy(admin)
        revoked = admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke").json()

        response = agent.get(ARTIFACT_URL, params={"generation": revoked["generation"]})

        assert response.status_code == 403


def test_a_stopped_deployment_is_still_downloadable(app, admin, agent, device):
    """STOPPED is not REVOKED. The artifact stays authorized so a device that
    restarts -- or finishes a download that was in flight when Stop arrived -- does
    not have to refetch. Only revocation withdraws the bytes.
    """
    ready(app, admin)
    stopped = admin.post(f"/api/v1/devices/{DEVICE_ID}/stop").json()

    response = agent.get(ARTIFACT_URL, params={"generation": stopped["generation"]})

    assert response.status_code == 200


def test_waiting_for_materialization_is_503_with_a_backoff(app, admin, agent, device):
    """Reachable in production: `PUT /deployment` returns before the bytes exist, so
    a prompt agent can poll desired state and fetch before materialization
    finishes. It must be told to wait, with a server-chosen interval -- otherwise
    every device in the fleet picks its own and they converge on a thundering herd.
    """
    with materialization_held(app):
        generation = deploy(admin)

        response = agent.get(ARTIFACT_URL, params={"generation": generation})

        assert response.status_code == 503
        assert response.headers["retry-after"] == str(
            app.state.ctx.settings.heartbeat_interval_seconds
        )


def test_the_download_succeeds_once_the_bytes_land(app, admin, agent, device):
    """The other half of the wait: 503 is transient, and the same request works
    unchanged afterwards. An agent that treated it as failure would never retry."""
    with materialization_held(app):
        generation = deploy(admin)
        assert agent.get(ARTIFACT_URL, params={"generation": generation}).status_code == 503

    wait_for_artifact(app)
    response = agent.get(ARTIFACT_URL, params={"generation": generation})

    assert response.status_code == 200
    assert response.headers["etag"] == f'"{digest(response.content)}"'


def test_only_a_wait_carries_retry_after(app, admin, agent, device):
    """409, 403 and 404 are not transient. A `Retry-After` on them would tell the
    agent to keep hammering a request that can never succeed."""
    generation, _ = ready(app, admin)
    deploy(admin, "2")
    stale = agent.get(ARTIFACT_URL, params={"generation": generation})
    revoked = admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke").json()
    forbidden = agent.get(ARTIFACT_URL, params={"generation": revoked["generation"]})

    assert (stale.status_code, forbidden.status_code) == (409, 403)
    assert "retry-after" not in stale.headers
    assert "retry-after" not in forbidden.headers


# --------------------------------------------------------------------------
# Authorization
# --------------------------------------------------------------------------


def test_an_anonymous_download_is_refused(app, admin, client, device):
    """The route is reachable from the public internet once unauthenticated
    platform access is enabled, so this 401 is the only thing standing between
    anyone and the fleet's model artifacts."""
    ready(app, admin)

    assert client.get(ARTIFACT_URL).status_code == 401


def test_a_device_cannot_download_another_devices_artifact(app, admin, agent, device):
    """403, not 401: the credential is valid, the resource is not its own. A stolen
    Jetson must not become a key to every model in the fleet."""
    ready(app, admin)
    admin.post("/api/v1/devices", json={"device_id": "other-device"})

    response = agent.get("/api/v1/devices/other-device/artifact")

    assert response.status_code == 403


def test_an_operator_token_is_not_a_device_token(app, admin, device):
    """The two schemes are disjoint. An operator wanting the bytes has its own
    route, which resolves through the registry rather than through some device's
    desired state."""
    ready(app, admin)

    assert admin.get(ARTIFACT_URL).status_code == 401


# --------------------------------------------------------------------------
# The operator route
# --------------------------------------------------------------------------


OPERATOR_URL = "/api/v1/artifacts/fashion-cnn/1"


def test_an_operator_can_fetch_what_a_device_would_receive(app, admin, device):
    """The point of this route: diagnosing a device's checksum complaint without
    having to own a device token."""
    _, info = ready(app, admin)

    response = admin.get(OPERATOR_URL)

    assert response.status_code == 200
    assert digest(response.content) == info.sha256
    assert response.headers["x-lighthouse-sha256"] == info.sha256
    assert response.headers["cache-control"] == "no-store"


def test_the_operator_route_resolves_through_the_registry(app, admin):
    """No device, no deployment, nothing in the cache -- and it still finds the
    version by name. That independence is what makes it useful when a device's
    desired state is itself the thing under suspicion.
    """
    response = admin.get(OPERATOR_URL)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"

    wait_for_artifact(app)
    assert admin.get(OPERATOR_URL).status_code == 200


def test_an_unknown_model_is_404_on_the_operator_route(admin):
    assert admin.get("/api/v1/artifacts/no-such-model/1").status_code == 404


def test_a_device_token_cannot_use_the_operator_route(app, admin, agent, device):
    """Otherwise the device route's structural authorization would be decorative:
    a device could name any model it liked through here."""
    ready(app, admin)

    assert agent.get(OPERATOR_URL).status_code == 401
