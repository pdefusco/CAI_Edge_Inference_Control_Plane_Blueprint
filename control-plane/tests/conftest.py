"""Fixtures for the control plane.

These tests drive the **real** app: real routes, real auth dependencies, real
SQLite (in memory), real artifact service, and the fake registry -- which is not a
stub but a producer of genuine gzipped MLflow tarballs with deterministic digests.
Nothing here mocks an HTTP layer, so a route that only works because a test
patched something around it cannot pass.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import pytest
from starlette.testclient import TestClient

from lighthouse.config import Settings
from lighthouse.main import create_app
from lighthouse.repositories import SqliteStore

ADMIN_TOKEN = "lha_testtoken0123456789abcdef"
DEVICE_ID = "jetson-orin-01"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        env="local",
        data_dir=tmp_path / "lighthouse",
        registry_impl="fake",
        admin_token=ADMIN_TOKEN,
    )


@pytest.fixture
def store() -> SqliteStore:
    store = SqliteStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def app(settings, store):
    return create_app(settings, store=store)


@pytest.fixture
def client(app):
    """Unauthenticated client. Use it to assert that routes actually refuse."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def admin(app):
    with TestClient(app, headers={"X-Lighthouse-Admin-Token": ADMIN_TOKEN}) as test_client:
        yield test_client


@pytest.fixture
def device(admin):
    """An enrolled device, plus the token that was returned exactly once."""
    response = admin.post(
        "/api/v1/devices",
        json={"device_id": DEVICE_ID, "display_name": "Bench Jetson", "platform": "jetson-orin"},
    )
    assert response.status_code == 201, response.text
    return response.json()["credentials"]["token"]


@pytest.fixture
def agent(app, device):
    """A client authenticated as the enrolled device."""
    with TestClient(app, headers={"Authorization": f"Bearer {device}"}) as test_client:
        yield test_client


def wait_for_artifact(app, cache_key: str | None = None, *, timeout: float = 10.0):
    """Block until materialization finishes.

    `PUT /deployment` kicks off a background thread by design (spec SS13: accept and
    return, never wait for the device), so a test that asserts on artifact bytes has
    to wait for the thread rather than assume it already ran.
    """
    import time

    context = app.state.ctx
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cache_key is None:
            rows = context.store.list_artifacts()
            if rows and all(row.status in ("READY", "FAILED") for row in rows):
                return rows
        elif context.artifacts.get_ready(cache_key) is not None:
            return context.artifacts.get_ready(cache_key)
        time.sleep(0.02)
    raise AssertionError(f"artifact {cache_key or '<any>'} never became ready")


@contextmanager
def materialization_held(app):
    """Hold the background materialization thread inside `open_artifact`.

    `artifact_ready: false` and the 503 it produces on the artifact route are real
    production states, but a 64 KiB fake artifact materializes in microseconds.
    Racing that window would make these tests pass or fail on scheduler luck, so it
    is held open explicitly instead.
    """
    registry = app.state.ctx.registry
    original = registry.open_artifact
    release = threading.Event()

    def blocking(mv):
        # Bounded, so a test that forgets to release fails as a test rather than
        # hanging the suite.
        release.wait(10)
        return original(mv)

    registry.open_artifact = blocking
    try:
        yield release
    finally:
        release.set()
        registry.open_artifact = original
        # Let the thread finish before the in-memory store is torn down under it.
        if app.state.ctx.store.list_artifacts():
            wait_for_artifact(app)
