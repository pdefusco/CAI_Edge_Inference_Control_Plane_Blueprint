-- Lighthouse control-plane schema (spec SS19).
--
-- SQLite for the MVP, but every column type and idiom here is chosen to port to
-- PostgreSQL unchanged: no SQLite-only types, timestamps stored as ISO-8601 UTC
-- text, booleans as INTEGER 0/1.
--
-- Two things are deliberately NOT stored:
--   * governance status -- derived from the desired/actual pair at read time;
--   * connectivity/online -- derived from last_seen age at read time.
-- Storing either would mean a cached value that is wrong for exactly as long as
-- no sweeper has run.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS device (
    device_id     TEXT PRIMARY KEY,
    display_name  TEXT,
    platform      TEXT,
    registered_at TEXT NOT NULL,
    last_seen     TEXT
);

-- Current desired state, one row per device. History lives in deployment_history
-- so that rollback detection has something to compare against.
CREATE TABLE IF NOT EXISTS desired_deployment (
    device_id        TEXT PRIMARY KEY REFERENCES device(device_id) ON DELETE CASCADE,
    generation       INTEGER NOT NULL,
    desired_state    TEXT NOT NULL,
    model_name       TEXT,
    model_version    TEXT,
    -- The registry-side s3a:// location, kept for audit and operator display.
    -- Never handed to a device; the device gets a control-plane-relative URL.
    registry_artifact_uri TEXT,
    artifact_sha256  TEXT,
    artifact_format  TEXT,
    -- Lineage identity of the artifact, which is what the cache is keyed by.
    model_id         TEXT,
    version_uuid     TEXT,
    updated_at       TEXT NOT NULL
);

-- Append-only record of every desired-state change. This is what lets the server
-- classify a deployment as a rollback by looking at where the device has been,
-- rather than trusting an operator to label it.
CREATE TABLE IF NOT EXISTS deployment_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id     TEXT NOT NULL REFERENCES device(device_id) ON DELETE CASCADE,
    generation    INTEGER NOT NULL,
    desired_state TEXT NOT NULL,
    model_name    TEXT,
    model_version TEXT,
    created_at    TEXT NOT NULL,
    UNIQUE (device_id, generation)
);

CREATE INDEX IF NOT EXISTS idx_deployment_history_device
    ON deployment_history (device_id, generation DESC);

-- Last reported actual state, one row per device.
CREATE TABLE IF NOT EXISTS actual_deployment (
    device_id           TEXT PRIMARY KEY REFERENCES device(device_id) ON DELETE CASCADE,
    observed_generation INTEGER NOT NULL DEFAULT 0,
    actual_state        TEXT NOT NULL DEFAULT 'UNKNOWN',
    model_name          TEXT,
    model_version       TEXT,
    artifact_sha256     TEXT,
    inference_running   INTEGER NOT NULL DEFAULT 0,
    message             TEXT,
    hardware_json       TEXT,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_event (
    event_id   TEXT PRIMARY KEY,
    timestamp  TEXT NOT NULL,
    device_id  TEXT,
    event_type TEXT NOT NULL,
    generation INTEGER,
    -- JSON object. Never contains token material: audit records token_id only.
    details    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_event_time ON audit_event (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_event_device ON audit_event (device_id, timestamp DESC);

-- Per-device bearer tokens.
--
-- token_id is the public half of the credential and is indexed, so verification
-- is one keyed lookup plus one constant-time compare. Without a public id,
-- verifying a token would mean scanning every row and hashing against each.
--
-- token_sha256 is a plain SHA-256, NOT bcrypt/argon2, and that is correct here:
-- these secrets are 256 bits of CSPRNG output, not passwords. Slow KDFs exist to
-- make low-entropy guessing expensive; against full entropy they buy nothing
-- while adding per-request CPU to a loop that runs every 10 seconds per device.
CREATE TABLE IF NOT EXISTS device_token (
    token_id     TEXT PRIMARY KEY,
    device_id    TEXT NOT NULL REFERENCES device(device_id) ON DELETE CASCADE,
    token_sha256 TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at   TEXT,
    label        TEXT
);

CREATE INDEX IF NOT EXISTS idx_device_token_device ON device_token (device_id);

-- Materialized artifacts, keyed by registry lineage rather than by the mutable
-- (name, version) label, so repointing a version label cannot collide with
-- previously cached bytes.
CREATE TABLE IF NOT EXISTS artifact_cache (
    cache_key     TEXT PRIMARY KEY,          -- "<model_id>/<version_uuid>"
    model_name    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_id      TEXT NOT NULL,
    version_uuid  TEXT NOT NULL,
    -- PENDING while a background thread is streaming; READY once hashed and
    -- renamed into place; FAILED with an error recorded.
    status        TEXT NOT NULL DEFAULT 'PENDING',
    sha256        TEXT,
    size_bytes    INTEGER,
    packaging     TEXT,
    entrypoint    TEXT,
    cache_path    TEXT,
    source_uri    TEXT,
    error         TEXT,
    created_at    TEXT NOT NULL,
    completed_at  TEXT,
    last_access   TEXT
);

CREATE INDEX IF NOT EXISTS idx_artifact_cache_access ON artifact_cache (last_access);
CREATE INDEX IF NOT EXISTS idx_artifact_cache_model ON artifact_cache (model_name, model_version);
