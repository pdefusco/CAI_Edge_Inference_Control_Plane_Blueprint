# CAI Edge Model Hub

## Claude Code Project Handoff

### Project Goal

Build a proof-of-concept **edge model governance and deployment control
plane** using **Cloudera AI (CAI)** and an **NVIDIA Jetson Orin**.

Models are trained and registered in CAI. A web application running as a
**CAI Application** acts as the Edge Model Hub and governance control
plane. A lightweight agent running on each edge device, initially a
single Jetson Orin, maintains an outbound connection/polling loop to the
control plane.

The system must provide a live view of:

-   which devices are online;
-   which model each device is supposed to run;
-   which model/version each device is actually running;
-   whether desired and actual state match;
-   when the device last checked in;
-   model deployment status;
-   model health;
-   whether a model has been stopped or revoked.

The CAI dashboard must also be able to change desired state so that the
edge device can **deploy, upgrade, stop, roll back, or revoke a model**.

The key architectural principle is:

> **CAI owns desired state. The edge agent owns reconciliation and
> reports actual state.**

Do not implement the CAI application by SSHing into devices or opening
inbound connections to the home/edge network. Edge devices initiate
outbound HTTPS connections to the control plane.

------------------------------------------------------------------------

# 1. High-Level Architecture

``` text
                            CLOUDERA AI
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Model Training                                                  │
│        │                                                          │
│        ▼                                                          │
│   MLflow / CAI Model Registry                                     │
│        │                                                          │
│        ▼                                                          │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │                  Edge Model Hub — CAI App                  │  │
│   │                                                            │  │
│   │  Device Inventory           Model Inventory                │  │
│   │  Desired State              Actual State                   │  │
│   │  Deployment Controls        Live Heartbeats                │  │
│   │  Audit Events               Governance Status              │  │
│   │                                                            │  │
│   │  [Deploy] [Upgrade] [Rollback] [Stop] [Revoke]            │  │
│   └────────────────────────────┬───────────────────────────────┘  │
│                                │                                  │
└────────────────────────────────┼──────────────────────────────────┘
                                 │ HTTPS
                                 │ outbound from edge
                                 ▼
                      ┌───────────────────────┐
                      │ NVIDIA Jetson Orin   │
                      │                       │
                      │ Edge Agent            │
                      │      │                │
                      │      ├─ heartbeat     │
                      │      ├─ desired state │
                      │      ├─ reconciliation│
                      │      └─ health        │
                      │                       │
                      │ Model Runtime         │
                      │ ONNX / TensorRT       │
                      └───────────────────────┘
```

The first version should support **one CAI deployment and one Jetson**,
but the data model and API must support multiple devices.

------------------------------------------------------------------------

# 2. Core Design Principle: Desired vs. Actual State

This project should behave like a small declarative control plane.

The server does not directly execute commands on devices.

Instead:

``` text
CAI Dashboard
     │
     │ modifies desired state
     ▼
Control Plane Database
     │
     │ edge agent polls
     ▼
Jetson Edge Agent
     │
     │ reconciles
     ▼
Local Model Runtime
     │
     │ reports result
     ▼
Control Plane
```

Example:

``` text
Desired Model Version: 8
Actual Model Version:  7
State:                 OUT_OF_SYNC
```

The edge agent sees the mismatch and reconciles.

After successful deployment:

``` text
Desired Model Version: 8
Actual Model Version:  8
State:                 HEALTHY
```

This pattern must be used for **deployment, upgrades, rollback,
stopping, and revocation**.

------------------------------------------------------------------------

# 3. MVP Scope

The initial MVP should prove this complete lifecycle:

``` text
Train model in CAI
        ↓
Register model
        ↓
Model appears in Edge Model Hub
        ↓
Register Jetson
        ↓
Assign model/version to Jetson
        ↓
Jetson discovers desired state
        ↓
Jetson downloads model
        ↓
Jetson starts local inference runtime
        ↓
Jetson heartbeat reports model/version
        ↓
Dashboard shows HEALTHY
        ↓
User clicks STOP MODEL
        ↓
Desired state becomes STOPPED
        ↓
Jetson reconciles
        ↓
Inference stops
        ↓
Jetson reports STOPPED
        ↓
Dashboard confirms STOPPED
```

The MVP should prioritize making this end-to-end flow work over adding
production-scale infrastructure.

------------------------------------------------------------------------

# 4. Model Lifecycle States

Use explicit desired and actual states.

## Desired states

``` text
RUNNING
STOPPED
REVOKED
```

A desired deployment also contains:

``` text
model_name
model_version
artifact_uri
artifact_format
artifact_checksum
generation
```

Example:

``` json
{
  "device_id": "jetson-orin-01",
  "desired_state": "RUNNING",
  "model_name": "fraud-detector",
  "model_version": "7",
  "artifact_format": "onnx",
  "artifact_uri": "<artifact location>",
  "artifact_checksum": "<sha256>",
  "generation": 42
}
```

## Actual states

At minimum:

``` text
UNKNOWN
IDLE
DOWNLOADING
DEPLOYING
RUNNING
STOPPING
STOPPED
REVOKING
REVOKED
FAILED
```

The server should derive governance/compliance status from desired
vs. actual state rather than asking the agent to decide governance
policy.

Example:

``` text
Desired: RUNNING / v7
Actual:  RUNNING / v7

=> HEALTHY
```

``` text
Desired: RUNNING / v7
Actual:  RUNNING / v6

=> OUT_OF_SYNC
```

``` text
Desired: STOPPED
Actual:  RUNNING

=> STOP_PENDING
```

``` text
Desired: REVOKED
Actual:  REVOKED

=> REVOKED
```

------------------------------------------------------------------------

# 5. Stop vs. Revoke

These are intentionally different operations.

## Stop

`STOPPED` means:

-   terminate local inference;
-   keep the model artifact on disk;
-   report that inference is no longer running;
-   allow the model to be restarted later.

Conceptually:

``` text
model.onnx      PRESENT
model.engine    PRESENT
inference       STOPPED
```

## Revoke

`REVOKED` means:

-   stop inference;
-   remove locally deployed model artifacts;
-   remove generated runtime artifacts such as TensorRT engines;
-   report that the model is no longer present;
-   prevent automatic restart unless a newer desired-state generation
    explicitly authorizes deployment.

Conceptually:

``` text
model.onnx      REMOVED
model.engine    REMOVED
inference       STOPPED
deployment      REVOKED
```

The MVP may implement Stop before Revoke, but the architecture must
accommodate both.

------------------------------------------------------------------------

# 6. Generation-Based Reconciliation

Every desired-state change should increment a monotonically increasing
`generation`.

Example:

``` text
Generation 40 -> deploy v6
Generation 41 -> deploy v7
Generation 42 -> stop
Generation 43 -> deploy v8
```

The agent reports:

``` json
{
  "observed_generation": 42
}
```

This allows the control plane to distinguish:

-   a device that has received the latest instruction;
-   a device still acting on stale state;
-   delayed heartbeats;
-   failed reconciliation.

Do not rely solely on timestamps for reconciliation ordering.

------------------------------------------------------------------------

# 7. Device Heartbeat

The Jetson should send a heartbeat periodically.

Suggested MVP interval:

``` text
10 seconds
```

Make this configurable.

Example:

``` http
POST /api/v1/devices/jetson-orin-01/heartbeat
```

Example payload:

``` json
{
  "device_id": "jetson-orin-01",
  "timestamp": "2026-10-01T20:05:31Z",
  "observed_generation": 42,
  "actual_state": "RUNNING",
  "model": {
    "name": "fraud-detector",
    "version": "7",
    "format": "onnx",
    "checksum": "..."
  },
  "runtime": {
    "inference_running": true,
    "pid": 1234
  },
  "hardware": {
    "platform": "NVIDIA Jetson Orin",
    "gpu_available": true
  }
}
```

Additional telemetry can later include:

``` text
GPU utilization
GPU memory
CPU utilization
RAM
temperature
disk space
inference latency
request rate
model health
JetPack version
CUDA version
TensorRT version
```

These are not required for the first working vertical slice.

------------------------------------------------------------------------

# 8. Desired-State API

The Jetson should retrieve its desired state through an outbound
request.

Example:

``` http
GET /api/v1/devices/jetson-orin-01/desired-state
```

Response:

``` json
{
  "generation": 43,
  "desired_state": "RUNNING",
  "model": {
    "name": "fraud-detector",
    "version": "8",
    "format": "onnx",
    "artifact_uri": "...",
    "sha256": "..."
  }
}
```

The edge agent compares this to local state and determines whether
reconciliation is required.

------------------------------------------------------------------------

# 9. Edge Agent Reconciliation Loop

The Jetson agent should have a simple deterministic reconciliation loop.

Pseudo-code:

``` python
while True:
    desired = get_desired_state()
    actual = inspect_local_state()

    if desired.generation > actual.observed_generation:
        reconcile(desired, actual)

    send_heartbeat()

    sleep(interval)
```

Reconciliation logic:

``` text
desired RUNNING
    │
    ├─ no model installed
    │      -> download
    │      -> verify
    │      -> deploy
    │      -> start
    │
    ├─ wrong version installed
    │      -> stop current
    │      -> download desired
    │      -> verify
    │      -> deploy
    │      -> start
    │
    └─ correct version running
           -> no action

desired STOPPED
    -> stop inference
    -> retain artifacts

desired REVOKED
    -> stop inference
    -> remove artifacts
```

The reconciliation logic should be idempotent.

Running it multiple times with the same desired state must be safe.

------------------------------------------------------------------------

# 10. Model Artifact Handling

The first supported artifact format should be:

``` text
ONNX
```

Do not make TensorRT conversion a blocker for the first end-to-end
implementation.

Initial flow:

``` text
CAI Model Registry
      ↓
ONNX artifact
      ↓
Jetson
      ↓
ONNX Runtime
```

Once the control-plane lifecycle works, add:

``` text
ONNX
  ↓
TensorRT compilation on Jetson
  ↓
TensorRT engine
  ↓
local inference
```

TensorRT engines should generally be treated as device/runtime-specific
derived artifacts rather than the canonical registry artifact.

------------------------------------------------------------------------

# 11. CAI / MLflow Integration

The CAI application should have a registry adapter rather than
scattering MLflow calls throughout the application.

Suggested abstraction:

``` python
class ModelRegistry:
    def list_models(self): ...
    def list_versions(self, model_name): ...
    def get_model_version(self, model_name, version): ...
    def get_artifact_uri(self, model_name, version): ...
```

First implementation:

``` text
MLflowModelRegistry
```

The rest of the control plane should depend on the interface, not
directly on MLflow.

For local development, provide a fake/mock registry implementation so
the complete system can run without CAI.

------------------------------------------------------------------------

# 12. Control Plane API

Implement a REST API.

Suggested endpoints:

``` text
GET    /api/v1/health

GET    /api/v1/devices
POST   /api/v1/devices
GET    /api/v1/devices/{device_id}

POST   /api/v1/devices/{device_id}/heartbeat
GET    /api/v1/devices/{device_id}/desired-state

PUT    /api/v1/devices/{device_id}/deployment

POST   /api/v1/devices/{device_id}/stop
POST   /api/v1/devices/{device_id}/revoke

GET    /api/v1/models
GET    /api/v1/models/{model_name}/versions

GET    /api/v1/events
GET    /api/v1/devices/{device_id}/events
```

Use FastAPI unless there is a strong technical reason not to.

Use Pydantic models for API contracts.

------------------------------------------------------------------------

# 13. Deployment Request

A deployment request might look like:

``` http
PUT /api/v1/devices/jetson-orin-01/deployment
```

``` json
{
  "model_name": "fraud-detector",
  "model_version": "7",
  "desired_state": "RUNNING"
}
```

The server should:

1.  validate the model/version;
2.  resolve its artifact information;
3.  increment generation;
4.  persist desired state;
5.  create an audit event;
6.  return the new desired state.

It should **not** synchronously wait for the Jetson to deploy the model.

The dashboard observes convergence through heartbeats.

------------------------------------------------------------------------

# 14. Dashboard

Build a simple but polished dashboard suitable for running as a CAI
Application.

The primary page should show fleet status.

Example:

``` text
EDGE MODEL GOVERNANCE
────────────────────────────────────────────────────────────────

Devices: 4       Healthy: 2       Out of Sync: 1       Offline: 1


DEVICE             MODEL             DESIRED   ACTUAL   STATUS
────────────────────────────────────────────────────────────────
jetson-orin-01     fraud-detector      v7        v7    HEALTHY
jetson-orin-02     fraud-detector      v7        v6    OUT OF SYNC
factory-cam-01     defect-detector    v12       v12    HEALTHY
factory-cam-02     defect-detector    v12        --    OFFLINE
```

Device detail:

``` text
jetson-orin-01
─────────────────────────────────────────

Connectivity        ONLINE
Last Pulse          4 seconds ago

Model               fraud-detector
Desired Version     7
Actual Version      7
Desired State       RUNNING
Actual State        RUNNING
Generation          42
Observed Generation 42

Artifact            VERIFIED
Inference           RUNNING

[ Deploy Version ] [ Stop Model ] [ Revoke Model ]
```

The UI should make **desired state vs. actual state** visually obvious.

Do not hide asynchronous transitions.

For example, after clicking Stop:

``` text
Desired: STOPPED
Actual:  RUNNING
Status:  STOPPING / PENDING
```

Only show `STOPPED` after the edge agent reports it.

------------------------------------------------------------------------

# 15. Offline Detection

The control plane should derive device connectivity from heartbeat age.

Suggested configurable thresholds:

``` text
ONLINE     < 30 seconds
STALE      30-60 seconds
OFFLINE    > 60 seconds
```

Do not store `online=true` as authoritative state.

Derive connectivity from `last_seen`.

------------------------------------------------------------------------

# 16. Audit Log

Every control-plane action should create an immutable audit event.

Examples:

``` text
DEVICE_REGISTERED
DEPLOYMENT_REQUESTED
MODEL_VERSION_CHANGED
STOP_REQUESTED
REVOKE_REQUESTED
DEVICE_HEARTBEAT
DEPLOYMENT_SUCCEEDED
DEPLOYMENT_FAILED
MODEL_STOPPED
MODEL_REVOKED
DEVICE_OFFLINE
```

Example event:

``` json
{
  "event_id": "...",
  "timestamp": "...",
  "device_id": "jetson-orin-01",
  "event_type": "STOP_REQUESTED",
  "generation": 42,
  "model_name": "fraud-detector",
  "model_version": "7"
}
```

Avoid logging a heartbeat as a full audit event forever if that creates
excessive data. Heartbeat history and governance/audit events may
eventually be separate storage concepts.

------------------------------------------------------------------------

# 17. Authentication and Security

Do not hard-code credentials.

Configuration should come from environment variables.

At minimum, design for:

``` text
CONTROL_PLANE_URL
DEVICE_ID
DEVICE_TOKEN

MLFLOW_TRACKING_URI
MLFLOW credentials / CAI environment configuration
```

For the MVP, device authentication can use a per-device bearer token.

Structure the code so this can later become:

``` text
mTLS
device certificates
short-lived tokens
device identity
signed artifacts
```

Artifact integrity must support SHA-256 verification before activation.

A failed checksum must prevent deployment.

------------------------------------------------------------------------

# 18. Jetson Runtime Abstraction

Do not couple the reconciliation engine directly to subprocess commands.

Create a runtime abstraction.

Example:

``` python
class ModelRuntime:
    def inspect(self): ...
    def deploy(self, artifact): ...
    def start(self): ...
    def stop(self): ...
    def revoke(self): ...
    def health(self): ...
```

Implementations might eventually include:

``` text
ProcessRuntime
DockerRuntime
ONNXRuntime
TritonRuntime
```

For the MVP, choose the simplest implementation that can reliably
demonstrate starting and stopping inference on the Jetson.

Provide a mock runtime for local development and tests.

------------------------------------------------------------------------

# 19. Persistence

For the MVP, use SQLite behind a repository/storage abstraction.

Store at minimum:

## Device

``` text
device_id
display_name
platform
registered_at
last_seen
```

## DesiredDeployment

``` text
device_id
generation
desired_state
model_name
model_version
artifact_uri
artifact_checksum
updated_at
```

## ActualDeployment

``` text
device_id
observed_generation
actual_state
model_name
model_version
artifact_checksum
inference_running
updated_at
```

## AuditEvent

``` text
event_id
timestamp
device_id
event_type
generation
details
```

Keep persistence isolated enough that PostgreSQL could replace SQLite
later.

------------------------------------------------------------------------

# 20. Suggested Repository Structure

Start with a monorepo.

``` text
cai-edge-model-hub/
│
├── README.md
├── CLAUDE.md
├── LICENSE
├── .gitignore
├── .env.example
├── Makefile
│
├── control-plane/
│   ├── README.md
│   ├── pyproject.toml
│   ├── src/
│   │   └── edge_model_hub/
│   │       ├── main.py
│   │       ├── api/
│   │       ├── models/
│   │       ├── services/
│   │       ├── repositories/
│   │       ├── registry/
│   │       └── config.py
│   └── tests/
│
├── edge-agent/
│   ├── README.md
│   ├── pyproject.toml
│   ├── src/
│   │   └── edge_agent/
│   │       ├── main.py
│   │       ├── client.py
│   │       ├── reconciler.py
│   │       ├── state.py
│   │       ├── artifact_manager.py
│   │       ├── runtime/
│   │       └── config.py
│   └── tests/
│
├── dashboard/
│   ├── README.md
│   └── ...
│
├── examples/
│   ├── mock-model/
│   └── mock-device/
│
├── scripts/
│   ├── dev.sh
│   ├── register_device.sh
│   └── simulate_device.sh
│
└── docs/
    ├── architecture.md
    ├── api.md
    ├── jetson-setup.md
    └── cai-deployment.md
```

Claude may adjust this structure when implementation details justify it,
but preserve the separation between:

``` text
control plane
edge agent
dashboard
registry integration
runtime integration
```

------------------------------------------------------------------------

# 21. Local Development Mode

The complete control-plane/reconciliation workflow must be testable
without a Jetson or CAI.

Provide:

``` text
FakeModelRegistry
MockModelRuntime
SimulatedEdgeDevice
```

A developer should be able to run:

``` bash
make dev
```

and get:

``` text
control plane
dashboard
simulated Jetson
mock model registry
```

Then demonstrate:

``` text
deploy -> running
stop -> stopped
deploy -> running
revoke -> revoked
```

This local simulation is important because it makes development
independent of access to the physical Jetson.

------------------------------------------------------------------------

# 22. Testing Requirements

Prioritize tests around state transitions and reconciliation.

Required cases:

``` text
no deployment -> deploy model
correct model already running -> no-op
wrong model version -> upgrade
RUNNING -> STOPPED
STOPPED -> RUNNING
RUNNING -> REVOKED
repeated same generation -> no-op
stale generation -> ignore
artifact checksum failure -> FAILED
runtime start failure -> FAILED
device heartbeat updates actual state
offline device detected from heartbeat age
```

The reconciler should be heavily unit tested because it is the core of
the system.

------------------------------------------------------------------------

# 23. Initial Implementation Phases

Claude should work incrementally and keep the project runnable after
each phase.

## Phase 1 --- Repository Bootstrap

Create:

``` text
repo structure
Python environments
FastAPI control plane
SQLite persistence
Pydantic API models
health endpoint
basic tests
Makefile
README
```

Do not implement CAI or Jetson-specific functionality yet.

## Phase 2 --- Device Control Plane

Implement:

``` text
device registration
device listing
desired state
heartbeat
actual state
generation numbers
offline detection
audit events
```

Add a simulated device.

## Phase 3 --- Reconciliation Agent

Implement:

``` text
edge agent
poll desired state
inspect local state
reconciliation loop
mock runtime
heartbeat reporting
```

Demonstrate:

``` text
RUNNING -> STOPPED -> RUNNING
```

entirely locally.

## Phase 4 --- Dashboard

Implement:

``` text
fleet view
device detail
desired vs actual state
last heartbeat
deploy control
stop control
revoke control
audit history
```

## Phase 5 --- Model Registry

Add:

``` text
ModelRegistry abstraction
FakeModelRegistry
MLflowModelRegistry
model listing
model version selection
artifact resolution
```

## Phase 6 --- Real Jetson

Run edge agent on NVIDIA Jetson Orin.

Implement:

``` text
real artifact download
SHA-256 validation
local ONNX inference
process/runtime lifecycle
hardware metadata
```

Prove remote Stop from CAI dashboard.

## Phase 7 --- CAI Deployment

Deploy the control plane/dashboard as a CAI Application.

Configure:

``` text
MLflow access
persistent configuration
device API authentication
public/reachable API endpoint
```

Connect the real Jetson through outbound HTTPS.

## Phase 8 --- TensorRT

After the lifecycle is stable:

``` text
download ONNX
compile/optimize for TensorRT
activate engine
report TensorRT runtime information
```

Do not make TensorRT part of the initial critical path.

------------------------------------------------------------------------

# 24. Definition of Done for First Demo

The first meaningful demo is complete when all of the following work:

1.  A model exists in the CAI/MLflow registry.
2.  The Edge Model Hub runs as a CAI Application.
3.  A physical Jetson Orin is registered.
4.  The dashboard shows the Jetson as online from live heartbeats.
5.  A registry model/version can be assigned to the Jetson.
6.  The Jetson discovers that desired state without an inbound
    connection from CAI.
7.  The Jetson downloads and verifies the artifact.
8.  The Jetson starts inference.
9.  The heartbeat reports the actual model/version.
10. The dashboard shows desired == actual and HEALTHY.
11. The user clicks **Stop Model** in CAI.
12. Desired state becomes STOPPED.
13. The Jetson sees the new generation and stops inference.
14. The Jetson reports actual state STOPPED.
15. The CAI dashboard confirms the model is no longer running.

This is the primary milestone.

------------------------------------------------------------------------

# 25. Demo Narrative

The demo should tell this story:

> A data scientist trains and registers a model in Cloudera AI. The
> model is approved for edge deployment and assigned to an NVIDIA Jetson
> Orin outside the CAI environment. The Jetson securely discovers the
> desired model, downloads it, runs it locally, and continuously reports
> its actual deployment state back to CAI.
>
> The CAI Edge Model Hub therefore knows not merely which models have
> been registered, but which model version is actually running on each
> physical edge device.
>
> If governance requirements change, an operator can stop or revoke the
> model from the CAI dashboard. The request changes the device's desired
> state. The Jetson reconciles that state locally and reports
> confirmation, closing the governance loop between the central model
> registry and edge inference infrastructure.

------------------------------------------------------------------------

# 26. Non-Goals for the Initial MVP

Do not initially build:

``` text
Kubernetes on the Jetson
large fleet orchestration
Kafka
complex message queues
service mesh
full RBAC system
multi-tenancy
OTA OS updates
custom model registry
custom training platform
automatic TensorRT optimization pipeline
high availability control plane
```

These can be considered later.

The purpose of the MVP is to prove the **model governance control
loop**.

------------------------------------------------------------------------

# 27. Engineering Principles

Follow these principles throughout implementation:

1.  **Desired state is authoritative.**
2.  **Actual state comes from the device.**
3.  **Never claim an action succeeded until the device confirms it.**
4.  **Edge devices initiate network communication.**
5.  **Reconciliation must be idempotent.**
6.  **Every desired-state change gets a new generation.**
7.  **Model artifacts must be integrity checked.**
8.  **Registry integration must be abstracted.**
9.  **Local runtime integration must be abstracted.**
10. **The project must work in simulation before requiring CAI or Jetson
    hardware.**
11. **Do not hard-code credentials.**
12. **Prefer a simple working vertical slice over premature
    infrastructure.**

------------------------------------------------------------------------

# 28. Instructions to Claude Code

Start by inspecting this document and creating an implementation plan.

Before writing substantial code:

1.  identify assumptions that materially affect architecture;
2.  inspect the existing repository before creating files;
3.  preserve any existing useful code;
4.  propose the smallest first vertical slice;
5.  keep dependencies minimal.

Then implement **Phase 1 and Phase 2 first**.

The first local milestone should be:

``` text
Control plane starts
      ↓
Simulated Jetson registers
      ↓
Simulated Jetson sends heartbeat
      ↓
Dashboard/API shows ONLINE
      ↓
User requests RUNNING
      ↓
generation increments
      ↓
simulated agent reconciles
      ↓
actual state becomes RUNNING
      ↓
user requests STOPPED
      ↓
simulated agent reconciles
      ↓
actual state becomes STOPPED
```

Do not jump immediately to TensorRT, CAI-specific deployment, or
physical Jetson integration.

Once the local control loop is reliable and tested, integrate the real
systems one boundary at a time.

------------------------------------------------------------------------

# 29. Suggested Repository Initialization

From the directory where the repository should live:

``` bash
mkdir cai-edge-model-hub
cd cai-edge-model-hub

git init

cp /path/to/this/spec.md ./PROJECT_SPEC.md

cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.env
*.db
*.sqlite
*.sqlite3
dist/
build/
*.egg-info/
.DS_Store
models/
artifacts/
EOF

git add PROJECT_SPEC.md .gitignore
git commit -m "Initialize CAI Edge Model Hub project"
```

Then start Claude Code from the repository root and give it:

``` text
Read PROJECT_SPEC.md completely.

Treat it as the architectural source of truth for this project.

First inspect the repository, then propose a concise implementation plan for
Phases 1 and 2. Identify any architectural assumptions you need me to decide.

Do not implement TensorRT, real Jetson integration, or CAI-specific deployment
yet. Build the smallest tested local vertical slice of the desired-state control
loop first.

Once the plan is agreed, begin implementation and keep the repository runnable
and tested after each milestone.
```
