"""lighthouse -- the Lighthouse control plane.

A FastAPI application that runs as a sidecar to the Cloudera AI model registry.
It owns desired state for a fleet of edge devices, brokers model artifacts out of
registry-backed object storage (which devices cannot reach themselves), and
renders a dashboard showing what each device is *actually* running versus what it
was told to run.
"""

__version__ = "0.1.0"
