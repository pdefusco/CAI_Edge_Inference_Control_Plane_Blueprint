"""keeper -- the Lighthouse edge agent.

Runs on the edge device (an NVIDIA Jetson Orin, in this project). Polls the
Lighthouse control plane for desired state over outbound-only HTTPS, downloads and
verifies ONNX artifacts, runs inference, and reports actual state by heartbeat.

It never listens. There is no inbound path into the network the device sits on,
which is what makes governing a box behind a home router tractable at all.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
