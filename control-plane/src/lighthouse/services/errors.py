"""Service-layer errors.

These live in one module so there is exactly one `DeviceNotFound`. Two classes
with the same name in sibling modules is the kind of thing that works until an
`except` clause in the API layer quietly stops matching the exception a service
actually raises -- which surfaces as a 500 on a path that should return 404.

The API layer maps these to status codes and imports nothing else from the
services' internals.
"""

from __future__ import annotations


class ServiceError(RuntimeError):
    """Base for everything the API layer is expected to translate."""


class DeviceNotFound(ServiceError):
    """No such device. -> 404"""


class DeviceAlreadyExists(ServiceError):
    """Duplicate enrollment. -> 409"""


class NothingDeployed(ServiceError):
    """Stop or revoke requested for a device with no deployment. -> 409"""
