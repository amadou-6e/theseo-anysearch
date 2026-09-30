"""Compatibility alias for portable routing sidecar contracts."""

import sys
from theseo_world_providers import routing_manifests as _module

sys.modules[__name__] = _module
