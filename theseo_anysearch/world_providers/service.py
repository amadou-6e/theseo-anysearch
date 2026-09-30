"""Compatibility alias for the provider-neutral service."""

import sys
from theseo_world_providers import service as _module

sys.modules[__name__] = _module
