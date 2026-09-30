"""Compatibility alias for portable world extent and identity contracts."""

import sys
from theseo_world_providers import world_manifest as _module

sys.modules[__name__] = _module
