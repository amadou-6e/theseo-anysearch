"""Compatibility alias for the version-1 provider API."""

import sys
from theseo_world_providers import api as _module

sys.modules[__name__] = _module
