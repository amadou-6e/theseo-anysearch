"""Compatibility alias for the provider source cache."""

import sys
from theseo_world_providers import source_cache as _module

sys.modules[__name__] = _module
