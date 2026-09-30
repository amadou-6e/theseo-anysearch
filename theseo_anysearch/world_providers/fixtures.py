"""Compatibility alias for the built-in provider fixture."""

import sys
from theseo_world_providers import fixtures as _module

sys.modules[__name__] = _module
