"""Compatibility alias for provider previews."""

import sys
from theseo_world_providers import render as _module

sys.modules[__name__] = _module
