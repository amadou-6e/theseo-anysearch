"""Compatibility alias for portable shape collision primitives."""

import sys
from theseo_world_providers import shape_collision as _module

sys.modules[__name__] = _module
