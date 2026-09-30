"""Compatibility alias for provider bundle admission."""

import sys
from theseo_world_providers import bundle as _module

sys.modules[__name__] = _module
