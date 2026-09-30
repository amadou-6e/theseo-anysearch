"""Compatibility alias for the AnySearch experiment attachment adapter."""

import sys
from theseo_anysearch.world_provider_adapter import selection as _module

sys.modules[__name__] = _module
