"""Compatibility alias for AnySearch world selection configuration."""

import sys
from theseo_anysearch.world_provider_adapter import models as _module

sys.modules[__name__] = _module
