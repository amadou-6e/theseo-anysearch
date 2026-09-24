"""Public RLlib exports, loaded only when a Torch model is requested."""

from importlib import import_module

__all__ = [
    "register_voxel_cnn_models",
    "build_rllib_model_dict",
    "build_rllib_rl_module_model_config",
    "VoxelHierarchicalBox3DCNN",
]


def __getattr__(name: str):
    if name in {"build_rllib_model_dict", "build_rllib_rl_module_model_config"}:
        module = import_module("theseo_anysearch.rllib.models.base")
    elif name in {"VoxelHierarchicalBox3DCNN", "register_voxel_cnn_models"}:
        module = import_module("theseo_anysearch.rllib.models.cnn")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value
