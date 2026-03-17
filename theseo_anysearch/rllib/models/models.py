from __future__ import annotations

from pydantic import ConfigDict

from theseo_anysearch.models import ModelConfig


class VoxelEncoderConfig(ModelConfig):
    """Config for the voxel encoder neural network."""
    model_config = ConfigDict(extra="forbid")

    # TODO: use_position_encoding is not yet wired into any model.
    # When implemented: reshape flat obs to (2r+1)³ grid, append (dx,dy,dz)
    # relative coordinates per cell, re-flatten, pass to FC layers.
    # Not needed for CNN models (VoxelBox2DCNN/3DCNN) — spatial structure is
    # preserved by convolution kernels. Only relevant for scalar obs + FC backbone.
    use_position_encoding: bool = True
    encoder_depth: int = 3


MODEL_CONFIGS: dict[str, type[ModelConfig]] = {
    "voxel_encoder": VoxelEncoderConfig,
}


# TODO! Dependencies should import MODEL_CONFIGS directly no wrapping function needed
def get_model_config_class(name: str) -> type[ModelConfig]:
    return MODEL_CONFIGS.get(name.lower(), ModelConfig)
