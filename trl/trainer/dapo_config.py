from dataclasses import dataclass, field
from typing import Optional

from .grpo_config import GRPOConfig


@dataclass
class DAPOConfig(GRPOConfig):
    r"""
    Configuration class for the [`DAPOTrainer`].

    This class extends GRPOConfig with DAPO-specific parameters. For details on other parameters, refer to the
    [`GRPOConfig`] documentation.

    Parameters:
        > Parameters specific to DAPO

        epsilon_low (`float`, *optional*, defaults to `0.2`):
            Lower clipping range for the Clip-Higher strategy.
        epsilon_high (`float`, *optional*, defaults to `0.28`):
            Higher clipping range for the Clip-Higher strategy.
        max_length (`int`, *optional*, defaults to `16384`):
            Expected maximum length for Overlong Reward Shaping.
        soft_punish_cache (`int`, *optional*, defaults to `4096`):
            Additional tokens allocated as soft punish cache for Overlong Reward Shaping.
        dynamic_sampling_min_accuracy (`float`, *optional*, defaults to `0.0`):
            Minimum accuracy threshold for Dynamic Sampling filtering.
        dynamic_sampling_max_accuracy (`float`, *optional*, defaults to `1.0`):
            Maximum accuracy threshold for Dynamic Sampling filtering.
        dynamic_sampling_buffer_size (`int`, *optional*, defaults to `512`):
            Size of the dynamic sampling buffer.
    """

    # Parameters specific to DAPO
    epsilon_low: float = field(
        default=0.2,
        metadata={"help": "Lower clipping range for the Clip-Higher strategy."},
    )
    epsilon_high: float = field(
        default=0.28,
        metadata={"help": "Higher clipping range for the Clip-Higher strategy."},
    )
    max_length: int = field(
        default=16384,
        metadata={"help": "Expected maximum length for Overlong Reward Shaping."},
    )
    soft_punish_cache: int = field(
        default=4096,
        metadata={"help": "Additional tokens allocated as soft punish cache for Overlong Reward Shaping."},
    )
    dynamic_sampling_min_accuracy: float = field(
        default=0.0,
        metadata={"help": "Minimum accuracy threshold for Dynamic Sampling filtering."},
    )
    dynamic_sampling_max_accuracy: float = field(
        default=1.0,
        metadata={"help": "Maximum accuracy threshold for Dynamic Sampling filtering."},
    )
    dynamic_sampling_buffer_size: int = field(
        default=512,
        metadata={"help": "Size of the dynamic sampling buffer."},
    ) 