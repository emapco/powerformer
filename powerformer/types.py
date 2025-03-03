from dataclasses import dataclass
from typing import Literal

import torch
import transformers
import transformers.activations
import transformers.modeling_outputs
from omegaconf import DictConfig


@dataclass
class PowerformerConfig(transformers.PretrainedConfig):
    d_model: int = 128
    d_ff: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout_rate: float = 0.3
    linear_dropout_rate: float = 0.3
    feed_forward_proj: str = "relu"
    layer_norm_epsilon: float = 1e-5
    context_len: int = 512
    prediction_len: int = 96
    patch_len: int = 16
    patch_stride: int = 8
    attn_locality_func: Literal[
        "similarity_power_law", "weighted_power_law", "butterworth_filter"
    ] = "similarity_power_law"
    attn_alpha: float = 1.0
    attn_butterworth_tc: float = 1.0
    attn_butterworth_order: int = 2
    # PretrainedConfig required args
    pruned_heads = False
    initializer_factor = 0.2


@dataclass
class PowerformerModelOutput(transformers.modeling_outputs.ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


@dataclass
class PowerformerDatasetConfig(DictConfig):
    train_path: str
    eval_path: str
    date_col: str | None = None
    feature_cols: list[str] | None = None
    sample_size: int | None = None
