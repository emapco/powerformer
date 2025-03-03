import logging
from typing import Literal

import numpy as np
import scipy.interpolate
import scipy.signal
import torch
import torch.nn.functional as F
import transformers
import transformers.activations
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from .types import PowerformerConfig, PowerformerModelOutput

logger = logging.getLogger(__file__)


# Model based on: "Powerformer: A Transformer with Weighted Causal Attention for Time-series Forecasting"
# https://arxiv.org/abs/2502.06151


# A.2 Code to Calculate the Gain
def butterworth_filter(scale, order, times):
    b, a = scipy.signal.butter(order, 0.8, "lowpass", analog=False)
    t, decay = scipy.signal.freqz(b, a)
    t = scale * t / 2
    dc = 5 * np.log(np.abs(decay))
    decay_interp = scipy.interpolate.interp1d(t, dc)
    return decay_interp(times)


# 3.1 Weighted Causal Multihead Attention
class WeightedCausalMultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        locality_func: Literal[
            "similarity_power_law", "weighted_power_law", "butterworth_filter"
        ] = ("similarity_power_law"),
        alpha=1.0,
        butterworth_tc=None,
        butterworth_order=2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** (-0.5)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q = torch.nn.Linear(d_model, d_model)
        self.k = torch.nn.Linear(d_model, d_model)
        self.v = torch.nn.Linear(d_model, d_model)
        self.o = torch.nn.Linear(d_model, d_model)

        self.f_type = locality_func
        self.alpha = alpha
        if locality_func == "butterworth_filter":
            assert butterworth_tc is not None, "Butterworth filter requires cutoff time"
            assert butterworth_order is not None, "Butterworth filter requires order"
            self.butterworth_tc = butterworth_tc
            self.butterworth_order = butterworth_order

    def split_heads(self, x):
        return rearrange(x, "B T (H D) -> (B H) T D", H=self.num_heads)

    def combine_heads(self, x):
        return rearrange(x, "(B H) T D -> B T (H D)", H=self.num_heads)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        Q = self.split_heads(self.q(x))
        K = self.split_heads(self.k(x))
        V = self.split_heads(self.v(x))

        # S_h
        S_h: torch.Tensor = torch.einsum("H I D, H J D -> H I J", K, Q) * self.scale

        time_idx = torch.arange(T, device=x.device)
        delta = time_idx.unsqueeze(1) - time_idx.unsqueeze(0)  # delta[i,j] = i - j
        if self.f_type == "similarity_power_law":
            decaying_mask = -(delta.float() ** self.alpha)
            decaying_mask = torch.where(
                delta < 0, torch.zeros_like(decaying_mask), decaying_mask
            )
        elif self.f_type == "weighted_power_law":
            safe_delta = torch.where(
                delta > 0, delta.float(), torch.ones_like(delta.float())
            )
            decaying_mask = -self.alpha * torch.log(safe_delta)
            decaying_mask = torch.where(
                delta <= 0, torch.zeros_like(decaying_mask), decaying_mask
            )
        elif self.f_type == "butterworth_filter":
            positive_delta = delta.float().clamp(min=0).cpu().numpy()
            butter_mask = butterworth_filter(
                self.butterworth_tc, self.butterworth_order, positive_delta
            )
            decaying_mask = torch.tensor(butter_mask, device=x.device, dtype=x.dtype)
        else:
            raise ValueError("Unknown locality function")

        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        # S_h^(C, D) = S_h + M^(C) + M^(D)
        S_h = S_h.masked_fill_(causal_mask, float("-inf")) + decaying_mask
        C_h = F.softmax(S_h, dim=-1)

        attn = torch.einsum("H I J, H J D -> H I D", C_h, V)
        attn = self.combine_heads(attn)
        return self.o(attn)


class SDPSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model,
        dropout_p=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout_p = dropout_p

        self.q = torch.nn.Linear(d_model, d_model)
        self.k = torch.nn.Linear(d_model, d_model)
        self.v = torch.nn.Linear(d_model, d_model)
        self.o = torch.nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        try:
            with sdpa_kernel(SDPBackend.MATH):
                attn = scaled_dot_product_attention(
                    self.q(x),
                    self.k(x),
                    self.v(x),
                    dropout_p=self.dropout_p,
                )
        except Exception:
            attn = scaled_dot_product_attention(
                self.q(x),
                self.k(x),
                self.v(x),
                dropout_p=self.dropout_p,
            )
        return self.o(attn)


# PatchTST patching mechanism
# https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_self_supervised/src/callback/patch_mask.py#L8
class Patch(torch.nn.Module):
    def __init__(self, seq_len, patch_size, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.num_patch = (max(seq_len, patch_size) - patch_size) // stride + 1
        tgt_len = patch_size + stride * (self.num_patch - 1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x d_vars]
        """
        x = x[:, self.s_begin :, :]
        x = x.unfold(
            dimension=1, size=self.patch_size, step=self.stride
        )  # xb: [bs x num_patch x d_vars x patch_size]
        x = rearrange(x, "B P D N -> (B D) P N")
        return x


class PowerformerInstanceNorm(torch.nn.Module):
    """
    Apply instance normalization along the last dimension (with NaN‐aware mean and std).
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        mean_std: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if mean_std is None:
            mean = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0).to(
                x.device
            )
            std = torch.nan_to_num(
                (x - mean).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0
            )
            std = torch.where(std == 0, torch.abs(mean) + self.eps, std)
        else:
            mean, std = mean_std
        return (x - mean) / std, (mean, std)

    @staticmethod
    def inverse(
        x: torch.Tensor, mean_std: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        mean, std = mean_std
        return x * std + mean


class PowerformerPredictionHead(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        prediction_len: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.pred_proj = torch.nn.Linear(in_dim, prediction_len)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.pred_proj(hidden_states)
        return hidden_states


class PowerformerDenseActDense(torch.nn.Module):
    def __init__(self, cfg: PowerformerConfig):
        super().__init__()
        self.wi = torch.nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wo = torch.nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = torch.nn.Dropout(cfg.dropout_rate)
        self.act = transformers.activations.ACT2FN[cfg.feed_forward_proj]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class PowerformerLayerFF(torch.nn.Module):
    def __init__(self, cfg: PowerformerConfig):
        super().__init__()
        self.dense_act_dense = PowerformerDenseActDense(cfg)
        self.layer_norm = torch.nn.LayerNorm(cfg.d_model, cfg.layer_norm_epsilon)
        self.dropout = torch.nn.Dropout(cfg.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.dense_act_dense(hidden_states)
        return self.layer_norm(
            hidden_states + self.dropout(forwarded_states)
        )  # add and (post) norm with residual connection


class PowerformerBlock(torch.nn.Module):
    def __init__(self, cfg: PowerformerConfig):
        super().__init__()
        self.cfg = cfg
        self.layer_norm = torch.nn.LayerNorm(cfg.d_model, cfg.layer_norm_epsilon)
        self.attn = WeightedCausalMultiheadAttention(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            locality_func=cfg.attn_locality_func,
            alpha=cfg.attn_alpha,
            butterworth_tc=cfg.attn_butterworth_tc,
            butterworth_order=cfg.attn_butterworth_order,
        )
        # self.attn = SDPSelfAttention(
        #     d_model=cfg.d_model,
        #     dropout_p=self.cfg.dropout_rate,
        # )
        self.ff = PowerformerLayerFF(cfg)

    def forward(self, hidden_states: torch.Tensor):
        attention_outputs = self.attn(hidden_states)
        # C.3 excerpt: We additionally remove the dropout applied to the attention weights
        hidden_states = self.layer_norm(
            hidden_states + attention_outputs
        )  # add and (post) norm with residual connection
        return self.ff(hidden_states)


# C.3 Powerformer
class Powerformer(transformers.modeling_utils.PreTrainedModel):
    def __init__(
        self,
        cfg: PowerformerConfig,
    ):
        super().__init__(cfg)
        self.config = cfg
        self.patch = Patch(
            seq_len=cfg.context_len,
            patch_size=cfg.patch_len,
            stride=cfg.patch_stride,
        )
        self.instance_norm = PowerformerInstanceNorm(eps=cfg.layer_norm_epsilon)
        self.transformer_encoder = torch.nn.ModuleList(
            [PowerformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.prediction_head = PowerformerPredictionHead(
            in_dim=self.patch.num_patch * cfg.d_model,
            prediction_len=cfg.prediction_len,
            dropout_p=cfg.linear_dropout_rate,
        )
        self.proj_embed = torch.nn.Linear(self.patch.patch_size, cfg.d_model)
        self.init_weights()

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = (
            self.config.initializer_factor
        )  # Used for testing weights initialization
        d_model = self.config.d_model
        n_heads = self.config.num_heads
        if isinstance(module, PowerformerPredictionHead):
            module.pred_proj.weight.data.normal_(
                mean=0.0, std=factor * (self.config.d_model**-0.5)
            )
            if hasattr(module.pred_proj, "bias") and module.pred_proj.bias is not None:
                module.pred_proj.bias.data.zero_()
        elif isinstance(module, PowerformerDenseActDense):
            module.wi.weight.data.normal_(
                mean=0.0, std=factor * (self.config.d_model**-0.5)
            )
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(
                mean=0.0, std=factor * (self.config.d_ff**-0.5)
            )
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, WeightedCausalMultiheadAttention | SDPSelfAttention):
            module.q.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            if hasattr(module.q, "bias") and module.q.bias is not None:
                module.q.bias.data.zero_()
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            if hasattr(module.k, "bias") and module.k.bias is not None:
                module.k.bias.data.zero_()
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            if hasattr(module.v, "bias") and module.v.bias is not None:
                module.v.bias.data.zero_()
            module.o.weight.data.normal_(mean=0.0, std=factor * (n_heads**-0.5))
            if hasattr(module.o, "bias") and module.o.bias is not None:
                module.o.bias.data.zero_()

    def forward(
        self,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> PowerformerModelOutput:
        # C.3 excerpt: The input data comes in with D variates and shape [B, D, Tseq],
        # where B is the batch size and Tseq is the input sequence length.
        B, DVAR, T = context.shape

        context_norm, mean_std = self.instance_norm(context)
        # rearrange for patch consumption
        context_norm = rearrange(context_norm, "B D T -> B T D")
        patched_context = self.patch(context_norm)  # B*D, P, N
        embeds = self.proj_embed(patched_context)  # B*D, P, d_model

        encoder_output = embeds
        for layer in self.transformer_encoder:
            encoder_output = layer(encoder_output)

        # C.3 excerpt: The encoder output (of shape [B × D, P, N]) is flattened along the last two
        # dimensions ([B × D, P × N]) in preparation for the linear readout head.
        logits: torch.Tensor = self.prediction_head(
            rearrange(encoder_output, "B P N -> B (P N)")
        )
        logits = rearrange(logits, "(B D) T -> B D T", B=B, D=DVAR)
        logits = self.instance_norm.inverse(logits, mean_std)
        # logits, _ = self.instance_norm(logits)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = F.mse_loss(logits, labels)

        return PowerformerModelOutput(
            loss=loss,
            logits=logits,
        )
