import pytest
import torch

from .model import (
    Patch,
    PowerformerInstanceNorm,
    WeightedCausalMultiheadAttention,
)


@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim, num_heads, alpha",
    [(2, 8, 16, 4, 1.0), (1, 10, 32, 8, 2.0), (3, 5, 12, 3, 0.5)],
)
def test_similarity_powerlaw(batch_size, seq_len, embed_dim, num_heads, alpha):
    model = WeightedCausalMultiheadAttention(
        embed_dim, num_heads, locality_func="similarity_power_law", alpha=alpha
    )
    x = torch.randn(batch_size, seq_len, embed_dim)
    out = model(x)
    assert out.shape == (batch_size, seq_len, embed_dim)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim, num_heads, alpha",
    [(2, 8, 16, 4, 1.0), (1, 10, 32, 8, 2.0), (3, 5, 12, 3, 0.5)],
)
def test_weighted_powerlaw(batch_size, seq_len, embed_dim, num_heads, alpha):
    model = WeightedCausalMultiheadAttention(
        embed_dim, num_heads, locality_func="weighted_power_law", alpha=alpha
    )
    x = torch.randn(batch_size, seq_len, embed_dim)
    out = model(x)
    assert out.shape == (batch_size, seq_len, embed_dim)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim, num_heads, butterworth_tc",
    [(2, 8, 16, 4, 5.0), (1, 10, 32, 8, 10.0), (3, 5, 12, 3, 3)],
)
def test_butterworth_filter(batch_size, seq_len, embed_dim, num_heads, butterworth_tc):
    model = WeightedCausalMultiheadAttention(
        embed_dim,
        num_heads,
        locality_func="butterworth_filter",
        butterworth_tc=butterworth_tc,
    )
    x = torch.randn(batch_size, seq_len, embed_dim)
    out = model(x)
    assert out.shape == (batch_size, seq_len, embed_dim)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "feat_dim, patch_size, stride",
    [(1, 4, 4), (1, 4, 6), (8, 16, 8), (12, 32, 16), (64, 64, 64)],
)
def test_patch(feat_dim, patch_size, stride):
    # Test parameters
    batch_size = 32
    seq_length = 128
    feat_dim = feat_dim
    patch_size = patch_size
    stride = stride

    patcher = Patch(
        seq_len=seq_length,
        patch_size=patch_size,
        stride=stride,
    )

    x = torch.randn(batch_size, seq_length, feat_dim)
    patches: torch.Tensor = patcher(x)

    expected_num_patches = (seq_length - patch_size) // stride + 1
    assert patcher.num_patch == expected_num_patches
    assert patches.shape == (batch_size * feat_dim, expected_num_patches, patch_size)
    assert not torch.isnan(patches).any()


@pytest.mark.parametrize("rank", range(1, 6))
def test_instance_norm(rank):
    rank_dims = [4**rank for rank in range(rank)]

    norm = PowerformerInstanceNorm()
    x = torch.randn(*rank_dims)
    x_norm, loc_scale = norm(x)
    inverse_x = norm.inverse(x_norm, loc_scale)

    assert torch.allclose(x, inverse_x)
    assert x_norm.shape == x.shape
    assert isinstance(loc_scale, tuple)
    assert len(loc_scale) == 2
    assert loc_scale[0].shape == tuple(rank_dims)
    assert loc_scale[1].shape == tuple(rank_dims)


@pytest.mark.parametrize("rank", [2, 3])  # 1d version only support rank 2 and 3 tensors
def test_instance_norm_custom(rank):
    rank_dims = [4**rank for rank in range(rank)]
    norm = PowerformerInstanceNorm()
    pt_norm = torch.nn.InstanceNorm1d(rank_dims[-1])

    x = torch.randn(*rank_dims)
    x_norm, _ = norm(x)
    x_norm_pt = pt_norm(x)
    assert torch.allclose(x_norm, x_norm_pt)


def test_scores_matmul():
    b, h, i, j, d = 2, 4, 10, 10, 64
    K = torch.randn(b, h, i, d)
    Q = torch.randn(b, h, j, d)
    factor = d**-0.5

    # Test einsum and matmul similarity score calculation
    scores_einsum = torch.einsum("b h i d, b h j d -> b h i j", K, Q) * factor
    scores_matmul = torch.matmul(K, Q.transpose(-2, -1)) * factor

    assert torch.allclose(scores_einsum, scores_matmul)
