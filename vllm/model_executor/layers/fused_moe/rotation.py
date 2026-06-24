# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online (runtime) Hadamard rotations for MoE experts.

SpinQuant-style rotations such as R4 are split into an offline part, which is
fused into the expert weights at checkpoint creation time, and an online part,
which must be applied to the activations at runtime. For MoE the only online
rotation is a block-diagonal Hadamard applied to the down-projection (``w2``)
input, i.e. the intermediate activation produced by the gate/up activation
function. This mirrors the dense-linear path implemented by
``HadamardTransform`` in ``compressed_tensors/transform/module.py``.
"""

import torch

import vllm._custom_ops as ops


def apply_moe_hadamard_rotation(
    x: torch.Tensor, block_size: int | None
) -> torch.Tensor:
    """Apply a block-diagonal (normalized) Hadamard transform along the last dim.

    The transform is applied independently to each contiguous block of
    ``block_size`` elements along the last dimension, matching the
    ``head_dim`` semantics used by ``compressed-tensors`` SpinQuant transforms.
    A ``block_size`` of ``None`` applies a single Hadamard over the full last
    dimension.

    The Hadacore kernel only supports ``float16``/``bfloat16`` and operates in
    place; the returned tensor shares storage with ``x``.

    Args:
        x: Activation tensor to rotate, shape ``(..., features)``.
        block_size: Hadamard block size; must divide ``features``.

    Returns:
        The rotated tensor (an alias of ``x``).
    """
    features = x.shape[-1]
    if block_size is None:
        block_size = features
    assert features % block_size == 0, (
        f"Hadamard block size {block_size} must divide feature dim {features}"
    )

    if block_size == features:
        return ops.hadacore_transform(x)

    out = ops.hadacore_transform(x.unflatten(-1, (-1, block_size)))
    return out.flatten(-2, -1)
