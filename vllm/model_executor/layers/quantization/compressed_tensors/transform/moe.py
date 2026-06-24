# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parsing of online MoE transforms (e.g. SpinQuant R4) from a TransformConfig.

The dense-linear equivalent lives in
``compressed_tensors/transform/linear.py``. For fused MoE experts the relevant
online transforms are input rotations on the gate/up projection (``w1``) and
the down projection (``w2``). Offline transforms (``weight_input`` /
``weight_output``) are already fused into the checkpoint weights and need no
runtime handling here.
"""

from compressed_tensors.transform import TransformConfig, TransformLocation
from compressed_tensors.utils import match_name


class MoEOnlineRotations:
    """Online Hadamard rotation block sizes for the two MoE projections.

    A value of ``None`` means no online input rotation; a positive value is the
    Hadamard block size (``head_dim``).
    """

    def __init__(self, w1_block: int | None, w2_block: int | None):
        self.w1_block = w1_block
        self.w2_block = w2_block

    @property
    def any(self) -> bool:
        return self.w1_block is not None or self.w2_block is not None


def _online_input_block(
    representative_name: str,
    transform_config: TransformConfig | None,
) -> int | None:
    """Return the Hadamard block size for an online input rotation, if any.

    Args:
        representative_name: A representative unfused module name to match
            transform targets against (e.g. ``...experts.0.down_proj``).
        transform_config: The parsed compressed-tensors transform config.

    Returns:
        ``None`` if there is no online input rotation for this module, otherwise
        the Hadamard ``head_dim`` block size.
    """
    for scheme in transform_config.config_groups.values():
        for args in scheme.apply:
            if args.location != TransformLocation.INPUT or not args.is_online():
                continue
            if not any(match_name(representative_name, t) for t in args.targets):
                continue
            if any(match_name(representative_name, ig) for ig in args.ignore):
                continue
            if scheme.type != "hadamard":
                raise NotImplementedError(
                    f"Online MoE transform of type {scheme.type!r} is not "
                    "supported; only 'hadamard' rotations are implemented."
                )
            if scheme.head_dim is None:
                raise NotImplementedError(
                    "Online MoE Hadamard rotations without an explicit head_dim "
                    "(full-width) are not supported."
                )
            return scheme.head_dim

    return None


def get_moe_online_rotations(
    layer_name: str,
    transform_config: TransformConfig | None,
) -> MoEOnlineRotations:
    """Detect online input rotations for a fused MoE layer's experts.

    Args:
        layer_name: The fused experts module prefix (e.g. ``...mlp.experts``).
        transform_config: The parsed compressed-tensors transform config.

    Returns:
        A ``MoEOnlineRotations`` describing the gate/up (``w1``) and down
        (``w2``) input rotation block sizes.
    """
    if transform_config is None:
        return MoEOnlineRotations(None, None)

    # Representative unfused expert names, matching how `get_moe_method`
    # constructs unfused projection names for scheme lookup.
    gate_name = f"{layer_name}.0.gate_proj"
    up_name = f"{layer_name}.0.up_proj"
    down_name = f"{layer_name}.0.down_proj"

    w1_gate = _online_input_block(gate_name, transform_config)
    w1_up = _online_input_block(up_name, transform_config)
    if w1_gate != w1_up:
        raise NotImplementedError(
            "Online input rotations must match for gate_proj and up_proj, got "
            f"{w1_gate} and {w1_up}."
        )

    w2 = _online_input_block(down_name, transform_config)

    return MoEOnlineRotations(w1_block=w1_gate, w2_block=w2)
