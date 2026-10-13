"""Kunlun-backed ``sgl_kernel.kvcacheio`` HiCache transfer ops.

Upstream ``MLATokenToKVPoolHost`` (DSA full-KV L2) and ``MambaPoolHost`` (KDA
state L2) both import ``transfer_kv_all_layer_mla`` / ``transfer_kv_per_layer_mla``
from ``sgl_kernel.kvcacheio`` and, in the ``layer_first`` + ``io_backend=kernel``
path, call them with **token-granular** indices and ``item_size`` = single-token
byte stride (``token_stride_size`` for MLA, ``_item_size_per_index`` for mamba).

The Kunlun ``xspeedgate_ops`` operators of the same name have exactly that
contract (token-granular index + single-item ``item_size``), so these thin
wrappers forward 1:1. Binding them into the ``sgl_kernel`` stub (see
``bootstrap/sgl_kernel_stub.py``) makes both host pools' L1<->L2 backup/load
work on Kunlun with no per-pool hook.

``xspeedgate_ops`` is imported lazily (first call) so this module stays
importable at the very early pre-shim stage where the stub binds it, before the
XPU runtime extension is loaded.
"""

from __future__ import annotations

import torch


def _ops():
    import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops)

    return torch.ops.xspeedgate_ops


def transfer_kv_all_layer_mla(
    src_layers,
    dst_layers,
    src_indices,
    dst_indices,
    item_size,
    num_layers,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    """All-layer D2H backup (device KV -> host KV), layer_first, token-granular."""
    _ops().transfer_kv_all_layer_mla(
        src_layers=src_layers,
        dst_layers=dst_layers,
        src_indices=src_indices,
        dst_indices=dst_indices,
        item_size=item_size,
        num_layers=num_layers,
        block_quota=block_quota,
        num_warps_per_block=num_warps_per_block,
    )


def transfer_kv_per_layer_mla(
    src,
    dst,
    src_indices,
    dst_indices,
    item_size,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
) -> None:
    """Single-layer, token-granular transfer, direction-aware.

    The two Kunlun ops are direction-specific:
      * ``transfer_kv_per_layer_mla`` is H2D only (requires ``src`` on CPU host);
      * ``transfer_kv_all_layer_mla`` is D2H (device -> host).

    Upstream calls this symbol for *both* directions:
      - MLA / mamba **load** (H2D): ``src`` = host buffer -> H2D op.
      - mamba (and layer-sharded MLA) **backup** (D2H): ``src`` = device tensor;
        there is no per-layer D2H op, so drive the all-layer D2H op with a
        single-layer pointer table (``num_layers=1``).
    """
    ops = _ops()
    if src.device.type == "cpu":
        ops.transfer_kv_per_layer_mla(
            src=src,
            dst=dst,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=item_size,
            block_quota=block_quota,
            num_warps_per_block=num_warps_per_block,
        )
        return

    # D2H backup of a single layer: ptr tables live on the device tensor's
    # device and hold the (device src / host dst) base addresses.
    src_layers = torch.tensor([src.data_ptr()], dtype=torch.uint64, device=src.device)
    dst_layers = torch.tensor([dst.data_ptr()], dtype=torch.uint64, device=src.device)
    ops.transfer_kv_all_layer_mla(
        src_layers=src_layers,
        dst_layers=dst_layers,
        src_indices=src_indices,
        dst_indices=dst_indices,
        item_size=item_size,
        num_layers=1,
        block_quota=block_quota,
        num_warps_per_block=num_warps_per_block,
    )
