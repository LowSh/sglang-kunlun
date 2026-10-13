"""Kunlun JIT HiCache kernel probe override.

When a HiCache host pool is created, ``*TokenToKVPoolHost.__init__`` probes
``can_use_hicache_jit_kernel`` / ``can_use_write_back_jit_kernel`` (in
``sglang.kernels.ops.kvcache.hicache``) to decide ``self.can_use_jit``. The
probe JIT-compiles a CUDA kernel with ``nvcc -std=c++20``.

Kunlun reports itself as CUDA sm80, so the probe runs, but its nvcc rejects
``c++20`` (``nvcc fatal: Value 'c++20' is not defined for option 'std'``). The
captured ninja build output is re-emitted as a multi-line
``logger.warning("Failed to load JIT HiCache kernel: ...")`` -- once per TP
rank, flooding the startup log.

The failure is benign: ``can_use_jit`` becomes ``False`` and the host pool
falls back to the ``io_backend="kernel"`` + ``layer_first`` path, which calls
``transfer_kv_per_layer_mla`` / ``transfer_kv_all_layer_mla`` -- the Kunlun
``xspeedgate_ops`` operators bound into the sgl_kernel stub. The JIT kernel can
never build here and is never used.

Replace both probes with ``REPLACE`` hooks that return ``False`` without
shelling out to nvcc. Behaviour is identical to the current failing probe
(``can_use_jit=False``), minus the noise and the per-rank build attempts.
``HookRegistry.apply_hooks()`` propagates the patch to every module that
imported the probe via ``from ... import`` (see ``_propagate_patch``).
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_HICACHE = "sglang.kernels.ops.kvcache.hicache"


@plugin_hook(f"{_HICACHE}.can_use_hicache_jit_kernel", type=HookType.REPLACE)
def can_use_hicache_jit_kernel_kunlun(*args, **kwargs) -> bool:
    """Kunlun nvcc has no c++20; the JIT HiCache kernel can never build."""
    return False


@plugin_hook(f"{_HICACHE}.can_use_write_back_jit_kernel", type=HookType.REPLACE)
def can_use_write_back_jit_kernel_kunlun(*args, **kwargs) -> bool:
    """Kunlun nvcc has no c++20; the staged write-back kernel can never build."""
    return False
