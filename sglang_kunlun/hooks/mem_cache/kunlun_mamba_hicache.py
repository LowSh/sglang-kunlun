"""Relax the MambaPoolHost layout guard so Kunlun can run the hybrid (KDA
state) HiCache host pool in ``layer_first``.

Background
----------
GLM-5.3-Flash is a hybrid KDA(linear) + DSA model, so enabling HiCache routes
through ``build_hybrid_mamba_stack`` and builds two host pools:
``MLATokenToKVPoolHost`` (DSA full KV) + ``MambaPoolHost`` (KDA state).

``MambaPoolHost.__init__`` asserts ``layout in {page_first, page_first_direct}``
-- upstream deprecated the ``layer_first`` path -- but the layer_first buffer
creation (``init_kv_buffer`` else-branch) and the layer_first transfer branch
(``_copy_tensor`` -> ``transfer_kv_per_layer_mla``) are both still present.

Kunlun's L1<->L2 transfer only exposes the single-pool, layer_first
``torch.ops.xspeedgate_ops.transfer_kv_per_layer_mla`` operator (no page_first
transpose kernel), and the MLA host pool already runs layer_first, so the mamba
host pool must be layer_first too. The only blocker is that one assert.

Approach
--------
A ``hook_registry`` hook cannot express this. The assert is a hardcoded literal
list *inside* ``__init__`` (no module-level constant to REPLACE), a BEFORE hook
rewriting ``layout`` would make ``init_kv_buffer()`` -- called from ``__init__``
-- allocate page_first buffers, and an AFTER hook comes too late. A REPLACE on
``__init__`` means vendoring ~90 lines of upstream body.

So instead regenerate ``__init__`` from its *live* source with ``"layer_first"``
added to the allowed list, then rebind it. Because newlines/indentation inside
``[...]`` are insignificant, we insert the extra element right after
``"page_first_direct",`` (first and only occurrence inside ``__init__``). This
auto-tracks upstream logic changes.

Failure is non-fatal: this module is imported unconditionally by the package
``__init__``, so raising here would break *every* startup, including runs with
HiCache disabled that never touch MambaPoolHost. On failure we log and leave the
upstream guard in place -- enabling HiCache on a hybrid model then fails later
with upstream's own ``Unsupported layout: layer_first`` assert.
"""

from __future__ import annotations

import inspect
import textwrap
import logging

logger = logging.getLogger(__name__)

_ANCHOR = '"page_first_direct",'
_INJECT = '"page_first_direct", "layer_first",'


def _install_mamba_layer_first_support() -> None:
    from sglang.srt.mem_cache.pool_host import mamba as _mamba

    cls = _mamba.MambaPoolHost
    if getattr(cls, "_kunlun_layout_relaxed", False):
        return

    src = textwrap.dedent(inspect.getsource(cls.__init__))
    if _ANCHOR not in src:
        raise RuntimeError(
            f"anchor {_ANCHOR!r} not found in MambaPoolHost.__init__ "
            "(upstream changed; update kunlun_mamba_hicache.py)"
        )

    patched = src.replace(_ANCHOR, _INJECT, 1)
    if '"layer_first"' not in patched:
        raise RuntimeError("failed to inject 'layer_first' into the layout guard")

    namespace: dict = {}
    exec(compile(patched, _mamba.__file__, "exec"), _mamba.__dict__, namespace)
    new_init = namespace["__init__"]
    new_init.__qualname__ = "MambaPoolHost.__init__"

    cls.__init__ = new_init
    cls._kunlun_layout_relaxed = True
    logger.info(
        "Kunlun HiCache: MambaPoolHost layout guard relaxed to allow "
        "layer_first (hybrid KDA state L2)."
    )


try:
    _install_mamba_layer_first_support()
except Exception as e:
    # Non-fatal: only HiCache on hybrid (mamba/KDA) models needs this. Everything
    # else -- including HiCache-disabled startups -- must keep working.
    logger.warning(
        "Kunlun HiCache: cannot relax the MambaPoolHost layout guard (%s). "
        "HiCache on hybrid KDA models will fail with upstream's "
        "'Unsupported layout: layer_first' assert; other paths are unaffected.",
        e,
    )
