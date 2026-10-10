# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for KDA ops.

Naming scheme (kda backend + wrappers):
- Device kernels (``@triton.jit``, launched directly) end with
  ``..._kernel_<variant>_npu`` (variant tags such as ``_fold``,
  ``_decay_table``, ``_nc2``, ``_ilp`` sit between the base kernel name and
  the ``_npu`` suffix).
- Host entry points are ``<public op name>_npu`` with no ``kernel`` token
  (e.g. ``chunk_kda_fwd_intra_npu`` wraps ``chunk_kda_fwd_kernel_*_npu``).
- Private device-side jit helpers and host helpers carry a leading ``_``;
  the ``_npu`` tag on them is optional.

FLA_KDA_* environment switches (complete list; all are read once at module
import or per call as noted, and all have permanent in-process fallbacks on
launch failure):

===============================  ==============  ==========================================
Switch                           Default         Effect
===============================  ==============  ==========================================
FLA_KDA_TAIL_ILP                 ``1`` (on)      chunk_fwd_tail: ``1`` selects the
                                                 ILP-rescheduled fused h+o tail kernel
                                                 (bit-identical); ``0``/``false``/``off``
                                                 restore the serial-schedule kernel.
FLA_KDA_FUSED_TAIL_BV            unset (=128)    chunk_fwd_tail: fused h+o tail V-block
                                                 width override for bench sweeps; only
                                                 16/32/64/128 are honoured, other or
                                                 unparsable values are ignored.
FLA_KDA_TAIL_GRID_CLAMP          ``1`` (on)      chunk_fwd_tail: m5 clamp hygiene —
                                                 ``0`` restores the grid=num_aicore
                                                 launch; the default launches
                                                 min(num_aicore, task_num) blocks so no
                                                 MIX block binds a cube core just to
                                                 run an empty task loop.
FLA_KDA_TAIL_BV_POLICY           ``0`` (off)     chunk_fwd_tail: ``1`` re-enables d1's
                                                 occupancy policy (BV 128->64 when the
                                                 128-wide grid under-fills the cube
                                                 cores). Default OFF: the C bench matrix
                                                 measured +14% e2e pollution from it.
FLA_KDA_DH_GRID_CLAMP            ``1`` (on)      chunk_delta_h (fwd_h oneslab/blockdim64
                                                 + bwd dhu): m5 clamp hygiene — ``0``
                                                 restores the grid=num_aicore launch on
                                                 all three persistent launches of that
                                                 module.
FLA_KDA_RETAIN_ACTIVATION_BUDGET 8 GiB           fla/ops/kda/chunk.py (wrapper, not this
                                                 package): activation retain budget in
                                                 bytes for the bwd recompute-skip
                                                 heuristic; ``0`` always recomputes.
FLA_KDA_ATEM_FUSE               ``1`` (on)      w1v2 aten-tail lever (combo18 default):
                                                 fwd chunk_local_cumsum persistent
                                                 producer dual-store a head-major
                                                 [B, H, T, S] fp32 copy of the cumsum'd
                                                 gate (w4 producer-copy doctrine); the
                                                 fwd recompute_w_u stage and the bwd wy
                                                 stage (via save_for_backward retention)
                                                 consume it directly, dropping both host
                                                 fp32 gate transposes (~1.7ms each
                                                 @t128k). ``0`` = byte-exact current
                                                 path; any dual failure latches off for
                                                 the process and the host transposes
                                                 run.
===============================  ==============  ==========================================
"""

from __future__ import annotations

import triton

from fla.ops.backends import BaseBackend

_SUPPORTED_INTRA_CHUNK_SIZES = (32, 64)
_SUB_CHUNK = 16


def _chunk_size_from(args, kwargs, default: int = 64) -> int:
    if 'chunk_size' in kwargs and kwargs['chunk_size'] is not None:
        return int(kwargs['chunk_size'])
    return default


def _verify_intra_chunk_size(args, kwargs):
    chunk_size = _chunk_size_from(args, kwargs)
    if chunk_size not in _SUPPORTED_INTRA_CHUNK_SIZES:
        return False, f'KDA Ascend intra only supports chunk_size in {_SUPPORTED_INTRA_CHUNK_SIZES}, got {chunk_size}'
    return True, None


def _verify_subchunk_aligned(args, kwargs):
    chunk_size = _chunk_size_from(args, kwargs)
    if chunk_size % _SUB_CHUNK != 0:
        return False, f'KDA Ascend bwd requires chunk_size % {_SUB_CHUNK} == 0, got {chunk_size}'
    return True, None


def _verify_bwd_intra_k(args, kwargs):
    # beyond the single-block UB envelope dispatch falls back to the mainline kernel
    k = args[1] if len(args) > 1 else kwargs['k']
    K = int(k.shape[-1])
    safe_gate = kwargs.get('safe_gate', False)
    limit = 512 if safe_gate else 256
    if triton.next_power_of_2(K) > limit:
        return False, f'KDA Ascend bwd_intra only supports K <= {limit} with safe_gate={safe_gate}, got K={K}'
    return True, None


class TritonAscendKDABackend(BaseBackend):
    """Ascend NPU backend for KDA gate, intra, WY, and backward kernels."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_kda_fwd_intra_verifier(self, *args, **kwargs):
        return _verify_intra_chunk_size(args, kwargs)

    def chunk_kda_fwd_intra(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_intra import chunk_kda_fwd_intra_npu
        return chunk_kda_fwd_intra_npu(*args, **kwargs)

    def chunk_kda_fwd_intra_token_parallel_verifier(self, *args, **kwargs):
        return _verify_intra_chunk_size(args, kwargs)

    def chunk_kda_fwd_intra_token_parallel(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_intra_token_parallel import (
            chunk_kda_fwd_intra_token_parallel_npu,
        )
        return chunk_kda_fwd_intra_token_parallel_npu(*args, **kwargs)

    def recompute_w_u_fwd_verifier(self, *args, **kwargs):
        return True, None

    def recompute_w_u_fwd(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.wy_fast import recompute_w_u_fwd_kda_npu
        return recompute_w_u_fwd_kda_npu(*args, **kwargs)

    def chunk_kda_bwd_intra_verifier(self, *args, **kwargs):
        ok, reason = _verify_intra_chunk_size(args, kwargs)
        if not ok:
            return ok, reason
        return _verify_bwd_intra_k(args, kwargs)

    def chunk_kda_bwd_intra(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_intra import chunk_kda_bwd_intra_npu
        return chunk_kda_bwd_intra_npu(*args, **kwargs)

    def chunk_kda_bwd_wy_dqkg_fused_verifier(self, *args, **kwargs):
        return _verify_subchunk_aligned(args, kwargs)

    def chunk_kda_bwd_wy_dqkg_fused(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused_npu
        return chunk_kda_bwd_wy_dqkg_fused_npu(*args, **kwargs)

    def chunk_kda_bwd_dAv_verifier(self, *args, **kwargs):
        return True, None

    def chunk_kda_bwd_dAv(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_bwd import chunk_kda_bwd_dAv_npu
        return chunk_kda_bwd_dAv_npu(*args, **kwargs)

    def kda_gate_fwd_verifier(self, *args, **kwargs):
        return True, None

    def kda_gate_fwd(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.gate import kda_gate_fwd_npu
        return kda_gate_fwd_npu(*args, **kwargs)

    def kda_gate_bwd_verifier(self, *args, **kwargs):
        return True, None

    def kda_gate_bwd(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.gate import kda_gate_bwd_npu
        return kda_gate_bwd_npu(*args, **kwargs)

    def kda_gate_chunk_cumsum_verifier(self, *args, **kwargs):
        return True, None

    def kda_gate_chunk_cumsum(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.gate import kda_gate_chunk_cumsum_npu
        return kda_gate_chunk_cumsum_npu(*args, **kwargs)

    def fused_kda_gate_verifier(self, *args, **kwargs):
        return True, None

    def fused_kda_gate(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.gate import fused_kda_gate_npu
        return fused_kda_gate_npu(*args, **kwargs)

    def fused_recurrent_kda_fwd_verifier(self, *args, **kwargs):
        return True, None

    def fused_recurrent_kda_fwd(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.fused_recurrent import fused_recurrent_kda_fwd_npu

        return fused_recurrent_kda_fwd_npu(*args, **kwargs)

    def chunk_kda_fwd_stage0_4_verifier(self, *args, **kwargs):
        import torch

        k = kwargs.get('k')
        v = kwargs.get('v')
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            return False, 'fused small-T stage0_4 needs k/v tensors'
        if kwargs.get('cu_seqlens') is not None or kwargs.get('chunk_indices') is not None:
            return False, 'fused small-T stage0_4 supports non-varlen inputs only'
        if kwargs.get('use_graph'):
            return False, 'fused small-T stage0_4 does not support use_graph'
        if not kwargs.get('safe_gate'):
            return False, 'fused small-T stage0_4 requires safe_gate'
        if kwargs.get('chunk_size') != 64:
            return False, f"fused small-T stage0_4 requires chunk_size 64, got {kwargs.get('chunk_size')}"
        if kwargs.get('scale') is None:
            return False, 'fused small-T stage0_4 requires an explicit scale'
        if k.dim() != 4:
            return False, f'fused small-T stage0_4 expects 4D k, got {k.dim()}D'
        B, T, H, K = k.shape
        HV, V = v.shape[2], v.shape[-1]
        if T not in (2048, 4096):
            return False, f'fused small-T stage0_4 is gated to the production small-T shapes T in (2048, 4096), got T={T} (bench: 2048 fwd 1.587x one-task-per-program grid; 4096 goes persistent-24, the pre-persistent grid benched 0.908x; other T keeps the staged pipeline so backend routing stays observable)'
        if K > 128 or V > 128:
            return False, f'fused small-T stage0_4 requires K,V <= 128, got K={K}, V={V}'
        if HV % H != 0:
            return False, f'fused small-T stage0_4 requires HV % H == 0, got HV={HV}, H={H}'
        # single launch: keep programs within the Ascend task-time budget and
        # every linear offset comfortably inside int32 (bases stay int64 anyway)
        if (T // 64) * B * HV > 4096:
            return False, 'fused small-T stage0_4 exceeds the 4096-program launch budget'
        if B * T * HV * max(K, V) > (1 << 31):
            return False, 'fused small-T stage0_4 exceeds int32-safe element budget'
        if k.dtype != v.dtype or k.dtype.is_floating_point is False:
            return False, f'fused small-T stage0_4 requires matching float dtypes, got {k.dtype}/{v.dtype}'
        return True, None

    def chunk_kda_fwd_stage0_4(self, *args, **kwargs):
        from fla.ops.kda.backends.triton_ascend.chunk_fwd_small_t import chunk_kda_fwd_stage0_4_npu
        return chunk_kda_fwd_stage0_4_npu(*args, **kwargs)
