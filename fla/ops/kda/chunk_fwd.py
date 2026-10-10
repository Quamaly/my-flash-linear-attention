# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import torch

from fla.ops.backends import dispatch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.cp import FLACPContext
from fla.ops.cp.chunk_delta_h import chunk_gated_delta_rule_fwd_h_pre_process, compress_h0
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import IS_NPU

# ---------------------------------------------------------------------------
# ATEM fuse (w1v2, FLA_KDA_ATEM_FUSE): the fwd gate producer
# (chunk_local_cumsum persistent kernel) dual-stores a head-major [B, H, T, S]
# fp32 copy of the cumsum'd gate (w4/y3 producer-copy doctrine). The copy
# feeds the T-contig gk reads of recompute_w_u (fwd) and, saved for backward,
# the wy dw/k parts' G_T_CONTIG reads (bwd) -- killing both host fp32 gate
# transposes (~1.7ms each @t128k). On by default (combo18); FLA_KDA_ATEM_FUSE=0
# restores the byte-exact pre-dual-store path; any dual failure latches the
# feature off for the process (host transposes run).
# ---------------------------------------------------------------------------
_ATEM_FUSE_ENV = 'FLA_KDA_ATEM_FUSE'
_atem_fuse_broken = False


def _atem_fuse_enabled() -> bool:
    if _atem_fuse_broken:
        return False
    return IS_NPU and os.environ.get(_ATEM_FUSE_ENV, '1').strip().lower() not in ('0', 'false', 'off')


def _atem_gate_t_buf(g: torch.Tensor, use_graph: bool) -> torch.Tensor | None:
    """Caller-allocated head-major dual-write buffer, or None = current path."""
    if not _atem_fuse_enabled() or use_graph:
        return None
    B, T, H, S = g.shape
    return g.new_empty(B, H, T, S, dtype=torch.float)


@dispatch('kda')
def chunk_kda_fwd_stage0_4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int,
    safe_gate: bool,
    use_gate_in_kernel: bool,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    disable_recompute: bool,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None,
    use_graph: bool,
):
    """Fused gate+cumsum+A+solve+recompute forward head (stages 0-4) in one launch.

    Returns ``(g_cumsum, w, u, qg, kg, Aqk, Akk)`` with the same contracts as
    the staged pipeline (``qg`` is ``None`` unless ``disable_recompute`` is
    set), or ``None`` when no fused implementation is available for the given
    inputs; callers then fall back to the standard multi-kernel pipeline.
    Backends must only accept small aligned non-varlen shapes (T <= 4096).
    """
    return None


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context: FLACPContext | None = None,
    use_graph: bool = False,
    chunk_offsets: torch.LongTensor | None = None,
    qk_l2norm_out: dict | None = None,
):
    # g_org is the PRE-cumsum gate on both paths (raw f_proj output when
    # use_gate_in_kernel, else the pre-transformed gate). The Ascend NPU
    # intra backend folds it into its persistent per-chunk kernel as a
    # decay-table prologue; every other backend ignores it.
    g_org = g

    # Small aligned non-varlen shapes may take the fully fused stage0-4 path
    # (gate + cumsum + intra A + solve + recompute in a single launch).
    # Returns None when no backend supports the shape; the staged pipeline
    # below is then unchanged. The stage0-4 kernel has no FOLD_L2N support,
    # so when the caller skipped its standalone l2norm (qk_l2norm_out bundle
    # active) it is gated off here; the staged pipeline below then either
    # folds the l2norm inside the NPU intra kernel or materializes it on the
    # host.
    fused = None
    if cp_context is None and qk_l2norm_out is None:
        fused = chunk_kda_fwd_stage0_4(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            disable_recompute=disable_recompute,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_graph=use_graph,
        )

    if fused is not None:
        g, w, u, qg, kg, Aqk, Akk = fused
        # The stage0-4 fused path skips intra entirely, so no WY producer ran
        # to emit qg_scaled; the tail wrapper computes it with torch ops.
        qg_scaled = None
        g_t = None
    else:
        g_t = None
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=RCP_LN2,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound,
                use_graph=use_graph,
            )
        else:
            g_t = _atem_gate_t_buf(g, use_graph)
            if g_t is not None:
                try:
                    g = chunk_local_cumsum(
                        g=g,
                        scale=RCP_LN2,
                        chunk_size=chunk_size,
                        cu_seqlens=cu_seqlens,
                        chunk_indices=chunk_indices,
                        use_graph=use_graph,
                        g_t_out=g_t,
                    )
                except Exception:
                    # permanent latch: dual-store path is dead for this
                    # process; rerun the plain launch so g stays byte-exact.
                    global _atem_fuse_broken
                    _atem_fuse_broken = True
                    g_t = None
                    g = chunk_local_cumsum(
                        g=g,
                        scale=RCP_LN2,
                        chunk_size=chunk_size,
                        cu_seqlens=cu_seqlens,
                        chunk_indices=chunk_indices,
                        use_graph=use_graph,
                    )
            else:
                g = chunk_local_cumsum(
                    g=g,
                    scale=RCP_LN2,
                    chunk_size=chunk_size,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    use_graph=use_graph,
                )

        # Fused h+o tail (Ascend NPU): the conservative input-space pre-check
        # runs BEFORE intra so the WY producer can emit the scaled gated query
        # (qg_scaled) in the same pass; the authoritative post-check on
        # (kg, u) runs below.
        fused_tail_candidate = False
        if IS_NPU:
            from fla.ops.kda.backends.triton_ascend.chunk_fwd_tail import (
                chunk_kda_fwd_h_o_fused_candidate,
            )

            fused_tail_candidate = chunk_kda_fwd_h_o_fused_candidate(
                q=q,
                k=k,
                v=v,
                T=q.shape[1],
                K=k.shape[-1],
                V=v.shape[-1],
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                state_v_first=state_v_first,
                use_graph=use_graph,
                cp_context=cp_context,
            )

        # qg is None unless disable_recompute is set; qg_scaled is not None
        # only on the NPU fused-tail candidate path (None from every other
        # backend). qk_l2norm_out and g_raw drive the NPU intra kernel's
        # l2norm fold and decay-table prologue; other backends ignore them.
        w, u, qg, kg, Aqk, Akk, qg_scaled = chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=v,
            gk=g,
            gk_t=g_t,
            beta=beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            safe_gate=safe_gate,
            disable_recompute=disable_recompute,
            use_graph=use_graph,
            store_qg_scaled=fused_tail_candidate,
            qk_l2norm_out=qk_l2norm_out,
            g_raw=g_org,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            gate_scale=RCP_LN2,
        )

    if qk_l2norm_out is not None:
        # the intra kernel (or its fallback) materialized the normalized q/k
        # into the bundle; the tail/o kernels below must read those
        q = qk_l2norm_out['q']

    if cp_context is not None:
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=kg,
            w=w,
            u=u,
            gk=g,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            context=cp_context,
            chunk_size=chunk_size,
            state_v_first=state_v_first,
            use_graph=use_graph,
        )

    # Fused h+o tail (Ascend NPU): carry the [BK, BV] fp32 state in registers
    # through the chunk loop and compute o in the same pass, skipping the
    # [B, NT, HV, K, V] h materialization (and v_new when unused). Fallback to
    # the unfused fwd_h + o pair for varlen/unaligned/exotic shapes.
    use_fused_tail = False
    if IS_NPU:
        from fla.ops.kda.backends.triton_ascend.chunk_fwd_tail import (
            chunk_kda_fwd_h_o_fused_npu,
            chunk_kda_fwd_h_o_fused_ok,
        )

        use_fused_tail = chunk_kda_fwd_h_o_fused_ok(
            q=q,
            k=kg,
            u=u,
            T=q.shape[1],
            K=kg.shape[-1],
            V=u.shape[-1],
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            state_v_first=state_v_first,
            use_graph=use_graph,
            cp_context=cp_context,
        )

    if use_fused_tail:
        # The bwd reads saved h/v_new only when disable_recompute=True.
        # qg_scaled was requested from the WY producer when the input-space
        # candidate check passed; otherwise the tail wrapper computes it with
        # torch ops (fallback, e.g. exotic strided inputs or dispatch off).
        o, h, v_new, final_state = chunk_kda_fwd_h_o_fused_npu(
            q=q,
            k=kg,
            w=w,
            u=u,
            g=g,
            A=Aqk,
            gk=g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            store_h=disable_recompute or return_intermediate_states,
            save_new_value=disable_recompute,
            qg_scaled=qg_scaled,
        )
    else:
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            chunk_size=chunk_size,
            state_v_first=state_v_first,
        )

        if cp_context is not None:
            # In Context Parallel (CP) mode, global initial states are not supported at the entry point.
            # The `initial_state` here is computed internally via inter-rank communication.
            # Since only the first sequence in the local batch can be a continuation of a cross-rank sequence,
            # only the first state in the tensor is relevant. We compress it to optimize memory for `save_for_backward`.
            initial_state = compress_h0(initial_state, context=cp_context)

        o = chunk_gla_fwd_o_gk(
            q=q,
            v=v_new,
            g=g,
            A=Aqk,
            h=h,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
            use_graph=use_graph,
        )
    if disable_recompute is False:
        # Delete to save memory
        w, u, qg, kg, v_new = None, None, None, None, None
        if not return_intermediate_states:
            h = None
        if use_gate_in_kernel:
            g = None
    # g_t: the producer-written head-major gate copy (None unless the ATEM
    # dual-store ran and use_gate_in_kernel left the plain-cumsum path); the
    # caller may retain it for the bwd wy consumers.
    return o, final_state, g, g_t, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state
