# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KDA chunk backward kernels for triton-ascend on Ascend NPU."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp2
from fla.utils import ascend_compile_kwargs, input_guard
from fla.utils.launch_fast import fast_launch
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    get_npu_properties,
    max_grid_axis_chunks,
)

_DAV_NUM_WARPS = 2
_DAV_MEM_MULT = 8.0
_DAV_SAFETY_MARGIN = 0.75
_DAV_FALLBACK_TILE = 8
_DAV_MAX_TILE = 64


def _get_dAv_bv(BT: int, V: int) -> int:
    return compute_row_tile_block_size(
        BT, V, _DAV_MEM_MULT,
        tiling_row=False,
        safety_margin=_DAV_SAFETY_MARGIN,
        fallback=_DAV_FALLBACK_TILE,
        min_block=8,
        max_block=min(_DAV_MAX_TILE, triton.next_power_of_2(V)),
    )


def _launch_dAv_2d_kernel(kernel, *, nt: int, bh_total: int, kernel_kwargs: dict) -> None:
    max_nt = max_grid_axis_chunks(nt, bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for nt_off in range(0, nt, max_nt):
        nt_len = min(max_nt, nt - nt_off)
        kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            fast_launch(kernel, (nt_len, bh_len), num_warps=_DAV_NUM_WARPS, **kernel_kwargs)


@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_dAv_npu(
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV
    T_seq = T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    v += (bos * HV + i_hv) * V
    do += (bos * HV + i_hv) * V
    # dv is emitted in the [B, HV, T, V] layout (per-head stride-1 rows) so
    # the dhu kernel loads it with contiguous row blocks and no host
    # transpose is needed between the two stages.
    if IS_VARLEN:
        dv += (tl.cast(i_hv, tl.int64) * T_seq + bos) * V
    else:
        dv += tl.cast(i_bh, tl.int64) * T_seq * V
    dA += (bos * HV + i_hv) * BT

    p_A = tl.make_block_ptr(A + (bos * HV + i_hv) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (V, T), (1, HV * V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do_c = b_do + 0.0
        b_dA = tl.dot(b_do, b_v, b_dA, allow_tf32=False)
        b_A_i = tl.load(p_A, boundary_check=(0, 1))
        b_A_i = tl.where(m_A, b_A_i, 0).to(b_do.dtype)
        b_dv = tl.dot(b_A_i, b_do_c, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    p_dA = tl.make_block_ptr(dA, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def chunk_kda_bwd_dAv_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    A: torch.Tensor | None = None,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_graph:
        raise NotImplementedError("use_graph is not supported on the Ascend NPU backend")
    B, T, HV, V = k.shape[0], k.shape[1], do.shape[2], do.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    BV = _get_dAv_bv(BT, V)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dA = v.new_empty(B, T, HV, BT, dtype=torch.float)
    # [B, HV, T, V] storage, returned as a [B, T, HV, V] view whose transpose
    # is already contiguous: the dhu kernel reads it with stride-1 rows and
    # downstream T-contiguous readers skip the host transpose copy.
    dv_storage = do.new_zeros(B, HV, T, V)
    dv = dv_storage.transpose(1, 2)

    _launch_dAv_2d_kernel(
        chunk_kda_bwd_kernel_dAv_npu,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            v=v,
            A=A,
            do=do,
            dv=dv_storage,
            dA=dA,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T=T,
            HV=HV,
            V=V,
            BT=BT,
            BV=BV,
            IS_VARLEN=cu_seqlens is not None,
            NT_OFFSET=0,
            BH_OFFSET=0,
        ),
    )
    return dA, dv


_BC = 16
_BWD_MEM_MULT = 10.0
_SAFETY_MARGIN = 0.80
_FALLBACK_TILE = 16
_MAX_TILE = 128
# Persistent wy launches never exceed this many programs.
_WY_MAX_PROGRAMS = 24

# wy fused launch (wave25, FLA_KDA_WY_FUSE): when every wy-consumer tensor is
# in the T-contig posture (the production bench posture) and NK == 1, the
# v_part / k_part / dw_part per-task chain runs as ONE launch
# (chunk_kda_bwd_kernel_wy_fused_npu): dA_acc/db flow in registers, dk/dg
# through the same-task L2-warm GM handoff. Default OFF; any gated-launch
# anomaly latches the feature off for the process and the stock three-launch
# path runs.
_WY_FUSE_ENV = 'FLA_KDA_WY_FUSE'
_wy_fuse_broken = False


def _wy_fuse_enabled() -> bool:
    if _wy_fuse_broken:
        return False
    return os.environ.get(_WY_FUSE_ENV, '1').strip().lower() not in ('0', 'false', 'off')

# Block-triangular wy_dw dot #2 zero-block skip (FLA_KDA_WY_TRI, wave25 w1):
# routes dw_part launches to the panelized twin; default off = legacy;
# BT == 64 + T-contig k/g required; failure latches off (proven-path relaunch).
_WY_TRI_ENV = 'FLA_KDA_WY_TRI'
_wy_tri_broken = False


def _wy_tri_enabled(BT: int, k_t_contig: bool, g_t_contig: bool) -> bool:
    # The non-T-contig FINALIZE tri form exceeds the 192KiB UB envelope at
    # compile (BiShengIR ub-overflow 4007936-bit requirement, w1 probe
    # 2026-10-05); the production wrapper always supplies T-contig k/g
    # (head_dim > 1), anything else keeps the legacy kernel.
    if _wy_tri_broken or BT != 64 or not (k_t_contig and g_t_contig):
        return False
    return os.environ.get(_WY_TRI_ENV, '0').strip().lower() in ('1', 'true', 'on')


def _get_bk(K: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        K,
        _BWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=16,
        max_block=min(_MAX_TILE, triton.next_power_of_2(K)),
    )


def _get_bv(V: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        V,
        _BWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=16,
        max_block=min(_MAX_TILE, triton.next_power_of_2(V)),
    )


def _t_contig_arg(x: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, bool]:
    """Transpose [B, T, H*, ...] → [B, H*, T, ...] so T-loads are stride-1.

    For tensors with no in-tree T-contig producer (v, dv, k: user/native
    inputs) — always materializes the transposed copy.
    """
    if head_dim == 1:
        return x, False
    return x.transpose(1, 2).contiguous(), True


def _maybe_t_contig_arg(x: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, bool]:
    """Layout-aware variant for tensors with an in-tree T-contig producer:
    inter_solve's Akk ([B, HV, T, BT]) and the orchestrator-prepared gate/beta
    copies pass through instead of re-transposing. Detection is
    shape[1] == head_dim; for the degenerate T == head_dim the two layouts
    share shape/strides and every in-tree NPU producer of these tensors is
    T-contig, so that reading is preferred. head_dim == 1 keeps the native
    stride-1 path.
    """
    if head_dim == 1:
        return x, False
    if x.shape[1] == head_dim:
        return (x.contiguous(), True) if not x.is_contiguous() else (x, True)
    return x.transpose(1, 2).contiguous(), True


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_kda_bwd_kernel_wy_v_part_npu(
    v,
    beta,
    A,
    dv,
    dv2,
    dA_acc,
    db_acc,
    cu_seqlens,
    chunk_indices,
    T,
    BH,
    task_num,
    num_core,
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
):
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
        else:
            bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

        if G_T_CONTIG:
            if IS_VARLEN:
                v_ptr = v + tl.cast(i_hv, tl.int64) * T_seq * V + bos * V
                dv_ptr = dv + tl.cast(i_hv, tl.int64) * T_seq * V + bos * V
                A_ptr = A + tl.cast(i_hv, tl.int64) * T_seq * BT + bos * BT
                beta_ptr = beta + tl.cast(i_hv, tl.int64) * T_seq + bos
            else:
                hv_off = tl.cast(i_b, tl.int64) * HV + i_hv
                v_ptr = v + hv_off * T_seq * V
                dv_ptr = dv + hv_off * T_seq * V
                A_ptr = A + hv_off * T_seq * BT
                beta_ptr = beta + hv_off * T_seq
            v_stride_t = V
            a_stride_t = BT
            beta_stride = 1
        else:
            v_ptr = v + (bos * HV + i_hv) * V
            dv_ptr = dv + (bos * HV + i_hv) * V
            A_ptr = A + (bos * HV + i_hv) * BT
            beta_ptr = beta + bos * HV + i_hv
            v_stride_t = HV * V
            a_stride_t = HV * BT
            beta_stride = HV

        dv2_ptr = dv2 + (bos * HV + i_hv) * V
        dA_ptr = dA_acc + (bos * HV + i_hv) * BT
        db_ptr = db_acc + bos * HV + i_hv

        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (BT, BT), (0, 1))
        p_beta = tl.make_block_ptr(beta_ptr, (T,), (beta_stride,), (i_t * BT,), (BT,), (0,))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_beta = tl.load(p_beta, boundary_check=(0,))

        b_dA = tl.zeros([BT, BT], dtype=tl.float32)
        b_db = tl.zeros([BT], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_dv = tl.make_block_ptr(dv_ptr, (T, V), (v_stride_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_v = tl.make_block_ptr(v_ptr, (T, V), (v_stride_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_dA = tl.dot(b_dv, tl.trans(b_v), b_dA, allow_tf32=False)
            # Ascend tl.dot clobbers lhs; copy A before every V-slab use.
            b_A_c = b_A + 0.0
            b_dvb = tl.dot(b_A_c, b_dv, allow_tf32=False)
            b_db += tl.sum(b_dvb * b_v, 1)
            p_dv2 = tl.make_block_ptr(dv2_ptr, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_dv2, (b_dvb * b_beta[:, None]).to(p_dv2.dtype.element_ty), boundary_check=(0, 1))

        p_dA = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_db = tl.make_block_ptr(db_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_kda_bwd_kernel_wy_k_part_npu(
    q,
    k,
    v_new,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dg,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    scale,
    T,
    BH,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    K_OFFSET: tl.constexpr,
):
    i_k = K_OFFSET
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + i_t.to(tl.int64)
        else:
            i_tg = tl.cast(i_b, tl.int64) * tl.cdiv(T, BT) + i_t
            bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

        q_ptr = q + (bos * H + i_h) * K
        k_ptr = k + (bos * H + i_h) * K
        v_new_ptr = v_new + (bos * HV + i_hv) * V
        if G_T_CONTIG:
            # g passed in the wy-consumer layout [B, HV, T, K]
            # (shared with recompute_w_u / dw_part: one host transpose total).
            if IS_VARLEN:
                g_ptr = g + tl.cast(i_hv, tl.int64) * T_seq * K + bos * K
            else:
                g_ptr = g + tl.cast(i_b * HV + i_hv, tl.int64) * T_seq * K
            g_stride_t = K
        else:
            g_ptr = g + (bos * HV + i_hv) * K
            g_stride_t = HV * K
        h_ptr = h + (i_tg * HV + i_hv) * K * V
        do_ptr = do + (bos * HV + i_hv) * V
        dh_ptr = dh + (i_tg * HV + i_hv) * K * V
        dq_ptr = dq + (bos * HV + i_hv) * K
        dk_ptr = dk + (bos * HV + i_hv) * K
        dg_ptr = dg + (bos * HV + i_hv) * K

        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_gn = g_ptr + (min(T, i_t * BT + BT) - 1).to(tl.int64) * g_stride_t + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        o_i = tl.arange(0, BC)
        n_sub = BT // BC
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                p_dh = tl.make_block_ptr(dh_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                p_dh = tl.make_block_ptr(dh_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            b_dgk += tl.sum(b_h * b_dh, axis=0)

        b_dgk *= exp2(b_gn)

        b_kdk_sum = tl.zeros([BK], dtype=tl.float32)
        for s in range(n_sub):
            i_tc_s = i_t * BT + s * BC
            m_s = (i_tc_s + o_i) < T

            p_k = tl.make_block_ptr(k_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g_ptr, (T, K), (g_stride_t, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

            b_dk = tl.zeros([BC, BK], dtype=tl.float32)
            for i_v in range(tl.cdiv(V, BV)):
                p_v_new = tl.make_block_ptr(v_new_ptr, (T, V), (HV * V, 1), (i_tc_s, i_v * BV), (BC, BV), (1, 0))
                if STATE_V_FIRST:
                    p_dh = tl.make_block_ptr(dh_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                else:
                    p_dh = tl.make_block_ptr(dh_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
                b_dh = tl.load(p_dh, boundary_check=(0, 1))
                b_dk = tl.dot(b_v_new, b_dh.to(b_v_new.dtype), b_dk, allow_tf32=False)

            b_dk = b_dk * tl.where(m_s[:, None], exp2(b_gn[None, :] - b_g), 0)
            b_kdk_sum += tl.sum(b_k * b_dk, axis=0)
            p_dk = tl.make_block_ptr(dk_ptr, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        b_dgk_total = b_dgk + b_kdk_sum

        for s in range(n_sub):
            i_tc_s = i_t * BT + s * BC
            m_last_s = (i_tc_s + o_i) == min(T, i_t * BT + BT) - 1

            p_k = tl.make_block_ptr(k_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g_ptr, (T, K), (g_stride_t, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_q = tl.make_block_ptr(q_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk_ptr, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)

            b_dq = tl.zeros([BC, BK], dtype=tl.float32)
            for i_v in range(tl.cdiv(V, BV)):
                p_do = tl.make_block_ptr(do_ptr, (T, V), (HV * V, 1), (i_tc_s, i_v * BV), (BC, BV), (1, 0))
                if STATE_V_FIRST:
                    p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                else:
                    p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                b_do = tl.load(p_do, boundary_check=(0, 1))
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_dq = tl.dot(b_do, b_h.to(b_do.dtype), b_dq, allow_tf32=False)

            b_dq = b_dq * exp2(b_g) * scale
            b_dg = b_q * b_dq - b_k * b_dk + m_last_s[:, None] * b_dgk_total

            p_dq = tl.make_block_ptr(dq_ptr, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_dg = tl.make_block_ptr(dg_ptr, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _wy_dw_part_tri_task_npu(
    k_ptr, k_stride_t,
    g_ptr, g_stride_t,
    beta_ptr, beta_stride,
    A_ptr, a_stride_t,
    h_ptr,
    dv_ptr, dv_stride_t,
    dA_ptr, db_ptr, dg_ptr, dk_ptr, dA_out_ptr, db_out_ptr,
    i_t, i_k, T,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NB: tl.constexpr,
    STATE_V_FIRST: tl.constexpr, FINALIZE: tl.constexpr,
):
    """Block-triangular wy_dw task body: 32x32 sub-block skip of dot #2.
    Loaded A = TRANSPOSED stored Akk (unit diag + strict upper); its
    rows 32..63 x cols 0..31 sub-block (bi > bj) is exact GM zeros ->
    eliding it is a multiply-by-zero elision (fp32 identity). dw/kg are
    BORN as 32-row panels (in-register row slicing unavailable: join/split
    blow the 192KiB UB envelope); #1/#3/#4/#5 + tail keep exact MAC counts
    and value DAGs — only tiling boundaries move; details inline."""
    # ---- dot #1 (dw accumulation), born as two 32-row panels -------------
    b_dw_lo = tl.zeros([NB, BK], dtype=tl.float32)
    b_dw_hi = tl.zeros([NB, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_dv_lo = tl.make_block_ptr(dv_ptr, (T, V), (dv_stride_t, 1), (i_t * BT, i_v * BV), (NB, BV), (1, 0))
        p_dv_hi = tl.make_block_ptr(dv_ptr, (T, V), (dv_stride_t, 1), (i_t * BT + NB, i_v * BV), (NB, BV), (1, 0))
        if STATE_V_FIRST:
            p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_dv_lo = tl.load(p_dv_lo, boundary_check=(0, 1))
        b_dv_hi = tl.load(p_dv_hi, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_h_a = b_h.to(b_dv_lo.dtype)
        b_dw_lo = tl.dot(b_dv_lo, b_h_a, b_dw_lo, allow_tf32=False)
        b_dw_hi = tl.dot(b_dv_hi, b_h_a, b_dw_hi, allow_tf32=False)
    p_k_lo = tl.make_block_ptr(k_ptr, (T, K), (k_stride_t, 1), (i_t * BT, i_k * BK), (NB, BK), (1, 0))
    p_k_hi = tl.make_block_ptr(k_ptr, (T, K), (k_stride_t, 1), (i_t * BT + NB, i_k * BK), (NB, BK), (1, 0))
    p_g_lo = tl.make_block_ptr(g_ptr, (T, K), (g_stride_t, 1), (i_t * BT, i_k * BK), (NB, BK), (1, 0))
    p_g_hi = tl.make_block_ptr(g_ptr, (T, K), (g_stride_t, 1), (i_t * BT + NB, i_k * BK), (NB, BK), (1, 0))
    p_beta_lo = tl.make_block_ptr(beta_ptr, (T,), (beta_stride,), (i_t * BT,), (NB,), (0,))
    p_beta_hi = tl.make_block_ptr(beta_ptr, (T,), (beta_stride,), (i_t * BT + NB,), (NB,), (0,))
    b_k_lo = tl.load(p_k_lo, boundary_check=(0, 1))
    b_k_hi = tl.load(p_k_hi, boundary_check=(0, 1))
    b_g_lo = tl.load(p_g_lo, boundary_check=(0, 1)).to(tl.float32)
    b_g_hi = tl.load(p_g_hi, boundary_check=(0, 1)).to(tl.float32)
    b_beta_lo, b_beta_hi = tl.load(p_beta_lo, boundary_check=(0,)), tl.load(p_beta_hi, boundary_check=(0,))
    b_gk_exp_lo, b_gk_exp_hi = exp2(b_g_lo), exp2(b_g_hi)
    b_kg_lo = b_k_lo * b_gk_exp_lo
    b_kg_hi = b_k_hi * b_gk_exp_hi
    b_gb_lo = b_gk_exp_lo * b_beta_lo[:, None]
    b_gb_hi = b_gk_exp_hi * b_beta_hi[:, None]
    # A diagonal sub-blocks for dot #2; the (1,0) block is never loaded.
    p_Au00 = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (NB, NB), (0, 1))
    p_Au01 = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT + NB), (NB, NB), (0, 1))
    p_Au11 = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (NB, i_t * BT + NB), (NB, NB), (0, 1))
    b_Au00 = tl.load(p_Au00, boundary_check=(0, 1))
    b_Au01 = tl.load(p_Au01, boundary_check=(0, 1))
    b_Au11 = tl.load(p_Au11, boundary_check=(0, 1))
    b_dw_lo = -b_dw_lo.to(b_Au00.dtype)
    b_dw_hi = -b_dw_hi.to(b_Au00.dtype)
    b_kg_lo_a = b_kg_lo.to(b_Au00.dtype)
    b_kg_hi_a = b_kg_hi.to(b_Au00.dtype)
    # dot #2: dkgb = A @ dw, (1,0) zero sub-block skipped
    b_dkgb_lo = tl.dot(b_Au00, b_dw_lo, allow_tf32=False)
    b_dkgb_lo = tl.dot(b_Au01, b_dw_hi, b_dkgb_lo, allow_tf32=False)
    b_dkgb_hi = tl.dot(b_Au11, b_dw_hi, allow_tf32=False)
    # dot #3: dA += dw @ kg^T as four 32x32 quadrants Q_ij =
    # dot(dw_i, trans(kg_j_a), loaded Q_ij); reduction stays BK-long.
    p_dAq00 = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (NB, NB), (1, 0))
    p_dAq01 = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT, NB), (NB, NB), (1, 0))
    p_dAq10 = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT + NB, 0), (NB, NB), (1, 0))
    p_dAq11 = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT + NB, NB), (NB, NB), (1, 0))
    b_kgT_lo, b_kgT_hi = tl.trans(b_kg_lo_a), tl.trans(b_kg_hi_a)
    # lhs-clobber law: dw panels are lhs twice each; second dot takes +0.0.
    b_dw_lo_c, b_dw_hi_c = b_dw_lo + 0.0, b_dw_hi + 0.0
    b_Q00 = tl.load(p_dAq00, boundary_check=(0, 1)).to(tl.float32)
    b_Q00 = tl.dot(b_dw_lo, b_kgT_lo, b_Q00, allow_tf32=False)
    b_Q01 = tl.load(p_dAq01, boundary_check=(0, 1)).to(tl.float32)
    b_Q01 = tl.dot(b_dw_lo_c, b_kgT_hi, b_Q01, allow_tf32=False)
    b_Q10 = tl.load(p_dAq10, boundary_check=(0, 1)).to(tl.float32)
    b_Q10 = tl.dot(b_dw_hi, b_kgT_lo, b_Q10, allow_tf32=False)
    b_Q11 = tl.load(p_dAq11, boundary_check=(0, 1)).to(tl.float32)
    b_Q11 = tl.dot(b_dw_hi_c, b_kgT_hi, b_Q11, allow_tf32=False)
    if FINALIZE:
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
        o_tl = i_t * BT + tl.arange(0, NB)
        o_th = o_tl + NB
        m_tl, m_th = o_tl < T, o_th < T
        m_Aq00 = (o_tl[:, None] > o_tl[None, :]) & (m_tl[:, None] & m_tl[None, :])
        m_Aq01 = (o_tl[:, None] > o_th[None, :]) & (m_tl[:, None] & m_th[None, :])
        m_Aq10 = (o_th[:, None] > o_tl[None, :]) & (m_th[:, None] & m_tl[None, :])
        m_Aq11 = (o_th[:, None] > o_th[None, :]) & (m_th[:, None] & m_th[None, :])
        b_Q00 = tl.where(m_Aq00, b_Q00 * b_beta_lo[None, :], 0)
        b_Q01 = tl.where(m_Aq01, b_Q01 * b_beta_hi[None, :], 0)
        b_Q10 = tl.where(m_Aq10, b_Q10 * b_beta_lo[None, :], 0)
        b_Q11 = tl.where(m_Aq11, b_Q11 * b_beta_hi[None, :], 0)
        # dot #4: mid = (mask*dA*beta) @ A; mid row panels via block matmul
        # over dA column quadrants against A row strips
        p_At = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (NB, BT), (0, 1))
        p_Ab = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (NB, i_t * BT), (NB, BT), (0, 1))
        b_At, b_Ab = tl.load(p_At, boundary_check=(0, 1)), tl.load(p_Ab, boundary_check=(0, 1))
        b_mid_lo = tl.dot(b_Q00.to(b_At.dtype), b_At, allow_tf32=False)
        b_mid_lo = tl.dot(b_Q01.to(b_At.dtype), b_Ab, b_mid_lo, allow_tf32=False)
        b_mid_hi = tl.dot(b_Q10.to(b_At.dtype), b_At, allow_tf32=False)
        b_mid_hi = tl.dot(b_Q11.to(b_At.dtype), b_Ab, b_mid_hi, allow_tf32=False)
        # dot #5: fin = A @ mid via A column blocks (A_l lower half = zeros)
        p_Al = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (BT, NB), (0, 1))
        p_Ar = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT + NB), (BT, NB), (0, 1))
        b_Al, b_Ar = tl.load(p_Al, boundary_check=(0, 1)), tl.load(p_Ar, boundary_check=(0, 1))
        b_fin = tl.dot(b_Al, b_mid_lo.to(b_Al.dtype), allow_tf32=False)
        b_fin = tl.dot(b_Ar, b_mid_hi.to(b_Al.dtype), b_fin, allow_tf32=False)
        p_dA_out = tl.make_block_ptr(dA_out_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_dA_out, tl.where(m_A, -b_fin, 0).to(p_dA_out.dtype.element_ty), boundary_check=(0, 1))
    else:
        tl.store(p_dAq00, b_Q00.to(p_dAq00.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dAq01, b_Q01.to(p_dAq01.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dAq10, b_Q10.to(p_dAq10.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dAq11, b_Q11.to(p_dAq11.dtype.element_ty), boundary_check=(0, 1))
    # tail: db/dk/dg RMW per panel
    p_db_acc_lo = tl.make_block_ptr(db_ptr, (T,), (HV,), (i_t * BT,), (NB,), (0,))
    p_db_acc_hi = tl.make_block_ptr(db_ptr, (T,), (HV,), (i_t * BT + NB,), (NB,), (0,))
    b_db_lo = tl.load(p_db_acc_lo, boundary_check=(0,)).to(tl.float32)
    b_db_hi = tl.load(p_db_acc_hi, boundary_check=(0,)).to(tl.float32)
    b_db_lo += tl.sum(b_dkgb_lo * b_kg_lo, 1)
    b_db_hi += tl.sum(b_dkgb_hi * b_kg_hi, 1)
    if FINALIZE:
        p_db_out_lo = tl.make_block_ptr(db_out_ptr, (T,), (HV,), (i_t * BT,), (NB,), (0,))
        p_db_out_hi = tl.make_block_ptr(db_out_ptr, (T,), (HV,), (i_t * BT + NB,), (NB,), (0,))
        tl.store(p_db_out_lo, b_db_lo.to(p_db_out_lo.dtype.element_ty), boundary_check=(0,))
        tl.store(p_db_out_hi, b_db_hi.to(p_db_out_hi.dtype.element_ty), boundary_check=(0,))
    else:
        tl.store(p_db_acc_lo, b_db_lo.to(p_db_acc_lo.dtype.element_ty), boundary_check=(0,))
        tl.store(p_db_acc_hi, b_db_hi.to(p_db_acc_hi.dtype.element_ty), boundary_check=(0,))

    p_dk_lo = tl.make_block_ptr(dk_ptr, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (NB, BK), (1, 0))
    p_dk_hi = tl.make_block_ptr(dk_ptr, (T, K), (HV * K, 1), (i_t * BT + NB, i_k * BK), (NB, BK), (1, 0))
    b_dk_lo = tl.load(p_dk_lo, boundary_check=(0, 1)).to(tl.float32)
    b_dk_hi = tl.load(p_dk_hi, boundary_check=(0, 1)).to(tl.float32)
    b_dk_lo = b_dk_lo + b_dkgb_lo * b_gb_lo
    b_dk_hi = b_dk_hi + b_dkgb_hi * b_gb_hi
    tl.store(p_dk_lo, b_dk_lo.to(p_dk_lo.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk_hi, b_dk_hi.to(p_dk_hi.dtype.element_ty), boundary_check=(0, 1))

    p_dg_lo = tl.make_block_ptr(dg_ptr, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (NB, BK), (1, 0))
    p_dg_hi = tl.make_block_ptr(dg_ptr, (T, K), (HV * K, 1), (i_t * BT + NB, i_k * BK), (NB, BK), (1, 0))
    b_dg_lo = tl.load(p_dg_lo, boundary_check=(0, 1)).to(tl.float32)
    b_dg_hi = tl.load(p_dg_hi, boundary_check=(0, 1)).to(tl.float32)
    b_dg_lo = b_dg_lo + b_kg_lo * b_dkgb_lo * b_beta_lo[:, None]
    b_dg_hi = b_dg_hi + b_kg_hi * b_dkgb_hi * b_beta_hi[:, None]
    tl.store(p_dg_lo, b_dg_lo.to(p_dg_lo.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg_hi, b_dg_hi.to(p_dg_hi.dtype.element_ty), boundary_check=(0, 1))

@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_kda_bwd_kernel_wy_dw_part_tri_npu(
    k,
    g,
    beta,
    A,
    h,
    dv,
    dA_acc,
    db_acc,
    dg,
    dk,
    dA,
    db,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    T,
    BH,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    K_T_CONTIG: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    K_OFFSET: tl.constexpr,
    FINALIZE: tl.constexpr,
):
    """Block-triangular twin of chunk_kda_bwd_kernel_wy_dw_part_npu."""
    i_k = K_OFFSET
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + i_t.to(tl.int64)
        else:
            i_tg = tl.cast(i_b, tl.int64) * tl.cdiv(T, BT) + i_t
            bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

        if K_T_CONTIG:
            if IS_VARLEN:
                k_ptr = k + tl.cast(i_h, tl.int64) * T_seq * K + bos * K
            else:
                k_ptr = k + (tl.cast(i_b, tl.int64) * H + i_h) * T_seq * K
            k_stride_t = K
        else:
            k_ptr = k + (bos * H + i_h) * K
            k_stride_t = H * K

        if G_T_CONTIG:
            if IS_VARLEN:
                g_ptr = g + tl.cast(i_hv, tl.int64) * T_seq * K + bos * K
                beta_ptr = beta + tl.cast(i_hv, tl.int64) * T_seq + bos
                A_ptr = A + tl.cast(i_hv, tl.int64) * T_seq * BT + bos * BT
                dv_ptr = dv + tl.cast(i_hv, tl.int64) * T_seq * V + bos * V
            else:
                hv_off = tl.cast(i_b, tl.int64) * HV + i_hv
                g_ptr = g + hv_off * T_seq * K
                beta_ptr = beta + hv_off * T_seq
                A_ptr = A + hv_off * T_seq * BT
                dv_ptr = dv + hv_off * T_seq * V
            g_stride_t = K
            a_stride_t = BT
            dv_stride_t = V
            beta_stride = 1
        else:
            g_ptr = g + (bos * HV + i_hv) * K
            beta_ptr = beta + bos * HV + i_hv
            A_ptr = A + (bos * HV + i_hv) * BT
            dv_ptr = dv + (bos * HV + i_hv) * V
            g_stride_t = HV * K
            a_stride_t = HV * BT
            dv_stride_t = HV * V
            beta_stride = HV

        h_ptr = h + (i_tg * HV + i_hv) * K * V
        dA_ptr = dA_acc + (bos * HV + i_hv) * BT
        db_ptr = db_acc + bos * HV + i_hv
        dg_ptr = dg + (bos * HV + i_hv) * K
        dk_ptr = dk + (bos * HV + i_hv) * K
        dA_out_ptr = dA + (bos * HV + i_hv) * BT
        db_out_ptr = db + bos * HV + i_hv

        _wy_dw_part_tri_task_npu(
            k_ptr, k_stride_t,
            g_ptr, g_stride_t,
            beta_ptr, beta_stride,
            A_ptr, a_stride_t,
            h_ptr,
            dv_ptr, dv_stride_t,
            dA_ptr, db_ptr, dg_ptr, dk_ptr, dA_out_ptr, db_out_ptr,
            i_t, i_k, T,
            HV=HV, K=K, V=V,
            BT=BT, BK=BK, BV=BV, NB=BT // 2,
            STATE_V_FIRST=STATE_V_FIRST, FINALIZE=FINALIZE,
        )


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_kda_bwd_kernel_wy_dw_part_npu(
    k,
    g,
    beta,
    A,
    h,
    dv,
    dA_acc,
    db_acc,
    dg,
    dk,
    dA,
    db,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    T,
    BH,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    K_T_CONTIG: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    K_OFFSET: tl.constexpr,
    FINALIZE: tl.constexpr,
):
    i_k = K_OFFSET
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + i_t.to(tl.int64)
        else:
            i_tg = tl.cast(i_b, tl.int64) * tl.cdiv(T, BT) + i_t
            bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

        if K_T_CONTIG:
            if IS_VARLEN:
                k_ptr = k + tl.cast(i_h, tl.int64) * T_seq * K + bos * K
            else:
                k_ptr = k + (tl.cast(i_b, tl.int64) * H + i_h) * T_seq * K
            k_stride_t = K
        else:
            k_ptr = k + (bos * H + i_h) * K
            k_stride_t = H * K

        if G_T_CONTIG:
            if IS_VARLEN:
                g_ptr = g + tl.cast(i_hv, tl.int64) * T_seq * K + bos * K
                beta_ptr = beta + tl.cast(i_hv, tl.int64) * T_seq + bos
                A_ptr = A + tl.cast(i_hv, tl.int64) * T_seq * BT + bos * BT
                dv_ptr = dv + tl.cast(i_hv, tl.int64) * T_seq * V + bos * V
            else:
                hv_off = tl.cast(i_b, tl.int64) * HV + i_hv
                g_ptr = g + hv_off * T_seq * K
                beta_ptr = beta + hv_off * T_seq
                A_ptr = A + hv_off * T_seq * BT
                dv_ptr = dv + hv_off * T_seq * V
            g_stride_t = K
            a_stride_t = BT
            dv_stride_t = V
            beta_stride = 1
        else:
            g_ptr = g + (bos * HV + i_hv) * K
            beta_ptr = beta + bos * HV + i_hv
            A_ptr = A + (bos * HV + i_hv) * BT
            dv_ptr = dv + (bos * HV + i_hv) * V
            g_stride_t = HV * K
            a_stride_t = HV * BT
            dv_stride_t = HV * V
            beta_stride = HV

        h_ptr = h + (i_tg * HV + i_hv) * K * V
        dA_ptr = dA_acc + (bos * HV + i_hv) * BT
        db_ptr = db_acc + bos * HV + i_hv
        dg_ptr = dg + (bos * HV + i_hv) * K
        dk_ptr = dk + (bos * HV + i_hv) * K
        dA_out_ptr = dA + (bos * HV + i_hv) * BT
        db_out_ptr = db + bos * HV + i_hv

        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_dv = tl.make_block_ptr(dv_ptr, (T, V), (dv_stride_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dw = tl.dot(b_dv, b_h.to(b_dv.dtype), b_dw, allow_tf32=False)

        p_k = tl.make_block_ptr(k_ptr, (T, K), (k_stride_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g_ptr, (T, K), (g_stride_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_beta = tl.make_block_ptr(beta_ptr, (T,), (beta_stride,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (BT, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_beta = tl.load(p_beta, boundary_check=(0,))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_gk_exp = exp2(b_g)
        b_kg = b_k * b_gk_exp
        b_gb = b_gk_exp * b_beta[:, None]
        # Match CUDA: downcast dw/kg to A.dtype before dA / dkgb GEMMs.
        b_dw = -b_dw.to(b_A.dtype)
        b_kg_a = b_kg.to(b_A.dtype)
        if FINALIZE:
            # b_A is the lhs of the dkgb dot and Ascend tl.dot clobbers its lhs;
            # compute on a copy so b_A stays live for the dA finalize epilogue.
            b_A_dkgb = b_A + 0.0
        else:
            b_A_dkgb = b_A
        b_dkgb = tl.dot(b_A_dkgb, b_dw, allow_tf32=False)

        p_dA_acc = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        b_dA = tl.load(p_dA_acc, boundary_check=(0, 1)).to(tl.float32)
        b_dw_c = b_dw + 0.0
        b_dA = tl.dot(b_dw_c, tl.trans(b_kg_a), b_dA, allow_tf32=False)
        if FINALIZE:
            # Folded dA finalize epilogue: on the last K-pass the in-register
            # dA_acc result goes straight to the final dA without a HBM
            # round-trip. The validity-extended causal mask equals the pure
            # causal mask whenever every row of the chunk is in range.
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
            b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
            # mid: (mask * dA_acc * beta) @ A. lhs clobbers b_dA (dead); b_A is rhs.
            b_mid = tl.dot(b_dA.to(b_A.dtype), b_A, allow_tf32=False)
            b_fin = tl.dot(b_A, b_mid.to(b_A.dtype), allow_tf32=False)
            p_dA_out = tl.make_block_ptr(dA_out_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
            tl.store(p_dA_out, tl.where(m_A, -b_fin, 0).to(p_dA_out.dtype.element_ty), boundary_check=(0, 1))
        else:
            tl.store(p_dA_acc, b_dA.to(p_dA_acc.dtype.element_ty), boundary_check=(0, 1))

        p_db_acc = tl.make_block_ptr(db_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_db = tl.load(p_db_acc, boundary_check=(0,)).to(tl.float32)
        b_db += tl.sum(b_dkgb * b_kg, 1)
        if FINALIZE:
            p_db_out = tl.make_block_ptr(db_out_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
            tl.store(p_db_out, b_db.to(p_db_out.dtype.element_ty), boundary_check=(0,))
        else:
            tl.store(p_db_acc, b_db.to(p_db_acc.dtype.element_ty), boundary_check=(0,))

        p_dk = tl.make_block_ptr(dk_ptr, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        b_dk = b_dk + b_dkgb * b_gb
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        p_dg = tl.make_block_ptr(dg_ptr, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dg = tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)
        b_dg = b_dg + b_kg * b_dkgb * b_beta[:, None]
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))




@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_kda_bwd_kernel_wy_vdw_fused_npu(
    k, g, beta, A, h, v, dv,
    dv2, dA, db, dg, dk,
    cu_seqlens, chunk_indices, chunk_offsets,
    T, BH, task_num, num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    K_T_CONTIG: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    FINALIZE: tl.constexpr,
    K_OFFSET: tl.constexpr,
):
    """v_part ⊕ dw_part per-task fused wy kernel (wave25 FLA_KDA_WY_FUSE).

    Form history (both device-gated dead, evidence in results/w1/meta.json):
    f_hyb v⊕k⊕dw — UB volume wall at the fp32 ATEM-g posture (1.84M bits);
    f_kdw k⊕dw — deterministic dkgb-dot reassociation (24/27, max 2.4e-2,
    on-vs-on byte-identical = codegen, not a race). This two-body form is
    the last matrix cell: the v phase's dA_acc/db hand off IN REGISTERS
    (fp32 GM round-trip was lossless, so the handoff is value-identical by
    construction) and the dw chain follows in the same task iteration with
    fresh [BT,BK] k/g loads (dw_part's own dataflow, verbatim dots).
    k_part runs as the stock launch BEFORE this kernel (it writes the dk/dg
    this kernel's dw phase RMWs; stock order v->k->dw preserved as
    k->v⊕dw). Launch gate: every wy consumer T-contig, NK == 1, env on;
    permanent latch-off on anomaly (STRICT env re-raises)."""
    i_k = K_OFFSET
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + i_t.to(tl.int64)
        else:
            i_tg = tl.cast(i_b, tl.int64) * tl.cdiv(T, BT) + i_t
            bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

        if K_T_CONTIG:
            if IS_VARLEN:
                k_ptr = k + tl.cast(i_h, tl.int64) * T_seq * K + bos * K
            else:
                k_ptr = k + (tl.cast(i_b, tl.int64) * H + i_h) * T_seq * K
            k_stride_t = K
        else:
            k_ptr = k + (bos * H + i_h) * K
            k_stride_t = H * K

        if G_T_CONTIG:
            if IS_VARLEN:
                g_ptr = g + tl.cast(i_hv, tl.int64) * T_seq * K + bos * K
                beta_ptr = beta + tl.cast(i_hv, tl.int64) * T_seq + bos
                A_ptr = A + tl.cast(i_hv, tl.int64) * T_seq * BT + bos * BT
                dv_ptr = dv + tl.cast(i_hv, tl.int64) * T_seq * V + bos * V
                v_ptr = v + tl.cast(i_hv, tl.int64) * T_seq * V + bos * V
            else:
                hv_off = tl.cast(i_b, tl.int64) * HV + i_hv
                g_ptr = g + hv_off * T_seq * K
                beta_ptr = beta + hv_off * T_seq
                A_ptr = A + hv_off * T_seq * BT
                dv_ptr = dv + hv_off * T_seq * V
                v_ptr = v + hv_off * T_seq * V
            g_stride_t = K
            a_stride_t = BT
            dv_stride_t = V
            v_stride_t = V
            beta_stride = 1
        else:
            g_ptr = g + (bos * HV + i_hv) * K
            beta_ptr = beta + bos * HV + i_hv
            A_ptr = A + (bos * HV + i_hv) * BT
            dv_ptr = dv + (bos * HV + i_hv) * V
            v_ptr = v + (bos * HV + i_hv) * V
            g_stride_t = HV * K
            a_stride_t = HV * BT
            dv_stride_t = HV * V
            v_stride_t = HV * V
            beta_stride = HV

        h_ptr = h + (i_tg * HV + i_hv) * K * V
        dv2_ptr = dv2 + (bos * HV + i_hv) * V
        dg_ptr = dg + (bos * HV + i_hv) * K
        dk_ptr = dk + (bos * HV + i_hv) * K
        dA_out_ptr = dA + (bos * HV + i_hv) * BT
        db_out_ptr = db + bos * HV + i_hv

        # ---- v phase (verbatim): dA/db REGISTER accumulators + dw fold; the
        #      dw dot takes dw_part's OWN b_dv load (no rhs->lhs tile reuse
        #      across the two spliced bodies) ----
        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (BT, BT), (0, 1))
        p_beta = tl.make_block_ptr(beta_ptr, (T,), (beta_stride,), (i_t * BT,), (BT,), (0,))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_beta = tl.load(p_beta, boundary_check=(0,))

        b_dA = tl.zeros([BT, BT], dtype=tl.float32)
        b_db = tl.zeros([BT], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_dv = tl.make_block_ptr(dv_ptr, (T, V), (dv_stride_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_v = tl.make_block_ptr(v_ptr, (T, V), (v_stride_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_dA = tl.dot(b_dv, tl.trans(b_v), b_dA, allow_tf32=False)
            # Ascend tl.dot clobbers lhs; copy A before every V-slab use.
            b_A_c = b_A + 0.0
            b_dvb = tl.dot(b_A_c, b_dv, allow_tf32=False)
            b_db += tl.sum(b_dvb * b_v, 1)
            p_dv2 = tl.make_block_ptr(dv2_ptr, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_dv2, (b_dvb * b_beta[:, None]).to(p_dv2.dtype.element_ty), boundary_check=(0, 1))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_dv_w = tl.load(p_dv, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dw = tl.dot(b_dv_w, b_h.to(b_dv_w.dtype), b_dw, allow_tf32=False)

        # ---- dw chain (verbatim); dA_acc/db from REGISTERS ----
        p_k = tl.make_block_ptr(k_ptr, (T, K), (k_stride_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g_ptr, (T, K), (g_stride_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_gk_exp = exp2(b_g)
        b_kg = b_k * b_gk_exp
        b_gb = b_gk_exp * b_beta[:, None]
        # Match CUDA: downcast dw/kg to A.dtype before dA / dkgb GEMMs.
        b_dw = -b_dw.to(b_A.dtype)
        b_kg_a = b_kg.to(b_A.dtype)
        if FINALIZE:
            # b_A is the lhs of the dkgb dot and Ascend tl.dot clobbers its lhs;
            # compute on a copy so b_A stays live for the dA finalize epilogue.
            b_A_dkgb = b_A + 0.0
        else:
            b_A_dkgb = b_A
        b_dkgb = tl.dot(b_A_dkgb, b_dw, allow_tf32=False)

        b_dw_c = b_dw + 0.0
        b_dA = tl.dot(b_dw_c, tl.trans(b_kg_a), b_dA, allow_tf32=False)
        if FINALIZE:
            # Folded dA finalize epilogue (verbatim): the register dA_acc from
            # the v phase takes the place of the GM load+round-trip.
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
            b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
            b_mid = tl.dot(b_dA.to(b_A.dtype), b_A, allow_tf32=False)
            b_fin = tl.dot(b_A, b_mid.to(b_A.dtype), allow_tf32=False)
            p_dA_out = tl.make_block_ptr(dA_out_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
            tl.store(p_dA_out, tl.where(m_A, -b_fin, 0).to(p_dA_out.dtype.element_ty), boundary_check=(0, 1))

        b_db += tl.sum(b_dkgb * b_kg, 1)
        if FINALIZE:
            p_db_out = tl.make_block_ptr(db_out_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
            tl.store(p_db_out, b_db.to(p_db_out.dtype.element_ty), boundary_check=(0,))

        p_dk = tl.make_block_ptr(dk_ptr, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        b_dk = b_dk + b_dkgb * b_gb
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        p_dg = tl.make_block_ptr(dg_ptr, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dg = tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)
        b_dg = b_dg + b_kg * b_dkgb * b_beta[:, None]
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

@input_guard(no_guard_contiguous=['dv'])
def chunk_kda_bwd_wy_dqkg_fused_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    use_graph: bool = False,
):
    if use_graph:
        raise NotImplementedError("use_graph is not supported on the Ascend NPU backend")
    B, T, H, K, HV, V = *k.shape, v.shape[2], v.shape[-1]
    BT = chunk_size
    if BT % _BC != 0:
        raise ValueError(f'KDA Ascend bwd requires chunk_size % {_BC} == 0, got {BT}')

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq = g.new_empty(B, T, HV, K, dtype=torch.float)
    dk = g.new_empty(B, T, HV, K, dtype=torch.float)
    dv2 = torch.empty_like(v)
    # dg/db/dA keep the native [B, T, H*, ...] layout: their readers (k_part's
    # dg/dk RMW, the dA finalize store, bwd_intra, reverse cumsum, the public
    # gradient outputs) all index natively, while the g/beta/A *inputs* may
    # arrive already T-contig from the producer/orchestrator — so empty_like
    # would propagate the wrong layout.
    dg = g.new_empty(B, T, HV, K, dtype=torch.float)
    db = beta.new_empty(B, T, HV, dtype=torch.float)
    dA = A.new_empty(B, T, HV, BT, dtype=torch.float)
    # dA/db double as the dA_acc/db_acc workspaces (output aliasing): wy_v_part
    # fully stores every (i_t, i_bh) tile of both accumulators (boundary-checked
    # block identical to the map wy_dw_part reads back under IS_VARLEN and dense),
    # so no accumulator element is read before its first write. The separate
    # fp32 workspaces and their zero fills drop (T*HV*(BT+1)*4B of dead writes
    # per call); the non-finalize wy_dw passes RMW the aliased buffers in place
    # and the FINALIZE pass loads the acc tile before storing the same addresses
    # out (tile ownership exclusive per task), so the acc/out pointer split
    # stays alias-safe.
    dA_acc = dA
    db_acc = db

    BK = _get_bk(K)
    BV = _get_bv(V)
    NK = triton.cdiv(K, BK)
    is_varlen = cu_seqlens is not None
    if chunk_offsets is None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if is_varlen else g.new_zeros(1, dtype=torch.int64)

    v_arg, v_t_contig = _t_contig_arg(v, HV)
    dv_arg = _t_contig_arg(dv, HV)[0]
    # Akk arrives [B, HV, T, BT] from inter_solve and the gate/beta copies may
    # come pre-transposed from chunk_kda_bwd: no aten Transpose for these.
    beta_arg, beta_t_contig = _maybe_t_contig_arg(beta, HV)
    A_arg, a_t_contig = _maybe_t_contig_arg(A, HV)
    g_arg, g_arg_t_contig = _maybe_t_contig_arg(g, HV)
    k_arg, k_t_contig = _t_contig_arg(k, H)

    bh_total = B * HV
    # At most 24 programs per launch. num_vectorcore is 24 on the 910B, but
    # clamp so a device reporting more cores cannot exceed the validated
    # launch envelope. m5 clamp hygiene: also never more programs than
    # tasks (degenerate at production shapes; one-task-per-program below),
    # matching the wy_fast / chunk_intra launcher convention — this one
    # num_core feeds the v_part / k_part / dw_part launches below.
    num_core = min(get_npu_properties()['num_vectorcore'], _WY_MAX_PROGRAMS)
    task_num = NT * bh_total
    # wave25 fused path (v_part ⊕ dw_part in ONE launch AFTER the stock
    # k_part launches — stock order v->k->dw preserved as k->v⊕dw; k_part
    # writes the dk/dg this kernel RMWs, the v phase hands dA_acc/db to the
    # dw chain in registers). Gate = the exact validated posture: every wy
    # consumer T-contig, NK == 1.
    if (
        _wy_fuse_enabled()
        and NK == 1
        and v_t_contig and a_t_contig and beta_t_contig and g_arg_t_contig
    ):
        try:
            k_part_kwargs_gated = dict(
                q=q,
                k=k,
                v_new=v_new,
                g=g_arg,
                h=h,
                do=do,
                dh=dh,
                dq=dq,
                dk=dk,
                dg=dg,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                scale=scale,
                T=T,
                BH=bh_total,
                task_num=task_num,
                num_core=num_core,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BT=BT,
                BC=32 if BT >= 32 else _BC,
                BK=BK,
                BV=BV,
                STATE_V_FIRST=state_v_first,
                IS_VARLEN=is_varlen,
                G_T_CONTIG=g_arg_t_contig,
                K_OFFSET=0,
            )
            fast_launch(chunk_kda_bwd_kernel_wy_k_part_npu, (num_core,), **k_part_kwargs_gated)
            fast_launch(
                chunk_kda_bwd_kernel_wy_vdw_fused_npu, (num_core,),
                k=k_arg,
                g=g_arg,
                beta=beta_arg,
                A=A_arg,
                h=h,
                v=v_arg,
                dv=dv_arg,
                dv2=dv2,
                dA=dA,
                db=db,
                dg=dg,
                dk=dk,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                T=T,
                BH=bh_total,
                task_num=task_num,
                num_core=num_core,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BT=BT,
                BK=BK,
                BV=BV,
                STATE_V_FIRST=state_v_first,
                IS_VARLEN=is_varlen,
                K_T_CONTIG=k_t_contig,
                G_T_CONTIG=True,
                FINALIZE=True,
                K_OFFSET=0,
                **ascend_compile_kwargs(),
            )
            return dq, dk, dv2, db, dg, dA
        except Exception:
            # permanent latch: this process falls back to the stock path.
            # FLA_KDA_WY_FUSE_STRICT=1 (verification runs only) re-raises so a
            # gated-path failure can never hide behind a vacuous stock pass.
            if os.environ.get('FLA_KDA_WY_FUSE_STRICT', '0').strip().lower() in ('1', 'true', 'on'):
                raise
            global _wy_fuse_broken
            _wy_fuse_broken = True

    if 0 < task_num < num_core:
        num_core = task_num
    # disable auto-multi-buffer on this V-slab launch
    fast_launch(
        chunk_kda_bwd_kernel_wy_v_part_npu, (num_core,),
        v=v_arg,
        beta=beta_arg,
        A=A_arg,
        dv=dv_arg,
        dv2=dv2,
        dA_acc=dA_acc,
        db_acc=db_acc,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        HV=HV,
        V=V,
        BT=BT,
        BV=BV,
        IS_VARLEN=is_varlen,
        G_T_CONTIG=v_t_contig,
        **ascend_compile_kwargs(),
    )

    k_part_kwargs = dict(
        q=q,
        k=k,
        v_new=v_new,
        g=g_arg,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BC=32 if BT >= 32 else _BC,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=is_varlen,
        G_T_CONTIG=g_arg_t_contig,
    )
    for k_off in range(NK):
        k_part_kwargs['K_OFFSET'] = k_off
        fast_launch(chunk_kda_bwd_kernel_wy_k_part_npu, (num_core,), **k_part_kwargs)

    dw_kwargs = dict(
        k=k_arg,
        g=g_arg,
        beta=beta_arg,
        A=A_arg,
        h=h,
        dv=dv_arg,
        dA_acc=dA_acc,
        db_acc=db_acc,
        dg=dg,
        dk=dk,
        dA=dA,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=is_varlen,
        K_T_CONTIG=k_t_contig,
        G_T_CONTIG=g_arg_t_contig,
    )
    # The last K-pass folds the dA finalize epilogue (mask(-A@((mask·dA·beta)A))
    # and the db copy) into the in-register dA_acc/db_acc RMW results.
    global _wy_tri_broken
    use_wy_tri = _wy_tri_enabled(BT, k_t_contig, g_arg_t_contig)
    for k_off in range(NK):
        dw_kwargs['K_OFFSET'] = k_off
        dw_kwargs['FINALIZE'] = k_off == NK - 1
        if use_wy_tri:
            try:
                fast_launch(chunk_kda_bwd_kernel_wy_dw_part_tri_npu, (num_core,), **dw_kwargs)
                continue
            except Exception:
                # Permanent latch + immediate legacy relaunch of the same
                # K-pass (wy_fast _FUSED_UW_LAUNCH_BROKEN pattern).
                _wy_tri_broken = True
                use_wy_tri = False
        fast_launch(chunk_kda_bwd_kernel_wy_dw_part_npu, (num_core,), **dw_kwargs)

    dv = dv2
    return dq, dk, dv, db, dg, dA
