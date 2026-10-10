# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KDA WY-representation kernels adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size
from fla.utils.launch_fast import fast_launch, num_aicore

# recompute_w_u_fwd: peak UB is max(u-slab, w-slab), not sum — tile BK/BV independently.
# u-slab: b_A[BT,BT] + b_v/b_vb/b_u[BT,BV]  (~3.5× BT×BV + BT×BT)
# w-slab: b_A + b_k/b_kb/b_gk/b_kg[/b_q/b_qg][BT,BK]  (4.5× or 6.5× BT×BK + BT×BT)
# STORE_QG_SCALED adds no slab: its store sits right after b_gk_exp, before the
# b_kb dot chain, where the live set is b_k/b_gk/b_gk_exp/b_q/b_qg_scaled
# (3.5× BT×BK fp32-units + bf16 halves) — under both existing slab budgets —
# and it is dead before the dot phase peaks, so _get_fwd_tiles is unchanged.
_RECOMPUTE_FWD_U_MEM_MULT = 3.5
_RECOMPUTE_FWD_W_MEM_MULT = 4.5
_RECOMPUTE_FWD_W_MEM_MULT_QG = 6.5
# FUSED_UW single full-width pass: both right-hand sides at full tile width
# (BK==K, BV==V) as two dots sharing ONE A load (pristine +0.0 copy for the
# second, per the Ascend lhs-clobber idiom). At the w-dot the
# live set is b_kb fp32 + b_w fp32 acc + A + A_pristine + 2-byte halves
# ~= 2 fp32-words per [BT, K+V] column-element; compile-verified to fit the
# 192KiB npubin envelope at BT<=64.
_RECOMPUTE_FWD_UW_FUSED_MEM_MULT = 2.0
# Safety envelope, deliberately narrow: 2-byte dtypes, K == V == 128,
# non-varlen, T a multiple of BT with T >= 8192, and BT <= 64 (BT=128
# full-width slabs would breach the ~150KB per-kernel UB envelope). A single
# [BT, K+V] wide-N dot (tl.join rhs / tl.split result) does not fit that
# budget: BiShengIR allocates 475KiB (BT=64) / 238KiB (BT=32) of UB against
# the hard 192KiB envelope on every shape this gate admits. The fused pass
# below keeps only what ships (one A load, no i_k/i_v loop machinery,
# hoisted offsets, adjacent qg/kg store issue). Everything else -- including
# every CI shape -- takes the unchanged legacy two-loop path, and a
# permanent try/except fallback in the launcher covers any compile surprise.
_RECOMPUTE_FWD_UW_FUSED_KV = 128
_RECOMPUTE_FWD_UW_FUSED_MIN_T = 8192
_RECOMPUTE_FWD_UW_FUSED_MAX_BT = 64
# Set (permanently, for this process) if the fused launch ever raises; see the
# try/except in recompute_w_u_fwd_kda_npu.
_FUSED_UW_LAUNCH_BROKEN = False
_SAFETY_MARGIN = 0.75
_FALLBACK_TILE = 8
_MAX_TILE_FWD = 128
_PREFERRED_TILE = 64


def _launch_wy_core_grid(kernel, *, task_num: int, kernel_kwargs: dict) -> None:
    num_core = num_aicore()
    # Persistent single launch: at most num_aicore programs (<= 24 on 910B
    # parts), never more programs than tasks. Coverage with grid < num_core:
    # every task t < task_num <= num_core is picked up by program t (stride
    # num_core lands past task_num), exactly once.
    fast_launch(kernel, (min(num_core, task_num),), task_num=task_num, num_core=num_core, **kernel_kwargs)


def _candidate_tiles(dim: int) -> list[int]:
    cap = min(_MAX_TILE_FWD, triton.next_power_of_2(dim))
    tiles = [b for b in (_PREFERRED_TILE, _MAX_TILE_FWD, 32, 16, 8) if b <= cap]
    return tiles or [_FALLBACK_TILE]


def _get_fwd_tiles(BT: int, K: int, V: int, *, store_qg: bool) -> tuple[int, int]:
    """Minimize V/K slab iterations under independent UB budgets for u- and w-slabs."""
    w_mult = _RECOMPUTE_FWD_W_MEM_MULT_QG if store_qg else _RECOMPUTE_FWD_W_MEM_MULT

    def _max_tile(fixed_dim: int, mem_mult: float) -> int:
        return compute_row_tile_block_size(
            BT, fixed_dim, mem_mult,
            tiling_row=False,
            safety_margin=_SAFETY_MARGIN,
            fallback=_FALLBACK_TILE,
            min_block=8,
            max_block=min(_MAX_TILE_FWD, triton.next_power_of_2(fixed_dim)),
        )

    max_bk = _max_tile(K, w_mult)
    max_bv = _max_tile(V, _RECOMPUTE_FWD_U_MEM_MULT)

    best_cost = None
    best_bk = max(8, min(max_bk, triton.next_power_of_2(K)))
    best_bv = max(8, min(max_bv, triton.next_power_of_2(V)))

    for bk in _candidate_tiles(K):
        if bk > max_bk:
            continue
        for bv in _candidate_tiles(V):
            if bv > max_bv:
                continue
            cost = triton.cdiv(V, bv) + triton.cdiv(K, bk)
            tie_break = bk + bv
            if best_cost is None or cost < best_cost or (cost == best_cost and tie_break > best_bk + best_bv):
                best_cost = cost
                best_bk, best_bv = bk, bv

    return best_bk, best_bv


def _use_fused_uw(
    BT: int,
    K: int,
    V: int,
    T: int,
    *,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    varlen: bool,
) -> bool:
    """Whether the single full-width FUSED_UW pass may replace the two-loop path.

    Requirements: the narrow production class -- 2-byte dtypes, K == V == 128,
    non-varlen, T a multiple of BT with T >= 8192, BT <= 64 -- plus full-width
    [BT, K] / [BT, V] slabs that fit the UB budget (the FUSED_UW constants
    above carry the envelope rationale). Everything else --
    including every CI shape -- takes the unchanged legacy two-loop path,
    whose compile surface is unchanged.
    """
    if varlen or T % BT != 0 or T < _RECOMPUTE_FWD_UW_FUSED_MIN_T:
        return False
    if BT > _RECOMPUTE_FWD_UW_FUSED_MAX_BT:
        return False
    if k_dtype != v_dtype or k_dtype.itemsize != 2 or K != V:
        return False
    if K != _RECOMPUTE_FWD_UW_FUSED_KV:
        return False
    if triton.next_power_of_2(K) != K or triton.next_power_of_2(K + V) != K + V:
        return False
    width = compute_row_tile_block_size(
        BT, K + V, _RECOMPUTE_FWD_UW_FUSED_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=8,
        max_block=triton.next_power_of_2(K + V),
    )
    return width >= K + V


def _hv_t_npu_arg(x: torch.Tensor, HV: int) -> tuple[torch.Tensor, bool]:
    """Return a [B, HV, T, ...]-contiguous read of ``x`` plus the T-contig flag.

    Layout-aware: when the producer already wrote the consumer's layout
    (shape[1] == HV, e.g. an orchestrator-prepared gate copy) the tensor is
    passed through instead of re-transposing. For T == HV the two layouts
    share shape/strides; the NPU mainline producers are T-contig, so that
    reading is preferred. HV == 1 stays on the native stride-1 path.
    """
    if HV == 1:
        return x, False
    if x.shape[1] == HV:
        return (x.contiguous(), True) if not x.is_contiguous() else (x, True)
    return x.transpose(1, 2).contiguous(), True


@triton.jit
def _beta_block_ptr(beta_ptr, T, t0, BT, BETA_T_CONTIG: tl.constexpr, HV: tl.constexpr):
    # t0 is the caller-hoisted chunk row offset (i_t * BT), shared by every
    # block ptr of the task so the scalar mul is issued once.
    if BETA_T_CONTIG:
        return tl.make_block_ptr(beta_ptr, (T,), (1,), (t0,), (BT,), (0,))
    return tl.make_block_ptr(beta_ptr, (T,), (HV,), (t0,), (BT,), (0,))


@triton.jit
def _gk_block_ptr(gk_ptr, T, K, t0, col_k, BT, BK, GK_T_CONTIG: tl.constexpr, HV: tl.constexpr):
    # t0/col_k are caller-hoisted row/column offsets (i_t * BT, i_k * BK).
    if GK_T_CONTIG:
        return tl.make_block_ptr(gk_ptr, (T, K), (K, 1), (t0, col_k), (BT, BK), (1, 0))
    return tl.make_block_ptr(gk_ptr, (T, K), (HV * K, 1), (t0, col_k), (BT, BK), (1, 0))


@triton.heuristics({
    "STORE_QG": lambda args: args["qg"] is not None,
    "STORE_QG_SCALED": lambda args: args["qg_scaled"] is not None,
    "STORE_KG": lambda args: args["kg"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T", "B", "task_num", "num_core"])
def recompute_w_u_fwd_kda_kernel_npu(
    q,
    k,
    qg,
    qg_scaled,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    task_num,
    num_core,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_QG_SCALED: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BETA_T_CONTIG: tl.constexpr,
    GK_T_CONTIG: tl.constexpr,
    FUSED_UW: tl.constexpr,
    A_T_CONTIG: tl.constexpr,
):
    T_max = T
    BH = B * HV
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t_o = task_id // BH
        i_bh = task_id % BH
        i_hv = i_bh % HV
        i_h = i_hv // (HV // H)
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t_o * 2).to(tl.int32), tl.load(
                chunk_indices + i_t_o * 2 + 1,
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
                cu_seqlens + i_n + 1,
            ).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            beta_bh = bos + i_hv * T_max
            gk_bh = i_hv * T_max * K + bos * K
        else:
            i_b = i_bh // HV
            i_t = i_t_o
            bos = tl.cast(i_b, tl.int64) * T
            beta_bh = (tl.cast(i_b, tl.int64) * HV + i_hv) * T_max
            gk_bh = (tl.cast(i_b, tl.int64) * HV + i_hv) * T_max * K

        k_ptr = k + (bos * H + i_h) * K
        v_ptr = v + (bos * HV + i_hv) * V
        u_ptr = u + (bos * HV + i_hv) * V
        w_ptr = w + (bos * HV + i_hv) * K
        if A_T_CONTIG:
            # Akk produced as [B, HV, T, BT] by inter_solve (wy-consumer layout).
            if IS_VARLEN:
                A_ptr = A + tl.cast(i_hv, tl.int64) * T_max * BT + bos * BT
            else:
                A_ptr = A + tl.cast(i_b * HV + i_hv, tl.int64) * T_max * BT
        else:
            A_ptr = A + (bos * HV + i_hv) * BT
        kg_ptr = kg + (bos * HV + i_hv) * K
        if BETA_T_CONTIG:
            beta_ptr = beta + beta_bh
        else:
            beta_ptr = beta + bos * HV + i_hv
        if GK_T_CONTIG:
            gk_ptr = gk + gk_bh
        else:
            gk_ptr = gk + (bos * HV + i_hv) * K
        if STORE_QG:
            q_ptr = q + (bos * H + i_h) * K
            qg_ptr = qg + (bos * HV + i_hv) * K
        if STORE_QG_SCALED:
            # b_q/qg_scaled companions of the STORE_QG pointers (the load is
            # CSE'd with the STORE_QG one when both flags are on).
            q_s_ptr = q + (bos * H + i_h) * K
            qg_scaled_ptr = qg_scaled + (bos * HV + i_hv) * K

        # Loop-invariant scalars, hoisted once per task: this kernel profiles
        # scalar-pipeline-bound on Ascend, so every redundant i_t*BT / K-offset
        # recompute inside the slab loops is pipeline bubbles. t0 is the chunk
        # row offset shared by every block ptr below.
        t0 = i_t * BT
        p_b = _beta_block_ptr(beta_ptr, T, t0, BT, BETA_T_CONTIG, HV)
        b_b = tl.load(p_b, boundary_check=(0,))

        a_stride_t = BT if A_T_CONTIG else HV * BT
        p_A = tl.make_block_ptr(A_ptr, (T, BT), (a_stride_t, 1), (t0, 0), (BT, BT), (1, 0))

        last_idx = min(t0 + BT, T) - 1
        # Row offset of the last valid gk row (kg byproduct); slab-loop-invariant.
        if GK_T_CONTIG:
            gn_row_off = last_idx * K
        else:
            gn_row_off = last_idx * HV * K

        if FUSED_UW:
            # Single full-width pass over both right-hand sides: v- and
            # k-slabs run as TWO full-width dots sharing ONE A load. (A
            # single [BT, K+V] wide-N dot via tl.join/tl.split does not fit:
            # BiShengIR allocates 475KiB of UB at BT=64 and 238KiB at BT=32
            # against the hard 192KiB npubin envelope, so it never shipped.)
            # Per-op math and op order match the two-loop path below exactly;
            # only the i_k/i_v loop machinery and the second A load are gone.
            p_v = tl.make_block_ptr(v_ptr, (T, V), (HV * V, 1), (t0, 0), (BT, BV), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
            b_A = tl.load(p_A, boundary_check=(0, 1))
            # Ascend tl.dot may clobber its lhs; keep a pristine copy for the
            # second dot (replaces the two-loop path's per-K-tile A reload).
            # The +0.0 copy promotes through fp32 (scalar constant), so cast
            # back: the second dot must see the same 2-byte lhs as the first
            # (chunk_delta_h's pristine-copy idiom).
            b_A_pristine = (b_A + 0.0).to(b_A.dtype)
            b_u = tl.dot(b_A, b_vb, allow_tf32=False)
            p_u = tl.make_block_ptr(u_ptr, (T, V), (HV * V, 1), (t0, 0), (BT, BV), (1, 0))
            tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

            p_k = tl.make_block_ptr(k_ptr, (T, K), (H * K, 1), (t0, 0), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kb = b_k * b_b[:, None]

            p_gk = _gk_block_ptr(gk_ptr, T, K, t0, 0, BT, BK, GK_T_CONTIG, HV)
            b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
            b_gk_exp = exp2(b_gk)
            b_kb = b_kb * b_gk_exp

            if STORE_QG_SCALED:
                # Scaled gated query for the fused h+o tail: bf16(q * exp2(gk) * scale),
                # the exact expression (and fp32->bf16 rounding point) the tail kernel
                # used to recompute per chunk from q and fp32 g. Placed here (before the
                # b_kb dot chain) so the extra live tiles never raise the slab peak.
                # FUSED_UW geometry: single full-width K tile (BK == K, no i_k loop),
                # caller-hoisted t0 row offset like every other block ptr here.
                p_qs = tl.make_block_ptr(q_s_ptr, (T, K), (H * K, 1), (t0, 0), (BT, BK), (1, 0))
                p_qgs = tl.make_block_ptr(qg_scaled_ptr, (T, K), (HV * K, 1), (t0, 0), (BT, BK), (1, 0))
                b_qs = tl.load(p_qs, boundary_check=(0, 1))
                tl.store(p_qgs, (b_qs * b_gk_exp * scale).to(p_qgs.dtype.element_ty), boundary_check=(0, 1))

            # qg/kg byproducts: values are computed first (independently, no
            # reassociation), then both stores are issued as one back-to-back
            # MTE burst before the dot, so their slabs die before the w
            # accumulator goes live and the store queue sees two adjacent
            # bursts instead of two scattered ones. q is loaded last so its
            # slab is not held across the kg compute.
            if STORE_KG:
                o_k = tl.arange(0, BK)
                m_k = o_k < K
                b_gn = tl.load(gk_ptr + gn_row_off + o_k, mask=m_k, other=0.0).to(tl.float32)
                b_kg = b_k * exp2(b_gn[None, :] - b_gk)
                p_kg = tl.make_block_ptr(kg_ptr, (T, K), (HV * K, 1), (t0, 0), (BT, BK), (1, 0))

            if STORE_QG:
                p_q = tl.make_block_ptr(q_ptr, (T, K), (H * K, 1), (t0, 0), (BT, BK), (1, 0))
                p_qg = tl.make_block_ptr(qg_ptr, (T, K), (HV * K, 1), (t0, 0), (BT, BK), (1, 0))
                b_q = tl.load(p_q, boundary_check=(0, 1))

            if STORE_QG:
                tl.store(p_qg, (b_q * b_gk_exp).to(p_qg.dtype.element_ty), boundary_check=(0, 1))
            if STORE_KG:
                tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

            b_w = tl.dot(b_A_pristine, b_kb.to(b_k.dtype), allow_tf32=False)
            p_w = tl.make_block_ptr(w_ptr, (T, K), (HV * K, 1), (t0, 0), (BT, BK), (1, 0))
            tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))
        else:
            for i_v in range(tl.cdiv(V, BV)):
                col_v = i_v * BV
                p_v = tl.make_block_ptr(v_ptr, (T, V), (HV * V, 1), (t0, col_v), (BT, BV), (1, 0))
                p_u = tl.make_block_ptr(u_ptr, (T, V), (HV * V, 1), (t0, col_v), (BT, BV), (1, 0))
                b_v = tl.load(p_v, boundary_check=(0, 1))
                b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
                # Ascend tl.dot may clobber the left operand; reload A each V tile.
                b_A = tl.load(p_A, boundary_check=(0, 1))
                b_u = tl.dot(b_A, b_vb, allow_tf32=False)
                tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

            for i_k in range(tl.cdiv(K, BK)):
                col_k = i_k * BK
                p_k = tl.make_block_ptr(k_ptr, (T, K), (H * K, 1), (t0, col_k), (BT, BK), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_kb = b_k * b_b[:, None]

                p_gk = _gk_block_ptr(gk_ptr, T, K, t0, col_k, BT, BK, GK_T_CONTIG, HV)
                b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
                b_gk_exp = exp2(b_gk)
                b_kb = b_kb * b_gk_exp

                if STORE_QG_SCALED:
                    # Scaled gated query for the fused h+o tail (two-loop path;
                    # the FUSED_UW branch above carries its own full-width copy):
                    # bf16(q * exp2(gk) * scale) at the exact fp32->bf16 rounding
                    # point the tail kernel used to recompute per chunk.
                    p_qs = tl.make_block_ptr(q_s_ptr, (T, K), (H * K, 1), (t0, col_k), (BT, BK), (1, 0))
                    p_qgs = tl.make_block_ptr(qg_scaled_ptr, (T, K), (HV * K, 1), (t0, col_k), (BT, BK), (1, 0))
                    b_qs = tl.load(p_qs, boundary_check=(0, 1))
                    tl.store(p_qgs, (b_qs * b_gk_exp * scale).to(p_qgs.dtype.element_ty), boundary_check=(0, 1))

                if STORE_QG:
                    p_q = tl.make_block_ptr(q_ptr, (T, K), (H * K, 1), (t0, col_k), (BT, BK), (1, 0))
                    p_qg = tl.make_block_ptr(qg_ptr, (T, K), (HV * K, 1), (t0, col_k), (BT, BK), (1, 0))
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    tl.store(p_qg, (b_q * b_gk_exp).to(p_qg.dtype.element_ty), boundary_check=(0, 1))

                if STORE_KG:
                    o_k = col_k + tl.arange(0, BK)
                    m_k = o_k < K
                    b_gn = tl.load(gk_ptr + gn_row_off + o_k, mask=m_k, other=0.0).to(tl.float32)
                    b_kg = b_k * exp2(b_gn[None, :] - b_gk)
                    p_kg = tl.make_block_ptr(kg_ptr, (T, K), (HV * K, 1), (t0, col_k), (BT, BK), (1, 0))
                    tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

                # Ascend tl.dot may clobber the left operand; reload A each K tile.
                b_A = tl.load(p_A, boundary_check=(0, 1))
                b_w = tl.dot(b_A, b_kb.to(b_k.dtype), allow_tf32=False)
                p_w = tl.make_block_ptr(w_ptr, (T, K), (HV * K, 1), (t0, col_k), (BT, BK), (1, 0))
                tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def recompute_w_u_fwd_kda_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    q: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_graph: bool = False,
    qg_scaled: torch.Tensor | None = None,
    scale: float | None = None,
    store_qg: bool | None = None,
    gk_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Optionally also fills a caller-allocated ``qg_scaled`` buffer with
    ``bf16(q * exp2(gk) * scale)`` — the precomputed per-token gate product
    consumed by the fused h+o tail kernel (``chunk_fwd_tail``), which would
    otherwise recompute the fp32 exp2/mul chain per chunk on the scalar pipe.

    ``store_qg`` (unscaled ``qg`` for the backward) defaults to ``q is not
    None``; pass it explicitly to decouple the two (the fused-tail path passes
    ``q`` for the scaled store only).
    """
    global _FUSED_UW_LAUNCH_BROKEN
    if use_graph:
        raise NotImplementedError("use_graph is not supported on the Ascend NPU backend")
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    BT = A.shape[-1]
    if store_qg is None:
        store_qg = q is not None
    store_qgs = qg_scaled is not None
    if store_qgs and q is None:
        raise ValueError("qg_scaled output requires the raw q tensor")
    if store_qgs and scale is None:
        raise ValueError("qg_scaled output requires scale")
    # Tile selection keys on the unscaled qg store only: the qg_scaled store
    # adds no slab peak (see the accounting note above _RECOMPUTE_FWD_U_MEM_MULT).
    BK, BV = _get_fwd_tiles(BT, K, V, store_qg=store_qg)
    fused_uw = (
        not _FUSED_UW_LAUNCH_BROKEN
        and _use_fused_uw(
            BT, K, V, T,
            k_dtype=k.dtype, v_dtype=v.dtype, varlen=cu_seqlens is not None,
        )
    )
    if fused_uw:
        # Single full-width tiles per side: the fused pass has no K/V sub-tiling.
        BK = BV = K

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Gate copies for T-contiguous reads: _hv_t_npu_arg materializes the
    # [B, HV, T] transposes (bit-identical layout permutation) and passes
    # tensors through when the producer already wrote that layout.
    beta, beta_t_contig = _hv_t_npu_arg(beta, HV)
    if gk_t is not None and gk_t.dim() == 4 and gk_t.is_contiguous() \
            and gk_t.shape[1] == HV and gk_t.dtype == gk.dtype:
        # producer-written head-major gate copy (w4 dual-store): pass through,
        # the host transpose drops
        gk, gk_t_contig = gk_t, True
    else:
        gk, gk_t_contig = _hv_t_npu_arg(gk, HV)
    # inter_solve produces Akk as [B, HV, T, BT]; a legacy [B, T, HV, BT] input
    # (shape[1] == T != HV) keeps the native strided path.
    a_t_contig = HV != 1 and A.dim() == 4 and A.shape[1] == HV
    if a_t_contig and not A.is_contiguous():
        A = A.contiguous()

    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    qg = k.new_empty(B, T, HV, K) if store_qg else None
    kg = k.new_empty(B, T, HV, K)
    # q pointer is read by either store branch; pass it whenever requested.
    q_arg = q if (store_qg or store_qgs) else None

    def _launch(fused: bool) -> None:
        _launch_wy_core_grid(
            recompute_w_u_fwd_kda_kernel_npu,
            task_num=NT * B * HV,
            kernel_kwargs=dict(
                q=q_arg,
                k=k,
                qg=qg,
                qg_scaled=qg_scaled,
                kg=kg,
                v=v,
                beta=beta,
                w=w,
                u=u,
                A=A,
                gk=gk,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                B=B,
                scale=float(scale) if scale is not None else 1.0,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BT=BT,
                BK=BK,
                BV=BV,
                BETA_T_CONTIG=beta_t_contig,
                GK_T_CONTIG=gk_t_contig,
                FUSED_UW=fused,
                A_T_CONTIG=a_t_contig,
            ),
        )

    if fused_uw:
        # If the fused specialization ever fails to build or launch (a
        # compile failure here would otherwise escape straight to the
        # caller), fall back to the proven two-loop path and disable the
        # fused attempt permanently for this process. Outputs are plain
        # full-coverage stores, so the relaunch fully overwrites any
        # partial write.
        try:
            _launch(fused=True)
            return w, u, qg, kg
        except Exception:
            _FUSED_UW_LAUNCH_BROKEN = True
            BK, BV = _get_fwd_tiles(BT, K, V, store_qg=store_qg)
    _launch(fused=False)
    return w, u, qg, kg
