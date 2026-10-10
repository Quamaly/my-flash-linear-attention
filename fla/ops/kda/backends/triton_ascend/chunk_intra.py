# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KDA chunk intra kernels for triton-ascend on Ascend NPU."""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

from fla.ops.kda.backends.triton_ascend.wy_fast import recompute_w_u_fwd_kda_npu as _recompute_w_u_fwd_kda_npu
from fla.ops.kda.chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp, exp2
from fla.ops.utils.softplus import softplus
from fla.utils import ascend_compile_kwargs, input_guard
from fla.utils.ascend_ub_manager import (
    compute_row_tile_block_size,
    get_npu_aiv_core_count,
)
from fla.utils.launch_fast import fast_launch, num_aicore

logger = logging.getLogger(__name__)

_BC = 32
_NUM_WARPS_SUB = 2
_NUM_WARPS_INTER = 2
# live-tile count scales with NC: BC=32/NC=2 holds half the tiles of the stock
# BC=16/NC=4 layout (one rhs pair + q/k/g rows instead of three pairs), so the
# stock 14.0 budget halves to 7.0 while keeping BK at 128.
_INTER_MEM_MULT = 7.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BK = 16
_MAX_INTER_BK = 128
# persistent core-count grids never exceed the cube-core count (910B1: 24).
_KDA_PERSISTENT_MAX_CORES = 24
# Fused intra+diag+inter envelope: non-varlen, NC==2 layouts only. K is
# capped so BK=next_pow2(K)<=128 keeps the fused kernel's live-tile set
# (6 raw [BC,BK] loads + fp32 derived tiles + 6 [BC,BC] accumulators) inside
# the 192KiB UB budget; everything else takes the legacy 3-launch pipeline.
_FUSED_INTRA_MAX_K = 128
# Fused-path T envelope (measured on the B1/H16/D128 benchmark shape): the
# single persistent launch wins the launch-bound regime (short sequences) but
# loses the throughput-bound regime, where the 3 pipelined persistent kernels
# have ample parallelism and the fused per-task serial chain (incl. the
# doubling-inverse dots) is longer. Gate at the measured crossover.
_FUSED_INTRA_MAX_T = 4096
# flipped off permanently if the fused kernel ever fails to compile/launch.
_FUSED_INTRA_OK = True
# h3 fold coverage: the fused small-T kernel folds the l2norm (FOLD_L2N)
# when the caller's qk_l2norm_out bundle is active, completing the t2v2
# fold's coverage of every NPU safe_gate trigger state (small T previously
# re-materialized the host l2norm inside this wrapper - 2 l2norm launches
# + 2 rstd copies). FLA_KDA_FUSED_FOLD=0 opts out; any host-side launch
# failure permanently reroutes to the proven host-fill + non-fold fused
# launch (bit-identical outputs either way).
_FUSED_FOLD_OK = os.environ.get('FLA_KDA_FUSED_FOLD', '1').strip().lower() not in ('0', 'false', 'off')


# disable auto-multi-buffer and AutoBlockify on the fused inter launch
_INTER_COMPILE_KWARGS = ascend_compile_kwargs(blacklist_auto_blockify=True)


def _get_inter_bk(K: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        K,
        _INTER_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=min(_MAX_INTER_BK, triton.next_power_of_2(K)),
    )


def _launch_core_grid(kernel, *, task_num: int, kernel_kwargs: dict, **compile_opts) -> None:
    # persistent core-count grid (cf. wy_fast._launch_wy_core_grid): one launch
    # with at most min(num_aicore, 24) programs; each program strides the task
    # queue (same structure as chunk_kda_bwd_kernel_intra_npu).
    num_core = min(num_aicore(), _KDA_PERSISTENT_MAX_CORES)
    fast_launch(kernel, (max(1, min(num_core, task_num)),), **compile_opts, **kernel_kwargs)


# Permanent kill switch for the decay-table sub-chunk intra kernel. Flipped to
# True by _run_decay_table_with_fallback on the FIRST failure of the
# decay-table launch (e.g. any triton-ascend compile error); once set, this
# process always takes the legacy path. Host-side only, so it is directly
# unit-testable.
_DECAY_TABLE_DISABLED = False


def _decay_table_eligible(
    *,
    safe_gate: bool,
    cu_seqlens,
    use_graph: bool,
    g_raw,
    gate_scale,
    T: int,
    chunk_size: int,
    K: int,
) -> bool:
    """Host-side gate for the decay-table sub-chunk intra kernel.

    Fast-path geometry only: non-varlen, T a multiple of the chunk, chunk_size
    64 (=> BC=32, NC=2 for K<=256), power-of-2 K<=128 (=> BK==K, unpredicated
    tiles incl. the K-major transposed k loads). g_raw must be supplied
    together with the gate scale the standalone cumsum kernel applied.
    """
    return (
        safe_gate
        and cu_seqlens is None
        and not use_graph
        and g_raw is not None
        and gate_scale is not None
        and chunk_size == 64
        and T % chunk_size == 0
        and K <= 128
        and triton.next_power_of_2(K) == K
    )


def _run_decay_table_with_fallback(launch_decay_table, launch_legacy) -> bool:
    """Try the decay-table launch once; on ANY failure fall back and kill it forever.

    Both arguments are zero-arg callables so the policy itself stays pure
    host-side code (unit-testable without an NPU). Returns True iff the
    decay-table path ran.
    """
    global _DECAY_TABLE_DISABLED
    if not _DECAY_TABLE_DISABLED:
        try:
            launch_decay_table()
            return True
        except Exception:
            # permanent: never attempt the decay-table kernel again in this process
            _DECAY_TABLE_DISABLED = True
    launch_legacy()
    return False


@triton.jit
def _kda_invert_diag_in_reg_npu(b_raw, o_i, m_A, m_I, valid_rows, BC: tl.constexpr):
    """In-register triangular inverse via 2^k factor doubling (all tl.dot).

    For strictly-lower nilpotent N (N^BC == 0), (I-N)^-1 == prod_k (I + N^(2^k))
    EXACTLY - a finite geometric series - so this matches diag_solve's
    forward-substitution result up to fp32 dot reassociation.

    Why not the substitution loop (device-proven, triton-ascend 3.2.0):
    the scalar row-by-row loop with the loop-carried tl.where update
    miscompiles whenever the containing kernel has ANY tl.dot, in any
    position - the loop-carried tile is clobbered by the dot pipelines'
    buffer allocation. Reproduced standalone: substitution-only kernel
    bit-correct; add dots before OR after (or +0.0 copies / static loop
    bounds / flattened dot nests) -> 1e-2..5e-1 errors in fp32 AND bf16,
    BC=32 and BC=16. The doubling form has no loop-carried scalar state,
    only cube dots, and device-verifies at 1.25e-7 (fp32) / output-rounding
    (bf16).

    Out-of-range rows keep identity rows: their N rows are zero (masked
    accumulators), so every factor leaves I there - same contract the
    standalone store/load round-trip had for in-range rows and the consumers
    expect (masked-zero operands / masked stores).
    """
    b_N = -tl.where(m_A, b_raw, 0)
    b_Ai = b_N + m_I  # I + N
    b_P = b_N
    # BC<=16 needs 4 factors (N^16 == 0); BC<=32 needs 5. Extra factors
    # contribute I and are exact no-ops, the count just trims dead dots.
    for k in tl.static_range(5 if BC > 16 else 4):
        b_P = tl.dot(b_P, b_P, allow_tf32=False)
        b_Ai = tl.dot(b_Ai, b_P + m_I, allow_tf32=False)
    return b_Ai


@triton.jit(do_not_specialize=['T', 'NT_TOTAL', 'BH_TOTAL'])
def chunk_kda_fwd_kernel_intra_diag_inter_fused_npu(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    T,
    NT_TOTAL,
    BH_TOTAL,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    q_n,
    k_n,
    q_rstd,
    k_rstd,
    FOLD_L2N: tl.constexpr = False,
    L2N_EPS: tl.constexpr = 1e-6,
):
    """Persistent fused replacement for intra_sub_chunk + diag_solve +
    inter_solve on the NC==2 envelope (host-gated: safe_gate, non-varlen,
    K<=128).

    One task per (chunk, bh): both diagonal Akk blocks are built from the
    q/k/g tiles that are loaded exactly once (the inter kernel's second GM
    read pass disappears), inverted in registers by the diag_solve forward
    substitution (_kda_invert_diag_in_reg_npu), merged with the single
    off-diagonal block pair, and stored straight into Aqk/Akk. Akkd never
    round-trips through GM on this path.

    Per-block math is a verbatim port of the kernels it replaces: the
    diagonal blocks keep intra_sub_chunk's raw-dtype midpoint gate
    subtraction and scale/beta/where order; the off-diagonal pair and the
    merge chain keep inter_solve_fused's fp32 row-0 reference, beta fold and
    (+0.0) lhs-clobber guard.

    FOLD_L2N (h3 fold-coverage completion): this kernel is the FIRST q/k
    consumer at small T, so it can fold the l2norm exactly like the
    persistent fold/decay-table sub-chunk kernels - rows are normalized
    in-register with the l2norm_fwd math (fp32 sum + eps, single strip
    covers all K since BK == next_pow2(K) >= K), the normalized tiles and
    the [B, H, T] stride-1 rstd rows are materialized into the caller's
    bundle for every later reader (recompute/tail via qg_scaled, backward,
    l2norm_bwd), and the in-kernel tiles are rebound through the store
    dtype so every dot below consumes the exact bytes the materialized path
    would have loaded from GM. Masked rows/blocks keep the load contract:
    boundary rows normalize to exact zeros and their rstd stores stay
    masked to real rows.
    """
    core_id = tl.program_id(0)
    num_core = tl.num_programs(0)
    # widen before multiplying; the task decomposition stays in int64 end to end
    task_num = tl.cast(NT_TOTAL, tl.int64) * BH_TOTAL
    for task_id in tl.range(core_id, task_num, num_core):
        # i_bh slowest / i_t fastest: the linearization of the former (NT, BH)
        # inter grid keeps a program's consecutive tasks inside one head for
        # L2 locality while the grid streams T in order.
        i_bh = task_id // NT_TOTAL
        i_t = tl.cast(task_id - i_bh * NT_TOTAL, tl.int32)
        i_b = tl.cast(i_bh // HV, tl.int32)
        i_hv = tl.cast(i_bh % HV, tl.int32)
        i_h = i_hv // (HV // H)

        bos = tl.cast(i_b, tl.int64) * T

        # padding-task guard (was the early return of the split-launch grid;
        # a persistent program must not return or it drops its remaining tasks)
        if i_t * BT < T:
            i_tc0 = i_t * BT
            i_tc1 = i_t * BT + BC

            # per-task pointer rebinding (ptr-arg += inside the task loop would
            # both accumulate offsets across tasks and break the loop-carried
            # typecheck)
            q_l = q + (bos * H + i_h) * K
            k_l = k + (bos * H + i_h) * K
            g_l = g + (bos * HV + i_hv) * K
            beta_l = beta + bos * HV + i_hv
            Aqk_l = Aqk + (bos * HV + i_hv) * BT
            # head-major Akk slab base ([B, HV, T, BT], the wy-consumer
            # layout; the fused path is non-varlen so the packed-length form
            # is exact)
            Akk_l = Akk + tl.cast(i_b * HV + i_hv, tl.int64) * T * BT

            if FOLD_L2N:
                # bundle rebases (non-varlen: bos == i_b * T): token-major
                # q_n/k_n planes and the [B, H, T] stride-1 rstd rows, the
                # only store shape this class of kernel's UB model accepts
                # (the t2v2 strided-rstd finding)
                q_n_l = q_n + (bos * H + i_h) * K
                k_n_l = k_n + (bos * H + i_h) * K
                qr_l = q_rstd + bos * H + i_h * T
                kr_l = k_rstd + bos * H + i_h * T

            o_i = tl.arange(0, BC)
            m_tc0 = (i_tc0 + o_i) < T
            m_tc1 = (i_tc1 + o_i) < T
            m_Aqk = o_i[:, None] >= o_i[None, :]
            m_A = o_i[:, None] > o_i[None, :]
            m_I = o_i[:, None] == o_i[None, :]

            b_b0 = tl.load(beta_l + tl.cast(i_tc0, tl.int64) * HV + o_i * HV, mask=m_tc0, other=0.0)
            b_b1 = tl.load(beta_l + tl.cast(i_tc1, tl.int64) * HV + o_i * HV, mask=m_tc1, other=0.0)

            b_Aqk00 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Akk00 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Aqk11 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Akk11 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

            for i_k in range(tl.cdiv(K, BK)):
                o_k = i_k * BK + tl.arange(0, BK)
                m_k = o_k < K
                m_k2 = m_k[None, :]

                # gate reference rows for this strip: per-subchunk midpoint
                # (diagonal blocks, intra_sub_chunk semantics; the row index
                # is always in [i_tcs, T-1] even for T-tails) and block-1
                # row 0 (off-diagonal pair, inter semantics)
                b_gm0 = tl.load(g_l + (i_tc0 + min(BC // 2, T - i_tc0 - 1)).to(tl.int64) * HV * K + o_k,
                                mask=m_k, other=0.0)
                b_gm1 = tl.load(g_l + (i_tc1 + min(BC // 2, T - i_tc1 - 1)).to(tl.int64) * HV * K + o_k,
                                mask=m_k & (i_tc1 < T), other=0.0)
                b_gn1 = tl.load(g_l + i_tc1.to(tl.int64) * HV * K + o_k,
                                mask=m_k & (i_tc1 < T), other=0.0).to(tl.float32)

                # ---- sub-chunk 0 diagonal (intra_sub_chunk math) ----
                b_g0 = tl.load(g_l + tl.cast(i_tc0, tl.int64) * (HV * K) + o_i[:, None] * (HV * K) + o_k[None, :],
                               mask=m_tc0[:, None] & m_k2, other=0.0)
                b_q0 = tl.load(q_l + tl.cast(i_tc0, tl.int64) * (H * K) + o_i[:, None] * (H * K) + o_k[None, :],
                               mask=m_tc0[:, None] & m_k2, other=0.0)
                b_k0 = tl.load(k_l + tl.cast(i_tc0, tl.int64) * (H * K) + o_i[:, None] * (H * K) + o_k[None, :],
                               mask=m_tc0[:, None] & m_k2, other=0.0)
                if FOLD_L2N:
                    b_q032 = b_q0.to(tl.float32)
                    b_k032 = b_k0.to(tl.float32)
                    b_qr0 = 1 / tl.sqrt(tl.sum(b_q032 * b_q032, 1) + L2N_EPS)
                    b_kr0 = 1 / tl.sqrt(tl.sum(b_k032 * b_k032, 1) + L2N_EPS)
                    b_q032 = b_q032 * b_qr0[:, None]
                    b_k032 = b_k032 * b_kr0[:, None]
                    tl.store(q_n_l + tl.cast(i_tc0, tl.int64) * (H * K) + o_i[:, None] * (H * K) + o_k[None, :],
                             b_q032.to(q_n.dtype.element_ty), mask=m_tc0[:, None] & m_k2)
                    tl.store(k_n_l + tl.cast(i_tc0, tl.int64) * (H * K) + o_i[:, None] * (H * K) + o_k[None, :],
                             b_k032.to(k_n.dtype.element_ty), mask=m_tc0[:, None] & m_k2)
                    tl.store(qr_l + i_tc0 + o_i, b_qr0, mask=m_tc0)
                    tl.store(kr_l + i_tc0 + o_i, b_kr0, mask=m_tc0)
                    # round-trip through the store dtype so every dot below
                    # consumes the exact bytes later readers fetch from GM
                    b_q0 = b_q032.to(q_n.dtype.element_ty)
                    b_k0 = b_k032.to(k_n.dtype.element_ty)
                # raw-dtype midpoint subtraction then fp32, exactly as the
                # sub_chunk kernel does it (input-dtype rounding preserved)
                b_gm0d = (b_g0 - b_gm0[None, :]).to(tl.float32)
                b_gq0 = tl.where(m_tc0[:, None], exp2(b_gm0d), 0.)
                b_gk0 = tl.where(m_tc0[:, None], exp2(-b_gm0d), 0.)
                b_qg0 = b_q0 * b_gq0
                b_kg0 = b_k0 * b_gq0
                b_kgt0 = tl.trans(b_k0 * b_gk0)
                b_Aqk00 = tl.dot(b_qg0, b_kgt0, b_Aqk00, allow_tf32=False)
                b_Akk00 = tl.dot(b_kg0, b_kgt0, b_Akk00, allow_tf32=False)

                # ---- sub-chunk 1 diagonal (intra_sub_chunk math) ----
                b_g1 = tl.load(g_l + tl.cast(i_tc1, tl.int64) * (HV * K) + o_i[:, None] * (HV * K) + o_k[None, :],
                               mask=m_tc1[:, None] & m_k2, other=0.0)
                b_q1 = tl.load(q_l + tl.cast(i_tc1, tl.int64) * (H * K) + o_i[:, None] * (H * K) + o_k[None, :],
                               mask=m_tc1[:, None] & m_k2, other=0.0)
                b_k1 = tl.load(k_l + tl.cast(i_tc1, tl.int64) * (H * K) + o_i[:, None] * (H * K) + o_k[None, :],
                               mask=m_tc1[:, None] & m_k2, other=0.0)
                if FOLD_L2N:
                    # a fully out-of-range second sub-chunk (i_tc1 >= T)
                    # loads zeros: rows normalize to exact zeros and every
                    # store stays masked, so the dots see the same zero
                    # tiles the materialized path would have loaded
                    b_q132 = b_q1.to(tl.float32)
                    b_k132 = b_k1.to(tl.float32)
                    b_qr1 = 1 / tl.sqrt(tl.sum(b_q132 * b_q132, 1) + L2N_EPS)
                    b_kr1 = 1 / tl.sqrt(tl.sum(b_k132 * b_k132, 1) + L2N_EPS)
                    b_q132 = b_q132 * b_qr1[:, None]
                    b_k132 = b_k132 * b_kr1[:, None]
                    tl.store(q_n_l + tl.cast(i_tc1, tl.int64) * (H * K) + o_i[:, None] * (H * K) + o_k[None, :],
                             b_q132.to(q_n.dtype.element_ty), mask=m_tc1[:, None] & m_k2)
                    tl.store(k_n_l + tl.cast(i_tc1, tl.int64) * (H * K) + o_i[:, None] * (H * K) + o_k[None, :],
                             b_k132.to(k_n.dtype.element_ty), mask=m_tc1[:, None] & m_k2)
                    tl.store(qr_l + i_tc1 + o_i, b_qr1, mask=m_tc1)
                    tl.store(kr_l + i_tc1 + o_i, b_kr1, mask=m_tc1)
                    b_q1 = b_q132.to(q_n.dtype.element_ty)
                    b_k1 = b_k132.to(k_n.dtype.element_ty)
                b_gm1d = (b_g1 - b_gm1[None, :]).to(tl.float32)
                b_gq1 = tl.where(m_tc1[:, None], exp2(b_gm1d), 0.)
                b_gk1 = tl.where(m_tc1[:, None], exp2(-b_gm1d), 0.)
                b_qg1 = b_q1 * b_gq1
                b_kg1 = b_k1 * b_gq1
                b_kgt1 = tl.trans(b_k1 * b_gk1)
                b_Aqk11 = tl.dot(b_qg1, b_kgt1, b_Aqk11, allow_tf32=False)
                b_Akk11 = tl.dot(b_kg1, b_kgt1, b_Akk11, allow_tf32=False)

                # ---- off-diagonal pair (inter_solve math) ----
                b_gqn1 = tl.where(m_tc1[:, None], exp2(b_g1.to(tl.float32) - b_gn1[None, :]), 0)
                b_qg1o = b_q1 * b_gqn1
                b_kg1o = b_k1 * b_gqn1
                b_kgto = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0.to(tl.float32)))
                b_Aqk10 = tl.dot(b_qg1o, b_kgto, b_Aqk10, allow_tf32=False)
                b_Akk10 = tl.dot(b_kg1o, b_kgto, b_Akk10, allow_tf32=False)

            # scale/beta exactly where sub_chunk/inter applied them
            b_Aqk00 = tl.where(m_Aqk, b_Aqk00 * scale, 0.0)
            b_Aqk11 = tl.where(m_Aqk, b_Aqk11 * scale, 0.0)
            b_Aqk10 = b_Aqk10 * scale
            b_Akk00 = tl.where(m_A, b_Akk00 * b_b0.to(tl.float32)[:, None], 0.0)
            b_Akk11 = tl.where(m_A, b_Akk11 * b_b1.to(tl.float32)[:, None], 0.0)
            b_Akk10 = b_Akk10 * b_b1.to(tl.float32)[:, None]

            # in-register diagonal inversion (replaces the diag_solve launch;
            # Akkd never touches GM)
            b_Ai00 = _kda_invert_diag_in_reg_npu(b_Akk00, o_i, m_A, m_I, T - i_tc0, BC)
            b_Ai11 = _kda_invert_diag_in_reg_npu(b_Akk11, o_i, m_A, m_I, T - i_tc1, BC)

            # off-diagonal merge (verbatim from inter_solve_fused, incl. the
            # (+0.0) lhs-clobber guard for the stored b_Ai11)
            b_Ai11_c = b_Ai11 + 0.0
            b_Ai10 = -tl.dot(
                tl.dot(b_Ai11, b_Akk10, allow_tf32=False),
                b_Ai00,
                allow_tf32=False,
            )

            p_Aqk0 = Aqk_l + tl.cast(i_tc0, tl.int64) * (HV * BT) + o_i[:, None] * (HV * BT)
            p_Aqk1 = Aqk_l + tl.cast(i_tc1, tl.int64) * (HV * BT) + o_i[:, None] * (HV * BT)
            tl.store(p_Aqk0 + o_i[None, :], b_Aqk00.to(Aqk.dtype.element_ty), mask=m_tc0[:, None])
            tl.store(p_Aqk1 + o_i[None, :], b_Aqk10.to(Aqk.dtype.element_ty), mask=m_tc1[:, None])
            tl.store(p_Aqk1 + BC + o_i[None, :], b_Aqk11.to(Aqk.dtype.element_ty), mask=m_tc1[:, None])

            p_Akk0 = Akk_l + tl.cast(i_tc0, tl.int64) * BT + o_i[:, None] * BT
            p_Akk1 = Akk_l + tl.cast(i_tc1, tl.int64) * BT + o_i[:, None] * BT
            tl.store(p_Akk0 + o_i[None, :], b_Ai00.to(Akk.dtype.element_ty), mask=m_tc0[:, None])
            tl.store(p_Akk1 + o_i[None, :], b_Ai10.to(Akk.dtype.element_ty), mask=m_tc1[:, None])
            tl.store(p_Akk1 + BC + o_i[None, :], b_Ai11_c.to(Akk.dtype.element_ty), mask=m_tc1[:, None])


@triton.jit(do_not_specialize=['T', 'B', 'NT_TOTAL'])
def chunk_kda_fwd_kernel_diag_solve_npu(
    Akkd,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    NT_TOTAL,
    HV: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    """Per-subchunk lower-triangular forward substitution into Akkd.

    Run before inter_solve so the fused inter kernel only merges off-diagonal
    blocks, keeping scalar BC loops off the large (NT, BH) grid.

    Persistent AIV launch: at most num_vectorcore (48) programs in a single
    launch; each program strides the task space linearized exactly as the old
    (NT, NC, BH) split-launch grid (bwd-intra decomposition: i_bh fastest).
    """
    NC = tl.cdiv(BT, BC)
    core_id = tl.program_id(0)
    num_core = tl.num_programs(0)
    # widen before multiplying; the task decomposition stays in int64 end to end
    task_num = tl.cast(NT_TOTAL, tl.int64) * NC * B * HV
    BH_TOTAL = B * HV
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id % BH_TOTAL
        rem = task_id // BH_TOTAL
        i_t = rem % NT_TOTAL
        i_i = rem // NT_TOTAL
        i_b, i_hv = i_bh // HV, i_bh % HV

        if cu_seqlens is not None:
            i_n, i_t = tl.load(chunk_indices + i_t * 2), tl.load(chunk_indices + i_t * 2 + 1)
            # int64 guarantees: cu_seqlens may arrive as int32 and chunk_indices
            # follows its dtype, but the global offsets must stay 64-bit
            bos, eos = tl.cast(tl.load(cu_seqlens + i_n), tl.int64), tl.cast(tl.load(cu_seqlens + i_n + 1), tl.int64)
        else:
            bos, eos = i_b * T, i_b * T + T
        # T is a loop-carried arg (int32); the reassignment must keep its type
        T = tl.cast(eos - bos, tl.int32)

        # skip out-of-range sub-chunks (unaligned T, varlen tails)
        # no-op cast when i_t is int64; guards int32 chunk_indices tables
        i_ti = tl.cast(i_t * BT + i_i * BC, tl.int64)
        if i_ti < T:
            # rebind pointer per task (ptr-arg += inside the task loop would both
            # accumulate offsets across tasks and break loop-carried typecheck)
            Akkd_l = Akkd + (bos * HV + i_hv) * BC
            o_i = tl.arange(0, BC)
            m_A = o_i[:, None] > o_i[None, :]
            m_I = o_i[:, None] == o_i[None, :]

            # block ptr offsets must stay int32 (i_ti is a within-sequence row
            # index <= T, always int32-safe); task math above stays int64
            p_Akk = tl.make_block_ptr(Akkd_l, (T, BC), (HV * BC, 1), (i_ti.to(tl.int32), 0), (BC, BC), (1, 0))
            b_Akk = tl.load(p_Akk, boundary_check=(0, 1)).to(tl.float32)
            b_Ai = -tl.where(m_A, b_Akk, 0)
            for i in range(2, min(BC, T - i_ti)):
                b_a = -tl.load(Akkd_l + (i_ti + i).to(tl.int64) * HV * BC + o_i)
                b_a = tl.where(o_i < i, b_a, 0.)
                b_a += tl.sum(b_a[:, None] * b_Ai, 0)
                b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
            b_Ai += m_I
            tl.store(p_Akk, b_Ai.to(Akkd_l.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T', 'NT_TOTAL', 'BH_TOTAL'])
def chunk_kda_fwd_kernel_intra_sub_chunk_npu(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    NT_TOTAL,
    BH_TOTAL,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Persistent core-count grid over the (i_bh, i_t) chunk queue.

    One launch with <= num_aicore programs; each program loops over its share
    of the whole chunks with stride num_programs (classic persistent style,
    same structure as chunk_kda_bwd_kernel_intra_npu). The task layout keeps
    i_t fastest and i_bh slowest, so a program's consecutive tasks stay within
    one head for L2 locality while the whole grid streams T in order.

    One task covers ALL NC sub-chunks of a chunk: the task decomposition
    (int64 divmods), the varlen lookup and the six int64 pointer rebinds
    depend only on (i_bh, i_t), so they run once per chunk instead of once
    per sub-chunk (at T=131072: 65536 -> 32768 tasks, ~2731 -> ~1365 setups
    per program on a 24-core grid). Per-sub-chunk tile math is unchanged - only
    the work distribution and the setup hoisting differ from the split
    launches.
    """
    core_id = tl.program_id(0)
    num_core = tl.num_programs(0)
    # widen before multiplying; the task decomposition stays in int64 end to end
    task_num = tl.cast(NT_TOTAL, tl.int64) * BH_TOTAL
    # sub-chunk-invariant tile constants: allocated once for the whole launch
    # instead of once per task loop iteration
    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id // NT_TOTAL
        i_t = tl.cast(task_id % NT_TOTAL, tl.int32)
        i_b = tl.cast(i_bh // HV, tl.int32)
        i_hv = tl.cast(i_bh % HV, tl.int32)
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            # T is a loop-carried arg (int32); the reassignment must keep its type
            T = (eos - bos).to(tl.int32)
        else:
            bos = tl.cast(i_b, tl.int64) * T
            eos = bos + T

        # rebind pointers once per chunk (ptr-arg += inside the task loop would
        # both accumulate offsets across tasks and break loop-carried typecheck)
        q_l = q + (bos * H + i_h) * K
        k_l = k + (bos * H + i_h) * K
        g_l = g + (bos * HV + i_hv) * K
        beta_l = beta + (bos * HV + i_hv)
        Aqk_l = Aqk + (bos * HV + i_hv) * BT
        Akk_l = Akk + (bos * HV + i_hv) * BC
        row_base = i_t * BT

        # static unroll (NC is a small constexpr, 2 in production): keeps the
        # per-task instruction stream flat like the former per-sub-chunk queue
        # (no inner back-edge for the pipeliner to treat as a scheduling scope
        # boundary) and folds i_i * BC into a per-copy constant.
        for i_i in tl.static_range(0, NC):
            i_ti = row_base + i_i * BC
            if i_ti < T:
                p_q = tl.make_block_ptr(q_l, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
                p_k = tl.make_block_ptr(k_l, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
                p_g = tl.make_block_ptr(g_l, (T, K), (HV * K, 1), (i_ti, 0), (BC, BK), (1, 0))
                # K-major companions of k/g, consumed ONLY on the BK == K fast path below,
                # where the [BK, BC] tile is exactly in-bounds (tile dim0 == shape dim0,
                # same transposed-load pattern as chunk_bwd): the dot RHS is assembled
                # straight in transposed layout instead of paying an in-UB fp32 tl.trans.
                # For BK != K (non-pow2 K, e.g. K=60 -> BK=64) the boundary_check masking
                # of the stride-1 transposed dim is not exercised anywhere else in the
                # codebase and mis-fills on Ascend (tail columns read into the next
                # head's rows), so that case keeps the original row-major masked code
                # entirely.
                p_kt = tl.make_block_ptr(k_l, (K, T), (1, H * K), (0, i_ti), (BK, BC), (0, 1))
                p_gt = tl.make_block_ptr(g_l, (K, T), (1, HV * K), (0, i_ti), (BK, BC), (0, 1))

                p_beta = tl.make_block_ptr(beta_l, (T,), (HV,), (i_ti,), (BC,), (0,))

                # Fast path requires BK == K (compile-time) AND all BC rows in bounds
                # (runtime). Production shapes (pow2 K, T a multiple of BC) take it for
                # every sub-chunk: loads/stores drop all boundary predicates, the midpoint
                # clamp collapses to the constant BC // 2, and the m_c row masks are
                # identity (dropping those wheres is exact on full tiles). Everything
                # else - non-pow2 K, each sequence's final sub-chunk - runs the original
                # masked code verbatim.
                b_full = BK == K and i_ti + BC <= T
                if b_full:
                    b_q = tl.load(p_q)
                    b_k = tl.load(p_k)
                    b_g = tl.load(p_g)
                    b_beta = tl.load(p_beta)
                    p_gn = g_l + (i_ti + BC // 2).to(tl.int64) * HV * K + tl.arange(0, BK)
                    b_gn = tl.load(p_gn)
                    b_gq = exp2((b_g - b_gn[None, :]).to(tl.float32))
                    # shared RHS assembled directly in transposed [BK, BC] layout
                    b_kgt = tl.load(p_kt) * exp2((b_gn[:, None] - tl.load(p_gt)).to(tl.float32))
                else:
                    o_c = i_ti + tl.arange(0, BC)
                    m_c = o_c < T

                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_g = tl.load(p_g, boundary_check=(0, 1))
                    b_beta = tl.load(p_beta, boundary_check=(0,))

                    p_gn = g_l + (i_ti + min(BC // 2, T - i_ti - 1)).to(tl.int64) * HV * K + tl.arange(0, BK)
                    b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)

                    b_gm = (b_g - b_gn[None, :]).to(tl.float32)

                    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.)
                    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.)

                    b_kgt = tl.trans(b_k * b_gk)

                b_Aqk = tl.dot(b_q * b_gq, b_kgt, allow_tf32=False) * scale
                b_Akk = tl.dot(b_k * b_gq, b_kgt, allow_tf32=False) * b_beta[:, None]

                b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
                b_Akk = tl.where(m_Akk, b_Akk, 0.0)

                p_Aqk = tl.make_block_ptr(Aqk_l, (T, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
                p_Akk = tl.make_block_ptr(Akk_l, (T, BC), (HV * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
                if b_full:
                    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty))
                    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty))
                else:
                    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
                    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))


# L2norm-folding variant of the persistent sub-chunk intra kernel above.
# Kept as a separate def instead of weaving FOLD_L2N branches into that
# kernel so both numerically verified bodies stay intact; the launcher
# dispatches on the qk_l2norm_out bundle.


@triton.jit(do_not_specialize=['T', 'NT_TOTAL', 'BH_TOTAL'])
def chunk_kda_fwd_kernel_intra_sub_chunk_fold_npu(
    q,
    k,
    q_n,
    k_n,
    q_rstd,
    k_rstd,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    NT_TOTAL,
    BH_TOTAL,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    FOLD_L2N: tl.constexpr,
    L2N_EPS: tl.constexpr,
):
    """Persistent core-count grid over the (i_bh, i_i, i_t) task queue.

    One launch with <= num_aicore programs; each program loops over its share
    of the tasks with stride num_programs (classic persistent style, same
    structure as chunk_kda_bwd_kernel_intra_npu). The task layout keeps i_t
    fastest and i_bh slowest - the linearization of the former (NT, NC, BH)
    grid - so a program's consecutive tasks stay within one head for L2
    locality while the whole grid streams T in order. Per-task tile math is
    unchanged; only the work distribution differs from the split launches.
    """
    core_id = tl.program_id(0)
    num_core = tl.num_programs(0)
    # widen before multiplying; the task decomposition stays in int64 end to end
    task_num = tl.cast(NT_TOTAL, tl.int64) * NC * BH_TOTAL
    per_bh = tl.cast(NT_TOTAL, tl.int64) * NC
    # the varlen branch below rebinds T to the per-sequence length; the rstd
    # working buffer is head-strided by the TOTAL token count, so keep it
    T_tot = T
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id // per_bh
        rem = task_id - i_bh * per_bh
        i_t = tl.cast(rem % NT_TOTAL, tl.int32)
        i_i = tl.cast(rem // NT_TOTAL, tl.int32)
        i_b = tl.cast(i_bh // HV, tl.int32)
        i_hv = tl.cast(i_bh % HV, tl.int32)
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            # T is a loop-carried arg (int32); the reassignment must keep its type
            T = (eos - bos).to(tl.int32)
        else:
            bos = tl.cast(i_b, tl.int64) * T
            eos = bos + T

        i_ti = i_t * BT + i_i * BC
        if i_ti < T:
            # rebind pointers per task (ptr-arg += inside the task loop would both
            # accumulate offsets across tasks and break loop-carried typecheck)
            q_l = q + (bos * H + i_h) * K
            k_l = k + (bos * H + i_h) * K
            g_l = g + (bos * HV + i_hv) * K
            beta_l = beta + (bos * HV + i_hv)
            Aqk_l = Aqk + (bos * HV + i_hv) * BT
            Akk_l = Akk + (bos * HV + i_hv) * BC

            p_q = tl.make_block_ptr(q_l, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k_l, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g_l, (T, K), (HV * K, 1), (i_ti, 0), (BC, BK), (1, 0))
            # K-major companions of k/g: the dot RHS is consumed transposed, so fetch it
            # as [BK, BC] straight from GM instead of computing b_k * gate row-major and
            # paying an in-UB fp32 tl.trans in every program. Same 256B-contiguous rows
            # on the wire; +16KB bf16 mte1 trades against a 16KB fp32 scalar permute.
            p_kt = tl.make_block_ptr(k_l, (K, T), (1, H * K), (0, i_ti), (BK, BC), (0, 1))
            p_gt = tl.make_block_ptr(g_l, (K, T), (1, HV * K), (0, i_ti), (BK, BC), (0, 1))

            p_beta = tl.make_block_ptr(beta_l, (T,), (HV,), (i_ti,), (BC,), (0,))

            if FOLD_L2N:
                # first q/k consumer: rows are l2-normalized in-register with the
                # exact l2norm_fwd math (fp32 accumulate, same eps) and both the
                # normalized tiles and the row rstd are materialized for every
                # later reader (inter/recompute/tail kernels, backward,
                # l2norm_bwd) - the host-side l2norm_fwd launches are skipped.
                q_n_l = q_n + (bos * H + i_h) * K
                k_n_l = k_n + (bos * H + i_h) * K
                if IS_VARLEN:
                    # varlen inputs are flattened to B == 1, so the (h, t)
                    # row of the [1, H, T] buffer sits at h*T_tot + t
                    qr_l = q_rstd + i_h * T_tot + bos
                    kr_l = k_rstd + i_h * T_tot + bos
                else:
                    # [B, H, T]: the (b, h, t) row sits at (bos*H + i_h*T + t)
                    qr_l = q_rstd + bos * H + i_h * T
                    kr_l = k_rstd + bos * H + i_h * T
                p_qn = tl.make_block_ptr(q_n_l, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
                p_kn = tl.make_block_ptr(k_n_l, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))

            # Fast path requires BK == K (compile-time) AND all BC rows in bounds
            # (runtime). Production shapes (pow2 K, T a multiple of BC) take it for
            # every sub-chunk: loads/stores drop all boundary predicates and the m_c
            # row masks are identity. Everything else - non-pow2 K (e.g. K=60, where
            # boundary_check on the stride-1 transposed dim mis-fills on ascend and
            # reads the next head's rows), each sequence's final sub-chunk - runs the
            # original masked code verbatim.
            b_full = BK == K and i_ti + BC <= T
            if b_full:
                b_q = tl.load(p_q)
                b_k = tl.load(p_k)
                if FOLD_L2N:
                    b_q32 = b_q.to(tl.float32)
                    b_k32 = b_k.to(tl.float32)
                    b_qr = 1 / tl.sqrt(tl.sum(b_q32 * b_q32, 1) + L2N_EPS)
                    b_kr = 1 / tl.sqrt(tl.sum(b_k32 * b_k32, 1) + L2N_EPS)
                    b_q32 = b_q32 * b_qr[:, None]
                    b_k32 = b_k32 * b_kr[:, None]
                    tl.store(p_qn, b_q32.to(q_n.dtype.element_ty))
                    tl.store(p_kn, b_k32.to(k_n.dtype.element_ty))
                    o_r = i_ti + tl.arange(0, BC)
                    tl.store(qr_l + o_r, b_qr)
                    tl.store(kr_l + o_r, b_kr)
                    # round-trip through the store dtype so the dot inputs stay
                    # bit-identical to the tiles later readers fetch from GM
                    b_q = b_q32.to(q_n.dtype.element_ty)
                    b_k = b_k32.to(k_n.dtype.element_ty)
                b_g = tl.load(p_g)
                b_beta = tl.load(p_beta)
                p_gn = g_l + (i_ti + BC // 2).to(tl.int64) * HV * K + tl.arange(0, BK)
                b_gn = tl.load(p_gn)
                b_gq = exp2((b_g - b_gn[None, :]).to(tl.float32))
                # shared RHS assembled directly in transposed [BK, BC] layout;
                # folded runs round the normalized K-major tile through the
                # store dtype first so it equals the materialized k_n bytes
                if FOLD_L2N:
                    b_kgt = (tl.load(p_kt).to(tl.float32) * b_kr[None, :]).to(k_n.dtype.element_ty) \
                        * exp2((b_gn[:, None] - tl.load(p_gt)).to(tl.float32))
                else:
                    b_kgt = tl.load(p_kt) * exp2((b_gn[:, None] - tl.load(p_gt)).to(tl.float32))
            else:
                o_c = i_ti + tl.arange(0, BC)
                m_c = o_c < T

                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if FOLD_L2N:
                    # boundary rows load as zeros, so their rstd is 1/sqrt(eps)
                    # and the normalized row is an exact zero - identical to the
                    # boundary-checked loads of the materialized path; the rstd
                    # store itself stays masked to real rows only
                    b_q32 = b_q.to(tl.float32)
                    b_k32 = b_k.to(tl.float32)
                    b_qr = 1 / tl.sqrt(tl.sum(b_q32 * b_q32, 1) + L2N_EPS)
                    b_kr = 1 / tl.sqrt(tl.sum(b_k32 * b_k32, 1) + L2N_EPS)
                    b_q32 = b_q32 * b_qr[:, None]
                    b_k32 = b_k32 * b_kr[:, None]
                    tl.store(p_qn, b_q32.to(q_n.dtype.element_ty), boundary_check=(0, 1))
                    tl.store(p_kn, b_k32.to(k_n.dtype.element_ty), boundary_check=(0, 1))
                    tl.store(qr_l + o_c, b_qr, mask=m_c)
                    tl.store(kr_l + o_c, b_kr, mask=m_c)
                    b_q = b_q32.to(q_n.dtype.element_ty)
                    b_k = b_k32.to(k_n.dtype.element_ty)
                b_g = tl.load(p_g, boundary_check=(0, 1))
                b_beta = tl.load(p_beta, boundary_check=(0,))

                p_gn = g_l + (i_ti + min(BC // 2, T - i_ti - 1)).to(tl.int64) * HV * K + tl.arange(0, BK)
                b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)

                b_gm = (b_g - b_gn[None, :]).to(tl.float32)

                b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.)
                b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.)

                b_kgt = tl.trans(b_k * b_gk)

            b_Aqk = tl.dot(b_q * b_gq, b_kgt, allow_tf32=False) * scale
            b_Akk = tl.dot(b_k * b_gq, b_kgt, allow_tf32=False) * b_beta[:, None]

            o_i = tl.arange(0, BC)
            m_Aqk = o_i[:, None] >= o_i[None, :]
            m_Akk = o_i[:, None] > o_i[None, :]

            b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
            b_Akk = tl.where(m_Akk, b_Akk, 0.0)

            p_Aqk = tl.make_block_ptr(Aqk_l, (T, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
            p_Akk = tl.make_block_ptr(Akk_l, (T, BC), (HV * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
            if b_full:
                tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty))
                tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty))
            else:
                tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T', 'NT_TOTAL', 'BH_TOTAL'])
def chunk_kda_fwd_kernel_intra_sub_chunk_decay_table_npu(
    q,
    k,
    q_n,
    k_n,
    q_rstd,
    k_rstd,
    g_raw,
    beta,
    Aqk,
    Akk,
    A_log,
    dt_bias,
    lower_bound,
    gate_scale,
    scale,
    T,
    NT_TOTAL,
    BH_TOTAL,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    GATE_RAW: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    HAS_GATE_SCALE: tl.constexpr,
    FOLD_L2N: tl.constexpr,
    L2N_EPS: tl.constexpr,
):
    """Persistent chunk-granular intra kernel with a per-chunk decay table.

    Builds the chunk-local decay table inside the kernel instead of reading
    back the standalone cumsum kernel's stored output.

    Task decomposition is chunk-granular (one task = one (i_bh, i_t) chunk,
    inner tl.static_range over the NC sub-chunks, hoisted o_i/m_Aqk/m_Akk,
    per-chunk int64 pointer rebinds) - the fused intra kernel in this file
    uses the same per-chunk form. A chunk PROLOGUE consumes the
    PRE-cumsum gate (the raw f_proj output when GATE_RAW, else the
    pre-transformed gate) and builds the chunk-local cumsum table ONCE per
    chunk, op-for-op the standalone kernels' sequence (load->fp32->bias->gate
    transform->tl.cumsum(axis=0)->*=gate_scale, cf.
    kda_gate_chunk_cumsum_vector_kernel_npu / chunk_local_cumsum_vector_kernel):

        both [BC, BK] row-halves of the chunk's raw g are loaded up front and
        each half is scanned from its OWN sub-chunk boundary; the cross-half
        prefix cancels exactly in every midpoint row-difference ((P+a)-(P+b)
        == a-b in real arithmetic), so it is dropped (proven sub-chunk-1
        semantics). A literal [BT, BK] register tile cannot be sliced into
        [BC, BK] dot operands without 3D reshape/permute/split patterns,
        which are unreliable on triton-ascend, hence the two-halves
        materialization.

    Each sub-chunk then derives ALL its gate inputs from the table by row
    differences vs its own midpoint row (BC//2 of its own half): b_gn via an
    exact one-hot masked sum (zeros add exactly in fp32), b_gq = exp2(gc-gn),
    and the K-major RHS factor kt*tl.trans(exp2(gn-gc)). This replaces the
    legacy fast path's three passes over the stored fp32 cumsum output (b_g
    tile + midpoint row + transposed tile) with one bf16 pass over the chunk's
    raw g rows, and pays the transform+scan once per chunk instead of the
    standalone-kernel GM round trip. The standalone cumsum kernel still runs
    for every other consumer (inter/wy/h-o tail/bwd).

    When FOLD_L2N is set, each sub-chunk body first runs the identical
    in-register l2norm math of the legacy fast path (fp32 accumulate, same
    eps, round-trip through the store dtype) and materializes q_n/k_n plus the
    row rstd for the later consumers - the l2norm transients die before the
    gate/dot phase, so they never coexist with the exp2/dot live set.

    Numerics vs the legacy stored-cumsum path: sub-chunk 0's table half starts
    at the chunk boundary, so its scanned values are bit-identical to the
    standalone [BT]-wide scan's first half (same per-column op order; stored
    gk is fp32) and its outputs are bit-identical to the legacy kernel's.
    Sub-chunk 1's rows drop the first-half prefix exactly in real arithmetic:
    only exp2-amplified fp32 association noise remains (measured <=3.7e-4
    abs at |Akkd|~489, relative ~5e-7, >=20x below the bf16 quantization of
    the Aqk/Akk outputs and well inside the existing tolerances).

    Geometry is fast-path only: the host gate (_decay_table_eligible)
    guarantees non-varlen, T % BT == 0, BK == K and NC == 2, so every
    load/store is full and unpredicated. Anything else takes the legacy
    kernel verbatim (incl. varlen, non-pow2 K and chunk_size 32).
    """
    tl.static_assert(NC == 2, "decay-table prologue is written for the NC==2 production layout")
    core_id = tl.program_id(0)
    num_core = tl.num_programs(0)
    # widen before multiplying; the task decomposition stays in int64 end to end
    task_num = tl.cast(NT_TOTAL, tl.int64) * BH_TOTAL
    # sub-chunk-invariant tile constants: allocated once for the whole launch
    # instead of once per task loop iteration
    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id // NT_TOTAL
        rem = task_id - i_bh * NT_TOTAL
        i_t = tl.cast(rem, tl.int32)
        i_b = tl.cast(i_bh // HV, tl.int32)
        i_hv = tl.cast(i_bh % HV, tl.int32)
        i_h = i_hv // (HV // H)

        # non-varlen only (host gate); keep the loop-carried int32 T contract
        bos = tl.cast(i_b, tl.int64) * T
        row_base = i_t * BT
        if row_base < T:
            # rebind pointers once per chunk (ptr-arg += inside the task loop
            # would both accumulate offsets across tasks and break the
            # loop-carried typecheck)
            q_l = q + (bos * H + i_h) * K
            k_l = k + (bos * H + i_h) * K
            graw_l = g_raw + (bos * HV + i_hv) * K
            beta_l = beta + (bos * HV + i_hv)
            Aqk_l = Aqk + (bos * HV + i_hv) * BT
            # Akk arg is the fp32 Akkd diagonal buffer ([B, T, HV, BC])
            Akk_l = Akk + (bos * HV + i_hv) * BC
            if FOLD_L2N:
                # non-varlen only (host gate): the [B, H, T] rstd working
                # buffer's (b, h, t) row sits at (bos*H + i_h*T + t)
                q_n_l = q_n + (bos * H + i_h) * K
                k_n_l = k_n + (bos * H + i_h) * K
                qr_l = q_rstd + bos * H + i_h * T
                kr_l = k_rstd + bos * H + i_h * T

            # ---- decay-table prologue: one pass over the chunk's raw g rows ----
            p_graw0 = tl.make_block_ptr(graw_l, (T, K), (HV * K, 1), (row_base, 0), (BC, BK), (1, 0))
            p_graw1 = tl.make_block_ptr(graw_l, (T, K), (HV * K, 1), (row_base + BC, 0), (BC, BK), (1, 0))
            if GATE_RAW:
                if HAS_BIAS:
                    # shared across both halves: loaded once per chunk
                    b_bias = tl.load(dt_bias + i_hv * K + tl.arange(0, BK)).to(tl.float32)
                b_A = tl.load(A_log + i_hv).to(tl.float32) if HAS_A else 1.0

            b_g0 = tl.load(p_graw0).to(tl.float32)
            b_g1 = tl.load(p_graw1).to(tl.float32)
            if GATE_RAW:
                if HAS_BIAS:
                    b_g0 = b_g0 + b_bias[None, :]
                    b_g1 = b_g1 + b_bias[None, :]
                if not USE_LOWER_BOUND:
                    b_t0 = -exp(b_A) * softplus(b_g0)
                    b_t1 = -exp(b_A) * softplus(b_g1)
                else:
                    b_t0 = lower_bound * tl.sigmoid((exp(b_A) if HAS_A else b_A) * b_g0)
                    b_t1 = lower_bound * tl.sigmoid((exp(b_A) if HAS_A else b_A) * b_g1)
            else:
                b_t0 = b_g0
                b_t1 = b_g1
            # one scale rounding per table value, same as the standalone
            # kernel's post-scan *= gate_scale
            if HAS_GATE_SCALE:
                b_gc0 = tl.cumsum(b_t0, axis=0) * gate_scale
                b_gc1 = tl.cumsum(b_t1, axis=0) * gate_scale
            else:
                b_gc0 = tl.cumsum(b_t0, axis=0)
                b_gc1 = tl.cumsum(b_t1, axis=0)

            # ---- per-sub-chunk bodies: everything derived from the table ----
            # static unroll (NC is a small constexpr, 2 in production): keeps
            # the per-task instruction stream flat like the former
            # per-sub-chunk queue
            for i_i in tl.static_range(0, NC):
                # constexpr select of this sub-chunk's table half (pruned at
                # compile time by the static_range unroll)
                if i_i == 0:
                    b_gc = b_gc0
                else:
                    b_gc = b_gc1
                i_ti = row_base + i_i * BC

                p_q = tl.make_block_ptr(q_l, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
                p_k = tl.make_block_ptr(k_l, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
                # K-major companion of k, consumed ONLY where BK == K (host
                # gate): the [BK, BC] tile is exactly in-bounds, assembled
                # straight in transposed layout (same pattern as the legacy
                # fast path's p_kt; BK != K keeps the legacy kernel entirely)
                p_kt = tl.make_block_ptr(k_l, (K, T), (1, H * K), (0, i_ti), (BK, BC), (0, 1))
                p_beta = tl.make_block_ptr(beta_l, (T,), (HV,), (i_ti,), (BC,), (0,))

                b_q = tl.load(p_q)
                b_k = tl.load(p_k)
                if FOLD_L2N:
                    # first q/k consumer: the exact l2norm_fwd math of the
                    # legacy fast path (fp32 accumulate, same eps), the
                    # normalized tiles + row rstd materialized for every later
                    # reader; dot inputs round-trip through the store dtype so
                    # they equal the materialized q_n/k_n bytes
                    p_qn = tl.make_block_ptr(q_n_l, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
                    p_kn = tl.make_block_ptr(k_n_l, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
                    b_q32 = b_q.to(tl.float32)
                    b_k32 = b_k.to(tl.float32)
                    b_qr = 1 / tl.sqrt(tl.sum(b_q32 * b_q32, 1) + L2N_EPS)
                    b_kr = 1 / tl.sqrt(tl.sum(b_k32 * b_k32, 1) + L2N_EPS)
                    b_q32 = b_q32 * b_qr[:, None]
                    b_k32 = b_k32 * b_kr[:, None]
                    tl.store(p_qn, b_q32.to(q_n.dtype.element_ty))
                    tl.store(p_kn, b_k32.to(k_n.dtype.element_ty))
                    tl.store(qr_l + i_ti + tl.arange(0, BC), b_qr)
                    tl.store(kr_l + i_ti + tl.arange(0, BC), b_kr)
                    b_q = b_q32.to(q_n.dtype.element_ty)
                    b_k = b_k32.to(k_n.dtype.element_ty)
                b_beta = tl.load(p_beta)

                # this sub-chunk's midpoint row from the chunk-local table:
                # exact one-hot masked sum (zeros add exactly in fp32)
                b_gn = tl.sum(tl.where(o_i[:, None] == BC // 2, b_gc, 0.0), 0)

                b_gq = exp2(b_gc - b_gn[None, :])
                # shared RHS assembled row-major from the table, then
                # transposed (the lowering the legacy masked path ships); the
                # folded run scales the raw K-major k tile by the row rstd and
                # rounds through the store dtype first, exactly like the
                # legacy fast path
                if FOLD_L2N:
                    b_kgt = (tl.load(p_kt).to(tl.float32) * b_kr[None, :]).to(k_n.dtype.element_ty) \
                        * tl.trans(exp2(b_gn[None, :] - b_gc))
                else:
                    b_kgt = tl.load(p_kt) * tl.trans(exp2(b_gn[None, :] - b_gc))

                b_Aqk = tl.dot(b_q * b_gq, b_kgt, allow_tf32=False) * scale
                b_Akk = tl.dot(b_k * b_gq, b_kgt, allow_tf32=False) * b_beta[:, None]

                b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
                b_Akk = tl.where(m_Akk, b_Akk, 0.0)

                p_Aqk = tl.make_block_ptr(Aqk_l, (T, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
                p_Akk = tl.make_block_ptr(Akk_l, (T, BC), (HV * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
                tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty))
                tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty))


@triton.jit(do_not_specialize=['T', 'NT_TOTAL', 'BH_TOTAL'])
def chunk_kda_fwd_kernel_inter_solve_fused_npu(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    NT_TOTAL,
    BH_TOTAL,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Persistent core-count grid over the (i_bh, i_t) task queue.

    One launch with <= num_aicore programs; each program loops over its share
    of the tasks with stride num_programs (classic persistent style, same
    structure as chunk_kda_bwd_kernel_intra_npu). The task layout keeps i_t
    fastest and i_bh slowest - the linearization of the former (NT, BH) grid -
    so a program's consecutive tasks stay within one head for L2 locality
    while the whole grid streams T in order. Per-task tile math is value-
    preserved; the task body takes a mask-free exact-tile fast path when
    NC == 2, BK == K and the chunk is fully in-bounds (comment at the branch).
    """
    # Diagonal Akkd blocks are inverted by diag_solve before this kernel.
    core_id = tl.program_id(0)
    num_core = tl.num_programs(0)
    # widen before multiplying; the task decomposition stays in int64 end to end
    task_num = tl.cast(NT_TOTAL, tl.int64) * BH_TOTAL
    # Akk is stored in the wy-consumer layout [B, HV, T, BT] (T-contiguous
    # rows): the bwd wy trio and recompute_w_u read it without a host
    # transpose. T_seq keeps the packed length before varlen overwrites T.
    T_seq = T
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id // NT_TOTAL
        rem = task_id - i_bh * NT_TOTAL
        i_t = tl.cast(rem % NT_TOTAL, tl.int32)
        i_b = tl.cast(i_bh // HV, tl.int32)
        i_hv = tl.cast(i_bh % HV, tl.int32)
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            # T is a loop-carried arg (int32); the reassignment must keep its type
            T = (eos - bos).to(tl.int32)
        else:
            bos = tl.cast(i_b, tl.int64) * T
            eos = bos + T

        # padding-task guard (was the early return of the split-launch grid)
        if i_t * BT < T:
            i_tc0 = i_t * BT
            i_tc1 = i_t * BT + BC
            i_tc2 = i_t * BT + 2 * BC
            i_tc3 = i_t * BT + 3 * BC

            q_l = q + (bos * H + i_h) * K
            k_l = k + (bos * H + i_h) * K
            g_l = g + (bos * HV + i_hv) * K
            Aqk_l = Aqk + (bos * HV + i_hv) * BT
            if IS_VARLEN:
                Akk_l = Akk + tl.cast(i_hv, tl.int64) * T_seq * BT + bos * BT
            else:
                Akk_l = Akk + tl.cast(i_b * HV + i_hv, tl.int64) * T_seq * BT
            Akkd_l = Akkd + (bos * HV + i_hv) * BC

            o_i = tl.arange(0, BC)
            m_tc0 = (i_tc0 + o_i) < T
            m_tc1 = (i_tc1 + o_i) < T
            m_tc2 = (i_tc2 + o_i) < T
            m_tc3 = (i_tc3 + o_i) < T

            b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

            b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

            b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

            # loop-invariant gate-row bases, hoisted out of the K loop
            # (int64 keep: i_tc * HV * K must not narrow at T=131072)
            p_gn1 = g_l + tl.cast(i_tc1, tl.int64) * (HV * K)
            if NC >= 3:
                p_gn2 = g_l + tl.cast(i_tc2, tl.int64) * (HV * K)
                if NC >= 4:
                    p_gn3 = g_l + tl.cast(i_tc3, tl.int64) * (HV * K)

            # NB: parenthesized - the triton 3.2.0 frontend rejects boolean
            # chains longer than two operands (chained-boolean compile error).
            if (NC == 2 and BK == K) and (i_tc1 + BC <= T):
                # Exact-tile fast path, manual addressing only - deliberately
                # NO make_block_ptr here: under this kernel's
                # blacklist_auto_blockify=True compile kwargs the block-ptr
                # form of this same body benched measurably slower on
                # long-sequence fwd shapes. On the production tile (BT=2*BC,
                # one BK tile spans K, chunk fully in-bounds) every load is
                # exact, so the row-mask wheres, the per-element load
                # predicates and the column masks all disappear while the
                # address arithmetic stays in manual-pointer form.
                o_k = tl.arange(0, BK)
                b_k0 = tl.load(k_l + tl.cast(i_tc0, tl.int64) * (H * K) + o_i[:, None] * (H * K) + o_k[None, :]).to(tl.float32)
                b_g0 = tl.load(g_l + tl.cast(i_tc0, tl.int64) * (HV * K) + o_i[:, None] * (HV * K) + o_k[None, :]).to(tl.float32)

                p_q1 = q_l + tl.cast(i_tc1, tl.int64) * (H * K)
                p_k1 = k_l + tl.cast(i_tc1, tl.int64) * (H * K)
                p_g1 = g_l + tl.cast(i_tc1, tl.int64) * (HV * K)
                b_q1 = tl.load(p_q1 + o_i[:, None] * (H * K) + o_k[None, :]).to(tl.float32)
                b_k1 = tl.load(p_k1 + o_i[:, None] * (H * K) + o_k[None, :]).to(tl.float32)
                b_g1 = tl.load(p_g1 + o_i[:, None] * (HV * K) + o_k[None, :]).to(tl.float32)
                b_gn1 = tl.load(p_gn1 + o_k).to(tl.float32)
                b_gqn = exp2(b_g1 - b_gn1[None, :])
                b_qg1 = b_q1 * b_gqn
                b_kg1 = b_k1 * b_gqn
                b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
                b_Aqk10 = tl.dot(b_qg1, b_kgt, b_Aqk10, allow_tf32=False)
                b_Akk10 = tl.dot(b_kg1, b_kgt, b_Akk10, allow_tf32=False)
            else:
                for i_k in range(tl.cdiv(K, BK)):
                    o_k = i_k * BK + tl.arange(0, BK)
                    m_k = o_k < K
                    m_k2 = m_k[None, :]

                    b_k0 = tl.load(k_l + tl.cast(i_tc0, tl.int64) * (H * K) + o_i[:, None] * (H * K) + o_k[None, :],
                                   mask=m_tc0[:, None] & m_k2, other=0.0).to(tl.float32)
                    b_g0 = tl.load(g_l + tl.cast(i_tc0, tl.int64) * (HV * K) + o_i[:, None] * (HV * K) + o_k[None, :],
                                   mask=m_tc0[:, None] & m_k2, other=0.0).to(tl.float32)

                    p_q1 = q_l + tl.cast(i_tc1, tl.int64) * (H * K)
                    p_k1 = k_l + tl.cast(i_tc1, tl.int64) * (H * K)
                    p_g1 = g_l + tl.cast(i_tc1, tl.int64) * (HV * K)
                    b_q1 = tl.load(p_q1 + o_i[:, None] * (H * K) + o_k[None, :], mask=m_tc1[:, None] & m_k2, other=0.0).to(tl.float32)
                    b_k1 = tl.load(p_k1 + o_i[:, None] * (H * K) + o_k[None, :], mask=m_tc1[:, None] & m_k2, other=0.0).to(tl.float32)
                    b_g1 = tl.load(p_g1 + o_i[:, None] * (HV * K) + o_k[None, :], mask=m_tc1[:, None] & m_k2, other=0.0).to(tl.float32)
                    b_gn1 = tl.load(p_gn1 + o_k, mask=m_k & (i_tc1 < T), other=0).to(tl.float32)
                    b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
                    b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
                    b_Aqk10 = tl.dot(b_q1 * b_gqn, b_kgt, b_Aqk10, allow_tf32=False)
                    b_Akk10 = tl.dot(b_k1 * b_gqn, b_kgt, b_Akk10, allow_tf32=False)

                    if NC >= 3:
                        p_q2 = q_l + tl.cast(i_tc2, tl.int64) * (H * K)
                        p_k2 = k_l + tl.cast(i_tc2, tl.int64) * (H * K)
                        p_g2 = g_l + tl.cast(i_tc2, tl.int64) * (HV * K)
                        b_q2 = tl.load(p_q2 + o_i[:, None] * (H * K) + o_k[None, :], mask=m_tc2[:, None] & m_k2, other=0.0).to(tl.float32)
                        b_k2 = tl.load(p_k2 + o_i[:, None] * (H * K) + o_k[None, :], mask=m_tc2[:, None] & m_k2, other=0.0).to(tl.float32)
                        b_g2 = tl.load(p_g2 + o_i[:, None] * (HV * K) + o_k[None, :], mask=m_tc2[:, None] & m_k2, other=0.0).to(tl.float32)
                        b_gn2 = tl.load(p_gn2 + o_k, mask=m_k & (i_tc2 < T), other=0).to(tl.float32)
                        b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                        b_qg2 = b_q2 * b_gqn2
                        b_kg2 = b_k2 * b_gqn2
                        b_qg2_c = b_qg2 + 0.0
                        b_kg2_c = b_kg2 + 0.0
                        b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                        b_Aqk20 = tl.dot(b_qg2, b_kgt, b_Aqk20, allow_tf32=False)
                        b_Akk20 = tl.dot(b_kg2, b_kgt, b_Akk20, allow_tf32=False)
                        b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                        b_Aqk21 = tl.dot(b_qg2_c, b_kgt, b_Aqk21, allow_tf32=False)
                        b_Akk21 = tl.dot(b_kg2_c, b_kgt, b_Akk21, allow_tf32=False)

                        if NC >= 4:
                            p_q3 = q_l + tl.cast(i_tc3, tl.int64) * (H * K)
                            p_k3 = k_l + tl.cast(i_tc3, tl.int64) * (H * K)
                            p_g3 = g_l + tl.cast(i_tc3, tl.int64) * (HV * K)
                            b_q3 = tl.load(p_q3 + o_i[:, None] * (H * K) + o_k[None, :],
                                           mask=m_tc3[:, None] & m_k2, other=0.0).to(tl.float32)
                            b_k3 = tl.load(p_k3 + o_i[:, None] * (H * K) + o_k[None, :],
                                           mask=m_tc3[:, None] & m_k2, other=0.0).to(tl.float32)
                            b_g3 = tl.load(p_g3 + o_i[:, None] * (HV * K) + o_k[None, :],
                                           mask=m_tc3[:, None] & m_k2, other=0.0).to(tl.float32)
                            b_gn3 = tl.load(p_gn3 + o_k, mask=m_k & (i_tc3 < T), other=0).to(tl.float32)
                            b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                            b_qg3 = b_q3 * b_gqn3
                            b_kg3 = b_k3 * b_gqn3
                            b_qg3_c1 = b_qg3 + 0.0
                            b_kg3_c1 = b_kg3 + 0.0
                            b_qg3_c2 = b_qg3 + 0.0
                            b_kg3_c2 = b_kg3 + 0.0
                            b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                            b_Aqk30 = tl.dot(b_qg3, b_kgt, b_Aqk30, allow_tf32=False)
                            b_Akk30 = tl.dot(b_kg3, b_kgt, b_Akk30, allow_tf32=False)
                            b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                            b_Aqk31 = tl.dot(b_qg3_c1, b_kgt, b_Aqk31, allow_tf32=False)
                            b_Akk31 = tl.dot(b_kg3_c1, b_kgt, b_Akk31, allow_tf32=False)
                            b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                            b_Aqk32 = tl.dot(b_qg3_c2, b_kgt, b_Aqk32, allow_tf32=False)
                            b_Akk32 = tl.dot(b_kg3_c2, b_kgt, b_Akk32, allow_tf32=False)

            p_Aqk1 = Aqk_l + tl.cast(i_tc1, tl.int64) * (HV * BT) + o_i[:, None] * (HV * BT)
            tl.store(p_Aqk1 + o_i[None, :], (b_Aqk10 * scale).to(Aqk_l.dtype.element_ty), mask=m_tc1[:, None])

            b_b1 = tl.load(beta + (bos * HV + i_hv) + tl.cast(i_tc1, tl.int64) * HV + o_i * HV, mask=m_tc1, other=0.0).to(tl.float32)
            b_Akk10 = b_Akk10 * b_b1[:, None]
            if NC >= 3:
                p_Aqk2 = Aqk_l + tl.cast(i_tc2, tl.int64) * (HV * BT) + o_i[:, None] * (HV * BT)
                tl.store(p_Aqk2 + o_i[None, :], (b_Aqk20 * scale).to(Aqk_l.dtype.element_ty), mask=m_tc2[:, None])
                tl.store(p_Aqk2 + BC + o_i[None, :], (b_Aqk21 * scale).to(Aqk_l.dtype.element_ty), mask=m_tc2[:, None])

                b_b2 = tl.load(beta + (bos * HV + i_hv) + tl.cast(i_tc2, tl.int64)
                               * HV + o_i * HV, mask=m_tc2, other=0.0).to(tl.float32)
                b_Akk20 = b_Akk20 * b_b2[:, None]
                b_Akk21 = b_Akk21 * b_b2[:, None]
            if NC >= 4:
                p_Aqk3 = Aqk_l + tl.cast(i_tc3, tl.int64) * (HV * BT) + o_i[:, None] * (HV * BT)
                tl.store(p_Aqk3 + o_i[None, :], (b_Aqk30 * scale).to(Aqk_l.dtype.element_ty), mask=m_tc3[:, None])
                tl.store(p_Aqk3 + BC + o_i[None, :], (b_Aqk31 * scale).to(Aqk_l.dtype.element_ty), mask=m_tc3[:, None])
                tl.store(p_Aqk3 + 2 * BC + o_i[None, :], (b_Aqk32 * scale).to(Aqk_l.dtype.element_ty), mask=m_tc3[:, None])

                b_b3 = tl.load(beta + (bos * HV + i_hv) + tl.cast(i_tc3, tl.int64)
                               * HV + o_i * HV, mask=m_tc3, other=0.0).to(tl.float32)
                b_Akk30 = b_Akk30 * b_b3[:, None]
                b_Akk31 = b_Akk31 * b_b3[:, None]
                b_Akk32 = b_Akk32 * b_b3[:, None]

            p_Akkd0 = Akkd_l + tl.cast(i_tc0, tl.int64) * (HV * BC) + o_i[:, None] * (HV * BC)
            p_Akkd1 = Akkd_l + tl.cast(i_tc1, tl.int64) * (HV * BC) + o_i[:, None] * (HV * BC)
            b_Ai00 = tl.load(p_Akkd0 + o_i[None, :], mask=m_tc0[:, None], other=0.0).to(tl.float32)
            b_Ai11 = tl.load(p_Akkd1 + o_i[None, :], mask=m_tc1[:, None], other=0.0).to(tl.float32)
            if NC >= 3:
                p_Akkd2 = Akkd_l + tl.cast(i_tc2, tl.int64) * (HV * BC) + o_i[:, None] * (HV * BC)
                b_Ai22 = tl.load(p_Akkd2 + o_i[None, :], mask=m_tc2[:, None], other=0.0).to(tl.float32)
            if NC >= 4:
                p_Akkd3 = Akkd_l + tl.cast(i_tc3, tl.int64) * (HV * BC) + o_i[:, None] * (HV * BC)
                b_Ai33 = tl.load(p_Akkd3 + o_i[None, :], mask=m_tc3[:, None], other=0.0).to(tl.float32)

            b_Ai11_c = b_Ai11 + 0.0
            if NC >= 3:
                b_Ai22_c = b_Ai22 + 0.0
                b_Ai22_c2 = b_Ai22 + 0.0
                b_Ai22_c3 = b_Ai22 + 0.0
            if NC >= 4:
                b_Ai33_c = b_Ai33 + 0.0
                b_Ai33_c2 = b_Ai33 + 0.0
                b_Ai33_c3 = b_Ai33 + 0.0
                b_Akk31_c = b_Akk31 + 0.0
                b_Akk32_c = b_Akk32 + 0.0

            b_Ai10 = -tl.dot(
                tl.dot(b_Ai11, b_Akk10, allow_tf32=False),
                b_Ai00,
                allow_tf32=False,
            )

            if NC >= 3:
                b_Ai21 = -tl.dot(
                    tl.dot(b_Ai22, b_Akk21, allow_tf32=False),
                    b_Ai11_c,
                    allow_tf32=False,
                )
                b_Ai20 = -tl.dot(
                    b_Ai22_c2,
                    tl.dot(b_Akk20, b_Ai00, allow_tf32=False) +
                    tl.dot(b_Akk21, b_Ai10, allow_tf32=False),
                    allow_tf32=False,
                )
            if NC >= 4:
                b_Ai32 = -tl.dot(
                    tl.dot(b_Ai33, b_Akk32, allow_tf32=False),
                    b_Ai22_c3,
                    allow_tf32=False,
                )
                b_Ai31 = -tl.dot(
                    b_Ai33_c2,
                    tl.dot(b_Akk31, b_Ai11_c, allow_tf32=False) +
                    tl.dot(b_Akk32, b_Ai21, allow_tf32=False),
                    allow_tf32=False,
                )
                b_Ai30 = -tl.dot(
                    b_Ai33_c3,
                    tl.dot(b_Akk30, b_Ai00, allow_tf32=False) +
                    tl.dot(b_Akk31_c, b_Ai10, allow_tf32=False) +
                    tl.dot(b_Akk32_c, b_Ai20, allow_tf32=False),
                    allow_tf32=False,
                )

            # T-contig Akk rows: stride BT along t within the [B, HV, T, BT] slab
            # (Akk_l already carries the per-head + bos offset).
            p_Akk0 = Akk_l + tl.cast(i_tc0, tl.int64) * BT + o_i[:, None] * BT
            p_Akk1 = Akk_l + tl.cast(i_tc1, tl.int64) * BT + o_i[:, None] * BT
            p_Akk2 = Akk_l + tl.cast(i_tc2, tl.int64) * BT + o_i[:, None] * BT
            p_Akk3 = Akk_l + tl.cast(i_tc3, tl.int64) * BT + o_i[:, None] * BT

            tl.store(p_Akk0 + o_i[None, :], b_Ai00.to(Akk_l.dtype.element_ty), mask=m_tc0[:, None])
            tl.store(p_Akk1 + o_i[None, :], b_Ai10.to(Akk_l.dtype.element_ty), mask=m_tc1[:, None])
            tl.store(p_Akk1 + BC + o_i[None, :], b_Ai11_c.to(Akk_l.dtype.element_ty), mask=m_tc1[:, None])
            if NC >= 3:
                tl.store(p_Akk2 + o_i[None, :], b_Ai20.to(Akk_l.dtype.element_ty), mask=m_tc2[:, None])
                tl.store(p_Akk2 + BC + o_i[None, :], b_Ai21.to(Akk_l.dtype.element_ty), mask=m_tc2[:, None])
                tl.store(p_Akk2 + 2 * BC + o_i[None, :], b_Ai22_c.to(Akk_l.dtype.element_ty), mask=m_tc2[:, None])
            if NC >= 4:
                tl.store(p_Akk3 + o_i[None, :], b_Ai30.to(Akk_l.dtype.element_ty), mask=m_tc3[:, None])
                tl.store(p_Akk3 + BC + o_i[None, :], b_Ai31.to(Akk_l.dtype.element_ty), mask=m_tc3[:, None])
                tl.store(p_Akk3 + 2 * BC + o_i[None, :], b_Ai32.to(Akk_l.dtype.element_ty), mask=m_tc3[:, None])
                tl.store(p_Akk3 + 3 * BC + o_i[None, :], b_Ai33_c.to(Akk_l.dtype.element_ty), mask=m_tc3[:, None])


@input_guard
def chunk_kda_fwd_intra_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
    use_graph: bool = False,
    store_qg_scaled: bool = False,
    qk_l2norm_out: dict | None = None,
    g_raw: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    gate_scale: float | None = None,
    gk_t: torch.Tensor | None = None,
):
    global _FUSED_INTRA_OK
    global _FUSED_FOLD_OK
    if use_graph:
        raise NotImplementedError("use_graph is not supported on the Ascend NPU backend")
    if qk_l2norm_out is not None and not safe_gate:
        # the token-parallel kernel reads cross-token k rows whose rstd other
        # programs own, so the fold is only wired into the persistent sub-chunk
        # kernel; the host keeps its standalone l2norm on that path
        raise NotImplementedError("qk_l2norm_out requires the safe_gate intra path")
    B, T, H, K, HV = *k.shape, gk.shape[2]
    BT = chunk_size
    if BT not in (32, 64):
        raise ValueError(f"KDA intra chunk kernel only supports chunk_size 32 or 64, got {BT}.")
    # BC=32 tiles need NC>=2 (chunk_size 32 keeps BC=16).
    # BC=32 tile envelope: sub BK is next_pow2(K), and [32, 512] sub tiles exceed the
    # UB budget (compile failure) - keep stock BC=16 for K > 256.
    BC = _BC if (BT >= 64 and K <= 256) else 16
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    is_varlen = cu_seqlens is not None

    Aqk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    # Akk in the wy-consumer layout [B, HV, T, BT] so the bwd wy trio and
    # recompute_w_u consume it without an aten Transpose (inter_solve writes
    # the same bytes either way). Zeros init retained even though the stores
    # fully cover the slab; only the address permutation changes.
    Akk = torch.zeros(B, HV, T, BT, device=k.device, dtype=k.dtype)

    # Fused persistent path: intra_sub_chunk + diag_solve + inter_solve in one
    # <=24-program launch. NC==2 layouts only, non-varlen, K<=128 (UB
    # envelope) and T<=4096 (launch-bound regime, see _FUSED_INTRA_MAX_T);
    # everything else - and any compile/launch failure - takes the persistent
    # fallback pipeline below (persistent sub_chunk, split diag_solve,
    # persistent inter).
    use_fused = (
        safe_gate
        and NC == 2
        and cu_seqlens is None
        and K <= _FUSED_INTRA_MAX_K
        and T <= _FUSED_INTRA_MAX_T
        and _FUSED_INTRA_OK
    )
    # h3 fold coverage: at small T the fused kernel is the first q/k
    # consumer, so try the FOLD_L2N specialization first - it materializes
    # the caller's bundle in-pass (normalized q/k + [B, H, T] rstd rows)
    # and the standalone host l2norm round trip disappears on this path
    # too. Value-safe fallback ladder: fold failure (likely a UB/compile
    # reject of the fatter live set) permanently reroutes to the proven
    # host l2norm fill + non-fold fused launch below; a non-fold fused
    # failure then falls to the persistent pipeline exactly as before.
    # Every rewrite in the ladder is an idempotent overwrite of
    # deterministic bytes, so rerunning after a failed launch is safe.
    fused_folded = False
    if use_fused and qk_l2norm_out is not None and _FUSED_FOLD_OK:
        try:
            _launch_core_grid(
                chunk_kda_fwd_kernel_intra_diag_inter_fused_npu,
                task_num=NT * B * HV,
                kernel_kwargs=dict(
                    q=q,
                    k=k,
                    g=gk,
                    beta=beta,
                    Aqk=Aqk,
                    Akk=Akk,
                    scale=scale,
                    T=T,
                    NT_TOTAL=NT,
                    BH_TOTAL=B * HV,
                    H=H,
                    HV=HV,
                    K=K,
                    BT=BT,
                    BC=BC,
                    BK=triton.next_power_of_2(K),
                    q_n=qk_l2norm_out['q'],
                    k_n=qk_l2norm_out['k'],
                    q_rstd=qk_l2norm_out['q_rstd'],
                    k_rstd=qk_l2norm_out['k_rstd'],
                    FOLD_L2N=True,
                    L2N_EPS=1e-6,
                ),
                num_warps=_NUM_WARPS_INTER,
                **_INTER_COMPILE_KWARGS,
            )
            fused_folded = True
        except Exception:
            _FUSED_FOLD_OK = False

    if use_fused and qk_l2norm_out is not None and not fused_folded:
        # fold unavailable at this shape: materialize the host l2norm into
        # the caller's bundle. After the rebind the fused launch (and any
        # launch-failure fallback) consumes normalized q/k, and bwd still
        # gets its rstd buffers.
        from fla.modules.l2norm import l2norm_fwd
        q_n, q_rstd = l2norm_fwd(q)
        k_n, k_rstd = l2norm_fwd(k)
        qk_l2norm_out['q'], qk_l2norm_out['k'] = q_n, k_n
        qk_l2norm_out['q_rstd'].copy_(q_rstd.transpose(1, 2))
        qk_l2norm_out['k_rstd'].copy_(k_rstd.transpose(1, 2))
        q, k = q_n, k_n
        qk_l2norm_out = None

    if use_fused and not fused_folded:
        try:
            _launch_core_grid(
                chunk_kda_fwd_kernel_intra_diag_inter_fused_npu,
                task_num=NT * B * HV,
                kernel_kwargs=dict(
                    q=q,
                    k=k,
                    g=gk,
                    beta=beta,
                    Aqk=Aqk,
                    Akk=Akk,
                    scale=scale,
                    T=T,
                    NT_TOTAL=NT,
                    BH_TOTAL=B * HV,
                    H=H,
                    HV=HV,
                    K=K,
                    BT=BT,
                    BC=BC,
                    BK=triton.next_power_of_2(K),
                    q_n=None,
                    k_n=None,
                    q_rstd=None,
                    k_rstd=None,
                ),
                num_warps=_NUM_WARPS_INTER,
                **_INTER_COMPILE_KWARGS,
            )
        except Exception:
            # permanently disable the fused kernel and rerun the unfused
            # pipeline (outputs are pure overwrites of the zero-filled
            # Aqk/Akk, so rerunning after a failed launch is value-safe)
            _FUSED_INTRA_OK = False
            use_fused = False

    if fused_folded:
        # the fused kernel just materialized the normalized q/k into the
        # bundle; rebind so recompute_w_u and every successor read the
        # normalized tensors (same-stream ordering serializes the launches)
        q = qk_l2norm_out['q']
        k = qk_l2norm_out['k']

    if not use_fused:
        Akkd = torch.zeros(B, T, HV, BC, device=k.device, dtype=torch.float32)

        if safe_gate:
            # the UB model for sub tiles is never binding for K<=128 (returns next_pow2(K)
            # unchanged); for K>128 it would shrink BK to 128 and change the Akkd accumulation
            # path (fp16 varlen tolerance) - so pin the stock next_pow2 for all K.
            sub_bk = triton.next_power_of_2(K)

            def _launch_fold_sub_chunk():
                # NC-granular tasks with the in-kernel l2norm fold that
                # materializes the caller's bundle (normalized q/k + rstd).
                # Only launched while the bundle is active.
                _launch_core_grid(
                    chunk_kda_fwd_kernel_intra_sub_chunk_fold_npu,
                    task_num=NT * NC * B * HV,
                    kernel_kwargs=dict(
                        q=q,
                        k=k,
                        q_n=qk_l2norm_out['q'] if qk_l2norm_out is not None else None,
                        k_n=qk_l2norm_out['k'] if qk_l2norm_out is not None else None,
                        q_rstd=qk_l2norm_out['q_rstd'] if qk_l2norm_out is not None else None,
                        k_rstd=qk_l2norm_out['k_rstd'] if qk_l2norm_out is not None else None,
                        g=gk,
                        beta=beta,
                        Aqk=Aqk,
                        Akk=Akkd,
                        scale=scale,
                        cu_seqlens=cu_seqlens,
                        chunk_indices=chunk_indices,
                        T=T,
                        NT_TOTAL=NT,
                        BH_TOTAL=B * HV,
                        H=H,
                        HV=HV,
                        K=K,
                        BT=BT,
                        BC=BC,
                        BK=sub_bk,
                        NC=NC,
                        IS_VARLEN=is_varlen,
                        FOLD_L2N=qk_l2norm_out is not None,
                        L2N_EPS=1e-6,
                    ),
                    num_warps=_NUM_WARPS_SUB,
                )

            def _launch_legacy_sub_chunk():
                # chunk-granular persistent tasks; expects q/k already
                # normalized.
                _launch_core_grid(
                    chunk_kda_fwd_kernel_intra_sub_chunk_npu,
                    task_num=NT * B * HV,
                    kernel_kwargs=dict(
                        q=q,
                        k=k,
                        g=gk,
                        beta=beta,
                        Aqk=Aqk,
                        Akk=Akkd,
                        scale=scale,
                        cu_seqlens=cu_seqlens,
                        chunk_indices=chunk_indices,
                        T=T,
                        NT_TOTAL=NT,
                        BH_TOTAL=B * HV,
                        H=H,
                        HV=HV,
                        K=K,
                        BT=BT,
                        BC=BC,
                        BK=sub_bk,
                        NC=NC,
                        IS_VARLEN=is_varlen,
                    ),
                    num_warps=_NUM_WARPS_SUB,
                )

            # decay-table path: fold the intra path's own gate preparation
            # into a persistent per-chunk kernel so it stops reading back the
            # standalone cumsum output (one bf16 pass over the chunk's raw g
            # rows replaces the b_g tile + midpoint row + transposed tile
            # fp32 loads; every sub-chunk gate input is derived from the
            # chunk-local cumsum table). The standalone kernel still runs for
            # the other consumers (inter/wy/h-o tail/bwd). Strict geometry
            # gate + permanent fallback on any decay-table launch failure to
            # the kernel matching the bundle state (fold kernel while the
            # bundle is active, the legacy kernel otherwise).
            decay_table_eligible = (
                g_raw is None or g_raw.shape == gk.shape
            ) and _decay_table_eligible(
                safe_gate=safe_gate,
                cu_seqlens=cu_seqlens,
                use_graph=use_graph,
                g_raw=g_raw,
                gate_scale=gate_scale,
                T=T,
                chunk_size=chunk_size,
                K=K,
            )
            if decay_table_eligible:
                def _launch_decay_table():
                    _launch_core_grid(
                        chunk_kda_fwd_kernel_intra_sub_chunk_decay_table_npu,
                        task_num=NT * B * HV,
                        kernel_kwargs=dict(
                            q=q,
                            k=k,
                            q_n=qk_l2norm_out['q'] if qk_l2norm_out is not None else None,
                            k_n=qk_l2norm_out['k'] if qk_l2norm_out is not None else None,
                            q_rstd=qk_l2norm_out['q_rstd'] if qk_l2norm_out is not None else None,
                            k_rstd=qk_l2norm_out['k_rstd'] if qk_l2norm_out is not None else None,
                            g_raw=g_raw,
                            beta=beta,
                            Aqk=Aqk,
                            Akk=Akkd,
                            A_log=A_log,
                            dt_bias=dt_bias,
                            lower_bound=lower_bound,
                            gate_scale=gate_scale,
                            scale=scale,
                            T=T,
                            NT_TOTAL=NT,
                            BH_TOTAL=B * HV,
                            H=H,
                            HV=HV,
                            K=K,
                            BT=BT,
                            BC=BC,
                            BK=sub_bk,
                            NC=NC,
                            GATE_RAW=bool(use_gate_in_kernel),
                            HAS_A=A_log is not None,
                            HAS_BIAS=dt_bias is not None,
                            USE_LOWER_BOUND=lower_bound is not None,
                            HAS_GATE_SCALE=gate_scale is not None,
                            FOLD_L2N=qk_l2norm_out is not None,
                            L2N_EPS=1e-6,
                        ),
                        num_warps=_NUM_WARPS_SUB,
                    )

                if qk_l2norm_out is not None:
                    _run_decay_table_with_fallback(_launch_decay_table, _launch_fold_sub_chunk)
                else:
                    _run_decay_table_with_fallback(_launch_decay_table, _launch_legacy_sub_chunk)
            elif qk_l2norm_out is not None:
                try:
                    _launch_fold_sub_chunk()
                except Exception:
                    # fold kernel unusable at this shape: materialize the host
                    # l2norm into the caller's bundle and rerun the legacy
                    # kernel on the normalized q/k; downstream sees identical
                    # bytes either way
                    from fla.modules.l2norm import l2norm_fwd
                    q_n, q_rstd = l2norm_fwd(q)
                    k_n, k_rstd = l2norm_fwd(k)
                    qk_l2norm_out['q'], qk_l2norm_out['k'] = q_n, k_n
                    qk_l2norm_out['q_rstd'].copy_(q_rstd.transpose(1, 2))
                    qk_l2norm_out['k_rstd'].copy_(k_rstd.transpose(1, 2))
                    q, k = q_n, k_n
                    qk_l2norm_out = None
                    _launch_legacy_sub_chunk()
            else:
                _launch_legacy_sub_chunk()
            if qk_l2norm_out is not None:
                # the persistent kernel just materialized the normalized q/k;
                # same-stream ordering serializes before every later launch here
                q = qk_l2norm_out['q']
                k = qk_l2norm_out['k']
        else:
            Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(
                q=q,
                k=k,
                gk=gk,
                beta=beta,
                Aqk=Aqk,
                Akk=Akkd,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_size=BT,
                sub_chunk_size=BC,
            )

        # Invert diagonal Akkd blocks first; inter then only merges off-diagonals.
        # Persistent single launch: <= AIV (vector) core count programs cover the
        # full (NT, NC, BH) task space (an earlier budget-split multi-launcher
        # was replaced by this single persistent launch).
        diag_task_num = NT * NC * B * HV
        diag_num_core = min(get_npu_aiv_core_count(), diag_task_num)
        fast_launch(chunk_kda_fwd_kernel_diag_solve_npu, (diag_num_core,),
            Akkd=Akkd,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            B=B,
            NT_TOTAL=NT,
            HV=HV,
            BT=BT,
            BC=BC,
            num_warps=_NUM_WARPS_SUB,
        )

        inter_bk = _get_inter_bk(K)
        _launch_core_grid(
            chunk_kda_fwd_kernel_inter_solve_fused_npu,
            task_num=NT * B * HV,
            kernel_kwargs=dict(
                q=q,
                k=k,
                g=gk,
                beta=beta,
                Aqk=Aqk,
                Akkd=Akkd,
                Akk=Akk,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                NT_TOTAL=NT,
                BH_TOTAL=B * HV,
                H=H,
                HV=HV,
                K=K,
                BT=BT,
                BC=BC,
                NC=NC,
                BK=inter_bk,
                IS_VARLEN=is_varlen,
            ),
            num_warps=_NUM_WARPS_INTER,
            **_INTER_COMPILE_KWARGS,
        )
    # Akkd is dead here (inter_solve was its last reader) but must NOT be
    # freed early: a stream-ordered free of the [B, T, HV, BC] fp32 buffer at
    # this point - letting the allocator hand the block to the w/u/qg/kg
    # outputs of the recompute stage below - crashed the NPU runtime at
    # T=131072 (async workspace reuse hazard). Let the tensor die at scope
    # exit instead.
    # qg_scaled: bf16(q * exp2(g) * scale) filled by the WY producer in the
    # same pass (its exp2(g) is already live there); consumed by the fused
    # h+o tail so the per-chunk fp32 exp2/mul gate chain leaves the tail loop.
    qg_scaled = k.new_empty(B, T, HV, K) if store_qg_scaled else None
    w, u, qg, kg = _recompute_w_u_fwd_kda_npu(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=q if (disable_recompute or store_qg_scaled) else None,
        gk=gk,
        gk_t=gk_t,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        qg_scaled=qg_scaled,
        scale=scale,
        store_qg=disable_recompute,
    )
    return w, u, qg, kg, Aqk, Akk, qg_scaled


@triton.jit(do_not_specialize=['B', 'T', 'NT_TOTAL'])
def chunk_kda_bwd_kernel_intra_npu(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    # host-proven: non-varlen, T % BT == 0, BT % BC == 0, BK == K -> every
    # lane of every tile (rows of the current/past/future BC blocks, columns,
    # dA quadrants) is in-bounds, so all load/store predicate masks and the
    # remaining row/col select predicates are vacuous and disappear (plain
    # unmasked MTE form). The masked body below stays statement-equivalent
    # as the permanent fallback.
    ALIGNED: tl.constexpr,
    # host-proven: ALIGNED & SAFE_GATE & BK<=128 -> the transposed (dkt) diag
    # site derives its dA tiles via tl.trans of the raw square the first diag
    # site already loads and inherits that site's exp_p/exp_n factors.
    # BK>128 keeps the two-site loads: the carried raw+factor tiles overflow
    # the 192KB UB across the future-blocks loop at BK=256.
    DIAG_REUSE: tl.constexpr,
    NT_TOTAL,
):
    NC = tl.cdiv(BT, BC)
    core_id = tl.program_id(0)
    num_core = tl.num_programs(0)
    # widen before multiplying; the task decomposition stays in int64 end to end
    task_num = tl.cast(NT_TOTAL, tl.int64) * NC * B * HV
    BH_TOTAL = B * HV
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id % BH_TOTAL
        rem = task_id // BH_TOTAL
        i_t = rem % NT_TOTAL
        i_i0 = rem // NT_TOTAL
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        if cu_seqlens is not None:
            i_n, i_t = tl.load(chunk_indices + i_t * 2), tl.load(chunk_indices + i_t * 2 + 1)
            # int64 guarantees: cu_seqlens may arrive as int32 and chunk_indices
            # follows its dtype, but the global offsets must stay 64-bit
            bos, eos = tl.cast(tl.load(cu_seqlens + i_n), tl.int64), tl.cast(tl.load(cu_seqlens + i_n + 1), tl.int64)
        else:
            bos, eos = i_b * T, i_b * T + T
        # T is a loop-carried arg (int32); the reassignment must keep its type
        T = tl.cast(eos - bos, tl.int32)

        # rebind pointers per task (ptr-arg += inside the task loop would both
        # accumulate offsets across tasks and break loop-carried typecheck)
        off_h = (bos * H + i_h) * K
        off_hv = bos * HV + i_hv
        q_l = q + off_h
        k_l = k + off_h
        g_l = g + off_hv * K
        beta_l = beta + off_hv
        dAqk_l = dAqk + off_hv * BT
        dAkk_l = dAkk + off_hv * BT
        dq_l = dq + off_hv * K
        dq2_l = dq2 + off_hv * K
        dk_l = dk + off_hv * K
        dk2_l = dk2 + off_hv * K
        dg_l = dg + off_hv * K
        dg2_l = dg2 + off_hv * K
        db_l = db + off_hv

        o_k = tl.arange(0, BK)
        o_i = tl.arange(0, BC)
        if ALIGNED:
            # host-proven (non-varlen, T%BT==0, BT%BC==0, BK==K): every chunk
            # holds exactly NC full sub-chunks, so the per-task cdiv bound and
            # the tail min()s collapse to constexpr values.
            NC_LOC = NC
        else:
            NC_LOC = min(NC, tl.cdiv(T - i_t * BT, BC))
        i_i = i_i0
        if i_i0 < NC_LOC:
            # no-op cast when i_t is int64; guards int32 chunk_indices tables
            i_ti = tl.cast(i_t * BT + i_i * BC, tl.int64)
            if not ALIGNED:
                m_k = o_k < K
                m_row = (i_ti + o_i) < T
                m_ik = m_row[:, None] & m_k[None, :]
            elif not SAFE_GATE:
                # the ALIGNED body is maskless, but the non-safe pairwise diag
                # path always loads masked and still needs these
                m_k = o_k < K
                m_row = (i_ti + o_i) < T
            a_row = i_i * BC + o_i

            # per-task offset chains: shared address arithmetic every
            # pointer site below would otherwise re-materialize
            row_hk = o_i[:, None] * (H * K)
            row_hvk = o_i[:, None] * (HV * K)
            row_dA = o_i[:, None] * (HV * BT)
            col_dAt = o_i[None, :] * (HV * BT)
            off_cur_hk = i_ti * (H * K)
            off_cur_hvk = i_ti * (HV * K)
            off_dA = i_ti * (HV * BT)
            off_beta = i_ti * HV
            base_t = i_t * BT

            p_g_cur = g_l + off_cur_hvk + row_hvk + o_k[None, :]
            p_q_cur = q_l + off_cur_hk + row_hk + o_k[None, :]
            p_k_cur = k_l + off_cur_hk + row_hk + o_k[None, :]
            if ALIGNED:
                b_g = tl.load(p_g_cur).to(tl.float32)
                b_b = tl.load(beta_l + off_beta + o_i * HV)
                b_q = tl.load(p_q_cur)
                b_k = tl.load(p_k_cur)
                b_k_f32 = b_k.to(tl.float32)
            else:
                b_g = tl.load(p_g_cur, mask=m_ik, other=0.0).to(tl.float32)
                b_b = tl.load(beta_l + off_beta + o_i * HV, mask=m_row, other=0.0)
                b_q = tl.load(p_q_cur, mask=m_ik, other=0.0)
                b_k = tl.load(p_k_cur, mask=m_ik, other=0.0)
                b_k_f32 = b_k.to(tl.float32)

            b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
            b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)

            # ---- inter blocks (j < i) ----
            if i_i > 0:
                if ALIGNED:
                    b_gn = tl.load(g_l + off_cur_hvk + o_k).to(tl.float32)[None, :]
                else:
                    b_gn = tl.load(g_l + off_cur_hvk + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]
                for i_j in range(0, i_i):
                    row_j = tl.cast(base_t + i_j * BC, tl.int64)
                    p_kj_l = k_l + row_j * (H * K) + row_hk + o_k[None, :]
                    p_gkj_l = g_l + row_j * (HV * K) + row_hvk + o_k[None, :]
                    if ALIGNED:
                        b_kj = tl.load(p_kj_l)
                        b_gkj = tl.load(p_gkj_l)
                    else:
                        m_rowj = (row_j + o_i) < T
                        m_ikj = m_rowj[:, None] & m_k[None, :]
                        b_kj = tl.load(p_kj_l, mask=m_ikj, other=0.0)
                        b_gkj = tl.load(p_gkj_l, mask=m_ikj, other=0.0)
                    b_kg = b_kj * exp2(b_gn - b_gkj.to(tl.float32))
                    p_dAqk_j = dAqk_l + off_dA + row_dA + (i_j * BC + o_i)[None, :]
                    p_dAkk_j = dAkk_l + off_dA + row_dA + (i_j * BC + o_i)[None, :]
                    if ALIGNED:
                        b_dAqk = tl.load(p_dAqk_j)
                        b_dAkk = tl.load(p_dAkk_j)
                    else:
                        m_ij = m_row[:, None] & ((i_j * BC + o_i)[None, :] < BT)
                        b_dAqk = tl.load(p_dAqk_j, mask=m_ij, other=0.0)
                        b_dAkk = tl.load(p_dAkk_j, mask=m_ij, other=0.0)
                    b_dq2 = tl.dot(b_dAqk.to(tl.float32), b_kg.to(tl.float32), b_dq2, allow_tf32=False)
                    b_dk2 = tl.dot(b_dAkk.to(tl.float32), b_kg.to(tl.float32), b_dk2, allow_tf32=False)
                b_gqn = exp2(b_g - b_gn)
                b_dq2 *= b_gqn
                b_dk2 *= b_gqn

            # ---- diagonal (SAFE_GATE midpoint path) ----
            if SAFE_GATE:
                if ALIGNED:
                    # i_ti + BC <= T host-proven, so the tail min() is BC//2
                    i_gm = i_ti + BC // 2
                    b_gm = tl.load(g_l + i_gm * HV * K + o_k).to(tl.float32)[None, :]
                else:
                    i_gm = i_ti + min(BC // 2, T - i_ti - 1)
                    b_gm = tl.load(g_l + i_gm * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]
                p_dAqk_d = dAqk_l + off_dA + row_dA + a_row[None, :]
                p_dAkk_d = dAkk_l + off_dA + row_dA + a_row[None, :]
                if ALIGNED:
                    # every row/col predicate is vacuous under ALIGNED: the
                    # select mask is the pure triangular and the gate-decay
                    # factors need no zeroing
                    m_i_d = o_i[:, None] >= o_i[None, :]
                    if DIAG_REUSE:
                        # keep the natural-orientation raw square live for the
                        # transposed dkt diag site below: it then derives its
                        # tiles via tl.trans and skips the duplicate b_gm row
                        # load, the b_g_d/exp2 recompute and both stride-HV*BT
                        # transposed gathers
                        b_dAqk_raw = tl.load(p_dAqk_d).to(tl.float32)
                        b_dAkk_raw = tl.load(p_dAkk_d).to(tl.float32)
                        b_dAqk_d = tl.where(m_i_d, b_dAqk_raw, 0.)
                        b_dAkk_d = tl.where(m_i_d, b_dAkk_raw, 0.)
                    else:
                        b_dAqk_d = tl.load(p_dAqk_d).to(tl.float32)
                        b_dAkk_d = tl.load(p_dAkk_d).to(tl.float32)
                        b_dAqk_d = tl.where(m_i_d, b_dAqk_d, 0.)
                        b_dAkk_d = tl.where(m_i_d, b_dAkk_d, 0.)
                    b_g_d = b_g - b_gm
                    exp_p = exp2(b_g_d)
                    exp_n = exp2(-b_g_d)
                else:
                    m_ij_d = m_row[:, None] & ((i_i * BC + o_i)[None, :] < BT)
                    b_dAqk_d = tl.load(p_dAqk_d, mask=m_ij_d, other=0.0).to(tl.float32)
                    b_dAkk_d = tl.load(p_dAkk_d, mask=m_ij_d, other=0.0).to(tl.float32)
                    m_i_d = (o_i[:, None] >= o_i[None, :]) & m_row[:, None] & m_row[None, :]
                    b_dAqk_d = tl.where(m_i_d, b_dAqk_d, 0.)
                    b_dAkk_d = tl.where(m_i_d, b_dAkk_d, 0.)
                    b_g_d = tl.where(m_row[:, None], b_g - b_gm, 0.)
                    exp_p = tl.where(m_row[:, None], exp2(b_g_d), 0.)
                    exp_n = tl.where(m_row[:, None], exp2(-b_g_d), 0.)
                b_k_exp = b_k_f32 * exp_n
                b_dq2 += tl.dot(b_dAqk_d, b_k_exp, allow_tf32=False) * exp_p
                b_dk2 += tl.dot(b_dAkk_d, b_k_exp, allow_tf32=False) * exp_p
            else:
                # pairwise per-column diag for unbounded (non-safe) gates
                o_dA_c = off_dA + o_i * (HV * BT) + i_i * BC
                for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
                    b_dAqk_j = tl.load(dAqk_l + o_dA_c + j, mask=m_row, other=0.0)
                    b_dAkk_j = tl.load(dAkk_l + o_dA_c + j, mask=m_row, other=0.0)
                    b_kj = tl.load(k_l + off_cur_hk + j * (H * K) + o_k, mask=m_k, other=0).to(tl.float32)
                    b_gkj = tl.load(g_l + off_cur_hvk + j * (HV * K) + o_k, mask=m_k, other=0).to(tl.float32)
                    m_i = o_i[:, None] >= j
                    b_gqk = exp2(b_g - b_gkj[None, :])
                    b_dq2 += tl.where(m_i, b_dAqk_j[:, None] * b_kj[None, :] * b_gqk, 0.)
                    b_dk2 += tl.where(m_i, b_dAkk_j[:, None] * b_kj[None, :] * b_gqk, 0.)

            # ---- first-half outputs: dq2/db (past + diag contributions) ----
            p_dq_cur = dq_l + off_cur_hvk + row_hvk + o_k[None, :]
            p_dq2_cur = dq2_l + off_cur_hvk + row_hvk + o_k[None, :]
            p_db_cur = db_l + off_beta + o_i * HV
            # issue the RMW load before the scalar epilogue so its latency is
            # covered and the two stores drain back-to-back
            if ALIGNED:
                b_dq_rm = tl.load(p_dq_cur)
            else:
                b_dq_rm = tl.load(p_dq_cur, mask=m_ik, other=0.0)
            b_db = tl.sum(b_dk2 * b_k_f32, 1)
            b_dk2 = b_dk2 * b_b.to(tl.float32)[:, None]
            b_dg2 = b_q.to(tl.float32) * b_dq2
            if ALIGNED:
                b_dq2 += b_dq_rm.to(tl.float32)
                tl.store(p_dq2_cur, b_dq2.to(dq2.dtype.element_ty))
                tl.store(p_db_cur, b_db.to(tl.float32))
            else:
                b_dq2 += b_dq_rm.to(tl.float32)
                tl.store(p_dq2_cur, b_dq2.to(dq2.dtype.element_ty), mask=m_ik)
                tl.store(p_db_cur, b_db.to(tl.float32), mask=m_row)

            # ---- future blocks (j > i) and diag-kk: dkt contribution ----
            b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
            if i_i < NC_LOC - 1:
                if ALIGNED:
                    # min(i_ti + BC, T) == i_ti + BC host-proven
                    b_gn_f = tl.load(g_l + (i_ti + BC - 1) * HV * K + o_k).to(tl.float32)[None, :]
                else:
                    b_gn_f = tl.load(g_l + (min(i_ti + BC, T) - 1) * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]
                for i_j in range(i_i + 1, NC_LOC):
                    row_j = tl.cast(base_t + i_j * BC, tl.int64)
                    p_qf_l = q_l + row_j * (H * K) + row_hk + o_k[None, :]
                    p_kf_l = k_l + row_j * (H * K) + row_hk + o_k[None, :]
                    p_gkf_l = g_l + row_j * (HV * K) + row_hvk + o_k[None, :]
                    if ALIGNED:
                        b_qf = tl.load(p_qf_l)
                        b_kf = tl.load(p_kf_l)
                        b_gkf = tl.load(p_gkf_l).to(tl.float32)
                        b_bf = tl.load(beta_l + row_j * HV + o_i * HV)
                    else:
                        m_rowj = (row_j + o_i) < T
                        m_ikj = m_rowj[:, None] & m_k[None, :]
                        b_qf = tl.load(p_qf_l, mask=m_ikj, other=0.0)
                        b_kf = tl.load(p_kf_l, mask=m_ikj, other=0.0)
                        b_gkf = tl.load(p_gkf_l, mask=m_ikj, other=0.0).to(tl.float32)
                        b_bf = tl.load(beta_l + row_j * HV + o_i * HV, mask=m_rowj, other=0.0)
                    # transposed dA tiles: element (a, b) at dA + a + b*(HV*BT)
                    p_dAqk_f = dAqk_l + row_j * (HV * BT) + a_row[:, None] + col_dAt
                    p_dAkk_f = dAkk_l + row_j * (HV * BT) + a_row[:, None] + col_dAt
                    if ALIGNED:
                        b_dAqk_f = tl.load(p_dAqk_f)
                        b_dAkk_f = tl.load(p_dAkk_f)
                        # m_rowj provably all-true: the zeroing selects on the
                        # shared gate-decay factor disappear
                        b_gkn = exp2(b_gkf - b_gn_f)
                        b_qg = b_qf * b_gkn
                        b_kbg = b_kf * b_bf.to(tl.float32)[:, None] * b_gkn
                    else:
                        m_t = ((row_j + o_i)[None, :] < T) & (a_row[:, None] < BT)
                        b_dAqk_f = tl.load(p_dAqk_f, mask=m_t, other=0.0)
                        b_dAkk_f = tl.load(p_dAkk_f, mask=m_t, other=0.0)
                        b_gkn = exp2(b_gkf - b_gn_f)
                        b_qg = b_qf * tl.where(m_rowj[:, None], b_gkn, 0)
                        b_kbg = b_kf * b_bf.to(tl.float32)[:, None] * tl.where(m_rowj[:, None], b_gkn, 0)
                    b_dkt = tl.dot(b_dAqk_f.to(tl.float32), b_qg.to(tl.float32), b_dkt, allow_tf32=False)
                    b_dkt = tl.dot(b_dAkk_f.to(tl.float32), b_kbg.to(tl.float32), b_dkt, allow_tf32=False)
                b_dkt *= exp2(b_gn_f - b_g)

            if SAFE_GATE:
                if ALIGNED and not DIAG_REUSE:
                    i_gm = i_ti + BC // 2
                    b_gm = tl.load(g_l + i_gm * HV * K + o_k).to(tl.float32)[None, :]
                elif not ALIGNED:
                    i_gm = i_ti + min(BC // 2, T - i_ti - 1)
                    b_gm = tl.load(g_l + i_gm * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]
                if ALIGNED:
                    # vacuous row predicates dropped (ALIGNED): pure triangular
                    # mask + un-zeroed decay factors
                    m_i_t = o_i[:, None] <= o_i[None, :]
                    if DIAG_REUSE:
                        # tl.trans(raw)[r, c] reads dA[i_ti + c, i_i*BC + r]:
                        # the exact element the legacy stride-HV*BT gather
                        # fetches (same addresses, same bits); exp_p/exp_n are
                        # the identical site-1 midpoint factors (b_g/b_gm are
                        # untouched between the two diag sites)
                        b_dAqk_t = tl.where(m_i_t, tl.trans(b_dAqk_raw), 0.)
                        b_dAkk_t = tl.where(m_i_t, tl.trans(b_dAkk_raw), 0.)
                    else:
                        p_dAqk_t = dAqk_l + off_dA + a_row[:, None] + col_dAt
                        p_dAkk_t = dAkk_l + off_dA + a_row[:, None] + col_dAt
                        b_dAqk_t = tl.load(p_dAqk_t).to(tl.float32)
                        b_dAkk_t = tl.load(p_dAkk_t).to(tl.float32)
                        b_dAqk_t = tl.where(m_i_t, b_dAqk_t, 0.)
                        b_dAkk_t = tl.where(m_i_t, b_dAkk_t, 0.)
                        b_g_d = b_g - b_gm
                        exp_p = exp2(b_g_d)
                        exp_n = exp2(-b_g_d)
                else:
                    p_dAqk_t = dAqk_l + off_dA + a_row[:, None] + col_dAt
                    p_dAkk_t = dAkk_l + off_dA + a_row[:, None] + col_dAt
                    m_t = (i_ti + o_i)[None, :] < T
                    b_dAqk_t = tl.load(p_dAqk_t, mask=m_t, other=0.0).to(tl.float32)
                    b_dAkk_t = tl.load(p_dAkk_t, mask=m_t, other=0.0).to(tl.float32)
                    m_i_t = (o_i[:, None] <= o_i[None, :]) & m_row[:, None] & m_row[None, :]
                    b_dAqk_t = tl.where(m_i_t, b_dAqk_t, 0.)
                    b_dAkk_t = tl.where(m_i_t, b_dAkk_t, 0.)
                    b_g_d = tl.where(m_row[:, None], b_g - b_gm, 0.)
                    exp_p = tl.where(m_row[:, None], exp2(b_g_d), 0.)
                    exp_n = tl.where(m_row[:, None], exp2(-b_g_d), 0.)
                b_q_exp = b_q.to(tl.float32) * exp_p
                b_kb_exp = b_k.to(tl.float32) * b_b.to(tl.float32)[:, None] * exp_p
                b_dkt += tl.dot(b_dAqk_t, b_q_exp, allow_tf32=False) * exp_n
                b_dkt += tl.dot(b_dAkk_t, b_kb_exp, allow_tf32=False) * exp_n
            else:
                # pairwise per-column diag (future side) for unbounded gates
                for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
                    b_dAqk_j = tl.load(dAqk_l + off_dA + j * (HV * BT) + i_i * BC + o_i)
                    b_dAkk_j = tl.load(dAkk_l + off_dA + j * (HV * BT) + i_i * BC + o_i)
                    b_qj = tl.load(q_l + off_cur_hk + j * (H * K) + o_k, mask=m_k, other=0).to(tl.float32)
                    b_kbj = tl.load(k_l + off_cur_hk + j * (H * K) + o_k, mask=m_k,
                                    other=0).to(tl.float32) * tl.load(beta_l + (i_ti + j) * HV)
                    b_gkj = tl.load(g_l + off_cur_hvk + j * (HV * K) + o_k, mask=m_k, other=0).to(tl.float32)
                    m_i = o_i[:, None] <= j
                    b_gkq = exp2(b_gkj[None, :] - b_g)
                    b_dkt += tl.where(m_i, b_dAqk_j[:, None] * b_qj[None, :] * b_gkq, 0.)
                    b_dkt += tl.where(m_i, b_dAkk_j[:, None] * b_kbj[None, :] * b_gkq, 0.)

            # ---- second-half outputs: dk2/dg2 (adds future/dkt contributions) ----
            p_dg_cur = dg_l + off_cur_hvk + row_hvk + o_k[None, :]
            p_dk_cur = dk_l + off_cur_hvk + row_hvk + o_k[None, :]
            p_dk2_cur = dk2_l + off_cur_hvk + row_hvk + o_k[None, :]
            p_dg2_cur = dg2_l + off_cur_hvk + row_hvk + o_k[None, :]
            # both RMW loads issue up front; stores drain back-to-back after
            # the epilogue compute
            if ALIGNED:
                b_dg_rm = tl.load(p_dg_cur)
                b_dk_rm = tl.load(p_dk_cur)
            else:
                b_dg_rm = tl.load(p_dg_cur, mask=m_ik, other=0.0)
                b_dk_rm = tl.load(p_dk_cur, mask=m_ik, other=0.0)
            b_dg2 += (b_dk2 - b_dkt) * b_k.to(tl.float32) + b_dg_rm
            b_dk2 += b_dk_rm.to(tl.float32)
            b_dk2 += b_dkt
            if ALIGNED:
                tl.store(p_dk2_cur, b_dk2.to(dk2.dtype.element_ty))
                tl.store(p_dg2_cur, b_dg2)
            else:
                tl.store(p_dk2_cur, b_dk2.to(dk2.dtype.element_ty), mask=m_ik)
                tl.store(p_dg2_cur, b_dg2, mask=m_ik)


# NC2 per-chunk merged bwd_intra. Flipped off permanently if the merged
# kernel ever fails to compile/launch; the two-task path stays as fallback.
_BWD_INTRA_NC2_OK = True


@triton.jit(do_not_specialize=['B', 'T', 'NT_TOTAL'])
def chunk_kda_bwd_kernel_intra_nc2_npu(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
    db,
    B,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NT_TOTAL,
):
    """NC==2 per-chunk merge of chunk_kda_bwd_kernel_intra_npu.

    Host-proven (the exact ALIGNED+SAFE_GATE+DIAG_REUSE specialization of the
    proven kernel, plus BT==2*BC): non-varlen, T % BT == 0, BK == K <= 128,
    so every lane of every tile is in-bounds, NC is statically 2 and the
    pairwise per-column diag paths never fire. One task owns a whole chunk
    and runs sub-chunk 0 then sub-chunk 1 statement-for-statement like the
    two original (i_bh, i_i, i_t) tasks; every load address, mask, exp2
    factor, dot operand and accumulator sequence is replicated verbatim, so
    the stored dq2/dk2/dg2/db bits are identical.

    What the merge buys: the second task re-read exactly what the first task
    already held - both [BC,BK] fp32 g halves (the inter b_gkj / own b_g
    tiles), the k/q bf16 halves, the beta rows and the cross-half [BC,BC]
    dA block (task 0 gathered its 1024 elements transposed for the future
    site, task 1 natural for the inter site; here it is loaded once natural
    and tl.trans'd) - ~4x[BC,BK] fp32-equivalents of duplicate GM reads
    saved per chunk, and the task queue halves. Both runtime-bound block
    loops (for i_j in range(0, i_i) / range(i_i+1, NC_LOC)) resolve to
    straight-line single-block code. UB live set matches the original
    per-task peak at every program point: phase 1 reuses tiles phase 0
    leaves live instead of adding new simultaneous ones (<=~150KB
    envelope, no new carries beyond the boundary, where the phase
    accumulators are already stored).
    """
    core_id = tl.program_id(0)
    num_core = tl.num_programs(0)
    # widen before multiplying; the task decomposition stays int64 end to end
    task_num = tl.cast(NT_TOTAL, tl.int64) * B * HV
    BH_TOTAL = B * HV
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id % BH_TOTAL
        i_t = task_id // BH_TOTAL
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        bos = i_b * T
        # rebind pointers per task (ptr-arg += inside the task loop would both
        # accumulate offsets across tasks and break loop-carried typecheck)
        off_h = (bos * H + i_h) * K
        off_hv = bos * HV + i_hv
        q_l = q + off_h
        k_l = k + off_h
        g_l = g + off_hv * K
        beta_l = beta + off_hv
        dAqk_l = dAqk + off_hv * BT
        dAkk_l = dAkk + off_hv * BT
        dq_l = dq + off_hv * K
        dq2_l = dq2 + off_hv * K
        dk_l = dk + off_hv * K
        dk2_l = dk2 + off_hv * K
        dg_l = dg + off_hv * K
        dg2_l = dg2 + off_hv * K
        db_l = db + off_hv

        o_k = tl.arange(0, BK)
        o_i = tl.arange(0, BC)
        base_t = i_t * BT
        cs = tl.cast(base_t, tl.int64)
        row_hk = o_i[:, None] * (H * K)
        row_hvk = o_i[:, None] * (HV * K)
        row_dA = o_i[:, None] * (HV * BT)
        off_hk0 = cs * (H * K)
        off_hk1 = (cs + BC) * (H * K)
        off_hvk0 = cs * (HV * K)
        off_hvk1 = (cs + BC) * (HV * K)
        off_dA0 = cs * (HV * BT)
        off_dA1 = (cs + BC) * (HV * BT)
        off_beta0 = cs * HV
        off_beta1 = (cs + BC) * HV
        # sub-chunk 1's diag dA columns start at BC
        a_row1 = BC + o_i
        m_i_d = o_i[:, None] >= o_i[None, :]
        m_i_t = o_i[:, None] <= o_i[None, :]

        # ================= sub-chunk 0 (i_i == 0, i_ti == cs) =================
        b_g0 = tl.load(g_l + off_hvk0 + row_hvk + o_k[None, :]).to(tl.float32)
        b_b0 = tl.load(beta_l + off_beta0 + o_i * HV)
        b_q0 = tl.load(q_l + off_hk0 + row_hk + o_k[None, :])
        b_k0 = tl.load(k_l + off_hk0 + row_hk + o_k[None, :])
        b_k0_f32 = b_k0.to(tl.float32)

        b_dq2_0 = tl.zeros([BC, BK], dtype=tl.float32)
        b_dk2_0 = tl.zeros([BC, BK], dtype=tl.float32)

        # ---- diagonal site 1 (SAFE_GATE midpoint; raw squares stay live for
        # ---- the transposed site 2 below, DIAG_REUSE idiom) ----
        b_gm0 = tl.load(g_l + (cs + BC // 2) * HV * K + o_k).to(tl.float32)[None, :]
        b_dAqk_raw0 = tl.load(dAqk_l + off_dA0 + row_dA + o_i[None, :]).to(tl.float32)
        b_dAkk_raw0 = tl.load(dAkk_l + off_dA0 + row_dA + o_i[None, :]).to(tl.float32)
        b_dAqk_d0 = tl.where(m_i_d, b_dAqk_raw0, 0.)
        b_dAkk_d0 = tl.where(m_i_d, b_dAkk_raw0, 0.)
        b_g_d0 = b_g0 - b_gm0
        exp_p0 = exp2(b_g_d0)
        exp_n0 = exp2(-b_g_d0)
        b_k_exp0 = b_k0_f32 * exp_n0
        b_dq2_0 += tl.dot(b_dAqk_d0, b_k_exp0, allow_tf32=False) * exp_p0
        b_dk2_0 += tl.dot(b_dAkk_d0, b_k_exp0, allow_tf32=False) * exp_p0

        # ---- first-half outputs: dq2/db (RMW load issued before the epilogue
        # ---- so its latency is covered) ----
        b_dq_rm0 = tl.load(dq_l + off_hvk0 + row_hvk + o_k[None, :])
        b_db0 = tl.sum(b_dk2_0 * b_k0_f32, 1)
        b_dk2_0 = b_dk2_0 * b_b0.to(tl.float32)[:, None]
        b_dg2_0 = b_q0.to(tl.float32) * b_dq2_0
        b_dq2_0 += b_dq_rm0.to(tl.float32)
        tl.store(dq2_l + off_hvk0 + row_hvk + o_k[None, :], b_dq2_0.to(dq2.dtype.element_ty))
        tl.store(db_l + off_beta0 + o_i * HV, b_db0.to(tl.float32))

        # ---- future block {1} (was: for i_j in range(1, NC_LOC)) ----
        b_gn_f = tl.load(g_l + (cs + BC - 1) * HV * K + o_k).to(tl.float32)[None, :]
        b_q1 = tl.load(q_l + off_hk1 + row_hk + o_k[None, :])
        b_k1 = tl.load(k_l + off_hk1 + row_hk + o_k[None, :])
        b_g1 = tl.load(g_l + off_hvk1 + row_hvk + o_k[None, :]).to(tl.float32)
        b_b1 = tl.load(beta_l + off_beta1 + o_i * HV)
        # cross-half dA block rows [cs+BC, cs+BT) x cols [0, BC), natural: the
        # original task 0 fetched the identical 1024 elements (same addresses,
        # same bits) through the stride-HV*BT transposed gather
        b_dAqk_s = tl.load(dAqk_l + off_dA1 + row_dA + o_i[None, :])
        b_dAkk_s = tl.load(dAkk_l + off_dA1 + row_dA + o_i[None, :])
        b_dAqk_f = tl.trans(b_dAqk_s)
        b_dAkk_f = tl.trans(b_dAkk_s)
        b_gkn = exp2(b_g1 - b_gn_f)
        b_qg = b_q1 * b_gkn
        b_kbg = b_k1 * b_b1.to(tl.float32)[:, None] * b_gkn
        b_dkt_0 = tl.zeros([BC, BK], dtype=tl.float32)
        b_dkt_0 = tl.dot(b_dAqk_f.to(tl.float32), b_qg.to(tl.float32), b_dkt_0, allow_tf32=False)
        b_dkt_0 = tl.dot(b_dAkk_f.to(tl.float32), b_kbg.to(tl.float32), b_dkt_0, allow_tf32=False)
        b_dkt_0 *= exp2(b_gn_f - b_g0)

        # ---- diagonal site 2 (kk transposed; reuses the raw squares and the
        # ---- site-1 midpoint factors) ----
        b_dAqk_t0 = tl.where(m_i_t, tl.trans(b_dAqk_raw0), 0.)
        b_dAkk_t0 = tl.where(m_i_t, tl.trans(b_dAkk_raw0), 0.)
        b_q_exp0 = b_q0.to(tl.float32) * exp_p0
        b_kb_exp0 = b_k0.to(tl.float32) * b_b0.to(tl.float32)[:, None] * exp_p0
        b_dkt_0 += tl.dot(b_dAqk_t0, b_q_exp0, allow_tf32=False) * exp_n0
        b_dkt_0 += tl.dot(b_dAkk_t0, b_kb_exp0, allow_tf32=False) * exp_n0

        # ---- second-half outputs: dk2/dg2 (RMW loads up front, stores drain
        # ---- back-to-back after the epilogue compute) ----
        b_dg_rm0 = tl.load(dg_l + off_hvk0 + row_hvk + o_k[None, :])
        b_dk_rm0 = tl.load(dk_l + off_hvk0 + row_hvk + o_k[None, :])
        b_dg2_0 += (b_dk2_0 - b_dkt_0) * b_k0_f32 + b_dg_rm0
        b_dk2_0 += b_dk_rm0.to(tl.float32)
        b_dk2_0 += b_dkt_0
        tl.store(dk2_l + off_hvk0 + row_hvk + o_k[None, :], b_dk2_0.to(dk2.dtype.element_ty))
        tl.store(dg2_l + off_hvk0 + row_hvk + o_k[None, :], b_dg2_0)

        # ================= sub-chunk 1 (i_i == 1, i_ti == cs+BC) =================
        # own tiles are the ones sub-chunk 0's future site already loaded
        b_g = b_g1
        b_b = b_b1
        b_q = b_q1
        b_k = b_k1
        b_k_f32 = b_k1.to(tl.float32)

        b_dq2_1 = tl.zeros([BC, BK], dtype=tl.float32)
        b_dk2_1 = tl.zeros([BC, BK], dtype=tl.float32)

        # ---- inter block {0} (was: for i_j in range(0, 1)) ----
        b_gn = tl.load(g_l + off_hvk1 + o_k).to(tl.float32)[None, :]
        # b_kj == sub-chunk 0's own k half; b_gkj == its own g half
        b_kg = b_k0 * exp2(b_gn - b_g0.to(tl.float32))
        # b_dAqk_s/b_dAkk_s in natural orientation are exactly the tiles the
        # original task 1's inter site loaded
        b_dq2_1 = tl.dot(b_dAqk_s.to(tl.float32), b_kg.to(tl.float32), b_dq2_1, allow_tf32=False)
        b_dk2_1 = tl.dot(b_dAkk_s.to(tl.float32), b_kg.to(tl.float32), b_dk2_1, allow_tf32=False)
        b_gqn = exp2(b_g - b_gn)
        b_dq2_1 *= b_gqn
        b_dk2_1 *= b_gqn

        # ---- diagonal site 1 ----
        b_gm1 = tl.load(g_l + (cs + BC + BC // 2) * HV * K + o_k).to(tl.float32)[None, :]
        b_dAqk_raw1 = tl.load(dAqk_l + off_dA1 + row_dA + a_row1[None, :]).to(tl.float32)
        b_dAkk_raw1 = tl.load(dAkk_l + off_dA1 + row_dA + a_row1[None, :]).to(tl.float32)
        b_dAqk_d1 = tl.where(m_i_d, b_dAqk_raw1, 0.)
        b_dAkk_d1 = tl.where(m_i_d, b_dAkk_raw1, 0.)
        b_g_d1 = b_g - b_gm1
        exp_p1 = exp2(b_g_d1)
        exp_n1 = exp2(-b_g_d1)
        b_k_exp1 = b_k_f32 * exp_n1
        b_dq2_1 += tl.dot(b_dAqk_d1, b_k_exp1, allow_tf32=False) * exp_p1
        b_dk2_1 += tl.dot(b_dAkk_d1, b_k_exp1, allow_tf32=False) * exp_p1

        # ---- first-half outputs: dq2/db ----
        b_dq_rm1 = tl.load(dq_l + off_hvk1 + row_hvk + o_k[None, :])
        b_db1 = tl.sum(b_dk2_1 * b_k_f32, 1)
        b_dk2_1 = b_dk2_1 * b_b.to(tl.float32)[:, None]
        b_dg2_1 = b_q.to(tl.float32) * b_dq2_1
        b_dq2_1 += b_dq_rm1.to(tl.float32)
        tl.store(dq2_l + off_hvk1 + row_hvk + o_k[None, :], b_dq2_1.to(dq2.dtype.element_ty))
        tl.store(db_l + off_beta1 + o_i * HV, b_db1.to(tl.float32))

        # ---- future blocks: none (i_i == NC-1); diagonal site 2 only ----
        b_dAqk_t1 = tl.where(m_i_t, tl.trans(b_dAqk_raw1), 0.)
        b_dAkk_t1 = tl.where(m_i_t, tl.trans(b_dAkk_raw1), 0.)
        b_q_exp1 = b_q.to(tl.float32) * exp_p1
        b_kb_exp1 = b_k.to(tl.float32) * b_b.to(tl.float32)[:, None] * exp_p1
        b_dkt_1 = tl.zeros([BC, BK], dtype=tl.float32)
        b_dkt_1 += tl.dot(b_dAqk_t1, b_q_exp1, allow_tf32=False) * exp_n1
        b_dkt_1 += tl.dot(b_dAkk_t1, b_kb_exp1, allow_tf32=False) * exp_n1

        # ---- second-half outputs: dk2/dg2 ----
        b_dg_rm1 = tl.load(dg_l + off_hvk1 + row_hvk + o_k[None, :])
        b_dk_rm1 = tl.load(dk_l + off_hvk1 + row_hvk + o_k[None, :])
        b_dg2_1 += (b_dk2_1 - b_dkt_1) * b_k_f32 + b_dg_rm1
        b_dk2_1 += b_dk_rm1.to(tl.float32)
        b_dk2_1 += b_dkt_1
        tl.store(dk2_l + off_hvk1 + row_hvk + o_k[None, :], b_dk2_1.to(dk2.dtype.element_ty))
        tl.store(dg2_l + off_hvk1 + row_hvk + o_k[None, :], b_dg2_1)


@input_guard
def chunk_kda_bwd_intra_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    use_graph: bool = False,
):
    if use_graph:
        raise NotImplementedError("use_graph is not supported on the Ascend NPU backend")
    B, T, H, K, HV = *k.shape, g.shape[2]
    BT = chunk_size
    BK = triton.next_power_of_2(K)
    if (safe_gate and BK > 512) or (not safe_gate and BK > 256):
        # Outside the validated single-block UB envelope. The dispatch verifier
        # routes these calls to the mainline kernel already; this covers direct
        # callers of the backend function.
        from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra as _mainline_bwd_intra
        # __wrapped__ skips re-dispatch back into this backend; bare fn when dispatch is disabled
        return getattr(_mainline_bwd_intra, '__wrapped__', _mainline_bwd_intra)(
            q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk, dq=dq, dk=dk, db=db, dg=dg,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=chunk_size, safe_gate=safe_gate)
    # The 192KB UB budget caps how many BCxBK fp32 tiles stay live, so BC must
    # follow the padded BK (non-pow2 K like 192 lands in the BK=256 tier).
    # SAFE_GATE dot path uses 32x32 cube tiles; the pairwise per-column diag
    # compiles only at BC<=16, and only at BC=8 once BK reaches 256.
    if safe_gate:
        BC = min(32, BT) if BK <= 256 else min(16, BT)
    else:
        BC = min(16, BT) if BK <= 128 else min(8, BT)
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    db2 = beta.new_empty(1, *beta.shape, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)
    num_core = num_aicore()
    # Full-tile fast path host gate: non-varlen with T%BT==0, BT%BC==0 and no
    # K padding (BK==K) makes every task's tiles provably in-bounds AND every
    # row/col predicate provably all-true (BT%BC==0 keeps a_row < BT on the
    # transposed dA tiles; non-pow2 chunk sizes like 48 fall back to masked),
    # so the kernel drops all load/store predicate masks and the vacuous
    # select/mask work around the dots (plain unmasked MTE ops).
    aligned = cu_seqlens is None and T % BT == 0 and BK == K and BT % BC == 0
    # Diag-tile reuse: only inside the ALIGNED compile, SAFE_GATE dot path
    # and the measured BK<=128 UB envelope - the carried raw+factor tiles
    # overflow the 192KB UB across the future-blocks loop at BK=256, which
    # keeps the two-site loads.
    diag_reuse = aligned and safe_gate and BK <= 128
    # NC2 per-chunk merged fast path: when the host proves the exact shape
    # the proven ALIGNED+SAFE_GATE+DIAG_REUSE specialization covers (diag_reuse
    # already implies aligned+safe_gate+BK<=128, aligned implies non-varlen +
    # T%BT==0 + BK==K) and the chunk is exactly two sub-chunks (BT==2*BC), one
    # persistent task per (chunk, bh) runs both sub-chunks with the second's
    # loads shared with the first - bitwise-identical outputs, fewer GM reads,
    # half the tasks, no runtime-bound block loops. Permanent fallback to
    # the proven two-task kernel on any compile/launch failure.
    global _BWD_INTRA_NC2_OK
    if _BWD_INTRA_NC2_OK and diag_reuse and cu_seqlens is None and BT == 2 * BC:
        try:
            fast_launch(
                chunk_kda_bwd_kernel_intra_nc2_npu, (num_core,),
                NT_TOTAL=NT,
                q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk,
                dq=dq, dq2=dq2, dk=dk, dk2=dk2, dg=dg, dg2=dg2, db=db2,
                B=B, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=BK,
            )
            return dq2, dk2, db2.sum(0).add_(db), dg2
        except Exception:
            _BWD_INTRA_NC2_OK = False
            logger.warning(
                "chunk_kda_bwd_intra_npu: NC2 merged launch failed; "
                "permanent fallback to the sub-chunk kernel",
                exc_info=True,
            )
    kernel_kwargs = dict(
        NT_TOTAL=NT,
        q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk,
        dq=dq, dq2=dq2, dk=dk, dk2=dk2, dg=dg, dg2=dg2, db=db2,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        B=B, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=BK,
        SAFE_GATE=safe_gate,
        DIAG_REUSE=diag_reuse,
    )
    try:
        fast_launch(chunk_kda_bwd_kernel_intra_npu, (num_core,), ALIGNED=aligned, **kernel_kwargs)
    except Exception:
        # Permanent fallback to the proven fully-masked path if the ALIGNED
        # specialization fails to compile/launch on this stack.
        logger.warning(
            "chunk_kda_bwd_intra_npu: ALIGNED=%s launch failed; retrying masked",
            aligned, exc_info=True,
        )
        kernel_kwargs['DIAG_REUSE'] = False
        fast_launch(chunk_kda_bwd_kernel_intra_npu, (num_core,), ALIGNED=False, **kernel_kwargs)
    return dq2, dk2, db2.sum(0).add_(db), dg2
