# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Single-kernel fused KDA forward stage0-4 for small aligned inputs (Ascend NPU).

Ports the FlagGems ``chunk_gated_delta_rule_fwd_stage0_4_kernel`` design to KDA
math: one launch per (chunk, v-head) computes gate activation -> chunk-local
cumsum -> intra A blocks (Aqk/Akkd, safe-gate midpoint form) -> diagonal
forward substitution -> off-diagonal merge -> recompute w/u (+ qg/kg),
replacing the 5-kernel pipeline (cumsum, sub_chunk, diag_solve, inter_solve,
recompute_w_u) plus the aten zeros/transposes in between.

Differences vs the standard pipeline that are deliberate:
- The chunk-local cumsum is evaluated per BC-row sub-block with a running
  carry (fp32 throughout). For the second sub-block this reassociates the
  fp32 summation order; accumulation stays fp32 and deviations are at fp32
  rounding level.
- Akkd is never materialized in GM: the diagonal solve reads the raw block
  from registers via masked row extraction.
- Aqk/Akk strictly-upper blocks are explicitly stored as zeros so the
  outputs stay bit-compatible with the zero-initialized pipeline tensors.

The triton-ascend backend is specialized: every ``tl.dot`` keeps the operand
shapes already used by ``chunk_kda_fwd_kernel_intra_sub_chunk_npu`` /
``chunk_kda_fwd_kernel_inter_solve_fused_npu`` (fp32 gated operands for the A
stage, bf16 casts for the w/u stage), and values consumed by several dots get
fresh ``.to()`` copies to survive the Ascend lhs-clobber behavior.

Dispatch accepts only the production small-T shapes (T in (2048, 4096),
non-varlen, chunk_size 64, K/V <= 128); other shapes keep the staged
pipeline. Grid policy: at or below ``_PERSISTENT_TASK_THRESHOLD`` tasks the
launch is one program per task (``(NT, B*HV)`` grid) — the T==2048
production shape (512 programs) measures 1.587x fwd vs the staged pipeline.
Above the threshold the same single kernel goes persistent (the
chunk_intra/wy_fast idiom): ``min(num_aicore, task_num)`` programs stride
over all (chunk, v-head) tasks with ``tl.range(core_id, task_num,
num_core)``. At T==4096 the one-task-per-program grid would be 1024
programs — ~43 serialized waves on the 24 cube cores, each wave paying the
full program dispatch + prologue, which measured 0.908x fwd — so the
persistent grid is used there instead. Every per-task GM base is rebound
inside the body (never ``+=`` on the arg pointers) and nothing is carried
across tasks. The task body itself is byte-identical in both modes, so
outputs are unchanged; only the task->program mapping differs.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp, exp2
from fla.ops.utils.softplus import softplus
from fla.utils.launch_fast import fast_launch
from fla.utils import ascend_compile_kwargs, input_guard
from fla.utils.ascend_ub_manager import get_npu_properties

logger = logging.getLogger(__name__)

_NUM_WARPS = 2
# BT=64/BC=32 fixed: NC=2 sub-blocks per chunk (the only dispatch-gated shape).
_BC = 32
# Above this many (chunk, v-head) tasks the launch switches from the
# one-task-per-program (NT, B*HV) grid to the persistent grid (see module
# docstring). 512 is exactly the T==2048 / B=1 / HV=16 task count (NT=32 *
# B*HV=16), so that shape keeps its measured-faster one-task-per-program
# launch.
_PERSISTENT_TASK_THRESHOLD = 512


@triton.jit
def _chunk_kda_fwd_stage0_4_body(
    i_t,
    i_bh,
    q,
    k,
    v,
    g,
    A_log,
    dt_bias,
    beta,
    gc,
    w,
    u,
    qg,
    kg,
    Aqk,
    Akk,
    scale,
    gscale,
    lower_bound,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    GATE_ACTIVATE: tl.constexpr,
    STORE_QG: tl.constexpr,
):
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    bos = tl.cast(i_b, tl.int64) * T
    i_t0 = tl.cast(i_t * BT, tl.int64)
    i_t1 = i_t0 + BC
    # int32 mirrors of the row bases for mask / runtime loop-bound arithmetic
    # (kept separate so the `min(BC, T - i_t0i)` loop bound stays int32,
    # exactly like the staged diag_solve kernel)
    i_t0i = i_t * BT
    i_t1i = i_t0i + BC

    q_l = q + (bos * H + i_h) * K
    k_l = k + (bos * H + i_h) * K
    v_l = v + (bos * HV + i_hv) * V
    g_l = g + (bos * HV + i_hv) * K
    beta_l = beta + (bos * HV + i_hv)
    gc_l = gc + (bos * HV + i_hv) * K
    w_l = w + (bos * HV + i_hv) * K
    u_l = u + (bos * HV + i_hv) * V
    kg_l = kg + (bos * HV + i_hv) * K
    Aqk_l = Aqk + (bos * HV + i_hv) * BT
    Akk_l = Akk + (bos * HV + i_hv) * BT
    if STORE_QG:
        qg_l = qg + (bos * HV + i_hv) * K

    o_c = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_k2 = m_k[None, :]
    m_v2 = m_v[None, :]
    m_c0 = (i_t0i + o_c) < T
    m_c1 = (i_t1i + o_c) < T

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_A = tl.load(A_log + i_hv).to(tl.float32) if HAS_A else 1.0

    # ---------------- block 0: gate activation + chunk-local cumsum -------- #
    b_gs0 = tl.load(g_l + i_t0 * (HV * K) + o_c[:, None] * (HV * K) + o_k[None, :],
                    mask=m_c0[:, None] & m_k2, other=0.0).to(tl.float32)
    if GATE_ACTIVATE:
        if HAS_BIAS:
            b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=m_k, other=0.0).to(tl.float32)
            b_gs0 += b_bias[None, :]
        if not USE_LOWER_BOUND:
            b_gate0 = -exp(b_A) * softplus(b_gs0)
        else:
            b_gate0 = lower_bound * tl.sigmoid((exp(b_A) if HAS_A else b_A) * b_gs0)
    else:
        b_gate0 = b_gs0
    b_gc0 = tl.cumsum(b_gate0, axis=0)
    b_gc0 *= gscale
    # gc at the block boundary (first row of block 1)
    b_carry = tl.sum(b_gate0, 0) * gscale
    tl.store(gc_l + i_t0 * (HV * K) + o_c[:, None] * (HV * K) + o_k[None, :],
             b_gc0, mask=m_c0[:, None] & m_k2)

    # ---------------- block 0: diagonal Aqk/Akkd (safe-gate midpoint) ------- #
    b_q0 = tl.load(q_l + i_t0 * (H * K) + o_c[:, None] * (H * K) + o_k[None, :],
                   mask=m_c0[:, None] & m_k2, other=0.0)
    b_k0 = tl.load(k_l + i_t0 * (H * K) + o_c[:, None] * (H * K) + o_k[None, :],
                   mask=m_c0[:, None] & m_k2, other=0.0)
    b_beta0r = tl.load(beta_l + i_t0 * HV + o_c * HV, mask=m_c0, other=0.0)
    b_beta0 = b_beta0r.to(tl.float32)

    i_mid0 = tl.minimum(BC // 2, T - i_t0i - 1)
    b_gm0 = tl.sum(tl.where((o_c == i_mid0)[:, None], b_gc0, 0.0), 0)
    b_gm0d = b_gc0 - b_gm0[None, :]
    b_gq0 = tl.where(m_c0[:, None], exp2(b_gm0d), 0.)
    b_gk0f = tl.where(m_c0[:, None], exp2(-b_gm0d), 0.)
    b_kgt0 = tl.trans(b_k0 * b_gk0f)
    b_Aqk00 = tl.dot(b_q0 * b_gq0, b_kgt0, allow_tf32=False) * scale
    b_Akkd0 = tl.dot(b_k0 * b_gq0, b_kgt0, allow_tf32=False) * b_beta0[:, None]
    b_Aqk00 = tl.where(m_Aqk, b_Aqk00, 0.0)
    b_Akkd0 = tl.where(m_Akk, b_Akkd0, 0.0)
    tl.store(Aqk_l + i_t0 * (HV * BT) + o_c[:, None] * (HV * BT) + o_c[None, :],
             b_Aqk00.to(Aqk.dtype.element_ty), mask=m_c0[:, None])
    # strictly-upper block must read as zeros (pipeline tensors were zero-init)
    tl.store(Aqk_l + i_t0 * (HV * BT) + o_c[:, None] * (HV * BT) + BC + o_c[None, :],
             tl.zeros([BC, BC], dtype=Aqk.dtype.element_ty), mask=m_c0[:, None])

    # ---------------- block 0: diagonal forward substitution ---------------- #
    b_Ai0 = -tl.where(m_Akk, b_Akkd0, 0)
    for i in range(2, min(BC, T - i_t0i)):
        b_a = tl.sum(tl.where((o_c == i)[:, None], b_Akkd0, 0.0), 0)
        b_a = -tl.where(o_c < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai0, 0)
        b_Ai0 = tl.where((o_c == i)[:, None], b_a, b_Ai0)
    b_Ai0 += m_I
    tl.store(Akk_l + i_t0 * (HV * BT) + o_c[:, None] * (HV * BT) + o_c[None, :],
             b_Ai0.to(Akk.dtype.element_ty), mask=m_c0[:, None])
    # strictly-upper block must read as zeros (pipeline tensors were zero-init)
    tl.store(Akk_l + i_t0 * (HV * BT) + o_c[:, None] * (HV * BT) + BC + o_c[None, :],
             tl.zeros([BC, BC], dtype=Akk.dtype.element_ty), mask=m_c0[:, None])

    # ---------------- block 0: recompute w/u ------------------------------- #
    b_v0 = tl.load(v_l + i_t0 * (HV * V) + o_c[:, None] * (HV * V) + o_v[None, :],
                   mask=m_c0[:, None] & m_v2, other=0.0)
    b_vb0 = (b_v0 * b_beta0r[:, None]).to(b_v0.dtype)
    b_kb0 = b_k0 * b_beta0r[:, None]
    b_kb0 = b_kb0 * exp2(b_gc0)
    b_kw0 = b_kb0.to(b_k0.dtype)
    b_u0 = tl.dot(b_Ai0.to(Akk.dtype.element_ty), b_vb0, allow_tf32=False)
    tl.store(u_l + i_t0 * (HV * V) + o_c[:, None] * (HV * V) + o_v[None, :],
             b_u0.to(u.dtype.element_ty), mask=m_c0[:, None] & m_v2)
    b_w0 = tl.dot(b_Ai0.to(Akk.dtype.element_ty), b_kw0, allow_tf32=False)
    tl.store(w_l + i_t0 * (HV * K) + o_c[:, None] * (HV * K) + o_k[None, :],
             b_w0.to(w.dtype.element_ty), mask=m_c0[:, None] & m_k2)

    # ---------------- block 1: gate activation + cumsum with carry ---------- #
    b_gs1 = tl.load(g_l + i_t1 * (HV * K) + o_c[:, None] * (HV * K) + o_k[None, :],
                    mask=m_c1[:, None] & m_k2, other=0.0).to(tl.float32)
    if GATE_ACTIVATE:
        if HAS_BIAS:
            b_gs1 += b_bias[None, :]
        if not USE_LOWER_BOUND:
            b_gate1 = -exp(b_A) * softplus(b_gs1)
        else:
            b_gate1 = lower_bound * tl.sigmoid((exp(b_A) if HAS_A else b_A) * b_gs1)
    else:
        b_gate1 = b_gs1
    b_gc1 = tl.cumsum(b_gate1, axis=0) * gscale + b_carry[None, :]
    tl.store(gc_l + i_t1 * (HV * K) + o_c[:, None] * (HV * K) + o_k[None, :],
             b_gc1, mask=m_c1[:, None] & m_k2)
    # gc at the chunk's last row (T % BT == 0 is guaranteed by the dispatch guard)
    b_glast = tl.sum(tl.where((o_c == BC - 1)[:, None], b_gc1, 0.0), 0)

    # ---------------- block 1: diagonal Aqk/Akkd ---------------------------- #
    b_q1 = tl.load(q_l + i_t1 * (H * K) + o_c[:, None] * (H * K) + o_k[None, :],
                   mask=m_c1[:, None] & m_k2, other=0.0)
    b_k1 = tl.load(k_l + i_t1 * (H * K) + o_c[:, None] * (H * K) + o_k[None, :],
                   mask=m_c1[:, None] & m_k2, other=0.0)
    b_beta1r = tl.load(beta_l + i_t1 * HV + o_c * HV, mask=m_c1, other=0.0)
    b_beta1 = b_beta1r.to(tl.float32)

    i_mid1 = tl.minimum(BC // 2, T - i_t1i - 1)
    b_gm1 = tl.sum(tl.where((o_c == i_mid1)[:, None], b_gc1, 0.0), 0)
    b_gm1d = b_gc1 - b_gm1[None, :]
    b_gq1 = tl.where(m_c1[:, None], exp2(b_gm1d), 0.)
    b_gk1f = tl.where(m_c1[:, None], exp2(-b_gm1d), 0.)
    b_kgt1 = tl.trans(b_k1 * b_gk1f)
    b_Aqk11 = tl.dot(b_q1 * b_gq1, b_kgt1, allow_tf32=False) * scale
    b_Akkd1 = tl.dot(b_k1 * b_gq1, b_kgt1, allow_tf32=False) * b_beta1[:, None]
    b_Aqk11 = tl.where(m_Aqk, b_Aqk11, 0.0)
    b_Akkd1 = tl.where(m_Akk, b_Akkd1, 0.0)
    tl.store(Aqk_l + i_t1 * (HV * BT) + o_c[:, None] * (HV * BT) + BC + o_c[None, :],
             b_Aqk11.to(Aqk.dtype.element_ty), mask=m_c1[:, None])

    # ---------------- block 1: diagonal solve ------------------------------- #
    b_Ai1 = -tl.where(m_Akk, b_Akkd1, 0)
    for i in range(2, min(BC, T - i_t1i)):
        b_a = tl.sum(tl.where((o_c == i)[:, None], b_Akkd1, 0.0), 0)
        b_a = -tl.where(o_c < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai1, 0)
        b_Ai1 = tl.where((o_c == i)[:, None], b_a, b_Ai1)
    b_Ai1 += m_I

    # ---------------- inter block: rows 1, cols 0 --------------------------- #
    # reference is gc at the first row of block 1 (inclusive cumsum, exactly
    # the row the staged inter_solve kernel loads); both exponent factors are
    # <= 1 for the monotone decreasing chunk-local cumsum.
    b_gref = tl.sum(tl.where((o_c == 0)[:, None], b_gc1, 0.0), 0)
    b_gqn1 = tl.where(m_c1[:, None], exp2(b_gc1 - b_gref[None, :]), 0)
    b_kgt0i = tl.trans(b_k0 * exp2(b_gref[None, :] - b_gc0))
    b_Aqk10 = tl.dot(b_q1 * b_gqn1, b_kgt0i, allow_tf32=False)
    b_Akk10 = tl.dot(b_k1 * b_gqn1, b_kgt0i, allow_tf32=False) * b_beta1[:, None]
    tl.store(Aqk_l + i_t1 * (HV * BT) + o_c[:, None] * (HV * BT) + o_c[None, :],
             (b_Aqk10 * scale).to(Aqk.dtype.element_ty), mask=m_c1[:, None])

    # Ascend tl.dot may clobber the left operand: snapshot b_Ai1 before it
    # feeds the merge dot (same pattern as the staged inter_solve kernel) and
    # serve every later consumer from the snapshot via fresh `.to()` copies.
    b_Ai1_c = b_Ai1 + 0.0
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai1, b_Akk10, allow_tf32=False),
        b_Ai0,
        allow_tf32=False,
    )
    tl.store(Akk_l + i_t1 * (HV * BT) + o_c[:, None] * (HV * BT) + o_c[None, :],
             b_Ai10.to(Akk.dtype.element_ty), mask=m_c1[:, None])
    tl.store(Akk_l + i_t1 * (HV * BT) + o_c[:, None] * (HV * BT) + BC + o_c[None, :],
             b_Ai1_c.to(Akk.dtype.element_ty), mask=m_c1[:, None])

    # ---------------- kg / qg ------------------------------------------------ #
    b_kg0 = b_k0 * exp2(b_glast[None, :] - b_gc0)
    tl.store(kg_l + i_t0 * (HV * K) + o_c[:, None] * (HV * K) + o_k[None, :],
             b_kg0.to(kg.dtype.element_ty), mask=m_c0[:, None] & m_k2)
    if STORE_QG:
        b_q0r = tl.load(q_l + i_t0 * (H * K) + o_c[:, None] * (H * K) + o_k[None, :],
                        mask=m_c0[:, None] & m_k2, other=0.0)
        tl.store(qg_l + i_t0 * (HV * K) + o_c[:, None] * (HV * K) + o_k[None, :],
                 (b_q0r * exp2(b_gc0)).to(qg.dtype.element_ty), mask=m_c0[:, None] & m_k2)

    b_kg1 = b_k1 * exp2(b_glast[None, :] - b_gc1)
    tl.store(kg_l + i_t1 * (HV * K) + o_c[:, None] * (HV * K) + o_k[None, :],
             b_kg1.to(kg.dtype.element_ty), mask=m_c1[:, None] & m_k2)
    if STORE_QG:
        tl.store(qg_l + i_t1 * (HV * K) + o_c[:, None] * (HV * K) + o_k[None, :],
                 (b_q1 * exp2(b_gc1)).to(qg.dtype.element_ty), mask=m_c1[:, None] & m_k2)

    # ---------------- block 1: recompute w/u -------------------------------- #
    b_v1 = tl.load(v_l + i_t1 * (HV * V) + o_c[:, None] * (HV * V) + o_v[None, :],
                   mask=m_c1[:, None] & m_v2, other=0.0)
    b_vb1 = (b_v1 * b_beta1r[:, None]).to(b_v1.dtype)
    b_kb1 = b_k1 * b_beta1r[:, None]
    b_kb1 = b_kb1 * exp2(b_gc1)
    b_kw1 = b_kb1.to(b_k1.dtype)
    b_u1 = tl.dot(b_Ai1_c.to(Akk.dtype.element_ty), b_vb1, allow_tf32=False) \
        + tl.dot(b_Ai10.to(Akk.dtype.element_ty), b_vb0, allow_tf32=False)
    tl.store(u_l + i_t1 * (HV * V) + o_c[:, None] * (HV * V) + o_v[None, :],
             b_u1.to(u.dtype.element_ty), mask=m_c1[:, None] & m_v2)
    b_w1 = tl.dot(b_Ai1_c.to(Akk.dtype.element_ty), b_kw1, allow_tf32=False) \
        + tl.dot(b_Ai10.to(Akk.dtype.element_ty), b_kw0, allow_tf32=False)
    tl.store(w_l + i_t1 * (HV * K) + o_c[:, None] * (HV * K) + o_k[None, :],
             b_w1.to(w.dtype.element_ty), mask=m_c1[:, None] & m_k2)


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_kda_fwd_stage0_4_small_t_kernel_npu(
    q,
    k,
    v,
    g,
    A_log,
    dt_bias,
    beta,
    gc,
    w,
    u,
    qg,
    kg,
    Aqk,
    Akk,
    scale,
    gscale,
    lower_bound,
    T,
    task_num,
    num_core,
    BH,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    GATE_ACTIVATE: tl.constexpr,
    STORE_QG: tl.constexpr,
    PERSISTENT: tl.constexpr,
):
    if PERSISTENT:
        # persistent (chunk_intra/wy_fast idiom): one program per AICore
        # strides over all (chunk, v-head) tasks; task-major mapping matches
        # chunk_bwd (i_t = task_id // BH, i_bh = task_id % BH) so consecutive
        # tasks on a core touch contiguous GM. All per-task state (GM bases,
        # masks, tiles) lives inside the body: nothing carries across tasks
        # and no arg pointer is ever advanced in place.
        core_id = tl.program_id(0)
        for task_id in tl.range(core_id, task_num, num_core):
            i_t = task_id // BH
            i_bh = task_id % BH
            _chunk_kda_fwd_stage0_4_body(
                i_t, i_bh,
                q, k, v, g, A_log, dt_bias, beta, gc, w, u, qg, kg, Aqk, Akk,
                scale, gscale, lower_bound, T,
                H, HV, K, V, BT, BC, BK, BV,
                HAS_A, HAS_BIAS, USE_LOWER_BOUND, GATE_ACTIVATE, STORE_QG,
            )
    else:
        # one task per program: the (NT, B*HV) grid
        i_t = tl.program_id(0)
        i_bh = tl.program_id(1)
        _chunk_kda_fwd_stage0_4_body(
            i_t, i_bh,
            q, k, v, g, A_log, dt_bias, beta, gc, w, u, qg, kg, Aqk, Akk,
            scale, gscale, lower_bound, T,
            H, HV, K, V, BT, BC, BK, BV,
            HAS_A, HAS_BIAS, USE_LOWER_BOUND, GATE_ACTIVATE, STORE_QG,
        )


@input_guard
def chunk_kda_fwd_stage0_4_npu(
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
    """Fused stage0-4 forward for the guarded small-T shape.

    Returns ``(gc, w, u, qg, kg, Aqk, Akk)`` mirroring the standard pipeline's
    tensors, or ``None`` when the kernel cannot be launched (callers fall back
    to the multi-kernel path).
    """
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    BT = chunk_size
    BC = _BC
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    NT = triton.cdiv(T, BT)
    task_num = NT * B * HV

    gc = torch.empty_like(g, dtype=torch.float32)
    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    qg = k.new_empty(B, T, HV, K) if disable_recompute else None
    kg = k.new_empty(B, T, HV, K)
    Aqk = k.new_empty(B, T, HV, BT)
    Akk = k.new_empty(B, T, HV, BT)

    # Grid policy: <= threshold tasks get one task per program (the T==2048
    # production launch); above it the same single launch goes persistent on
    # <= num_aicore programs (the T==4096 production launch).
    persistent = task_num > _PERSISTENT_TASK_THRESHOLD
    if persistent:
        num_core = get_npu_properties()["num_aicore"]
        grid = (min(num_core, task_num),)
    else:
        num_core = 1
        grid = (NT, B * HV)

    try:
        fast_launch(chunk_kda_fwd_stage0_4_small_t_kernel_npu, grid,
            q=q,
            k=k,
            v=v,
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            beta=beta,
            gc=gc,
            w=w,
            u=u,
            qg=qg,
            kg=kg,
            Aqk=Aqk,
            Akk=Akk,
            scale=scale,
            gscale=RCP_LN2,
            lower_bound=lower_bound,
            T=T,
            task_num=task_num,
            num_core=num_core,
            BH=B * HV,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BC=BC,
            BK=BK,
            BV=BV,
            HAS_A=A_log is not None,
            HAS_BIAS=dt_bias is not None,
            USE_LOWER_BOUND=lower_bound is not None,
            GATE_ACTIVATE=use_gate_in_kernel,
            STORE_QG=disable_recompute,
            PERSISTENT=persistent,
            num_warps=_NUM_WARPS,
            **ascend_compile_kwargs(blacklist_auto_blockify=True),
        )
    except Exception:
        logger.warning(
            "chunk_kda_fwd_stage0_4_npu: fused small-T kernel failed to launch; "
            "falling back to the standard pipeline",
            exc_info=True,
        )
        return None
    return gc, w, u, qg, kg, Aqk, Akk
