# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused h-state + output tail for the KDA chunk forward on Ascend NPU.

Replaces the ``chunk_gated_delta_rule_fwd_kernel_h_oneslab_npu`` h recurrence
(with its [B, NT, HV, K, V] materialization) + ``chunk_gla_fwd_kernel_o_npu``
(o = q@h_{i-1} + A@v_new) pair with one persistent-core task per (B*HV,
V-block) that carries the [BK, BV] fp32 state ``b_h1`` in registers across
the whole chunk loop (the persistent-tail design of FlagGems'
``chunk_fused_tail_vblock``). Per chunk, with the *pre-update* state in hand,

    b_v   = u - w @ b_h1                      (v_new, fp32, on chip)
    b_o   = qg_scaled @ b_h1 + A @ b_v        (qg_scaled = bf16(q*exp2(g)*scale))
    b_h1  = diag(exp2(gk_last)) * b_h1 + k^T @ b_v

so the per-chunk h round-trip through HBM (~1 GiB write + read at T=131072)
and the v_new round-trip when it is not consumed by the backward disappear.

The fused body is scalar-pipe heavy (~64% aic-scalar share), so per-chunk
scalar work is minimized:

* The per-chunk gate broadcast product ``b_q * exp2(b_g) * scale`` ([BT, BK]
  fp32 exp2 + two muls + bf16 cast per chunk) is precomputed once per token by
  the existing WY producer (``recompute_w_u_fwd_kda_kernel_npu``,
  STORE_QG_SCALED — it already holds ``exp2(g)`` in registers), with the exact
  same expression and fp32->bf16 rounding point, so the tail loads a bf16 tile
  instead of q (bf16) + g (fp32) and recomputing the exp2 chain.
* The causal ``tl.where(m_s, b_A, 0)`` epilogue on A is dropped: the NPU intra
  producers zero-init Aqk and only store causal (lower-triangular incl. diag)
  blocks, so the strict upper triangle is exact zeros and the mask was a
  [BT, BT] select + fp32 round-trip per chunk over already-zero data.
* Loop-control scalars are hoisted: one ``row = i_t * BT`` and one
  ``i_t64 = tl.cast(i_t, tl.int64)`` serve every block pointer in the body.
* ``gk_last_exp`` is already laid out [B, HV, NT, K] fp32 contiguous by
  ``_prepare_gk_last_exp`` — per-chunk row loads are contiguous 512B and
  consecutive chunks stream sequentially, so no producer re-layout is needed.

Math is op-for-op the two original kernels (same dot operand precisions:
fp32 for the h recurrence, bf16 for the o dots, with the same
fp32->bf16 rounding points), so outputs match up to fp32 accumulation
grouping. ``STORE_H``/``SAVE_NEW_VALUE`` keep the disable_recompute=True
path (bwd reads saved h/v_new) working at full fidelity.

The default schedule is the ILP variant: profile attribution shows every dot
in this loop costs critical-path time regardless of flops, so its body
issues the chain-independent dot/load work into the serial chain's shadow
(bitwise-equal outputs; see its docstring). ``FLA_KDA_TAIL_ILP=0`` or any
host-side launch failure falls back to the serial-schedule kernel.

m5 clamp hygiene: the launch grid is clamped to
``min(num_aicore, task_num)`` (d1's clamp, bit-level launch alignment only —
zero tiling / zero work-division change; e1 proved the identical clamp on
dhu at bit level). d1's BV=64 occupancy policy is carried but default OFF
(``FLA_KDA_TAIL_BV_POLICY=1`` re-enables): the C bench matrix measured +14%
e2e pollution from the policy, so BV stays 128. ``FLA_KDA_TAIL_GRID_CLAMP=0``
or any host-side launch failure permanently restores the full-grid launch.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from fla.utils import input_guard
from fla.utils.launch_fast import fast_launch, num_aicore

# Same compile flags as the unfused oneslab fwd_h launch: the live set
# (b_h1 + per-chunk w/v/k/qg/A/o tiles) packs under UB with buffer-saving
# enabled.
_FUSED_TAIL_COMPILE = dict(enable_ubuf_saving=True, unit_flag=True)
# V-block width. The o-dots want V-wide operands: at the production shape
# (B=1, HV=16, V=128) the fused tail measured 0.785x at BV=32, 0.917x at
# BV=64 and 1.029x at BV=128 vs the unfused kernel pair, so the V-wide
# default (16 tasks) is the proven configuration. FLA_KDA_FUSED_TAIL_BV
# overrides everything for sweeps (16/32/64/128 honoured).
_FUSED_TAIL_BV_DEFAULT = 128
try:
    _env_bv = int(os.environ.get('FLA_KDA_FUSED_TAIL_BV', '0') or 0)
except ValueError:
    _env_bv = 0
if _env_bv in (16, 32, 64, 128):
    _FUSED_TAIL_BV_DEFAULT = _env_bv

# d1 occupancy switches as carried by m5 (clamp hygiene unification):
# * FLA_KDA_TAIL_GRID_CLAMP (default on): grid = min(num_aicore, task_num)
#   so no MIX block binds a cube core just to run an empty task loop —
#   pure launch alignment, zero tiling/work-division change.
# * FLA_KDA_TAIL_BV_POLICY (default OFF): d1's policy that drops the tail
#   V-block width to 64 when the 128-wide grid leaves cube cores idle.
#   The C bench matrix measured +14% e2e pollution from the BV=64 policy,
#   so m5 defaults it off; '1' re-enables it for same-card A/B only.
_TAIL_BV_POLICY = os.environ.get('FLA_KDA_TAIL_BV_POLICY', '0').strip().lower() in ('1', 'true', 'on')
_TAIL_GRID_CLAMP = os.environ.get('FLA_KDA_TAIL_GRID_CLAMP', '1').strip().lower() not in ('0', 'false', 'off')

# Permanent in-process kill flags (campaign rule: every new path falls back
# to the proven path on first host-side launch failure).
_GRID_CLAMP_DISABLED = False
_TAIL_BV64_DISABLED = False


def _disable_tail_bv64() -> None:
    """Permanently opt this process out of the BV=64 occupancy policy."""
    global _TAIL_BV64_DISABLED
    _TAIL_BV64_DISABLED = True


def _fused_tail_bv(*, B: int, HV: int, V: int) -> int:
    """Per-call V-block width, deterministic in (B, HV, V) so the eligibility
    gates and the launch wrapper always agree on the same BV.

    Env override wins. Otherwise 128 (the proven width); the opt-in d1
    occupancy policy (FLA_KDA_TAIL_BV_POLICY=1) returns 64 when the 128-wide
    task count (B*HV*(V//128), whole-NT persistent tasks) under-fills the
    cube cores — V % 128 == 0 implies 64 | V. Shapes with V % 128 != 0 keep
    128, so the eligibility set is unchanged either way.
    """
    if _env_bv in (16, 32, 64, 128):
        return _env_bv
    if (
        _TAIL_BV_POLICY
        and not _TAIL_BV64_DISABLED
        and V % 128 == 0
        and B * HV * (V // 128) < num_aicore()
    ):
        return 64
    return _FUSED_TAIL_BV_DEFAULT

# ILP schedule: issue the dot2/qg work that is independent of the serial
# w@h -> v_new -> k^T@v_new chain INTO that chain's shadow instead of after
# it. Default on (bit-identical math, same op count); '0' selects the
# serial-schedule kernel, and any host-side launch failure of the ILP
# kernel falls back to it permanently.
_FUSED_TAIL_ILP = os.environ.get('FLA_KDA_TAIL_ILP', '1').strip().lower() not in ('0', 'false', 'off')


def _launch_core_grid(kernel, *, task_num: int, kernel_kwargs: dict, **compile_opts) -> None:
    """Persistent-core launch with the m5 clamp-hygiene grid (d1).

    The persistent idiom ``tl.range(core_id, task_num, num_core)`` only
    needs ``min(num_aicore, task_num)`` programs; extra blocks run an empty
    range but each still binds its MIX block (1 cube + 2 vector cores) for
    the whole kernel duration. At the production shape (B=1, HV=16, V=128
    -> 16 tasks) the old grid=num_aicore launch bound 24 AIC / 48 AIV with
    8 cube cores idle-looping through the entire tail; grid=(16,) with the
    num_core arg == grid makes blocks match tasks (the chunk_intra.py /
    wy_fast.py launcher convention, stride arg == launched grid) with
    byte-identical kernel math. Any host-side failure permanently restores
    the proven grid=num_aicore launch.
    """
    global _GRID_CLAMP_DISABLED
    num_core = num_aicore()
    if _TAIL_GRID_CLAMP and not _GRID_CLAMP_DISABLED and 0 < task_num < num_core:
        try:
            fast_launch(kernel, (task_num,), task_num=task_num, num_core=task_num, **compile_opts, **kernel_kwargs)
            return
        except Exception:
            _GRID_CLAMP_DISABLED = True
    fast_launch(kernel, (num_core,), task_num=task_num, num_core=num_core, **compile_opts, **kernel_kwargs)


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "STORE_H": lambda args: args["h"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "task_num", "num_core"])
def chunk_kda_fwd_kernel_h_o_fused_npu(
    qg_scaled,
    k,
    w,
    u,
    v_new,
    A,
    gk_last_exp,
    o,
    h,
    h0,
    ht,
    T,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_H: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
):
    """Aligned oneslab fusion: launched only when non-varlen, T % BT == 0,
    K == BK (power of two) and V % BV == 0, so no boundary checks are needed
    and the per-K gate decay is always host-precomputed (``gk_last_exp``).

    ``qg_scaled`` is the WY-producer-precomputed bf16(q * exp2(g) * scale)
    [B, T, HV, K] tile (see the module docstring)."""
    core_id = tl.program_id(0)
    DH_CS: tl.constexpr = HV * K * V
    stride_k: tl.constexpr = HV * K
    stride_v: tl.constexpr = HV * V
    stride_a: tl.constexpr = HV * BT
    NV: tl.constexpr = tl.cdiv(V, BV)
    NT = T // BT
    o_k = tl.arange(0, BK)

    for task_id in tl.range(core_id, task_num, num_core):
        i_v, i_nh = task_id % NV, (task_id // NV).to(tl.int64)
        i_n, i_hv = i_nh // HV, i_nh % HV
        bos = i_n * T
        v_start = i_v * BV

        # Rebind GM bases each task; do not in-place ``ptr +=`` across ``i_t``
        # (Ascend MTE OOB). ``h`` rebases each chunk via ``i_t64 * DH_CS``.
        qgs_base = qg_scaled + (bos * HV + i_hv) * K
        k_base = k + (bos * HV + i_hv) * K
        w_base = w + (bos * HV + i_hv) * K
        u_base = u + (bos * HV + i_hv) * V
        o_base = o + (bos * HV + i_hv) * V
        a_base = A + (bos * HV + i_hv) * BT
        if SAVE_NEW_VALUE:
            v_new_base = v_new + (bos * HV + i_hv) * V
        if STORE_H:
            h_nh = h + (i_n * NT * HV + i_hv) * K * V
        gk_last_exp_nh = gk_last_exp + (i_n * HV + i_hv) * NT * K

        b_h1 = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            h0_ptr = h0 + i_nh * K * V
            p_h0_1 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (0, v_start), (BK, BV), (1, 0))
            b_h1 += tl.load(p_h0_1).to(tl.float32)

        for i_t in range(NT):
            # Hoisted loop-control scalars: one row base + one int64 chunk
            # index serve the whole body.
            row = i_t * BT
            i_t64 = tl.cast(i_t, tl.int64)

            # ---- h kernel: v_new = u - w @ h_{i_t-1} with the pre-update state ----
            if STORE_H:
                h_base = h_nh + i_t64 * DH_CS
                p_h1 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start), (BK, BV), (1, 0))
                tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty))

            p_w1 = tl.make_block_ptr(w_base, (T, K), (stride_k, 1), (row, 0), (BT, BK), (1, 0))
            b_w = tl.load(p_w1).to(tl.float32)
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))

            p_u = tl.make_block_ptr(u_base, (T, V), (stride_v, 1), (row, v_start), (BT, BV), (1, 0))
            b_v = tl.load(p_u).to(tl.float32) - b_v
            if SAVE_NEW_VALUE:
                p_v_new = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (row, v_start), (BT, BV), (1, 0))
                tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty))

            # ---- o kernel: o = qg_scaled @ h_{i_t-1} + A @ v_new ----
            # b_h1 is still the pre-update state == the h[i_t] tile the standalone
            # o kernel loads from HBM; b_v.to(bf16) is the same rounding the v_new
            # store applies, so the bf16 dot operands are bit-identical.
            b_o = tl.zeros([BT, BV], dtype=tl.float32)
            p_qg = tl.make_block_ptr(qgs_base, (T, K), (stride_k, 1), (row, 0), (BT, BK), (1, 0))
            b_qg = tl.load(p_qg)
            b_o = tl.dot(b_qg, b_h1.to(b_qg.dtype), b_o)

            # A's strict upper triangle is exact zeros from the NPU intra
            # producers (zero-init Aqk, causal-only stores), so the causal
            # tl.where epilogue is dropped — the tile feeds the dot as loaded.
            p_a = tl.make_block_ptr(a_base, (T, BT), (stride_a, 1), (row, 0), (BT, BT), (1, 0))
            b_A = tl.load(p_a)
            b_o = tl.dot(b_A, b_v.to(b_A.dtype), b_o)
            p_o = tl.make_block_ptr(o_base, (T, V), (stride_v, 1), (row, v_start), (BT, BV), (1, 0))
            tl.store(p_o, b_o.to(p_o.dtype.element_ty))

            # ---- h update: decay first (pre-update state was needed above), then k^T @ v_new ----
            b_gk_last1 = tl.load(gk_last_exp_nh + i_t64 * K + o_k).to(tl.float32)
            b_h1 *= b_gk_last1[:, None]

            p_k1 = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, row), (BK, BT), (0, 1))
            b_k = tl.load(p_k1).to(tl.float32)
            b_h1 = tl.dot(b_k, b_v, b_h1)

        if STORE_FINAL_STATE:
            ht_ptr = ht + i_nh * K * V
            p_ht1 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start), (BK, BV), (1, 0))
            tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty))


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "STORE_H": lambda args: args["h"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "task_num", "num_core"])
def chunk_kda_fwd_kernel_h_o_fused_ilp_npu(
    qg_scaled,
    k,
    w,
    u,
    v_new,
    A,
    gk_last_exp,
    o,
    h,
    h0,
    ht,
    T,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_H: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
):
    """ILP reschedule of ``chunk_kda_fwd_kernel_h_o_fused_npu``: same ops,
    same operand bits, same accumulation order — o/h/v_new/ht verified
    bitwise-equal against the serial kernel across bf16/fp16 and BT/BV
    configurations.

    The loop-carried serial chain is ``w@h -> v_new -> k^T@v_new -> h``; the
    codegen wall serializes the per-chunk MTE2->cube->MTE3 segments in program
    order (every dot costs critical-path time regardless of flops), so
    independent work placed AFTER the chain still pays full serial time.
    This body issues it INTO the chain's shadow instead:

    * all GM block-ptr descriptors for the chunk are built in one scalar batch
      at the loop top (scalar/pointer pipelining), so no make_block_ptr ALU
      sits in the inter-dot gaps;
    * ``qg`` is loaded before dot1 and ``dot2 = qg @ h_pre`` issues directly
      after dot1 — it reads the same pre-update state (h_pre-dependent but
      dot1-independent) and so fills dot1's writeback shadow instead of
      waiting behind v_new as in the serial body;
    * the ``u`` load issues between the dot1/dot2 issues and the v_new
      subtract; ``A`` loads between the v_new consumers' issues;
    * the two v_new consumers (``A@v_new`` for o, ``k^T@v_new`` for h) issue
      adjacently at the bottom so dot4 fills dot3's shadow.

    Not attempted: cross-iteration tile prefetch (loop-carried tiles do not
    compile on this backend) and the algebraic term-split
    ``h += k^T@u - k^T@(w@h)`` (it adds a cube segment per chunk without
    shortening the serial chain, overflows the UB envelope at BV=128, and its
    reassociated fp32 cancellation error measures 3-80 bf16 steps against
    the <=1/32-step acceptance bar).
    """
    core_id = tl.program_id(0)
    DH_CS: tl.constexpr = HV * K * V
    stride_k: tl.constexpr = HV * K
    stride_v: tl.constexpr = HV * V
    stride_a: tl.constexpr = HV * BT
    NV: tl.constexpr = tl.cdiv(V, BV)
    NT = T // BT
    o_k = tl.arange(0, BK)

    for task_id in tl.range(core_id, task_num, num_core):
        i_v, i_nh = task_id % NV, (task_id // NV).to(tl.int64)
        i_n, i_hv = i_nh // HV, i_nh % HV
        bos = i_n * T
        v_start = i_v * BV

        qgs_base = qg_scaled + (bos * HV + i_hv) * K
        k_base = k + (bos * HV + i_hv) * K
        w_base = w + (bos * HV + i_hv) * K
        u_base = u + (bos * HV + i_hv) * V
        o_base = o + (bos * HV + i_hv) * V
        a_base = A + (bos * HV + i_hv) * BT
        if SAVE_NEW_VALUE:
            v_new_base = v_new + (bos * HV + i_hv) * V
        if STORE_H:
            h_nh = h + (i_n * NT * HV + i_hv) * K * V
        gk_last_exp_nh = gk_last_exp + (i_n * HV + i_hv) * NT * K

        b_h1 = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            h0_ptr = h0 + i_nh * K * V
            p_h0_1 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (0, v_start), (BK, BV), (1, 0))
            b_h1 += tl.load(p_h0_1).to(tl.float32)

        for i_t in range(NT):
            row = i_t * BT
            i_t64 = tl.cast(i_t, tl.int64)

            # b_h1 is still the pre-update state h_{i_t-1}: it feeds STORE_H,
            # dot1 (w@h) and dot2 (qg@h) below, then the decayed update.
            if STORE_H:
                h_base = h_nh + i_t64 * DH_CS
                p_h1 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start), (BK, BV), (1, 0))
                tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty))

            # Scalar batch: every descriptor the chunk body needs is built
            # here, off the inter-dot critical path.
            p_w1 = tl.make_block_ptr(w_base, (T, K), (stride_k, 1), (row, 0), (BT, BK), (1, 0))
            p_qg = tl.make_block_ptr(qgs_base, (T, K), (stride_k, 1), (row, 0), (BT, BK), (1, 0))
            p_u = tl.make_block_ptr(u_base, (T, V), (stride_v, 1), (row, v_start), (BT, BV), (1, 0))
            p_a = tl.make_block_ptr(a_base, (T, BT), (stride_a, 1), (row, 0), (BT, BT), (1, 0))
            p_o = tl.make_block_ptr(o_base, (T, V), (stride_v, 1), (row, v_start), (BT, BV), (1, 0))
            p_k1 = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, row), (BK, BT), (0, 1))
            if SAVE_NEW_VALUE:
                p_v_new = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (row, v_start), (BT, BV), (1, 0))

            # ---- dot pair on h_pre: dot1 heads the serial chain, dot2 is
            # independent of it (reads h_pre, not v_new) and issues in its
            # shadow. qg/u loads are the MTE2 fills between the issues. ----
            b_w = tl.load(p_w1).to(tl.float32)
            b_qg = tl.load(p_qg)
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
            b_o = tl.zeros([BT, BV], dtype=tl.float32)
            b_o = tl.dot(b_qg, b_h1.to(b_qg.dtype), b_o)
            b_v = tl.load(p_u).to(tl.float32) - b_v
            if SAVE_NEW_VALUE:
                tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty))

            # ---- v_new consumer 1 (o path, ends within the chunk) ----
            b_A = tl.load(p_a)
            b_o = tl.dot(b_A, b_v.to(b_A.dtype), b_o)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty))

            # ---- v_new consumer 2 (loop-carried state) ----
            b_gk_last1 = tl.load(gk_last_exp_nh + i_t64 * K + o_k).to(tl.float32)
            b_h1 *= b_gk_last1[:, None]
            b_k = tl.load(p_k1).to(tl.float32)
            b_h1 = tl.dot(b_k, b_v, b_h1)

        if STORE_FINAL_STATE:
            ht_ptr = ht + i_nh * K * V
            p_ht1 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start), (BK, BV), (1, 0))
            tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty))


@input_guard
def chunk_kda_fwd_h_o_fused_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    store_h: bool = False,
    save_new_value: bool = False,
    qg_scaled: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Fused replacement for ``chunk_gated_delta_rule_fwd_h`` + ``chunk_gla_fwd_o_gk``.

    ``k`` is the gated key tensor (``kg``). Returns ``(o, h, v_new, final_state)``
    with ``h``/``v_new`` allocated (and written) only when requested.

    ``qg_scaled`` is the WY-producer-precomputed ``bf16(q * exp2(g) * scale)``
    [B, T, HV, K] tensor; when it is not supplied (e.g. a direct call that did
    not go through ``chunk_kda_fwd``'s producer wiring) the wrapper computes it
    with torch ops as a fallback — same expression and rounding point.
    """
    from fla.ops.common.backends.triton_ascend.chunk_delta_h import _prepare_gk_last_exp

    B, T, H, K = q.shape
    HV, V = u.shape[2], u.shape[-1]
    BT = chunk_size
    N, NT = B, triton.cdiv(T, BT)

    if qg_scaled is None:
        # Fallback (direct callers without the producer wiring): same
        # expression and rounding point, with the GQA head mapping the kernel
        # path gets from the producer (i_h = i_hv // (HV // H)).
        q_hv = q if H == HV else q.repeat_interleave(HV // H, dim=2)
        qg_scaled = (q_hv.float() * torch.exp2(g.float()) * scale).to(q.dtype)
    assert qg_scaled.dtype == q.dtype and qg_scaled.is_contiguous()

    o = torch.empty_like(u)
    h = k.new_empty(B, NT, HV, K, V) if store_h else None
    v_new = torch.empty_like(u) if save_new_value else None
    final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None
    # Guard guarantees non-varlen + T % BT == 0, so this always precomputes.
    _, gk_last_exp = _prepare_gk_last_exp(gk, B=B, T=T, HV=HV, K=K, BT=BT, cu_seqlens=None)
    assert gk_last_exp is not None

    BK = triton.next_power_of_2(K)
    BV = _fused_tail_bv(B=B, HV=HV, V=V)
    # The eligibility gate guarantees divisibility on the chunk_kda_fwd path;
    # direct callers get the same host check instead of a device OOB.
    assert V % BV == 0

    def _launch(bv: int) -> None:
        kwargs = dict(
            qg_scaled=qg_scaled,
            k=k,
            w=w,
            u=u,
            v_new=v_new,
            A=A,
            gk_last_exp=gk_last_exp,
            o=o,
            h=h,
            h0=initial_state,
            ht=final_state,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=bv,
        )
        if _FUSED_TAIL_ILP:
            try:
                _launch_core_grid(
                    chunk_kda_fwd_kernel_h_o_fused_ilp_npu,
                    task_num=N * HV * triton.cdiv(V, bv),
                    kernel_kwargs=kwargs,
                    **_FUSED_TAIL_COMPILE,
                )
                return
            except Exception:
                # Permanent fallback: rerun the serial-schedule kernel in
                # full. Host-side failures only (async device errors are not
                # caught).
                pass
        _launch_core_grid(
            chunk_kda_fwd_kernel_h_o_fused_npu,
            task_num=N * HV * triton.cdiv(V, bv),
            kernel_kwargs=kwargs,
            **_FUSED_TAIL_COMPILE,
        )

    if BV != 128:
        try:
            _launch(BV)
            return o, h, v_new, final_state
        except Exception:
            # Permanent BV-policy fallback: any host-side failure of the
            # narrowed block relaunches the proven-width schedule in full.
            _disable_tail_bv64()
    _launch(128 if V % 128 == 0 else BV)
    return o, h, v_new, final_state


def chunk_kda_fwd_h_o_fused_ok(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None,
    state_v_first: bool,
    use_graph: bool,
    cp_context,
) -> bool:
    """Eligibility for the fused h+o tail (caller already checked IS_NPU)."""
    if cu_seqlens is not None or state_v_first or use_graph or cp_context is not None:
        return False
    if chunk_size not in (32, 64) or T % chunk_size != 0:
        return False
    if K < 16 or K > 256 or K & (K - 1) != 0:
        return False
    if V % _fused_tail_bv(B=q.shape[0], HV=u.shape[2], V=V) != 0:
        return False
    # The A@v_new dot casts both operands to the q dtype (matching the o kernel's
    # v_new-dtype cast); require the unified bf16/fp16 production layout.
    if not (q.dtype == k.dtype == u.dtype):
        return False
    if not (q.is_contiguous() and k.is_contiguous() and u.is_contiguous()):
        return False
    return True


def chunk_kda_fwd_h_o_fused_candidate(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None,
    state_v_first: bool,
    use_graph: bool,
    cp_context,
) -> bool:
    """Conservative input-space mirror of ``chunk_kda_fwd_h_o_fused_ok``.

    Runs BEFORE ``chunk_kda_fwd_intra`` so the WY producer can be asked for the
    scaled gated query (``qg_scaled``) in the same pass. The authoritative
    post-check runs on the produced ``kg``/``u`` as before; ``kg`` is always a
    fresh contiguous tensor with ``k``'s dtype and ``u = empty_like(v)``, so
    this pre-check can only be more conservative (exotic strided inputs fall
    back to the wrapper's torch-computed ``qg_scaled``), never more permissive.
    """
    if cu_seqlens is not None or state_v_first or use_graph or cp_context is not None:
        return False
    if chunk_size not in (32, 64) or T % chunk_size != 0:
        return False
    if K < 16 or K > 256 or K & (K - 1) != 0:
        return False
    if V % _fused_tail_bv(B=q.shape[0], HV=v.shape[2], V=V) != 0:
        return False
    if not (q.dtype == k.dtype == v.dtype):
        return False
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return False
    return True
