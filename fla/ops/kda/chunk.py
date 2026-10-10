# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Related files are modified and supported by the Moonshot AI Team

import os
import warnings

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.backends import dispatch
from fla.ops.common.gate import fused_beta_sigmoid, fused_beta_sigmoid_bwd
from fla.ops.cp import FLACPContext
from fla.ops.kda.chunk_bwd import chunk_kda_bwd
from fla.ops.kda.chunk_fwd import chunk_kda_fwd, _atem_fuse_enabled
from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_indices_static
from fla.utils import IS_NPU, autocast_custom_bwd, autocast_custom_fwd, input_guard

# Backward-retention heuristic (NPU): with ``disable_recompute=False`` the
# forward frees w/u/qg/kg/v_new/h and backward rebuilds them
# (kda_gate_chunk_cumsum + recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h,
# i.e. one extra full fwd_h state pass per backward). When the extra retained
# activation memory fits a budget we instead keep those tensors and take the
# existing ``disable_recompute=True`` save path, skipping the whole recompute
# block. The retained tensors come from the same deterministic kernels, so
# numerics are unchanged. Override the budget (bytes) with
# FLA_KDA_RETAIN_ACTIVATION_BUDGET; set it to 0 to always recompute.
_KDA_RETAIN_BUDGET_BYTES = 8 * 1024**3
_KDA_RETAIN_BUDGET_ENV = "FLA_KDA_RETAIN_ACTIVATION_BUDGET"


def _kda_retention_budget() -> int:
    raw = os.environ.get(_KDA_RETAIN_BUDGET_ENV)
    if raw is not None:
        try:
            value = int(raw)
            if value >= 0:
                return value
        except ValueError:
            pass
    return _KDA_RETAIN_BUDGET_BYTES


def _kda_retention_extra_bytes(
    q: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    keep_g: bool,
    keep_gate_t: bool = False,
) -> int:
    """Activation bytes held alive for backward when the recompute is skipped."""
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    esz = q.element_size()
    NT = (T + chunk_size - 1) // chunk_size
    # w, qg, kg are [B, T, HV, K]; u, v_new are [B, T, HV, V];
    # h is [B, NT, HV, K, V]; the fp32 gate chunk-cumsum is only kept
    # additionally when the gate is computed in kernel; keep_gate_t adds the
    # ATEM dual-store head-major fp32 gate copy (FLA_KDA_ATEM_FUSE).
    per_token = HV * (3 * K + 2 * V) * esz + (HV * K * 4 if keep_g else 0) \
        + (HV * K * 4 if keep_gate_t else 0)
    return B * (T * per_token + NT * HV * K * V * esz)


class ChunkKDAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        chunk_size: int = 64,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        cp_context: FLACPContext | None = None,
        use_graph: bool = False,
        max_num_seqs: int | None = None,
    ):
        # Apply l2norm
        q_rstd, k_rstd = None, None
        qk_l2norm_out = None
        if use_qk_l2norm_in_kernel:
            if IS_NPU and safe_gate and chunk_size in (32, 64) and not use_graph:
                # NPU fold: the NPU intra head kernel (persistent sub-chunk
                # pipeline or the small-T fused intra kernel) is the
                # first q/k consumer - it normalizes rows in-register (the exact
                # l2norm_fwd math) and materializes the normalized q/k plus the
                # row rstd for every later consumer (inter_solve, recompute,
                # h/o tail, backward, l2norm_bwd). The two standalone
                # l2norm_fwd launches and their raw q/k read pass are skipped.
                # rstd lands in a [2, B, H, T] working buffer (stride-1 per
                # head, the only store shape this kernel's UB model accepts)
                # and is transposed to the [B, T, H] l2norm interface in one
                # fused copy after the fwd pass.
                Bq, Tq, Hq = q.shape[0], q.shape[1], q.shape[2]
                rstd_buf = q.new_empty((2, Bq, Hq, Tq), dtype=torch.float32)
                qk_l2norm_out = {
                    'q': torch.empty_like(q),
                    'k': torch.empty_like(k),
                    'q_rstd': rstd_buf[0],
                    'k_rstd': rstd_buf[1],
                    'rstd_buf': rstd_buf,
                }
            else:
                q, q_rstd = l2norm_fwd(q)
                k, k_rstd = l2norm_fwd(k)

        beta_raw = beta
        if use_beta_sigmoid_in_kernel:
            beta = fused_beta_sigmoid(beta_raw, scale=2.0 if allow_neg_eigval else 1.0)

        chunk_indices, chunk_offsets = None, None
        if cu_seqlens is not None:
            if use_graph:
                if max_num_seqs is None:
                    max_num_seqs = cu_seqlens.shape[0] - 1
                assert cu_seqlens.shape[0] - 1 == max_num_seqs, (
                    f"cu_seqlens must be padded with zero-length tail sequences to exactly "
                    f"max_num_seqs + 1 entries, got {cu_seqlens.shape[0] - 1} sequences "
                    f"with max_num_seqs={max_num_seqs}"
                )
                nt_max = (q.shape[1] + chunk_size - 1) // chunk_size + max_num_seqs - 1
                chunk_indices, chunk_offsets = prepare_chunk_indices_static(cu_seqlens, chunk_size, nt_max)
            else:
                chunk_indices = prepare_chunk_indices(
                    cu_seqlens,
                    chunk_size,
                    cu_seqlens_cpu=cu_seqlens_cpu,
                )

        g_input = g

        # NPU: skip the backward recompute when keeping w/u/qg/kg/v_new/h
        # (plus the fp32 gate chunk-cumsum) for backward fits the budget.
        # Uses the existing disable_recompute=True save path unchanged.
        # T >= 8192 gate: the recompute skip only pays off at production-scale
        # sequence lengths; small-T runs keep the recompute path (and its
        # sub-op dispatch) intact.
        if (
            not disable_recompute
            and IS_NPU
            and q.shape[1] >= 8192
            and not return_intermediate_states
            and not use_graph
            and cp_context is None
            and _kda_retention_extra_bytes(
                q, v, chunk_size, use_gate_in_kernel,
                # the ATEM head-major gate copy is only produced on the
                # plain-cumsum path (use_gate_in_kernel=False)
                keep_gate_t=_atem_fuse_enabled() and not use_gate_in_kernel,
            ) <= _kda_retention_budget()
        ):
            disable_recompute = True

        (o, final_state, g_cumsum, g_t, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state) = chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g_input,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=chunk_size,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_intermediate_states,
            cp_context=cp_context,
            state_v_first=state_v_first,
            use_graph=use_graph,
            chunk_offsets=chunk_offsets,
            qk_l2norm_out=qk_l2norm_out,
        )

        if qk_l2norm_out is not None:
            # swap in the normalized q/k materialized inside the fwd pass and
            # produce the [B, T, H] rstd interface tensors (single fused copy);
            # the backward (chunk_kda_bwd + l2norm_bwd) consumes them exactly
            # as it consumed the standalone-l2norm outputs before
            q, k = qk_l2norm_out['q'], qk_l2norm_out['k']
            rstd = qk_l2norm_out['rstd_buf'].transpose(2, 3).contiguous()
            q_rstd, k_rstd = rstd[0], rstd[1]

        if return_intermediate_states:
            assert torch.is_inference_mode_enabled(), "return_intermediate_states is only allowed in inference mode"
            assert disable_recompute is False, "return_intermediate_states must be used with disable_recompute=False"
            return o.type_as(q), final_state, h

        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g_cumsum, g_t, g_input, beta_raw, beta, A_log, dt_bias, Aqk, Akk,
            w, u, qg, kg, v_new, h,
            initial_state, cu_seqlens, chunk_indices, chunk_offsets
        )
        ctx.use_graph = use_graph
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.allow_neg_eigval = allow_neg_eigval
        ctx.disable_recompute = disable_recompute
        ctx.cp_context = cp_context
        ctx.state_v_first = state_v_first
        return o.type_as(q), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        (q, q_rstd, k, k_rstd, v, g_cumsum, g_t, g_input, beta_raw, beta, A_log, dt_bias, Aqk, Akk,
         w, u, qg, kg, v_new, h,
         initial_state, cu_seqlens, chunk_indices, chunk_offsets) = (
            ctx.saved_tensors
        )

        dq, dk, dv, db, dg, dh0, dA, dbias = chunk_kda_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            g=g_cumsum,
            g_t=g_t,
            g_org=g_input if ctx.use_gate_in_kernel else None,
            state_v_first=ctx.state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=ctx.chunk_size,
            safe_gate=ctx.safe_gate,
            lower_bound=ctx.lower_bound,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            disable_recompute=ctx.disable_recompute,
            cp_context=ctx.cp_context,
            w=w,
            u=u,
            qg=qg,
            kg=kg,
            v_new=v_new,
            h=h,
            use_graph=ctx.use_graph,
            chunk_offsets=chunk_offsets,
            # gate backward needs fp32 dg; otherwise the reverse cumsum can
            # store the grad dtype directly and the return cast becomes a no-op
            dg_dtype=torch.float if ctx.use_gate_in_kernel else g_input.dtype,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        if ctx.use_beta_sigmoid_in_kernel:
            db = fused_beta_sigmoid_bwd(beta_raw, db, scale=2.0 if ctx.allow_neg_eigval else 1.0)

        return (dq.to(q), dk.to(k), dv.to(v), dg.to(g_input), db.to(beta_raw), dA, dbias, None, dh0,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None)


@torch.compiler.disable
@dispatch('kda')
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    cp_context: FLACPContext = None,
    use_graph: bool = False,
    max_num_seqs: int | None = None,
    **kwargs,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            values of shape ``[B, T, HV, V]``.
            GVA (Grouped Value Attention) is applied if ``HV > H``, where ``HV`` must be divisible by ``H``.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape ``[B, T, HV, K]``.
            When ``use_gate_in_kernel=False`` (default), ``g`` should be the pre-computed decay value.
            When ``use_gate_in_kernel=True``, ``g`` is the raw input before gate activation;
            the kernel fuses ``-exp(A_log) * softplus(g + dt_bias)`` + chunk cumsum internally.
        beta (torch.Tensor):
            betas of shape ``[B, T, HV]``.
        scale (Optional[float]):
            Scale factor for the KDA attention scores.
            If not provided, it will default to ``1 / sqrt(K)``. Default: ``None``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[N, HV, K, V]`` for ``N`` input sequences.
            For equal-length input sequences, ``N`` equals the batch size ``B``.
            Default: ``None``.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape ``[N, HV, K, V]``. Default: ``False``.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q,k tensor internally. Default: ``False``.
        use_gate_in_kernel (bool):
            Whether to compute the log-space KDA decay internally.
            - If ``True``:
              The passed ``g`` acts as the raw input for ``-exp(A_log) * softplus(g + dt_bias.view(HV, K))``.
              Note that as part of the input arguments,
              ``A_log`` (shape ``[HV]``) and the optional ``dt_bias`` (shape ``[HV * K]``) should be provided.
              When ``lower_bound`` is set, ``A_log`` may be ``None``,
              in which case the gate is ``lower_bound * sigmoid(g + dt_bias)``.
            - If ``False``, ``g`` is expected to be the pre-computed decay value.
            Default: ``False``.
        use_beta_sigmoid_in_kernel (bool):
            Whether to apply ``torch.sigmoid(beta)`` before launching the chunk kernel.
            - If ``True``, the passed ``beta`` acts as the raw beta logits.
            - If ``False``, ``beta`` is expected to already be in post-sigmoid space.
            Default: ``False``.
        allow_neg_eigval (bool):
            Whether to allow negative eigenvalues by scaling ``beta`` to ``[0, 2)``.
            Only takes effect together with ``use_beta_sigmoid_in_kernel=True``, in which case
            the kernel computes ``2 * sigmoid(beta)`` instead of ``sigmoid(beta)``.
            Default: ``False``.
        safe_gate (bool):
            Whether to clamp the gate to ``[lower_bound, 0)`` and enable M=16 TensorCore
            acceleration for higher throughput. Requires ``lower_bound`` to be set.
            Default: ``False``.
        lower_bound (Optional[float]):
            Lower bound for the forget gate (in log space). When set together with
            ``safe_gate=True``, changes the gate activation from
            ``-exp(A_log) * softplus(g + dt_bias)`` to
            ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``,
            which naturally clamps the output to ``[lower_bound, 0)``.
            Recommended value: ``-5`` (i.e., per-step decay ``exp(-5) ≈ 0.0067``).
            Default: ``None``.
        disable_recompute (bool):
            Whether to disable gradient recomputation in the kernel. When ``True``, the kernel
            will save all intermediate activations for backward pass, which is beneficial
            for training small models at the cost of increased memory usage. Default: ``False``.
        return_intermediate_states (bool):
            If True, returns intermediate state ``h`` for inference scenarios (e.g., vLLM).
            Must be used within ``torch.inference_mode()`` and will return a 3-tuple instead of 2-tuple.
            This is not intended for training as it bypasses autograd. Default: ``False``.
        state_v_first (Optional[bool]):
            Store the recurrent state in V-first ``[V, K]`` layout instead of the default ``[K, V]``. Default: ``False``.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape ``[N+1]`` used for variable-length training,
            consistent with the FlashAttention API.
        cu_seqlens_cpu (torch.LongTensor):
            Cumulative sequence lengths of shape ``[N+1]`` used for variable-length training,
            consistent with the FlashAttention API.
        cp_context (Optional[FLACPContext]):
            Context parallel context for distributed training across multiple devices.
            When provided, ``initial_state`` and ``output_final_state`` are not supported,
            and ``cu_seqlens`` will be overridden by the context. Default: ``None``.
        use_graph (bool):
            Whether to build the varlen chunk list on-device at a fixed shape so the
            forward/backward can be recorded by a platform graph (CUDA graph, NPU graph)
            and replayed after only the contents of ``cu_seqlens`` change. Requires:

            - ``cu_seqlens`` padded with zero-length tail sequences up to exactly
              ``max_num_seqs + 1`` entries; kernels return immediately on sentinel rows
              (segment id < 0), so output/gradient rows not covered by real tokens
              are left undefined.
            - ``initial_state`` and the incoming ``dht`` gradient (when used) shaped
              ``[max_num_seqs, ...]``.
            - the captured step warmed up eagerly beforehand so that kernel autotuning
              (and, with ``cp_context``, NCCL communicator setup) happens outside the graph.

            With ``cp_context``, the cross-rank state all-gathers are recorded into the
            graph, which additionally requires:

            - a graph-capturable (non-blocking) NCCL communicator, with all ranks
              capturing and replaying the same collectives in the same order;
            - ``cp_context.cu_seqlens`` being the persistent padded buffer, refreshed
              in place before each replay;
            - ``cp_context.pre_num_ranks_dev``/``post_num_ranks_dev``, persistent int32
              device scalars refreshed in place before each replay.

            ``disable_recompute=True`` is not supported. Default: ``False``.
        max_num_seqs (Optional[int]):
            Static upper bound of the sequence count used together with ``use_graph``.
            Defaults to ``cu_seqlens.shape[0] - 1``. Default: ``None``.

    Returns:
        - Normal mode (return_intermediate_states=False): A tuple (o, final_state)
            o (torch.Tensor):
                Outputs of shape ``[B, T, HV, V]``.
            final_state (torch.Tensor):
                Final state of shape ``[N, HV, K, V]`` if ``output_final_state=True`` else ``None``.
        - Inference mode (return_intermediate_states=True): A tuple (o, final_state, h)
            o (torch.Tensor):
                Outputs of shape ``[B, T, HV, V]``.
            final_state (torch.Tensor):
                Final state of shape ``[N, HV, K, V]`` if ``output_final_state=True`` else ``None``.
            h (torch.Tensor):
                Intermediate states of shape ``[B, NT, HV, K, V]`` and dtype ``bfloat16``.
                - For equal-length sequences: ``NT = ceil(T / chunk_size)``
                - For variable-length sequences (cu_seqlens): B is always 1 (flattened),
                  NT is the total number of chunks across all sequences.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.kda import chunk_kda
        # inputs with equal lengths (no GVA, HV == H)
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda')
        >>> g = torch.rand(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> A_log = torch.randn(H, dtype=torch.float32, device='cuda')
        >>> dt_bias = torch.randn(H * K, dtype=torch.float32, device='cuda')
        >>> o, ht = chunk_kda(
            q, k, v, g, beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            initial_state=h0,
            output_final_state=True
        )
        # GVA mode (HV > H)
        >>> HV = 8  # 2x more value heads than qk heads
        >>> v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device='cuda')
        >>> g = torch.rand(B, T, HV, K, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, HV, dtype=torch.bfloat16, device='cuda')
        >>> h0 = torch.randn(B, HV, K, V, dtype=torch.bfloat16, device='cuda')
        >>> A_log = torch.randn(HV, dtype=torch.float32, device='cuda')
        >>> dt_bias = torch.randn(HV * K, dtype=torch.float32, device='cuda')
        >>> o, ht = chunk_kda(
            q, k, v, g, beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_kda(
            q, k, v, g, beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if 'transpose_state_layout' in kwargs:
        if state_v_first:
            raise ValueError("Cannot pass both `state_v_first` and the deprecated `transpose_state_layout`.")
        warnings.warn(
            "`transpose_state_layout` is deprecated and renamed to `state_v_first`.",
            DeprecationWarning,
            stacklevel=2,
        )
        state_v_first = kwargs.pop('transpose_state_layout')

    if cp_context is not None:
        assert initial_state is None, "Initial state is not supported for CP"
        assert output_final_state is False, "Output final state is not supported for CP"
        assert cp_context.cu_seqlens is not None, "cu_seqlens is required for CP"
        # Override cu_seqlens and cu_seqlens_cpu with the ones from the context
        cu_seqlens = cp_context.cu_seqlens
        if cp_context.cu_seqlens_cpu is not None:
            cu_seqlens_cpu = cp_context.cu_seqlens_cpu

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        A_log, dt_bias = kwargs.get("A_log"), kwargs.get("dt_bias")
        if A_log is None and lower_bound is None:
            raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True` and `lower_bound` is not set.")

    chunk_size = kwargs.pop("chunk_size", 64)
    if chunk_size not in (32, 64):
        raise ValueError(f"`chunk_size` must be either 32 or 64 for KDA, got {chunk_size}.")

    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")

    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")

    # Validate head dimensions for GVA
    B, T, H, K, HV = *q.shape, v.shape[2]
    assert q.shape == k.shape, f"q and k must have the same shape, got q={q.shape} vs k={k.shape}"
    assert K <= 256, f"Currently we only support key headdim <=256 for KDA, got {K}."
    assert HV % H == 0, (
        f"For GVA, num_v_heads (HV={HV}) must be evenly divisible by num_qk_heads (H={H}), "
        f"but got HV % H = {HV % H}"
    )
    assert g.shape == (B, T, HV, K), f"g must have shape [B, T, HV, K]={[B, T, HV, K]}, got {list(g.shape)}"
    assert beta.shape == (B, T, HV), f"beta must have shape [B, T, HV]={[B, T, HV]}, got {list(beta.shape)}"

    if scale is None:
        scale = K ** -0.5
    return ChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
        safe_gate,
        lower_bound,
        chunk_size,
        disable_recompute,
        return_intermediate_states,
        cp_context,
        use_graph,
        max_num_seqs,
    )
