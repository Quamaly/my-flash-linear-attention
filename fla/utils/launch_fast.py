# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side launch replay for triton kernels (small-T host-wall fix).

Why: on triton-ascend 3.2.0 a ``kernel[grid](...)`` call costs ~190us of host
time (JIT binder + cache-key string build + autotune/heuristics wrapper
layers + device/stream queries), measured on the chunk_kda t2k workload where
the whole bench iteration is host-bound (device sum ~1.26ms vs ~3.0ms wall,
~7 launches per forward iteration => ~1.4ms/iter of pure python launch
overhead).

What: for every exact call signature (kernel identity, grid, scalar values,
tensor dtype and 16B-alignment class) the first call goes through the
standard triton path while a one-shot ``pre_run_hook`` records the resolved
(post-autotune/heuristics) argument mapping and the binder recomputes the
exact JIT cache key, identifying the CompiledKernel deterministically.
Subsequent calls with the same signature replay through the low-level
launcher (``CompiledKernel.run``) directly after cheap equivalence checks:

- grid values and every scalar argument match exactly (they are in the
  signature key; tensor extents are fully described by these scalars plus
  dtype, since the wrappers derive them from the same shapes);
- tensor arguments keep dtype and their ``data_ptr() % 16`` class
  (triton bakes 16-byte divisibility into the specialization key);
- the current device matches and no triton launch hooks are installed
  (profilers keep the fully instrumented standard path).

Because the signature key pins every scalar by value, all non-tensor
runtime arguments of a plan are constants: the plan stores a pre-assembled
argument template with the scalar values already embedded and only the
tensor slots are refilled from the current call (wrapper pass-through is
verified object-identical at capture time; wrapper-injected scalars from
heuristics/autotune are deterministic functions of the keyed values, so
they are embedded as constants too). Replay is therefore a list copy plus
one direct fill per tensor -- no per-launch name->value dict rebuild.
The previous per-tensor alignment re-check is subsumed by the key
comparison itself: the objects whose alignment would be re-checked are the
very objects whose (dtype, align) signature was just hashed into the key.

Any mismatch or error falls back to the standard triton call for that
invocation; signatures whose compiled kernel cannot be identified fall back
permanently (plan stored as ``None``). The replayed binary is the one the
standard path selects for the same arguments, so numerics are unchanged by
construction. The standard ``used_global_vals`` mutation check is not
replayed: kernels whose module globals change after first launch must not
use this helper (none do here).

Disable everything with ``FLA_FAST_LAUNCH=0``.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import torch
from triton.compiler.compiler import CompiledKernel
from triton.runtime import driver
from triton.runtime.jit import JITFunction

_ENABLED = os.environ.get('FLA_FAST_LAUNCH', '1').strip().lower() not in ('0', 'false', 'off')

# Fast raw-C stream/device queries (torch_npu); fall back to the triton
# driver round trips when unavailable.
try:
    import torch_npu
    _raw_stream = torch_npu._C._npu_getCurrentRawStream
    _cur_device = torch_npu._C._npu_getDevice
except Exception:  # pragma: no cover - non-NPU platforms
    _raw_stream = None
    _cur_device = None


def _current_stream(device: int):
    if _raw_stream is not None:
        return _raw_stream(device)
    return driver.active.get_current_stream(device)


def _current_device() -> int:
    if _cur_device is not None:
        return _cur_device()
    return driver.active.get_current_device()


def _unwrap_jit(kernel: Any) -> JITFunction | None:
    """Peel autotune/heuristics wrappers down to the JITFunction."""
    for _ in range(4):
        if isinstance(kernel, JITFunction):
            return kernel
        kernel = getattr(kernel, 'fn', None)
        if kernel is None:
            return None
    return None


def _sig1(v: Any):
    """Cheap hashable signature of one call argument.

    Tensors contribute dtype + 16B-alignment class only: their extent is
    fully described by the scalar arguments the wrappers pass alongside
    (T, BH, task_num, ...), which are keyed exactly.
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, torch.Tensor):
        # alignment class is baked into the triton specialization key
        return (v.dtype, v.data_ptr() & 15)
    return ('raw', repr(v))


# tensor dtypes -> small type code; ``code << 4 | align`` packs the whole
# tensor signature into one small int (cheap to build, cheap to hash)
_dtype_codes: dict[Any, int] = {}


def _sigtuple(vals) -> tuple:
    """Inline signature loop for the per-launch key (hot path).

    Same semantics as ``tuple(map(_sig1, vals))`` for the argument shapes
    that occur in practice, minus one python call per argument; exact
    ``torch.Tensor`` instances (the overwhelmingly common case) encode as
    ``dtype_code << 4 | (data_ptr() & 15)`` instead of a nested tuple.
    """
    out = []
    ap = out.append
    for v in vals:
        c = v.__class__
        if c is torch.Tensor:
            code = _dtype_codes.get(v.dtype)
            if code is None:
                code = _dtype_codes[v.dtype] = (len(_dtype_codes) + 1) << 4
            ap(code | (v.data_ptr() & 15))
        elif c is int or c is float or c is str or c is bool or v is None:
            ap(v)
        else:
            ap(_sig1(v))
    return tuple(out)


def _grid3(grid):
    # fast path: plain int tuples (what every launch site here passes)
    if type(grid) is tuple:
        n = len(grid)
        if n == 1:
            g0 = grid[0]
            if type(g0) is int:
                return (g0, 1, 1)
        elif n == 2:
            g0, g1 = grid
            if type(g0) is int and type(g1) is int:
                return (g0, g1, 1)
        elif n == 3:
            g0, g1, g2 = grid
            if type(g0) is int and type(g1) is int and type(g2) is int:
                return grid
    try:
        if callable(grid):
            return None
        g0 = int(grid[0])
        g1 = int(grid[1]) if len(grid) > 1 else 1
        g2 = int(grid[2]) if len(grid) > 2 else 1
        return (g0, g1, g2)
    except Exception:
        return None


class _Plan:
    __slots__ = ('device', 'jit', 'krun', 'kfunc', 'kmeta', 'g0', 'g1', 'g2',
                 'tpl', 'tsrc_a', 'tsrc_k')

    def __init__(self, device, jit, kernel_ck, grid, tpl, tsrc_a, tsrc_k):
        self.device = device
        self.jit = jit
        # raw C launch function, skipping NPULauncher.__call__'s per-call
        # flag checks (both flags are evaluated once here; when either is
        # set the plan is not built at all)
        self.krun = kernel_ck.__dict__['run'].launch
        self.kfunc = kernel_ck.__dict__['function']
        self.kmeta = kernel_ck.__dict__['packed_metadata']
        self.g0, self.g1, self.g2 = grid
        # pre-assembled runtime-argument template: scalar values embedded
        # (pinned exactly by the signature key), tensor slots set to None
        self.tpl = tpl
        # tensor refill slots: (index_in_run_args, arg_index) for positional
        # tensors and (index_in_run_args, param_name) for keyword tensors
        self.tsrc_a = tsrc_a
        self.tsrc_k = tsrc_k


_plans: dict[Any, _Plan | None] = {}
_replay_hits = 0
_MISSING = object()
_build_lock = threading.Lock()
_capture_resolved: tuple | None = None


def _on_capture(*cargs, **ckwargs):
    """One-shot pre_run_hook: record the resolved (post-wrapper) call args."""
    global _capture_resolved
    _capture_resolved = (cargs, ckwargs)


_SCALARS = (bool, int, float, str)


def _try_build_plan(jit: JITFunction, device: int, g3, user_args, user_kwargs) -> _Plan | None:
    """Locate the CompiledKernel the just-finished standard call selected."""
    if _capture_resolved is None or getattr(jit, "binder", None) is None:
        return None
    resolved_args, resolved_kwargs = _capture_resolved
    try:
        # wrapper pass-through: positional args must map index-to-index and
        # every user kwarg must survive unwrapped into the resolved kwargs
        # (heuristics/autotune only ADD deterministic entries)
        if len(resolved_args) != len(user_args):
            return None
        for a, b in zip(resolved_args, user_args):
            if a is not b:
                return None
        for k, v in user_kwargs.items():
            if resolved_kwargs.get(k, _MISSING) is not v:
                return None
        bound_args, sig_and_spec, constexpr_vals, _nonconst, excess_kwargs = jit.binder(
            *resolved_args, **resolved_kwargs
        )
        key = ''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
        kernel_ck = jit.cache[device].get(key)
        if kernel_ck is None:
            return None
        if hasattr(kernel_ck, 'result'):
            kernel_ck = kernel_ck.result()
        if not isinstance(kernel_ck, CompiledKernel):
            return None
        kernel_ck._init_handles()
        launcher_obj = kernel_ck.__dict__.get('run')
        # the raw-C bypass is only valid when the launcher wrapper's
        # per-call flags are off (compile-only mode / msprof registration);
        # otherwise keep the full NPULauncher path
        if not (
            hasattr(launcher_obj, 'launch')
            and not getattr(launcher_obj, 'compile_only', False)
            and not getattr(launcher_obj, 'enable_msprof_register_tensor', False)
        ):
            return None
        param_names = [p.name for p in jit.params]
        pos_of = {n: i for i, n in enumerate(param_names)}
        merged = dict(zip(param_names, resolved_args))
        merged.update(resolved_kwargs)
        runtime_names = [p.name for p in jit.params if not p.is_constexpr]
        if not all(n in merged for n in runtime_names):
            return None
        # build the replay template: constants for every non-tensor runtime
        # argument (values pinned by the signature key), refill slots for
        # tensors (objects re-supplied -- and re-signatured -- every call)
        n_pos = len(user_args)
        tpl: list = []
        tsrc_a: list = []
        tsrc_k: list = []
        for ri, n in enumerate(runtime_names):
            val = merged[n]
            if pos_of[n] < n_pos:
                # bound by the caller positionally (pass-through verified)
                if isinstance(val, torch.Tensor):
                    tsrc_a.append((ri, pos_of[n]))
                    tpl.append(None)
                elif val is None or isinstance(val, _SCALARS):
                    tpl.append(val)
                else:
                    return None
            elif user_kwargs.get(n, _MISSING) is val:
                # bound by the caller as a keyword argument
                if isinstance(val, torch.Tensor):
                    tsrc_k.append((ri, n))
                    tpl.append(None)
                elif val is None or isinstance(val, _SCALARS):
                    tpl.append(val)
                else:
                    return None
            elif val is None or isinstance(val, _SCALARS):
                # wrapper-injected scalar (heuristics/autotune): a
                # deterministic function of the keyed values -> constant
                tpl.append(val)
            else:
                # injected tensor / exotic runtime object: not replayable
                return None
        return _Plan(device, jit, kernel_ck, g3, tpl, tuple(tsrc_a), tuple(tsrc_k))
    except Exception:
        return None


def fast_launch(kernel, grid, /, *args, **kwargs):
    """Drop-in replacement for ``kernel[grid](*args, **kwargs)``.

    Replays the cached CompiledKernel for a repeated exact call signature;
    the first call and any anomaly take the standard triton path (see the
    module docstring for the exact safety checks).
    """
    if not _ENABLED:
        return kernel[grid](*args, **kwargs)

    g3 = _grid3(grid)
    if g3 is None:
        return kernel[grid](*args, **kwargs)

    # values-only signature: the plan pins (binary, grid) only — replay
    # refills only the tensor slots from the CURRENT call, so names are not
    # needed in the key (dict order is stable per call site; a rare order
    # change just builds a second equivalent plan).
    key = (
        id(kernel), g3,
        _sigtuple(args),
        _sigtuple(kwargs.values()),
    )
    plan = _plans.get(key, _MISSING)
    if plan is _MISSING:
        with _build_lock:
            plan = _plans.get(key, _MISSING)
            if plan is _MISSING:
                # runs this invocation exactly once through the standard
                # triton path, then captures the plan (or None = permanent
                # fallback for this signature)
                _plans[key] = _build_and_run(kernel, grid, g3, *args, **kwargs)
                return None
        # another thread built the plan mid-flight: replay below
    if plan is None:
        return kernel[grid](*args, **kwargs)

    try:
        if CompiledKernel.launch_enter_hook is not None or CompiledKernel.launch_exit_hook is not None:
            return kernel[grid](*args, **kwargs)
        if _current_device() != plan.device:
            return kernel[grid](*args, **kwargs)
        # template replay: scalar constants are pinned by the key (which
        # also verified every tensor's dtype + 16B alignment during the
        # lookup), so only the tensor slots are refilled -- no name->value
        # dict rebuild and no second data_ptr() pass per launch
        run_args = plan.tpl.copy()
        for ri, p in plan.tsrc_a:
            run_args[ri] = args[p]
        for ri, n in plan.tsrc_k:
            run_args[ri] = kwargs[n]
        plan.krun(plan.g0, plan.g1, plan.g2, _current_stream(plan.device),
                  plan.kfunc, plan.kmeta, None, None, None, *run_args)
        global _replay_hits
        _replay_hits += 1
        return None
    except Exception:
        _plans[key] = None
        return kernel[grid](*args, **kwargs)


def _build_and_run(kernel, grid, g3, *args, **kwargs) -> _Plan | None:
    """One standard triton call with a one-shot capture hook installed."""
    global _capture_resolved
    jit = _unwrap_jit(kernel)
    hook_installed = False
    _capture_resolved = None
    if jit is not None:
        try:
            jit.pre_run_hooks.append(_on_capture)
            hook_installed = True
        except Exception:
            hook_installed = False
    try:
        kernel[grid](*args, **kwargs)
    finally:
        if hook_installed:
            try:
                jit.pre_run_hooks.remove(_on_capture)
            except ValueError:
                pass
    if hook_installed and _capture_resolved is not None:
        try:
            device = _current_device()
        except Exception:
            device = None
        if device is not None:
            return _try_build_plan(jit, device, g3, args, kwargs)
    return None


# ---------------------------------------------------------------------------
# Cached device properties (num_aicore etc.) — launch helpers used to query
# these through python round trips on every call.
# ---------------------------------------------------------------------------
_props_cache: dict[int, dict] = {}


def device_props(device: int | None = None) -> dict:
    """Cached ``driver.active.utils.get_device_properties``."""
    try:
        if device is None:
            device = torch.npu.current_device() if torch.npu.is_available() else 0
        props = _props_cache.get(device)
        if props is None:
            props = driver.active.utils.get_device_properties(device)
            _props_cache[device] = props
        return props
    except Exception:
        return driver.active.utils.get_device_properties(device if device is not None else 0)


def num_aicore(device: int | None = None) -> int:
    try:
        return int(device_props(device)['num_aicore'])
    except Exception:
        return 24
