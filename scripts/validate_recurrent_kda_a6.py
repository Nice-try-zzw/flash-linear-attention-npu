#!/usr/bin/env python3
"""Standalone CPU/NPU precision runner for recurrent_kda on A6.

This entry point deliberately does not import ATK or call acl APIs directly.
It invokes the public ``fla_npu.ops.ascendc.recurrent_kda`` Python entry and
compares its outputs with an independent FP64 PyTorch reference.

Typical use on the A6 camodel host:

    python scripts/validate_recurrent_kda_a6.py --suite core --device 0

The process must already have CANN, torch_npu, camodel and fla_npu configured.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F


DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass(frozen=True)
class Case:
    name: str
    suites: tuple[str, ...]
    layout: str = "BSND"
    batch: int = 1
    tokens: int = 2
    heads: int = 1
    value_heads: int = 1
    value_dim: int = 128
    gate_dtype: str = "fp32"
    beta_dtype: str = "fp32"
    state_dtype: str = "fp32"
    state_v_first: bool = True
    use_cu_seqlens: bool = True
    lengths: tuple[int, ...] | None = None
    raw_gate: bool = False
    safe_gate: bool = False
    normalize_qk: bool = False
    beta_sigmoid: bool = False
    allow_neg_eigval: bool = False
    inplace: bool = False
    output_final_state: bool = True
    initial_none: bool = False
    indices: str | None = None
    indices_dtype: str = "int64"
    accepted: int | None = None
    strided: bool = False
    repeat: int = 1


CASES: tuple[Case, ...] = (
    Case("smoke_bsnd_fp32", ("smoke", "core", "full")),
    Case(
        "smoke_bsnd_fp32_inplace",
        ("core", "full"),
        output_final_state=True,
        inplace=True,
    ),
    Case(
        "dense_no_cu",
        ("core", "full"),
        batch=2,
        tokens=3,
        use_cu_seqlens=False,
    ),
    Case(
        "raw_gate_gva_bf16",
        ("smoke", "core", "full"),
        tokens=4,
        heads=2,
        value_heads=4,
        gate_dtype="bf16",
        beta_dtype="bf16",
        raw_gate=True,
        normalize_qk=True,
        beta_sigmoid=True,
        allow_neg_eigval=True,
    ),
    Case(
        "safe_gate_fp16_v256",
        ("core", "full"),
        batch=2,
        tokens=2,
        value_heads=2,
        value_dim=256,
        gate_dtype="fp16",
        beta_dtype="fp16",
        raw_gate=True,
        safe_gate=True,
        normalize_qk=True,
        beta_sigmoid=True,
    ),
    Case(
        "kfirst_bf16_state",
        ("core", "full"),
        tokens=4,
        heads=2,
        value_heads=4,
        state_dtype="bf16",
        state_v_first=False,
    ),
    Case(
        "varlen_empty_sequences",
        ("core", "full"),
        layout="TND",
        tokens=17,
        heads=2,
        value_heads=4,
        lengths=(1, 0, 7, 0, 9),
        normalize_qk=True,
        gate_dtype="bf16",
    ),
    Case(
        "packed_indices_repeat",
        ("core", "full"),
        tokens=8,
        heads=2,
        value_heads=4,
        indices="packed_repeat",
        accepted=8,
        inplace=True,
        state_dtype="bf16",
    ),
    Case(
        "noncontiguous_qkv_state",
        ("core", "full"),
        tokens=4,
        heads=2,
        value_heads=4,
        strided=True,
        normalize_qk=True,
        raw_gate=True,
        gate_dtype="fp16",
        beta_dtype="bf16",
    ),
    Case(
        "speculative_indices_int32",
        ("full",),
        batch=2,
        tokens=4,
        heads=1,
        value_heads=2,
        indices="spec_unique",
        indices_dtype="int32",
        accepted=3,
    ),
    Case(
        "no_final_state_non_inplace",
        ("full",),
        tokens=2,
        output_final_state=False,
        normalize_qk=True,
    ),
    Case(
        "initial_state_none",
        ("full",),
        tokens=3,
        heads=2,
        value_heads=4,
        initial_none=True,
        normalize_qk=True,
        raw_gate=True,
        safe_gate=True,
        gate_dtype="bf16",
    ),
    Case(
        "determinism_gva",
        ("full",),
        batch=2,
        tokens=8,
        heads=2,
        value_heads=4,
        gate_dtype="bf16",
        beta_sigmoid=True,
        repeat=2,
    ),
)


def _flatten_layout(x: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "TND":
        return x
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def _restore_layout(x: torch.Tensor, reference: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "TND":
        return x
    return x.reshape(reference.shape)


def _sequence_ranges(total_tokens: int, offsets: Sequence[int]) -> list[tuple[int, int]]:
    values = [int(value) for value in offsets]
    if len(values) < 2 or values[0] != 0 or values[-1] != total_tokens:
        raise ValueError(f"invalid cu_seqlens {values} for {total_tokens} tokens")
    if any(end < start for start, end in zip(values, values[1:])):
        raise ValueError(f"cu_seqlens must be nondecreasing: {values}")
    return list(zip(values, values[1:]))


def _state_slot(
    indices: torch.Tensor | None,
    sequence: int,
    start: int,
    token: int,
) -> int:
    if indices is None:
        return sequence
    if indices.ndim == 1:
        return int(indices[token].item())
    if indices.ndim == 2:
        return int(indices[sequence, token - start].item())
    raise ValueError(f"ssm_state_indices must be rank 1 or 2, got {indices.shape}")


def recurrent_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    cu_seqlens: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    layout: str,
    state_v_first: bool,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
    safe_gate: bool,
    lower_bound: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent FP64 recurrence matching the public operator semantics."""

    q_flat = _flatten_layout(q, layout).to(torch.float64)
    k_flat = _flatten_layout(k, layout).to(torch.float64)
    v_flat = _flatten_layout(v, layout).to(torch.float64)
    g_flat = _flatten_layout(g, layout).to(torch.float64)
    beta_flat = _flatten_layout(beta, layout).to(torch.float64)
    total_tokens, query_heads, key_dim = q_flat.shape
    _, value_heads, value_dim = v_flat.shape

    if use_qk_l2norm_in_kernel:
        q_norm = torch.sqrt(torch.sum(q_flat * q_flat, dim=-1, keepdim=True))
        k_norm = torch.sqrt(torch.sum(k_flat * k_flat, dim=-1, keepdim=True))
        q_flat = q_flat / q_norm.clamp_min(torch.finfo(torch.float64).tiny)
        k_flat = k_flat / k_norm.clamp_min(torch.finfo(torch.float64).tiny)
    q_flat = q_flat * scale

    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("A_log is required for raw-gate mode")
        gate = g_flat
        if dt_bias is not None:
            gate = gate + dt_bias.to(torch.float64).reshape(value_heads, key_dim).unsqueeze(0)
        exp_a = torch.exp(A_log.to(torch.float64)).reshape(1, value_heads, 1)
        gate = lower_bound * torch.sigmoid(exp_a * gate) if safe_gate else -exp_a * F.softplus(gate)
    else:
        gate = g_flat
    decay = torch.exp(gate)

    beta_eff = beta_flat
    if use_beta_sigmoid_in_kernel:
        beta_eff = torch.sigmoid(beta_eff)
        if allow_neg_eigval:
            beta_eff *= 2.0

    if cu_seqlens is None:
        if layout == "BSND":
            values = [index * q.shape[1] for index in range(q.shape[0] + 1)]
        else:
            values = [0, total_tokens]
    else:
        values = [int(item) for item in cu_seqlens.tolist()]
    ranges = _sequence_ranges(total_tokens, values)

    if initial_state is None:
        state = torch.zeros(
            (len(ranges), value_heads, value_dim, key_dim),
            dtype=torch.float64,
            device=q.device,
        )
        output_state_dtype = torch.float32
    else:
        state = initial_state.to(torch.float64).clone()
        output_state_dtype = initial_state.dtype
        if not state_v_first:
            state = state.transpose(-1, -2).contiguous()

    out_flat = torch.zeros_like(v_flat, dtype=torch.float64)
    for sequence, (start, end) in enumerate(ranges):
        if start == end:
            continue
        current_slot = _state_slot(ssm_state_indices, sequence, start, start)
        if num_accepted_tokens is not None:
            accepted = int(num_accepted_tokens[sequence].item())
            current_slot = _state_slot(ssm_state_indices, sequence, start, start + accepted - 1)
        for head in range(value_heads):
            query_head = head // (value_heads // query_heads)
            current = state[current_slot, head].clone()
            for token in range(start, end):
                current *= decay[token, head].unsqueeze(0)
                delta = v_flat[token, head] - torch.mv(current, k_flat[token, query_head])
                delta *= beta_eff[token, head]
                current += torch.outer(delta, k_flat[token, query_head])
                out_flat[token, head] = torch.mv(current, q_flat[token, query_head])
                output_slot = _state_slot(ssm_state_indices, sequence, start, token)
                state[output_slot, head] = current

    if not state_v_first:
        state = state.transpose(-1, -2).contiguous()
    out = _restore_layout(out_flat.to(v.dtype), v, layout)
    return out, state.to(output_state_dtype)


def _make_offsets(case: Case) -> tuple[int, ...] | None:
    if not case.use_cu_seqlens:
        return None
    if case.lengths is not None:
        offsets = [0]
        for length in case.lengths:
            offsets.append(offsets[-1] + length)
        return tuple(offsets)
    return tuple(index * case.tokens for index in range(case.batch + 1))


def _make_indices(case: Case, offsets: tuple[int, ...], max_length: int) -> tuple[torch.Tensor | None, int]:
    if case.indices is None:
        return None, len(offsets) - 1
    dtype = torch.int32 if case.indices_dtype == "int32" else torch.int64
    lengths = [end - start for start, end in zip(offsets, offsets[1:])]
    if case.indices == "packed_repeat":
        values = [token % 3 for token in range(offsets[-1])]
        return torch.tensor(values, dtype=dtype), 3
    if case.indices == "spec_unique":
        indices = torch.zeros((len(lengths), max_length), dtype=dtype)
        cursor = 0
        for sequence, length in enumerate(lengths):
            for token in range(length):
                indices[sequence, token] = cursor + token
            cursor += max(length, 1)
        return indices, cursor
    raise ValueError(f"unknown indices mode {case.indices}")


def build_cpu_inputs(case: Case) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(20261021)
    normal = lambda shape, scale: torch.randn(shape, generator=generator) * scale
    offsets = _make_offsets(case)
    if case.lengths is not None:
        capacity = sum(case.lengths)
        prefix_shape = (capacity,) if case.layout == "TND" else (1, capacity)
    else:
        capacity = case.batch * case.tokens
        prefix_shape = (capacity,) if case.layout == "TND" else (case.batch, case.tokens)
    max_length = max(
        (end - start for start, end in zip(offsets, offsets[1:])),
        default=case.tokens,
    ) if offsets is not None else case.tokens

    q = normal((*prefix_shape, case.heads, 128), 0.08).to(torch.bfloat16)
    k = normal((*prefix_shape, case.heads, 128), 0.035).to(torch.bfloat16)
    v = normal((*prefix_shape, case.value_heads, case.value_dim), 0.2).to(torch.bfloat16)
    gate = normal((*prefix_shape, case.value_heads, 128), 0.3) if case.raw_gate else (
        -torch.rand((*prefix_shape, case.value_heads, 128), generator=generator) * 0.04
    )
    beta = normal((*prefix_shape, case.value_heads), 0.6) if case.beta_sigmoid else (
        0.1 + torch.rand((*prefix_shape, case.value_heads), generator=generator) * 0.6
    )

    indices = None
    state_capacity = (
        (case.batch if case.layout == "BSND" else 1)
        if offsets is None
        else len(offsets) - 1
    )
    accepted = None
    if offsets is not None:
        indices, state_capacity = _make_indices(case, offsets, max_length)
        if indices is not None:
            lengths = [end - start for start, end in zip(offsets, offsets[1:])]
            accepted = torch.tensor(
                [min(case.accepted or length, length) if length else 0 for length in lengths],
                dtype=indices.dtype,
            )

    tail = (case.value_dim, 128) if case.state_v_first else (128, case.value_dim)
    initial_state = normal((state_capacity, case.value_heads, *tail), 0.02).to(
        DTYPES[case.state_dtype]
    )
    inputs: dict[str, Any] = {
        "q": q,
        "k": k,
        "v": v,
        "g": gate.to(DTYPES[case.gate_dtype]),
        "beta": beta.to(DTYPES[case.beta_dtype]),
        "initial_state": None if case.initial_none else initial_state,
        "cu_seqlens": None if offsets is None else torch.tensor(offsets, dtype=torch.int64),
        "ssm_state_indices": indices,
        "num_accepted_tokens": accepted,
        "A_log": None,
        "dt_bias": None,
    }
    if case.raw_gate:
        inputs["A_log"] = normal((case.value_heads,), 0.1) - 2.0
        inputs["dt_bias"] = normal((case.value_heads, 128), 0.05)
    return inputs


def _padded_view(x: torch.Tensor) -> torch.Tensor:
    storage = torch.zeros((*x.shape[:-1], x.shape[-1] * 2), dtype=x.dtype, device=x.device)
    view = storage[..., : x.shape[-1]]
    view.copy_(x)
    return view


def _head_gapped_state(x: torch.Tensor) -> torch.Tensor:
    storage = torch.zeros((x.shape[0], x.shape[1] + 1, *x.shape[2:]), dtype=x.dtype, device=x.device)
    view = storage[:, : x.shape[1]]
    view.copy_(x)
    return view


def materialize_npu_inputs(inputs: dict[str, Any], case: Case, device: torch.device) -> dict[str, Any]:
    materialized = {
        name: value.to(device).clone() if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }
    if case.strided:
        for name in ("q", "k", "v"):
            materialized[name] = _padded_view(materialized[name])
        if materialized["initial_state"] is not None:
            materialized["initial_state"] = _head_gapped_state(materialized["initial_state"])
    return materialized


def call_kwargs(case: Case) -> dict[str, Any]:
    return {
        "layout": case.layout,
        "scale": 128.0 ** -0.5,
        "output_final_state": case.output_final_state,
        "inplace_final_state": case.inplace,
        "use_qk_l2norm_in_kernel": case.normalize_qk,
        "use_gate_in_kernel": case.raw_gate,
        "use_beta_sigmoid_in_kernel": case.beta_sigmoid,
        "allow_neg_eigval": case.allow_neg_eigval,
        "safe_gate": case.safe_gate,
        "lower_bound": -5.0,
        "state_v_first": case.state_v_first,
    }


def run_reference(case: Case, inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    reference_kwargs = call_kwargs(case)
    reference_kwargs.pop("output_final_state")
    reference_kwargs.pop("inplace_final_state")
    return recurrent_kda_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["initial_state"],
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=inputs["ssm_state_indices"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        num_accepted_tokens=inputs["num_accepted_tokens"],
        **reference_kwargs,
    )


def run_npu_once(case: Case, device: torch.device) -> dict[str, Any]:
    from fla_npu.ops.ascendc import recurrent_kda
    import torch_npu  # noqa: F401

    cpu_inputs = build_cpu_inputs(case)
    npu_inputs = materialize_npu_inputs(cpu_inputs, case, device)
    before = (
        npu_inputs["initial_state"].clone()
        if npu_inputs["initial_state"] is not None
        else None
    )
    outputs = recurrent_kda(
        npu_inputs["q"],
        npu_inputs["k"],
        npu_inputs["v"],
        npu_inputs["g"],
        npu_inputs["beta"],
        npu_inputs["initial_state"],
        cu_seqlens=npu_inputs["cu_seqlens"],
        ssm_state_indices=npu_inputs["ssm_state_indices"],
        A_log=npu_inputs["A_log"],
        dt_bias=npu_inputs["dt_bias"],
        num_accepted_tokens=npu_inputs["num_accepted_tokens"],
        **call_kwargs(case),
    )
    torch.npu.synchronize()
    out, final_state = outputs
    after = npu_inputs["initial_state"]
    return {
        "out": out,
        "final_state": final_state,
        "input_state_after": after,
        "input_state_before": before,
    }


def tensor_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "passed": False,
            "shape_actual": list(actual.shape),
            "shape_expected": list(expected.shape),
            "reason": "shape mismatch",
        }
    actual_fp64 = actual.detach().float().cpu().to(torch.float64)
    expected_fp64 = expected.detach().float().cpu().to(torch.float64)
    finite = bool(torch.isfinite(actual_fp64).all().item() and torch.isfinite(expected_fp64).all().item())
    if not finite:
        return {"passed": False, "finite": False, "reason": "non-finite value"}

    difference = actual_fp64 - expected_fp64
    absolute = difference.abs()
    expected_abs = expected_fp64.abs()
    actual_norm = torch.linalg.vector_norm(actual_fp64)
    expected_norm = torch.linalg.vector_norm(expected_fp64)
    difference_norm = torch.linalg.vector_norm(difference)
    if float(actual_norm) == 0.0 and float(expected_norm) == 0.0:
        cosine = 1.0 if float(absolute.max()) == 0.0 else 0.0
    else:
        cosine = float(
            torch.sum(actual_fp64 * expected_fp64)
            / (actual_norm * expected_norm).clamp_min(torch.finfo(torch.float64).tiny)
        )
    relative_l2 = float(difference_norm / expected_norm.clamp_min(torch.finfo(torch.float64).tiny))
    max_abs = float(absolute.max()) if absolute.numel() else 0.0
    mean_abs = float(absolute.mean()) if absolute.numel() else 0.0
    rmse = float(torch.sqrt(torch.mean(difference * difference))) if difference.numel() else 0.0
    tolerance = atol + rtol * expected_abs
    matched_ratio = float(torch.mean((absolute <= tolerance).to(torch.float64))) if absolute.numel() else 1.0
    flat_index = int(absolute.argmax()) if absolute.numel() else 0
    max_index = []
    cursor = flat_index
    for size in reversed(absolute.shape):
        max_index.append(cursor % size)
        cursor //= size
    max_index = list(reversed(max_index))

    passed = (
        finite
        and cosine >= 0.999
        and relative_l2 <= 0.01
        and max_abs <= 0.1
        and matched_ratio >= 0.99
    )
    return {
        "passed": passed,
        "finite": finite,
        "cosine": cosine,
        "relative_l2": relative_l2,
        "actual_norm": float(actual_norm),
        "expected_norm": float(expected_norm),
        "difference_norm": float(difference_norm),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "matched_ratio": matched_ratio,
        "rtol": rtol,
        "atol": atol,
        "max_index": max_index,
        "actual_at_max": float(actual_fp64.flatten()[flat_index]) if actual_fp64.numel() else 0.0,
        "expected_at_max": float(expected_fp64.flatten()[flat_index]) if expected_fp64.numel() else 0.0,
    }


def compare_case(case: Case, reference: tuple[torch.Tensor, torch.Tensor], runs: list[dict[str, Any]]) -> dict[str, Any]:
    actual = runs[0]
    output_metrics = tensor_metrics(actual["out"], reference[0], rtol=1e-2, atol=2e-2)
    output_metrics["name"] = "out"

    state_metrics = None
    if case.output_final_state:
        if actual["final_state"] is None:
            state_metrics = {"name": "final_state", "passed": False, "reason": "missing final_state"}
        else:
            state_metrics = tensor_metrics(
                actual["final_state"],
                reference[1],
                rtol=1e-2,
                atol=2e-2,
            )
            state_metrics["name"] = "final_state"
    elif case.inplace:
        state_metrics = tensor_metrics(
            actual["input_state_after"],
            reference[1],
            rtol=1e-2,
            atol=2e-2,
        )
        state_metrics["name"] = "inplace_state"

    mutation_ok = True
    if not case.inplace and actual["input_state_before"] is not None:
        mutation_ok = bool(torch.equal(actual["input_state_after"], actual["input_state_before"]))

    repeat_ok = True
    if len(runs) > 1:
        repeat_ok = bool(torch.equal(runs[0]["out"], runs[1]["out"]))
        if case.output_final_state and runs[0]["final_state"] is not None:
            repeat_ok = repeat_ok and bool(torch.equal(runs[0]["final_state"], runs[1]["final_state"]))

    metrics = [output_metrics]
    if state_metrics is not None:
        metrics.append(state_metrics)
    passed = all(item.get("passed", False) for item in metrics) and mutation_ok and repeat_ok
    return {
        "name": case.name,
        "passed": passed,
        "mutation_contract_ok": mutation_ok,
        "deterministic": repeat_ok,
        "outputs": metrics,
    }


def _print_metrics(result: dict[str, Any]) -> None:
    for item in result["outputs"]:
        if "cosine" not in item:
            print(f"  [FAIL] {item['name']}: {item.get('reason', 'invalid metric')}")
            continue
        status = "PASS" if item["passed"] else "FAIL"
        print(
            f"  [{status}] {item['name']}: cosine={item['cosine']:.9f} "
            f"rel_l2={item['relative_l2']:.6e} max_abs={item['max_abs']:.6e} "
            f"matched={item['matched_ratio']:.6f} "
            f"norms={item['actual_norm']:.6e}/{item['expected_norm']:.6e} "
            f"max_idx={item['max_index']}"
        )
    print(
        f"  mutation_contract={'PASS' if result['mutation_contract_ok'] else 'FAIL'} "
        f"deterministic={'PASS' if result['deterministic'] else 'FAIL'}"
    )


def select_cases(args: argparse.Namespace) -> list[Case]:
    selected = [case for case in CASES if args.suite in case.suites]
    if args.case:
        wanted = set(args.case)
        selected = [case for case in selected if case.name in wanted]
        missing = wanted - {case.name for case in selected}
        if missing:
            raise SystemExit(f"unknown or suite-excluded cases: {sorted(missing)}")
    return selected


def cpu_self_check(cases: list[Case]) -> int:
    failures = 0
    comparator_checked = False
    for case in cases:
        try:
            inputs = build_cpu_inputs(case)
            golden = run_reference(case, inputs)
            result = compare_case(
                case,
                golden,
                [{
                    "out": golden[0],
                    "final_state": golden[1] if case.output_final_state else None,
                    "input_state_before": inputs["initial_state"],
                    "input_state_after": inputs["initial_state"],
                }],
            )
            if not all(item.get("passed", False) for item in result["outputs"]):
                raise AssertionError("golden does not compare equal to itself")
            if not comparator_checked and golden[0].numel():
                bad = golden[0].clone()
                bad.fill_(float(bad.float().max()) + 1.0)
                if tensor_metrics(golden[0], bad, rtol=1e-2, atol=2e-2)["passed"]:
                    raise AssertionError("metric comparator accepted a grossly wrong output")
                comparator_checked = True
            print(f"[CPU-SELF-CHECK PASS] {case.name}")
        except Exception as error:
            failures += 1
            print(f"[CPU-SELF-CHECK FAIL] {case.name}: {error}", file=sys.stderr)
    if comparator_checked and failures == 0:
        print("[METRIC-SELF-CHECK PASS] gross mismatch rejected")
    return 0 if failures == 0 and comparator_checked else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "core", "full"), default="core")
    parser.add_argument("--case", action="append", help="run only the named case; may be repeated")
    parser.add_argument("--device", default="0", help="torch_npu device index or npu:<index>")
    parser.add_argument("--json", type=Path, help="write machine-readable metrics")
    parser.add_argument("--list", action="store_true", help="list selected cases and exit")
    parser.add_argument(
        "--cpu-self-check",
        action="store_true",
        help="validate case generation/reference/metrics without importing torch_npu or fla_npu",
    )
    args = parser.parse_args()
    cases = select_cases(args)
    if args.list:
        for case in cases:
            print(f"{case.name} suites={','.join(case.suites)} layout={case.layout} T={case.tokens}")
        return 0
    if args.cpu_self_check:
        return cpu_self_check(cases)

    import torch_npu

    device_text = args.device if ":" in args.device else f"npu:{args.device}"
    device = torch.device(device_text)
    torch_npu.npu.set_device(device)
    print(f"DEVICE={device} CASES={len(cases)}")

    results = []
    for case in cases:
        print(f"[CASE] {case.name}")
        try:
            cpu_inputs = build_cpu_inputs(case)
            reference = run_reference(case, cpu_inputs)
            runs = [run_npu_once(case, device) for _ in range(case.repeat)]
            result = compare_case(case, reference, runs)
        except Exception as error:
            result = {"name": case.name, "passed": False, "error": repr(error), "outputs": []}
        results.append(result)
        _print_metrics(result)
        print(f"[{'PASS' if result['passed'] else 'FAIL'}] {case.name}")

    summary = {
        "soc": "Ascend960DT_968B7",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "device": str(device),
        "camodel": {
            "enabled": os.getenv("FLA_NPU_CAMODEL_MODE") == "1",
            "sim_dir": os.getenv("FLA_NPU_CAMODEL_SIM_DIR"),
            "shim": os.getenv("FLA_NPU_CAMODEL_SHIM"),
        },
        "fla_npu": _package_provenance(),
        "backend": _backend_provenance(),
        "criteria": {
            "cosine_min": 0.999,
            "relative_l2_max": 0.01,
            "max_abs_max": 0.1,
            "matched_ratio_min": 0.99,
            "rtol": 0.01,
            "atol": 0.02,
        },
        "cases": results,
        "passed": all(item["passed"] for item in results),
    }
    print(
        f"RESULT: {'PASS' if summary['passed'] else 'FAIL'} "
        f"{sum(item['passed'] for item in results)}/{len(results)} cases"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON={args.json.resolve()}")
    return 0 if summary["passed"] else 1


def _package_provenance() -> dict[str, Any]:
    import fla_npu

    return {
        "version": getattr(fla_npu, "__version__", "unknown"),
        "file": getattr(fla_npu, "__file__", None),
        "stable_abi_env": os.getenv("FLA_NPU_STABLE_ABI"),
    }


def _backend_provenance() -> dict[str, Any]:
    from fla_npu.ops import ascendc

    name = "npu_recurrent_kda"
    return {
        "selected": ascendc.BACKENDS.get(name),
        "fallbacks": ascendc.FALLBACKS.get(name, 0),
    }


if __name__ == "__main__":
    raise SystemExit(main())
