"""recurrent_kda 的 ATK executor（历史 A5 简化版）。

只支持旧的简单 spec：固定 `[B,T]`/`[B*T]` 均匀分段、无 `lengths/padding/indices`、
g 固定为 step log gate（无 A_log/dt_bias）、无 `qkv_view/strided_state`。
`atk_recurrent_kda_a6.json` 等结构矩阵包请使用 `executor_recurrent_kda.py` 执行。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _case_spec,
    _finite_tuple,
    _int_tensor,
    _kda_gate,
    _marker_device,
    _rand,
    _randn,
    _zeros,
)


OP_NAME = "recurrent_kda"


def build_inputs(spec, device, high_precision=False):
    calc_dtype = torch.float64 if high_precision else torch.bfloat16
    seed = int(spec.get("seed", 20260817))
    B, T, H, HV, K, V = (int(spec[x]) for x in ("B", "T", "H", "HV", "K", "V"))
    layout = str(spec.get("layout", "BSND"))

    cpu = torch.device("cpu")

    # TND: 把 batch 维展平到 token 维，q/k=[B*T,H,K]，v=[B*T,HV,V]
    # BSND: 保持 [B,T,H,K] / [B,T,HV,V]
    if layout.upper() == "TND":
        shape_bt = (B * T,)
    else:
        shape_bt = (B, T)

    inputs = {
        "q": _randn((*shape_bt, H, K), "bf16", calc_dtype, cpu, seed + 1),
        "k": _randn((*shape_bt, H, K), "bf16", calc_dtype, cpu, seed + 2),
        "v": _randn((*shape_bt, HV, V), "bf16", calc_dtype, cpu, seed + 3),
        "g": _kda_gate((*shape_bt, HV, K), "fp32",
                       torch.float64 if high_precision else torch.float32,
                       cpu, seed + 4),
        "beta": _rand((*shape_bt, HV), "fp32",
                      torch.float64 if high_precision else torch.float32,
                      cpu, seed + 5, 0.1, 0.9),
        # state pool：V-first [seq_num,HV,V,K] / K-first [seq_num,HV,K,V]，必须跟随 state_v_first
        "initial_state": _zeros(
            (B, HV) + ((V, K) if bool(spec.get("state_v_first", True)) else (K, V)),
            "fp32",
            torch.float64 if high_precision else torch.float32,
            cpu),
        "cu_seqlens": _int_tensor([i * T for i in range(B + 1)], cpu, torch.int64),
        "scale": float(spec.get("scale", 1.0 / math.sqrt(K))),
        "layout": layout,
    }

    if device.type != "cpu":
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
    return inputs


# CPU reference copied from the operator PTA reference so this executor is self-contained.
def _flatten_bsnd(x: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "TND":
        return x
    if layout != "BSND":
        raise ValueError("layout must be BSND or TND")
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def _restore_layout(x: torch.Tensor, ref: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "TND":
        return x
    return x.reshape(ref.shape)


def _seq_ranges(total_tokens: int, cu_seqlens: Sequence[int]):
    if len(cu_seqlens) < 2:
        raise ValueError("cu_seqlens must contain at least two cumulative offsets")
    offsets = [int(offset) for offset in cu_seqlens]
    if offsets[0] != 0:
        raise ValueError("cu_seqlens must start at zero")
    if any(end < start for start, end in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")
    if offsets[-1] != total_tokens:
        raise ValueError("the last cu_seqlens offset must equal the packed token count")
    return list(zip(offsets, offsets[1:]))


def _state_slot(ssm_state_indices: torch.Tensor, seq_idx: int, start: int, token: int) -> int:
    if ssm_state_indices.ndim == 1:
        return int(ssm_state_indices[token].item())
    if ssm_state_indices.ndim == 2:
        return int(ssm_state_indices[seq_idx, token - start].item())
    raise ValueError("ssm_state_indices must be packed [T] or speculative [seq_num,max_step]")


def recurrent_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    *,
    cu_seqlens: Optional[Sequence[int]] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    layout: str = "BSND",
    scale: Optional[float] = None,
    output_final_state: bool = True,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    state_v_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del output_final_state, inplace_final_state

    work_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q_flat = _flatten_bsnd(q, layout).to(work_dtype)
    k_flat = _flatten_bsnd(k, layout).to(work_dtype)
    v_flat = _flatten_bsnd(v, layout).to(work_dtype)
    g_flat = _flatten_bsnd(g, layout).to(work_dtype)
    beta_flat = _flatten_bsnd(beta, layout).to(work_dtype)
    total_tokens, h, dk = q_flat.shape
    _, hv, dv = v_flat.shape
    scale = (dk ** -0.5) if scale is None else scale

    if use_qk_l2norm_in_kernel:
        q_flat = q_flat / torch.sqrt((q_flat * q_flat).sum(dim=-1, keepdim=True) + 1e-6)
        k_flat = k_flat / torch.sqrt((k_flat * k_flat).sum(dim=-1, keepdim=True) + 1e-6)
    q_flat = q_flat * scale

    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("A_log is required when use_gate_in_kernel=True")
        gate = g_flat
        if dt_bias is not None:
            gate = gate + dt_bias.to(work_dtype).reshape(hv, dk).unsqueeze(0)
        exp_a = torch.exp(A_log.to(work_dtype)).reshape(1, hv, 1)
        if safe_gate:
            gate = lower_bound * torch.sigmoid(exp_a * gate)
        else:
            gate = -exp_a * F.softplus(gate)
    else:
        gate = g_flat
    gate_decay = torch.exp(gate.to(work_dtype))

    beta_eff = beta_flat
    if use_beta_sigmoid_in_kernel:
        beta_eff = torch.sigmoid(beta_eff)
        if allow_neg_eigval:
            beta_eff = beta_eff * 2.0

    if cu_seqlens is None:
        if layout.upper() == "BSND":
            dense_seq_len = q.shape[1]
            cu_seqlens = [i * dense_seq_len for i in range(q.shape[0] + 1)]
        else:
            cu_seqlens = [0, total_tokens]
    ranges = _seq_ranges(total_tokens, cu_seqlens)
    state_dtype = initial_state.dtype if initial_state is not None else work_dtype
    if initial_state is None:
        state = torch.zeros((len(ranges), hv, dv, dk), dtype=work_dtype, device=q.device)
    else:
        state = initial_state.to(work_dtype).clone()
        if not state_v_first:
            state = state.transpose(-1, -2).contiguous()
    out_flat = torch.zeros_like(v_flat, dtype=work_dtype)

    for seq_idx, (start, end) in enumerate(ranges):
        if start == end:
            continue
        state_slot = seq_idx
        if ssm_state_indices is not None:
            token = start
            if num_accepted_tokens is not None:
                token = start + int(num_accepted_tokens[seq_idx].item()) - 1
            state_slot = _state_slot(ssm_state_indices, seq_idx, start, token)
        for hv_idx in range(hv):
            h_idx = hv_idx // (hv // h)
            state_cur = state[state_slot, hv_idx].clone()
            for token in range(start, end):
                state_cur = state_cur * gate_decay[token, hv_idx].unsqueeze(0)
                delta = v_flat[token, hv_idx] - torch.mv(state_cur, k_flat[token, h_idx])
                delta = delta * beta_eff[token, hv_idx]
                state_cur = state_cur + torch.outer(delta, k_flat[token, h_idx])
                out_flat[token, hv_idx] = torch.mv(state_cur, q_flat[token, h_idx])
                out_slot = _state_slot(ssm_state_indices, seq_idx, start, token) if ssm_state_indices is not None else seq_idx
                state[out_slot, hv_idx] = state_cur

    if not state_v_first:
        state = state.transpose(-1, -2).contiguous()
    return _restore_layout(out_flat.to(q.dtype), v, layout), state.to(state_dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """Run the CPU reference at original or fp64 precision.

    无论内部计算精度如何，输出 dtype 始终与 NPU 对齐：
      - out   -> bfloat16
      - state -> float32
    这样 NPU、benchmark、golden 三者的输出 dtype 完全一致，
    ATK 的 dtype 校验不会拦，数值精度差异会体现在比对结果里。
    """
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    out, state = recurrent_kda_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["initial_state"],
        cu_seqlens=inputs["cu_seqlens"],
        layout=inputs["layout"],
        scale=inputs["scale"],
        output_final_state=True,
        inplace_final_state=False,
        state_v_first=bool(spec.get("state_v_first", True)),
    )
    return out.to(torch.bfloat16), state.to(torch.float32)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.recurrent_kda(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["initial_state"],
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=None,
        layout=inputs["layout"],
        scale=inputs["scale"],
        output_final_state=True,
        inplace_final_state=False,
        state_v_first=bool(spec.get("state_v_first", True)),
    )


@register("executor_recurrent_kda")
class FunctionApi(BaseApi):
    """ATK 执行入口，支持单标杆与双标杆。

    角色约定：
      - NPU/pyaclnn：原始精度，输出 bf16 out + fp32 state。
      - CPU benchmark（is_benchmark_task=True）：原始精度，输出 dtype 与 NPU 完全对齐。
      - CPU golden（is_benchmark_task=False）：fp64 高精度，作为真值参考。
      - 单标杆（无 is_benchmark_task）：CPU 默认走高精度。
    """

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        
        if self.device in {"npu", "pyaclnn"}:
            self.high_precision = False
            self.is_golden = False
            return

        if self.device != "cpu":
            self.high_precision = False
            self.is_golden = False
            return

        is_benchmark = getattr(task_result, "is_benchmark_task", None)
        if is_benchmark is None:
            # 单标杆：CPU 走高精度
            self.high_precision = True
            self.is_golden = True
        else:
            # 双标杆：benchmark 原始精度，golden 高精度
            self.high_precision = not bool(is_benchmark)
            self.is_golden = bool(is_benchmark)
        print(f"[KDA] device={self.device} "
            f"is_benchmark_task={getattr(task_result, 'is_benchmark_task', None)} "
            f"high_precision={self.high_precision}")

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)

        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
            return _finite_tuple(outputs, golden=False)

        if self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
            # 关键：golden=False，不让 _finite_tuple 改 dtype；
            # 角色区分交给 ATK 的 is_benchmark_task，不再靠 golden 参数。
            return _finite_tuple(outputs, golden=False)

        raise RuntimeError(
            f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}"
        )