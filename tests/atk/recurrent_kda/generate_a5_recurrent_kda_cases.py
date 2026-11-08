#!/usr/bin/env python3
"""生成 ascend960dt（A6）场景下的 recurrent_kda ATK 用例。

A6 约束与覆盖目标（见 docs/design_960dt.md）：
  - K 只能 128；V=128 走静态 UB 路径，**不限每段长度**，kernel 按 32-token（FP32 K-first 16）窗口递推。
  - V=256 走兼容路径，每段长度必须 <= 8。
  - HV 必须能被 H 整除；H/HV <= 256。
  - 存储模式由 cu_seqlens 是否存在决定：有 -> rank3 packed（layout=TND），无 -> rank4（layout=BSND）。

与旧版只做 "小维度 × 多 seed" 不同，本脚本按场景矩阵生成：
  - B（段数）覆盖 1/2/3/4/7/8/16/32/64；
  - S（每段长度）覆盖 32/16-token 窗口边界（15/16/17、31/32/33…）与长序列（最大 1024）；
  - cu_seqlens 覆盖 dense（不传）、均匀分段、ragged（含 0 长度段）、capacity > 有效 token（padding）、
    单段长序列、大量超短段；
  - 另覆盖 state 布局/dtype、gate/beta 组合、packed/speculative slot 索引、QKV view / 非连续 state、
    initial_state=None、inplace/output 关闭等分支。
  seed 只作为每个结构的固定输入种子（默认 1 个），不再用多 seed 堆数量。

用法：
    python generate_a5_recurrent_kda_cases.py --output atk_recurrent_kda_a6.json
    python generate_a5_recurrent_kda_cases.py --soc ascend950 --output atk_recurrent_kda_a5.json
    python generate_a5_recurrent_kda_cases.py --sim   # 仿真用最小精度包（T<=33）-> atk_recurrent_kda_a6_sim.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

K = 128
ROUTE = "ascendc"
API_TYPE = "executor_recurrent_kda"

# V=256 兼容路径的每段上限；V=128 静态路径不设上限（A6 X=32（FP32 K-first 16）/ A5 X=8）。
V256_MAX_SEG = 8
MAX_CAPACITY = 8192  # 单个用例的 packed token 上限，避免生成过大的输入

# S：每段长度。含 32/16-token 窗口的 ±1 边界、8-token 边界与常见长序列档位。
S_EDGES = [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33,
           63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024]
# B：段数。长序列只搭配小段数，短序列才用大段数。
B_VALUES = [1, 2, 3, 4, 7, 8, 16, 32, 64]
H_HV = [(1, 1), (2, 2), (2, 4), (3, 6), (4, 4), (1, 16)]
SEED = 20260817
SEEDS = [SEED]


def _segments_for(s: int, b: int) -> int:
    """控制 packed 规模：段长 >=256 时段数收缩，保证 B*S <= MAX_CAPACITY。"""
    if s >= 512:
        b = min(b, 2)
    elif s >= 256:
        b = min(b, 4)
    elif s >= 128:
        b = min(b, 8)
    elif s >= 64:
        b = min(b, 16)
    return max(1, b)


def iter_specs():
    """按场景矩阵产出 spec（每个 dict 一条用例，不再用 seed 扩展数量）。"""
    # 1) dense（不传 cu_seqlens）：rank4 BSND，S 全档位，B 轮换
    for i, s in enumerate(S_EDGES):
        b = _segments_for(s, [1, 2, 4, 3, 8][i % 5])
        yield dict(name=f"dense_S{s}_B{b}", T=s, B=b, H=2, HV=4, V=128,
                   layout="BSND", cu=False, state_v_first=(i % 2 == 0),
                   state_dtype="bf16" if i % 3 == 0 else "fp32")

    # 2) 均匀分段（cu_seqlens = [0, S, 2S, ...]）：rank3 packed，B 覆盖 1..64
    for i, s in enumerate(S_EDGES):
        b = _segments_for(s, [1, 2, 3, 4, 7, 8, 16, 32, 64][i % 9])
        yield dict(name=f"uniform_S{s}_B{b}", T=s, B=b, H=2, HV=4, V=128,
                   layout="TND", cu=True, state_v_first=(i % 2 == 1),
                   state_dtype="fp32" if i % 2 else "bf16")

    # 3) ragged：非等长、含空段、首尾短段
    ragged = [
        [1, 16, 33, 0, 65],
        [0, 1, 17, 0, 33, 7],
        [15, 0, 0, 16, 17],
        [9, 8, 17, 32, 1, 64],
        [129, 0, 3, 256, 0, 33],
        [1024, 0, 1, 511],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ]
    for i, lengths in enumerate(ragged):
        total = sum(lengths)
        if total > MAX_CAPACITY:
            continue
        yield dict(name=f"ragged{i}_B{len(lengths)}_cap{total}", T=max(lengths),
                   B=len(lengths), lengths=lengths, H=2, HV=4, V=128,
                   layout="TND", cu=True, state_v_first=(i % 2 == 0),
                   state_dtype="bf16" if i % 2 else "fp32", inplace=(i % 2 == 0))

    # 4) capacity > 有效 token（padding）：末项 offset < capacity
    for i, (lengths, pad) in enumerate([
        ([9, 17], 7), ([1, 8, 9], 5), ([16, 33, 1], 9),
        ([129, 17], 11), ([513, 2], 3), ([1, 1, 1, 1], 4),
    ]):
        yield dict(name=f"padding{i}_cap{sum(lengths) + pad}", T=max(lengths), B=len(lengths),
                   lengths=lengths, padding=pad, H=2, HV=4, V=128,
                   layout="TND", cu=True, state_v_first=(i % 2 == 0),
                   state_dtype="fp32" if i % 3 else "bf16", output=(i % 3 != 0))

    # 5) 单段长序列：B=1，S 到 1024
    for i, s in enumerate([17, 33, 129, 257, 513, 1024]):
        yield dict(name=f"single_S{s}", T=s, B=1, H=1, HV=1, V=128,
                   layout="TND", cu=True, state_v_first=(i % 2 == 0),
                   state_dtype="bf16" if i % 2 else "fp32", use_qk_l2norm_in_kernel=(i % 2 == 1))

    # 6) 大量超短段：B=32/64，S=1/2/3
    for i, (s, b) in enumerate([(1, 64), (2, 32), (3, 32), (1, 32), (2, 64), (4, 16)]):
        yield dict(name=f"many_S{s}_B{b}", T=s, B=b, H=2, HV=4, V=128,
                   layout="TND", cu=True, state_v_first=(i % 2 == 0),
                   state_dtype="bf16" if i % 2 else "fp32", inplace=(i % 2 == 1))

    # 7) V=256 兼容路径：每段 <= 8，dense 与 packed 各覆盖若干档
    for i, s in enumerate([1, 2, 4, 8]):
        yield dict(name=f"v256_dense_S{s}", T=s, B=[1, 2, 3, 4][i], H=2, HV=4, V=256,
                   layout="BSND", cu=False, state_v_first=(i % 2 == 0),
                   state_dtype="fp32" if i % 2 else "bf16")
        yield dict(name=f"v256_uniform_S{s}", T=s, B=[1, 2, 4, 8][i], H=2, HV=4, V=256,
                   layout="TND", cu=True, state_v_first=(i % 2 == 1),
                   state_dtype="bf16" if i % 2 else "fp32")
    yield dict(name="v256_ragged_mixed", T=8, B=5, lengths=[8, 0, 4, 1, 8],
               H=2, HV=4, V=256, layout="TND", cu=True, state_v_first=True,
               state_dtype="bf16")
    yield dict(name="v256_padding", T=8, B=3, lengths=[8, 2, 8], padding=4,
               H=2, HV=4, V=256, layout="TND", cu=True, state_v_first=False,
               state_dtype="fp32")

    # 8) state 布局 / dtype / 原位 / 无初始 state：长尾窗口
    for i, (vf, sd, inplace, initial_none) in enumerate([
        (True, "fp32", False, False),
        (True, "bf16", True, False),
        (False, "fp32", True, False),
        (False, "bf16", False, False),
        (True, "fp32", True, True),
        (False, "bf16", True, True),
    ]):
        yield dict(name=f"state_vf{int(vf)}_{sd}_ip{int(inplace)}_none{int(initial_none)}",
                   T=17, B=2, H=2, HV=4, V=128, layout="TND", cu=True,
                   state_v_first=vf, state_dtype=sd,
                   inplace=(inplace and not initial_none), initial_none=initial_none,
                   strided_state=(i % 2 == 0))

    # 9) gate / beta 组合：raw+safe+sigmoid+negative、dtype 三分支
    # 注意：executor 使用公开属性名 use_qk_l2norm_in_kernel / use_gate_in_kernel /
    # use_beta_sigmoid_in_kernel / allow_neg_eigval，以及 g_dtype / beta_dtype。
    gate_cases = [
        dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, safe_gate=True,
             use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True,
             g_dtype="fp32", beta_dtype="fp32"),
        dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, safe_gate=False,
             use_beta_sigmoid_in_kernel=True,
             g_dtype="bf16", beta_dtype="fp16"),
        dict(use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, safe_gate=False,
             use_beta_sigmoid_in_kernel=False,
             g_dtype="fp16", beta_dtype="bf16"),
        dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
             use_beta_sigmoid_in_kernel=True, g_dtype="fp32", beta_dtype="bf16"),
        dict(use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
             g_dtype="bf16", beta_dtype="fp32"),
        dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, safe_gate=True,
             use_beta_sigmoid_in_kernel=False,
             g_dtype="fp32", beta_dtype="fp16", large_gate=True),
        dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
             use_beta_sigmoid_in_kernel=True, dt_bias_none=True,
             g_dtype="fp32", beta_dtype="fp32"),
        dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, safe_gate=True,
             g_dtype="bf16", beta_dtype="bf16", pattern="small"),
    ]
    for i, extra in enumerate(gate_cases):
        yield dict(name=f"gate{i}", T=[9, 17, 33, 16, 17, 65, 17, 129][i], B=1,
                   H=2, HV=4, V=128, layout="TND", cu=True, state_v_first=(i % 2 == 0),
                   state_dtype="bf16" if i % 2 else "fp32", **extra)

    # 10) slot 索引：packed / speculative、accepted token、两种布局
    for i, (mode, cu, accepted) in enumerate([
        ("packed", True, 9), ("packed", True, 1), ("speculative", True, 9),
        ("speculative", True, 17), ("packed", False, 9), ("speculative", False, 1),
        ("packed", True, 33), ("speculative", True, 5),
    ]):
        yield dict(name=f"slots_{mode}_{i}", T=17, B=2, H=2, HV=4, V=128,
                   layout="TND" if cu else "BSND", cu=cu, indices=mode, accepted=accepted,
                   metadata_dtype="int32" if i % 2 else "int64",
                   state_v_first=(i % 2 == 0), state_dtype="bf16" if i % 3 == 0 else "fp32",
                   inplace=True, output=(i % 2 == 1))

    # 11) QKV view / 非连续输入：fused、head_gap、feature_gap、batch_gap
    for i, view in enumerate(["fused", "head_gap", "feature_gap"]):
        yield dict(name=f"view_{view}_B2", T=17, B=2, H=2, HV=4, V=128,
                   layout="TND", cu=True, qkv_view=view, strided_state=True,
                   state_v_first=(i % 2 == 0), state_dtype="fp32" if i % 2 else "bf16")
    for i, view in enumerate(["fused", "head_gap", "feature_gap", "batch_gap"]):
        yield dict(name=f"view_{view}_dense", T=9, B=2, H=2, HV=4, V=128,
                   layout="BSND", cu=False, qkv_view=view,
                   state_v_first=(i % 2 == 1), state_dtype="bf16")

    # 12) 输出/状态写回语义：output=False、首 token 单段、空段
    for i, (out_flag, inplace, lengths) in enumerate([
        (False, True, [17]), (False, False, None), (True, False, [1]),
        (True, True, [0, 17, 0]), (False, True, [16]), (True, False, [0, 1, 0, 33]),
    ]):
        spec = dict(name=f"out{int(out_flag)}_ip{int(inplace)}_{i}", T=17, B=1,
                    H=2, HV=4, V=128, layout="TND", cu=True,
                    state_v_first=(i % 2 == 0), state_dtype="fp32" if i % 2 else "bf16",
                    output=out_flag, inplace=inplace)
        if lengths is not None:
            spec.update(lengths=lengths, B=len(lengths), T=max(lengths))
        yield spec

    # 13) H/HV 比例与多 head 聚合
    for i, (h, hv) in enumerate(H_HV):
        yield dict(name=f"ratio_H{h}_HV{hv}", T=33, B=2, H=h, HV=hv, V=128,
                   layout="TND", cu=True, state_v_first=(i % 2 == 0),
                   state_dtype="fp32" if i % 3 else "bf16", use_qk_l2norm_in_kernel=True)


def iter_sim_specs():
    """仿真友好的最小精度矩阵：所有用例 T<=33、单段 <=33 token，避免仿真超时。

    覆盖原则（与全量矩阵一一对应，仅去掉大 T 档位）：
      - 窗口边界：A6 窗口 32（BF16 / FP32 V-first）用 T=33 跨界，FP32 K-first 窗口 16 用 T=17 跨界；
      - 分支：dense / uniform / ragged(含 0 段) / padding / 单头 / 超多段、V=256 兼容路径、
        state 布局×dtype×inplace×initial_none×strided、gate safe/raw/dt_bias_none/large_gate、
        slots packed/speculative × int32/int64、QKV view、output/inplace 语义、H/HV 比例。
    """

    def base(name, T, B, vf, sd, **extra):
        spec = dict(name=name, T=T, B=B, H=2, HV=4, V=128,
                    layout="TND", cu=True, state_v_first=vf, state_dtype=sd)
        spec.update(extra)
        return spec

    # 1) dense（不传 cu_seqlens）：无 ssm_state_indices 的 state 写回合并路径
    yield base("dense_S1_B2", 1, 2, True, "bf16", layout="BSND", cu=False)
    yield base("dense_S8_B2", 8, 2, False, "fp32", layout="BSND", cu=False)
    yield base("dense_S17_B2", 17, 2, False, "fp32", layout="BSND", cu=False)   # FP32 K-first 跨 16 窗口
    yield base("dense_S33_B2_vf1", 33, 2, True, "bf16", layout="BSND", cu=False)  # 跨 32 窗口
    yield base("dense_S33_B2_vf0", 33, 2, False, "bf16", layout="BSND", cu=False)  # BF16 K-first 跨 32 窗口

    # 2) 均匀分段 packed
    yield base("uniform_S9_B4", 9, 4, True, "fp32")
    yield base("uniform_S17_B3", 17, 3, False, "bf16")

    # 3) ragged（含 0 长度段 + 跨 32 窗口段）
    yield base("ragged_B5_cap51", 33, 5, True, "fp32", lengths=[0, 1, 17, 0, 33])

    # 4) padding：capacity > 有效 token，且 output 关闭
    yield base("padding_B2_cap33", 17, 2, False, "bf16", lengths=[9, 17], padding=7, output=False)

    # 5) 单头 H=HV=1 + l2norm 开/关
    yield base("single_S33_l2", 33, 1, False, "bf16", H=1, HV=1, use_qk_l2norm_in_kernel=True)
    yield base("single_S17_nol2", 17, 1, True, "fp32", H=1, HV=1, use_qk_l2norm_in_kernel=False)

    # 6) 大量超短段：多任务调度 / bank 轮换
    yield base("many_S2_B32", 2, 32, False, "bf16", inplace=True)

    # 7) V=256 兼容路径（每段 <= 8）
    yield base("v256_dense_S4", 4, 3, True, "bf16", V=256, layout="BSND", cu=False)
    yield base("v256_dense_S8", 8, 2, False, "fp32", V=256, layout="BSND", cu=False)
    yield base("v256_ragged", 8, 5, True, "bf16", V=256, lengths=[8, 0, 4, 1, 8])
    yield base("v256_padding", 8, 3, False, "fp32", V=256, lengths=[8, 2, 8], padding=4)

    # 8) state 布局 / dtype / inplace / initial_none / strided
    yield base("state_vf1_fp32_ip0_str", 17, 2, True, "fp32", inplace=False, strided_state=True)
    yield base("state_vf1_bf16_ip1", 17, 2, True, "bf16", inplace=True)
    yield base("state_vf0_fp32_ip1_str", 17, 2, False, "fp32", inplace=True, strided_state=True)
    yield base("state_vf0_bf16_none", 17, 2, False, "bf16", inplace=False, initial_none=True)

    # 9) gate / beta 分支（全部 T<=17，large_gate 用小 T）
    yield base("gate_safe_fp32", 9, 1, True, "fp32",
               use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, safe_gate=True,
               use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True,
               g_dtype="fp32", beta_dtype="fp32")
    yield base("gate_raw_bf16fp16", 17, 1, False, "bf16",
               use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, safe_gate=False,
               use_beta_sigmoid_in_kernel=True, g_dtype="bf16", beta_dtype="fp16")
    yield base("gate_raw_fp16bf16_nol2", 17, 1, True, "fp32",
               use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, safe_gate=False,
               use_beta_sigmoid_in_kernel=False, g_dtype="fp16", beta_dtype="bf16")
    yield base("gate_off_sig", 16, 1, False, "bf16",
               use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
               use_beta_sigmoid_in_kernel=True, g_dtype="fp32", beta_dtype="bf16")
    yield base("gate_off_all", 17, 1, True, "fp32",
               use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
               g_dtype="bf16", beta_dtype="fp32")
    yield base("gate_dt_bias_none", 17, 1, True, "fp32",
               use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
               use_beta_sigmoid_in_kernel=True, dt_bias_none=True,
               g_dtype="fp32", beta_dtype="fp32")
    yield base("gate_large", 9, 1, False, "bf16",
               use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, safe_gate=True,
               use_beta_sigmoid_in_kernel=False, g_dtype="fp32", beta_dtype="fp16",
               large_gate=True)

    # 10) slot 索引：逐 token 落盘路径（packed/speculative × int32/int64 × output 开关）
    yield base("slots_packed_i64", 17, 2, True, "bf16", indices="packed", accepted=9,
               metadata_dtype="int64", inplace=True, output=False)
    yield base("slots_packed_i32", 17, 2, False, "fp32", indices="packed", accepted=1,
               metadata_dtype="int32", inplace=True, output=True)
    yield base("slots_speculative_i32", 17, 2, False, "fp32", indices="speculative", accepted=17,
               metadata_dtype="int32", inplace=True, output=True)
    # dense（不传 cu_seqlens）+ indices 的组合
    yield base("slots_packed_dense_i64", 17, 2, True, "bf16", indices="packed", accepted=9,
               metadata_dtype="int64", inplace=True, output=False, layout="BSND", cu=False)

    # 11) QKV view / 非连续
    yield base("view_fused_str", 17, 2, True, "bf16", qkv_view="fused", strided_state=True)
    yield base("view_head_gap_str", 17, 2, False, "fp32", qkv_view="head_gap", strided_state=True)
    yield base("view_feature_gap_dense", 9, 2, False, "bf16", qkv_view="feature_gap",
               layout="BSND", cu=False)
    yield base("view_batch_gap_dense", 9, 2, True, "bf16", qkv_view="batch_gap",
               layout="BSND", cu=False)

    # 12) output/inplace 语义：全关、原位写回（推理 decode 常用路径）、全开 + 0 长度段
    yield base("out0_ip0", 17, 2, False, "fp32", output=False, inplace=False)
    yield base("out0_ip1", 17, 1, True, "bf16", output=False, inplace=True, lengths=[17])
    yield base("out1_ip1_zl", 17, 3, False, "bf16", output=True, inplace=True, lengths=[0, 17, 0])

    # 13) H/HV 比例（l2norm 开，T=33 跨 32 窗口）
    yield base("ratio_H1_HV1", 33, 2, True, "bf16", H=1, HV=1, use_qk_l2norm_in_kernel=True)
    yield base("ratio_H2_HV2", 33, 2, False, "bf16", H=2, HV=2, use_qk_l2norm_in_kernel=True)
    yield base("ratio_H2_HV4", 33, 2, True, "fp32", H=2, HV=4, use_qk_l2norm_in_kernel=True)
    yield base("ratio_H1_HV16", 33, 2, False, "fp32", H=1, HV=16, use_qk_l2norm_in_kernel=True)


def _segments(spec):
    """返回每段长度列表（用于 V=256 上限校验）。"""
    if "lengths" in spec:
        return list(spec["lengths"])
    return [int(spec["T"])] * int(spec["B"])


def check(spec):
    v = int(spec["V"])
    if v not in (128, 256):
        raise ValueError(f"V 只支持 128/256，当前 {v}")
    if v == 256 and max(_segments(spec)) > V256_MAX_SEG:
        raise ValueError(f"V=256 每段长度必须 <= {V256_MAX_SEG}: {_segments(spec)}")
    h, hv = int(spec["H"]), int(spec["HV"])
    if hv % h != 0 or h <= 0 or hv // h > 256:
        raise ValueError(f"H/HV 非法：H={h}, HV={hv}")
    if sum(_segments(spec)) + int(spec.get("padding", 0)) > MAX_CAPACITY:
        raise ValueError(f"capacity 超过 {MAX_CAPACITY}: {spec['name']}")
    if spec.get("padding", 0) < 0:
        raise ValueError(f"padding 必须非负: {spec['name']}")
    return True


def set_attr(case, name, value):
    for inp in case["inputs"]:
        if inp["name"] == name:
            inp["range_values"] = value
            return
    raise KeyError(f"template 缺少 input: {name}")


def build_case(template, case_id, spec, seed, soc, family="a6"):
    case = copy.deepcopy(template)
    # A5/A6 包的精度标准统一为 cv_fused_double_benchmark（与既有 a5/a6 包一致），
    # 模板来自 atk_recurrent_kda.json 的 mixed_tolerance_bm 不适用。
    case["standard"]["acc"] = {"cv_fused_double_benchmark": {
        "max_re_ratio": 5, "avg_re_ratio": 1.5,
        "root_mean_squared_ratio": 1.5, "number_count_radio": 4}}
    lengths = spec.get("lengths")
    cu = bool(spec.get("cu", lengths is not None))
    layout = "TND" if cu else "BSND"
    full = dict(spec, K=K, dtype="bf16", op="recurrent_kda", route=ROUTE, soc=soc,
                seed=seed, case_id=case_id, cu=cu, layout=layout)
    full.pop("name", None)
    name = spec.get("name", f"case{case_id}")
    full["name"] = name
    case["id"] = case_id
    case["default_seed"] = seed
    case["name"] = f"recurrent_kda_{family}_{case_id:04d}_{name}"
    case["api_type"] = API_TYPE
    set_attr(case, "case_spec", json.dumps(full, ensure_ascii=False))
    set_attr(case, "dtype", "bf16")
    set_attr(case, "B", int(spec["B"]))
    set_attr(case, "T", int(spec["T"]))
    set_attr(case, "H", int(spec["H"]))
    set_attr(case, "HV", int(spec["HV"]))
    set_attr(case, "K", K)
    set_attr(case, "V", int(spec["V"]))
    set_attr(case, "layout", layout)
    set_attr(case, "state_v_first", bool(spec.get("state_v_first", True)))
    set_attr(case, "case_id", case_id)
    set_attr(case, "seed", seed)
    set_attr(case, "soc", soc)
    set_attr(case, "route", ROUTE)
    return case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="atk_recurrent_kda.json")
    parser.add_argument("--output", default=None,
                        help="默认 atk_recurrent_kda_{a5|a6}.json；--sim 时为 atk_recurrent_kda_{a5|a6}_sim.json")
    parser.add_argument("--sim", action="store_true",
                        help="生成仿真友好的最小精度矩阵（T<=33，覆盖全部特性分支），"
                             "用于仿真器上验证精度，避免大用例超时")
    parser.add_argument("--soc", default="ascend960dt",
                        help="ascend960dt（默认）/ascend950；A5 的 V=128 同样支持长序列")
    parser.add_argument("--max-cases", type=int, default=400)
    parser.add_argument("--seed-count", type=int, default=1,
                        help="每个结构重复的种子数；默认 1，不靠多 seed 堆用例")
    args = parser.parse_args()

    family = "a5" if args.soc.endswith("950") else "a6"
    if args.output is None:
        args.output = f"atk_recurrent_kda_{family}_sim.json" if args.sim else f"atk_recurrent_kda_{family}.json"

    template = json.loads(Path(args.template).read_text(encoding="utf-8"))[0]

    cases, seen = [], set()
    long_v128 = 0
    specs = iter_sim_specs() if args.sim else iter_specs()
    for spec in specs:
        spec = dict(spec)
        check(spec)
        lengths = spec.get("lengths")
        key = (spec.get("name"), int(spec["B"]), int(spec["T"]), lengths and tuple(lengths),
               int(spec["V"]), spec.get("layout"), bool(spec.get("state_v_first", True)),
               spec.get("indices"), spec.get("qkv_view"), spec.get("padding"))
        if key in seen:
            continue
        seen.add(key)
        is_long = int(spec["V"]) == 128 and max(_segments(spec)) > 8
        for i in range(max(1, args.seed_count)):
            if len(cases) >= args.max_cases:
                break
            cases.append(build_case(template, len(cases), spec, SEED + i, args.soc, family))
            long_v128 += 1 if is_long else 0
        if len(cases) >= args.max_cases:
            break

    # 与仓内其它 ATK 包一致：一个用例一行，便于按 case 做 diff。
    text = "[\n" + ",\n".join(
        json.dumps(c, ensure_ascii=False, separators=(",", ":")) for c in cases) + "\n]\n"
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"CASES_GENERATED {len(cases)} -> {args.output}")
    print(f"unique_structs={len(seen)}  long_seq_v128={long_v128}  soc={args.soc}")


if __name__ == "__main__":
    main()
