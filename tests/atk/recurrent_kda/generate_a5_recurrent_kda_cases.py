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
                   state_dtype="bf16" if i % 2 else "fp32", norm=(i % 2 == 1))

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
    gate_cases = [
        dict(norm=True, raw_gate=True, safe_gate=True, beta_sigmoid=True, negative=True,
             gate_dtype="fp32", beta_dtype="fp32"),
        dict(norm=True, raw_gate=True, safe_gate=False, beta_sigmoid=True, negative=False,
             gate_dtype="bf16", beta_dtype="fp16"),
        dict(norm=False, raw_gate=True, safe_gate=False, beta_sigmoid=False, negative=False,
             gate_dtype="fp16", beta_dtype="bf16"),
        dict(norm=True, raw_gate=False, beta_sigmoid=True, gate_dtype="fp32", beta_dtype="bf16"),
        dict(norm=False, raw_gate=False, gate_dtype="bf16", beta_dtype="fp32"),
        dict(norm=True, raw_gate=True, safe_gate=True, beta_sigmoid=False,
             gate_dtype="fp32", beta_dtype="fp16", large_gate=True),
        dict(norm=True, raw_gate=True, beta_sigmoid=True, dt_bias_none=True,
             gate_dtype="fp32", beta_dtype="fp32"),
        dict(norm=True, raw_gate=True, safe_gate=True, gate_dtype="bf16", beta_dtype="bf16",
             pattern="small"),
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
                   state_dtype="fp32" if i % 3 else "bf16", norm=True)


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
    parser.add_argument("--output", default="atk_recurrent_kda_a6.json")
    parser.add_argument("--soc", default="ascend960dt",
                        help="ascend960dt（默认）/ascend950；A5 的 V=128 同样支持长序列")
    parser.add_argument("--max-cases", type=int, default=400)
    parser.add_argument("--seed-count", type=int, default=1,
                        help="每个结构重复的种子数；默认 1，不靠多 seed 堆用例")
    args = parser.parse_args()

    template = json.loads(Path(args.template).read_text(encoding="utf-8"))[0]

    cases, seen = [], set()
    long_v128 = 0
    family = "a5" if args.soc.endswith("950") else "a6"
    for spec in iter_specs():
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
