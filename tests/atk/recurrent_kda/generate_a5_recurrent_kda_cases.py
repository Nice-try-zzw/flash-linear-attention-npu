#!/usr/bin/env python3
"""生成 A5 平台 recurrent_kda 泛化 ATK 用例。

A5 约束（见 docs/当前限制）：
  - 每段序列长度必须 <= 8。
  - K 只能 128，V 只能 128 或 256。
  - HV 必须能被 H 整除，H/HV <= 256。
  - layout 只支持 BSND / TND。

本脚本按上述约束生成 ATK JSON 用例，覆盖：
  - 每段长度 T ∈ [1..8]（含边界 1 和 8）
  - 段数 B（多段 packed，cu_seqlens 均匀分段）
  - layout ∈ {BSND, TND}
  - H/HV 组合（HV 能被 H 整除）
  - V ∈ {128, 256}
  - 多 seed

注意：当前 executor_recurrent_kda.build_inputs 固定 initial_state 为
V-first 布局 (B, HV, V, K)，因此默认只生成 state_v_first=True 的用例。
若需要覆盖 state_v_first=False，请先扩展 build_inputs，使其在
state_v_first=False 时生成 (B, HV, K, V) 布局，并把
STATE_V_FIRST 改为 [True, False]。

用法：
  python generate_a5_recurrent_kda_cases.py \
      --template atk_recurrent_kda.json \
      --output atk_recurrent_kda_a5.json \
      --max-cases 200
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

K = 128
SOC = "ascend910b"
ROUTE = "ascendc"
API_TYPE = "executor_recurrent_kda"
A5_MAX_SEG = 8

# 覆盖维度（可按需扩展）
T_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]          # 每段长度边界
B_VALUES = [1, 2, 3]                         # 段数
LAYOUTS = ["BSND", "TND"]
STATE_V_FIRST = [True]                       # 见文件头说明
H_HV = [(1, 1), (2, 2), (2, 4), (4, 4), (2, 8)]
V_VALUES = [128, 256]
SEEDS = [20260817, 20260818, 20260819]


def iter_specs():
    """生成泛化 spec 组合，每段长度 = T <= A5_MAX_SEG。"""
    # 1) 每段长度边界：单段遍历所有 T
    for T in T_VALUES:
        yield dict(T=T, B=1, H=2, HV=4, V=128, layout="BSND", state_v_first=True)

    # 2) 段数变化：多段 packed
    for B in B_VALUES:
        for T in [1, 4, 8]:
            yield dict(T=T, B=B, H=2, HV=4, V=128, layout="BSND", state_v_first=True)

    # 3) layout 变化
    for layout in LAYOUTS:
        yield dict(T=4, B=1, H=2, HV=4, V=128, layout=layout, state_v_first=True)

    # 4) H/HV 组合（HV 能被 H 整除）
    for H, HV in H_HV:
        yield dict(T=4, B=1, H=H, HV=HV, V=128, layout="BSND", state_v_first=True)

    # 5) V 变化
    for V in V_VALUES:
        yield dict(T=4, B=1, H=2, HV=4, V=V, layout="BSND", state_v_first=True)

    # 6) TND + 多段 + V=256 组合
    for layout in LAYOUTS:
        for B in [2, 3]:
            yield dict(T=4, B=B, H=2, HV=4, V=256, layout=layout, state_v_first=False
                       ) if False else None  # 占位，避免误生成；如需 v_first=False 打开上方 STATE_V_FIRST
    for layout in LAYOUTS:
        for B in [2, 3]:
            yield dict(T=4, B=B, H=2, HV=4, V=256, layout=layout, state_v_first=True)


def check(spec):
    if spec["T"] > A5_MAX_SEG:
        raise ValueError(f"A5 约束：每段长度 {spec['T']} > {A5_MAX_SEG}")
    if spec["HV"] % spec["H"] != 0:
        raise ValueError(f"HV={spec['HV']} 不能被 H={spec['H']} 整除")
    if spec["H"] == 0 or spec["HV"] // spec["H"] > 256:
        raise ValueError(f"H/HV 比例超限：H={spec['H']}, HV={spec['HV']}")
    if spec["V"] not in (128, 256):
        raise ValueError(f"A5 只支持 V=128 或 256，当前 {spec['V']}")
    return True


def set_attr(case, name, value):
    for inp in case["inputs"]:
        if inp["name"] == name:
            inp["range_values"] = value
            return
    raise KeyError(f"template 缺少 input: {name}")


def build_case(template, case_id, spec, seed):
    case = copy.deepcopy(template)
    full = dict(
        spec, K=K, seed=seed, case_id=case_id,
        name=f"a5_T{spec['T']}_B{spec['B']}_H{spec['H']}_HV{spec['HV']}_V{spec['V']}_{spec['layout']}",
        dtype="bf16", op="recurrent_kda", route=ROUTE, soc=SOC,
    )
    case["id"] = case_id
    case["default_seed"] = seed
    case["name"] = f"recurrent_kda_a5_{case_id:04d}_T{spec['T']}_B{spec['B']}_{spec['layout']}"
    case["api_type"] = API_TYPE
    set_attr(case, "case_spec", json.dumps(full, ensure_ascii=False))
    set_attr(case, "dtype", "bf16")
    set_attr(case, "B", spec["B"])
    set_attr(case, "T", spec["T"])
    set_attr(case, "H", spec["H"])
    set_attr(case, "HV", spec["HV"])
    set_attr(case, "K", K)
    set_attr(case, "V", spec["V"])
    set_attr(case, "layout", spec["layout"])
    set_attr(case, "state_v_first", spec["state_v_first"])
    set_attr(case, "case_id", case_id)
    set_attr(case, "seed", seed)
    set_attr(case, "soc", SOC)
    set_attr(case, "route", ROUTE)
    return case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="atk_recurrent_kda.json")
    parser.add_argument("--output", default="atk_recurrent_kda_a5.json")
    parser.add_argument("--max-cases", type=int, default=200)
    args = parser.parse_args()

    template = json.loads(Path(args.template).read_text(encoding="utf-8"))[0]

    cases, seen = [], set()
    for spec in iter_specs():
        if spec is None:
            continue
        check(spec)
        key = (spec["T"], spec["B"], spec["H"], spec["HV"], spec["V"],
               spec["layout"], spec["state_v_first"])
        if key in seen:
            continue
        seen.add(key)
        for seed in SEEDS:
            if len(cases) >= args.max_cases:
                break
            cases.append(build_case(template, len(cases), spec, seed))
        if len(cases) >= args.max_cases:
            break

    Path(args.output).write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"A5_CASES_GENERATED {len(cases)} -> {args.output}")
    print(f"unique_structs={len(seen)}  A5_MAX_SEG={A5_MAX_SEG}")


if __name__ == "__main__":
    main()