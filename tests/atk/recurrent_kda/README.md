# RecurrentKda ATK 工程

本目录提供 `recurrent_kda` 的 ATK 单算子工程，包含 `executor_recurrent_kda.py`、`gen_recurrent_kda.py`、`recurrent_kda.yaml`、`atk_recurrent_kda.json`。

## 输入约束

- `layout` 支持 `BSND/TND`；`BSND` 下 `q/k=[B,T,H,K]`，`v/out=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`。
- `TND` 下 `q/k=[T,H,K]`，`v/out=[T,HV,V]`，`g=[T,HV,K]`，`beta=[T,HV]`。
- `initial_state` 是 state pool；`state_v_first=true` 时为 `[state_capacity,HV,V,K]`，否则为 `[state_capacity,HV,K,V]`。
- `cu_seqlens` 可选；传入时为同设备 `INT32/INT64` tensor，shape 为 `[seq_num+1]`，首项为 `0`，末项不超过 token capacity，offset 单调不减。
- `q/k/v/out` 仅支持 `BFLOAT16`；`g/beta` 支持 `FLOAT/BFLOAT16/FLOAT16`，`A_log/dt_bias` 支持 `FLOAT`。
- `HV % H == 0`，`H/HV <= 256`；当前支持 `K=128,V=128` 或 `K=128,V=256`。
- A2/A3 支持多时间窗口长序列；A5/A6 每段不超过 8；显式 `ssm_state_indices` 可使 `state_capacity > seq_num`。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

tests/reference/recurrent_kda_reference.py; fla/ops/ascendc/kda/recurrent_kda/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_recurrent_kda.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93`、`ascend950` 和 `ascend960dt`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950|ascend960dt` 使用。

## A6（ascend960dt）适配说明

- A6 走 `op_kernel/arch92/recurrent_kda.h`（vector 三发射 + 384KB UB 的独立实现），用例侧通过
  `-soc=ascend960dt`（或 `-soc=a6`）选择。host tiling 的 `isRegbase` 同时覆盖 A5/A6，`isA6` 只在 A6 上
  加深 `state/attn` 输出 ring 并降低 `repeat`，用例不需要额外参数。
- q/k L2 归一化：A6 在 kernel 内使用 `x / sqrt(sum(x*x) + 1e-6)`，与本目录 `executor_recurrent_kda.py`
  的 CPU 标杆口径一致；A5 保持无 epsilon 实现。两者相对差异约 `1e-6 / (sum + 1e-6)`，常规随机输入约
  1e-8（`mixed_tolerance_bm` 判定不受影响），q/k 极小的边界输入差异显著——该边界由
  `scripts/test_l2norm_epsilon.py` 单独覆盖：A6 应直接通过，A5 需带 `--expect-mismatch` 作为已知基线。
- A6 每段序列长度仍受 `RKDA_MAX_MTP=8` 限制；`atk_recurrent_kda_longseq.json`、
  `atk_recurrent_kda_longseq_mss.json` 的长序列用例只覆盖 A2/A3 的多时间窗口路径，不应在 A6 上作为通过项。
- 分支覆盖：A6 的融合 VF 只对 `K=128` 生效；`V=128/V=256`、`state_v_first` 两种布局、raw/safe gate、
  beta sigmoid、`allow_neg_eigval` 与 varlen/MTP 均可复用现有短序列矩阵
  （`atk_recurrent_kda.json`、`atk_recurrent_kda_mss.json`、`atk_recurrent_kda_perf.json`）。
- 同步风险：A6 会按 384KB 余量把 `state/attn` 输出 ring 加深到 2 级（`DeepenOutRingForA6`），
  MTEB3 写回与下一轮 V 计算跨 buffer 重叠；A6 验收建议额外执行 `-scope=determinism` 与
  `-scope=mssanitizer`，覆盖双缓冲下的 V/MTEB3 同步。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 2, "T": 2, "H": 2, "HV": 4, "K": 128, "V": 128, "layout": "BSND", "state_v_first": true, "op": "recurrent_kda", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -soc=ascend960dt -npu_device_id=0 -scope=all
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。