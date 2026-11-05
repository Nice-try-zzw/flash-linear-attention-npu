# RecurrentKda ATK 工程

本目录提供 `recurrent_kda` 的 ATK 单算子工程，包含 `executor_recurrent_kda.py`、`gen_recurrent_kda.py`、
`recurrent_kda.yaml`、`atk_recurrent_kda*.json`。

## 输入约束

- 存储模式由 `cu_seqlens` 是否传入决定，与 `layout` 属性无关：
  - 不传时为 rank4，`q/k=[B,T,H,K]`、`v/out=[B,T,HV,V]`、`g=[B,T,HV,K]`、`beta=[B,T,HV]`；
  - 传入时为 rank3 packed，`q/k=[capacity,H,K]`、`v/out=[capacity,HV,V]`、`g=[capacity,HV,K]`、`beta=[capacity,HV]`。
  - 执行器按 `cu_seqlens` 是否存在推导传给算子的 `layout`（packed→`TND`、rank4→`BSND`），
    用例 spec 里的 `layout` 仅作兼容参数，避免 rank 与 layout 不一致被校验拒绝。
- `initial_state` 是 state pool；`state_v_first=true` 时为 `[state_capacity,HV,V,K]`，否则为 `[state_capacity,HV,K,V]`。
- `cu_seqlens` 可选；传入时为同设备 `INT32/INT64` tensor，shape 为 `[seq_num+1]`，首项为 `0`，末项不超过 token capacity，offset 单调不减。
- `q/k/v/out` 仅支持 `BFLOAT16`；`g/beta` 支持 `FLOAT/BFLOAT16/FLOAT16`，state 支持 `FP32/BF16`，`A_log/dt_bias` 支持 `FLOAT`。
- `HV % H == 0`；`K=128`，`V=128` 或 `V=256`。
- A2/A3 支持多时间窗口长序列；**A5/A6 的 V=128 静态路径同样不限制每段长度**（A5 按 8-token、A6 按 16-token
  窗口递推，状态驻留 UB），`V=256` 在 A5/A6 上仍走兼容路径，每段长度必须 <=8。
- 显式 `ssm_state_indices` 可使 `state_capacity > seq_num`；Q/K/V 支持独立的合法 batch stride，
  不满足直接搬运条件时 aclnn 层回退连续化。

## 标杆来源

CPU recurrence 对齐 [PR #917](https://github.com/flashserve/flash-linear-attention-npu/pull/917)
提交 `082b0c704407d0b38909fd4b0f58703b8ab52873`。同一原始 dtype 量化输入提升 FP64 后
计算，最终 golden 按 ATK 规则转 FP32；L2 使用 `x/sqrt(sum(x*x)+1e-6)`。
标准沿用 `mixed_tolerance_bm`。完整 state pool 参与比较，仅排除 API 未定义的 out padding tail。

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_recurrent_kda.py` 中实现；
公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93`、`ascend950` 和 `ascend960dt`，可配合统一脚本的
`-soc=ascend910b|ascend910_93|ascend950|ascend960dt` 使用。用例 spec 内的 `soc` 只用于 CPU 标杆的
state 存储口径，实际平台由 `-soc` 指定。

## 用例包

| 包 | 数量 | 覆盖 |
| --- | --- | --- |
| `atk_recurrent_kda.json` | 44 结构 × 3 种子 = 132 | 小 shape、8 种融合开关、长序列、布局、dtype、state、slot 与边界 |
| `atk_recurrent_kda_perf.json` | 36 | B=32/64/72，HK=16，HV=32/48/128，T=1/2/4/8 |
| `atk_recurrent_kda_mss.json` | 11 | short/window/tail/transpose/slots/packed/fallback/独立 batch stride |
| `atk_recurrent_kda_alignment.json` | 54 | 方案对齐用例 |
| `atk_recurrent_kda_longseq.json` / `_mss.json` | 独立包 | `executor_recurrent_kda_longseq.py` 使用，覆盖每段 9…1024 |
| `atk_recurrent_kda_a6.json` | 结构矩阵生成的 A6 场景包 | `generate_a5_recurrent_kda_cases.py` 生成：B=1…64、S=1…1024（覆盖 16-token 窗口边界）、dense/均匀/ragged/padding/单段长序列/大量超短段，以及 state 布局、gate 分支、slot 索引、QKV view 等 |
| `atk_recurrent_kda_a5.json` | 历史 A5 短序列包 | `executor_recurrent_kda_a5.py` 使用 |

`python tests/atk/recurrent_kda/gen_recurrent_kda.py` 可重建正式包与新增对齐包；
`python tests/atk/recurrent_kda/generate_a5_recurrent_kda_cases.py --soc ascend960dt`
按结构矩阵重建 A6 场景包（默认输出 `atk_recurrent_kda_a6.json`，`--soc ascend950` 可生成 A5 版本）。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 2, "T": 2, "H": 2, "HV": 4, "K": 128, "V": 128, "layout": "BSND", "state_v_first": true, "op": "recurrent_kda", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -soc=ascend950 -npu_device_id=6 -scope=all
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -soc=ascend960dt -npu_device_id=0 -scope=all
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，
生成器会把不支持 FP16 的算子改回合法 BF16 用例。

## A5（ascend950）适配说明

- A5 的 V=128 走 `op_kernel/arch35/` 静态 UB 实现（Mutex 资源锁 + 逐 token VF）：窗口 8 token、
  3 个 State bank、248KB 可用 UB；V=256 走 `arch35/recurrent_kda_legacy.h`，每段 <=8。
- q/k L2 归一化使用 `x/sqrt(sum(x*x)+1e-6)`，与 A2/A3、A6 一致；`scripts/test_l2norm_epsilon.py`
  在 A5 上应直接通过，不再需要 `--expect-mismatch`。

## A6（ascend960dt）适配说明

- A6 走 `op_kernel/arch92/`（与 A5 同一套算法，按架构常量特化）：384KB UB 下窗口 16、State bank 4；
  用例侧通过 `-soc=ascend960dt`（或 `-soc=a6`）选择，不需要额外参数。
- A6 的 V=128 路径支持超过 `RKDA_MAX_MTP=8` 的长序列（16-token 窗口递推、状态驻留 UB），
  因此 `atk_recurrent_kda_longseq.json`、`atk_recurrent_kda_longseq_mss.json` 中以 `V=128` 为目标的长序列
  用例可在 A6 上执行；`V=256` 的长序列仍只覆盖 A2/A3 的多时间窗口路径，不应在 A6 上作为通过项。
- 分支覆盖：A5/A6 的静态 VF 只对 `K=128` 生效；`V=128/V=256`、`state_v_first` 两种布局、raw/safe gate、
  beta sigmoid、`allow_neg_eigval` 与 varlen/MTP 均可复用现有短序列矩阵
  （`atk_recurrent_kda.json`、`atk_recurrent_kda_mss.json`、`atk_recurrent_kda_perf.json`）。
- 同步风险：A5/A6 静态路径用 Mutex 资源锁替代 SetFlag/WaitFlag，State bank（A5 3 份 / A6 4 份）与
  输入/out 双槽跨 MTE2/V/MTE3 重叠；验收建议额外执行 `-scope=determinism` 与 `-scope=mssanitizer`，
  覆盖末 token 初态预取、逐 token 快照在飞与双槽输入输出的同步。
