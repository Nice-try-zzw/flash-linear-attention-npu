# RecurrentKda ATK 工程

本目录提供 `recurrent_kda` 的 ATK 单算子工程，包含 `executor_recurrent_kda.py`、`gen_recurrent_kda.py`、
`recurrent_kda.yaml`、精度包 `atk_recurrent_kda.json`、性能包 `atk_recurrent_kda_perf.json`、
内存/确定性包 `atk_recurrent_kda_mss.json`，以及结构矩阵用例包 `atk_recurrent_kda_a6_sim.json`
（由 `generate_a5_recurrent_kda_cases.py` 生成）。`atk_recurrent_kda_longseq*.json` 是独立的
长序列手工包，使用 `executor_recurrent_kda_longseq.py`，不由 `run_test_cpu.sh` 直接调度。

## 输入约束

- 存储模式由 `case_spec` 的 `cu` 决定（与 `layout` 属性无关）：`cu=false` 时不传 `cu_seqlens`，
  输入为 rank4，`q/k=[B,T,H,K]`，`v/out=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`；
  `cu=true` 时传入 `cu_seqlens`，输入为 rank3 packed，`q/k=[capacity,H,K]`，`v/out=[capacity,HV,V]`，
  `g=[capacity,HV,K]`，`beta=[capacity,HV]`。执行器按 `cu` 推导下发给算子的 `layout`（packed→`TND`、
  rank4→`BSND`），用例 spec 里的 `layout` 仅作兼容参数，避免 rank 与 layout 不一致被校验拒绝。
- `initial_state` 是 state pool；`state_v_first=true` 时为 `[state_capacity,HV,V,K]`，否则为 `[state_capacity,HV,K,V]`。
- `cu_seqlens` 可选；传入时为同设备 `INT32/INT64` tensor，shape 为 `[seq_num+1]`，首项为 `0`，末项不超过 token capacity，offset 单调不减。
- `q/k/v/out` 仅支持 `BFLOAT16`；`g/beta` 支持 `FLOAT/BFLOAT16/FLOAT16`，state 支持 `FP32/BF16`，`A_log/dt_bias` 支持 `FLOAT`。
- `HV % H == 0`，`H/HV <= 256`；当前支持 `K=128,V=128` 或 `K=128,V=256`。
- A2/A3 支持多时间窗口长序列；A5 每段仍不超过 8；A6 的 `V=128` 与 `V=256` 都由 arch92 静态 UB
  引擎按 token 递推、不限制每段长度（`V=256` 在引擎内把 State 的 V 维按 128 切片串行处理）；
  显式 `ssm_state_indices` 可使 `state_capacity > seq_num`。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

tests/reference/recurrent_kda_reference.py; fla/ops/ascendc/kda/recurrent_kda/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_recurrent_kda.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93`、`ascend950` 和 `ascend960dt`。统一脚本 `run_test_cpu.sh`
的 `-soc` 只接受 `ascend910b|ascend910_93|ascend950`（用于 GM 初始化判定），不包含 `ascend960dt`；
A6 用例需在本目录按后文的 `atk node ...` 命令直接执行。`gen_recurrent_kda.py` 生成的用例不在
`case_spec` 中写死 `soc`，CPU 标杆按平台无关口径计算；结构矩阵包（`atk_recurrent_kda_a6_sim.json`）
与 `_mss.json` 的 A6 用例写入 `soc=ascend960dt`，该字段只影响 CPU 标杆的 state 存储口径，
实际平台由运行的 NPU 决定。

## 默认用例

- BF16 长序列用例（默认，覆盖 `T > TOKEN_TILE(32)`、多 tile、bank 轮换、FP32 state），固定 3 个种子
  （case id 0/1/2，seed 20260817/20260818/20260819）：
  `{"dtype": "bf16", "B": 2, "T": 48, "H": 2, "HV": 4, "K": 128, "V": 128, "layout": "BSND", "cu": false, "state_v_first": true, "op": "recurrent_kda", "case_id": 0, "seed": 20260817, "route": "ascendc"}`
- 备用小用例（`PROFILES` 第二项替换为默认档时使用）：`{"B": 2, "T": 2, "H": 2, "HV": 4, "K": 128, "V": 128, "layout": "BSND", "cu": false, "state_v_first": true}`

注意：`-scope=gen_cases` 只生成默认档的单 seed 模板，覆盖 `atk_recurrent_kda.json` 前需按上面的 3 seed 约定补齐。

## 正式验收用例与 TilingKey 映射

三份正式 JSON 用途不可互相替代：`atk_recurrent_kda.json`（精度，3 个固定种子）、
`atk_recurrent_kda_perf.json`（性能）、`atk_recurrent_kda_mss.json`（内存/同步）；
后两份为手工构造的精简用例。

模型 case → 性能 case id：

| 模型 case | 形状 | 性能 case id |
| --- | --- | --- |
| `bf16_small`（推理 decode 典型小形状） | `B=2,T=2,H=2,HV=4,K=128,V=128`，BSND，bf16，`state_v_first=true` | `atk_recurrent_kda_perf.json` 0 |

可达 TilingKey → 精度/内存用例（A6 的 key 由 `op_host/recurrent_kda_tiling.cpp` 的条件选择；
`实际选择证据` 在完成上板运行后按 TilingKey 日志/埋点回填，当前仅为条件推导）：

| TilingKey | 选择条件 | 精度普通用例 | 精度边界用例 | `_mss.json` 用例 | 适用 SoC | 实际选择证据 |
| --- | --- | --- | --- | --- | --- | --- |
| `RKDA_TPL_COMPAT`(0) | 非 A6 平台（A2/A3/A5）全部 | `atk_recurrent_kda.json` 0-2 | — | 0 | A2/A3/A5 | 待运行时确认 |
| `RKDA_TPL_V_FIRST`(1) | A6、`state_v_first`、非 fused（V=128/V=256 同规则） | `a6_sim` 0/5/10/31（V=128）、12/14/43/44（V=256） | `a6_sim` 38/40（T=33 跨 32-token tile） | 2 | A6 | 待运行时确认 |
| `RKDA_TPL_K_FIRST`(2) | A6、`!state_v_first`、非 fused（V=128/V=256 同规则） | `a6_sim` 1/6/19/32（V=128）、13/15/42/45（V=256） | `a6_sim` 4/9/39/41（T=33） | 3 | A6 | 待运行时确认 |
| `RKDA_TPL_V_FIRST_FUSED`(3) | A6、`state_v_first`，且 norm+raw gate+dt_bias+beta sigmoid、无负特征值、非 safe gate | `a6_sim` 46 | `a6_sim` 48（T=33） | 4 | A6 | 待运行时确认 |
| `RKDA_TPL_K_FIRST_FUSED`(4) | 同上一行，但 `!state_v_first` | `a6_sim` 21 | `a6_sim` 47（T=33） | 5 | A6 | 待运行时确认 |


## 结构矩阵用例包

| 包 | 数量 | 覆盖 |
| --- | --- | --- |
| `atk_recurrent_kda_a6_sim.json` | 49 | A6 仿真最小精度包，`T<=33`、单用例 `<=66` token：dense/均匀/ragged（含 0 长度段）/padding/单头/超多段；`V=256` 按 128 切片递推（每段 9…33）；state 布局×dtype×inplace×initial_none×strided；gate safe/raw/dt_bias_none/large_gate 与 `g/beta` 三种 dtype；融合 gate（norm+raw gate+dt_bias+beta sigmoid）在 `V_FIRST_FUSED`/`K_FIRST_FUSED` 两个 key 下的普通与 T=33 边界；slot packed/speculative×INT32/INT64；QKV view / 非连续输入；`inplace_final_state` 写回与 non-inplace 不变式；H/HV 比例 |

按分支对应的 case id（`id` 即用例序号）：

| 分支 | case id |
| --- | --- |
| dense 短序列与 32-token tile 边界（1/8/17/33） | 0-4 |
| packed 均匀分段 | 5-6 |
| ragged（含 0 长度段）/ padding 补齐 | 7-8 |
| 单头路径（L2 norm 开/关）与 `B=32` 多段 | 9-11 |
| `V=256` 短段（每段 ≤8，走 V 切片） | 12-15 |
| `V=256` 长段（每段 9…33，含 gate 预处理） | 42-45 |
| state 布局×dtype×inplace×initial_none×strided | 16-19 |
| gate safe/raw/off/dt_bias_none/large 与 `g/beta` dtype | 20-26 |
| slot packed/speculative × INT32/INT64 | 27-30 |
| QKV view / 非连续输入 | 31-34 |
| `inplace_final_state` 写回与 non-inplace 不变式 | 35-37 |
| H/HV 比例（1/1、2/2、2/4、1/16） | 38-41 |
| 融合 gate（`V_FIRST_FUSED` 普通 / `K_FIRST_FUSED` 边界 / `V_FIRST_FUSED` 边界） | 46-48 |

该包必须用 `executor_recurrent_kda.py` 执行（`executor_recurrent_kda_longseq.py` 只适配 longseq 的 spec 字段），
由本目录的生成脚本按场景矩阵产出，新增结构后重新生成即可。每个结构只保留 1 个 seed（`--seed-count` 可调），
目的是控制仿真耗时；正式验收要求的多 seed 覆盖仍由 `atk_recurrent_kda.json` 承担：

```bash
cd tests/atk/recurrent_kda
python generate_a5_recurrent_kda_cases.py --sim   # 默认以本目录 atk_recurrent_kda.json[0] 为模板
```

`--sim` 输出 `atk_recurrent_kda_a6_sim.json`（`--soc ascend950` 时输出 A5 版本，并跳过 `V=256` 每段 `>8` 的档位）；
去掉 `--sim` 可生成全量 A6 结构矩阵（含 `T=1…1024`，仿真耗时长，不建议在仿真器上全跑）。
该包按 A6（`case_spec.soc=ascend960dt`）设计，其中 `V=256` 每段 `>8` 的档位在 A5 上不适用。
执行方式与 `run_test_cpu.sh` 的精度阶段一致（A6 不走统一脚本的 `-soc`）：

```bash
atk node --name npu_dut --backend npu --devices 0 --output_path ./result/recurrent_kda/a6_sim \
  node --name cpu_golden --backend cpu --output_path ./result/recurrent_kda/a6_sim \
  task -c ./atk_recurrent_kda_a6_sim.json --task accuracy --bm_device cpu \
       -p ./executor_recurrent_kda.py -to 14400
```


## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。