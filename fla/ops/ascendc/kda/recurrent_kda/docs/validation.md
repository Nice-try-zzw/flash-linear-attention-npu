# RecurrentKda 开发期验证记录

本文件只记录开发期验证、差异与风险；接口以 `docs/api.md` 为准，方案以 `docs/design.md` 为准。

## 1. 本次差异与回归范围

| 项 | 内容 |
| --- | --- |
| 变更类型 | 既有算子的平台适配与性能优化（新增 A6 `ascend960dt` 的 `V=128` 静态 UB 引擎） |
| 算法基线 | 主线 PR #1096 的 A5 引擎（逐 token VF、两遍读写、State bank 轮换、资源 Mutex） |
| 不改动 | A5 `op_kernel/arch35/*`、A2/A3 `op_kernel/recurrent_kda.h`、`op_host` 的既有 legacy UB 规则 |
| 新增 | `op_kernel/arch92/recurrent_kda_engine.h`（V=128 静态 UB 引擎）、
`op_kernel/arch92/recurrent_kda_legacy.h` + `op_kernel/arch92/recurrent_kda_regbase.h`（V=256 RegBase 兼容路径）、
`op_kernel/recurrent_kda_tiling_key.h` |
| 共享文件增量 | `recurrent_kda.cpp`（模板入口 + A6 分派）、`recurrent_kda_struct.h`（3 个 batch stride）、
`recurrent_kda_tiling*.{h,cpp}`（`isA6`、`PlanA6Ub`、A6 tiling key）、`recurrent_kda_def.cpp` /
`op_host/CMakeLists.txt`（`ascend960dt` 配置） |
| 回归范围 | A2/A3、A5 全部既有用例；A5 的 tiling key 恒为 `COMPAT`，kernel 路径不变 |

## 2. 验证矩阵

| 维度 | 取值 | 覆盖目的 | 状态 |
| --- | --- | --- | --- |
| SOC | `ascend960dt` | 新路径 | 待实测 |
| SOC | `ascend910b` / `ascend910_93` / `ascend950` | 共享文件改动回归 | 待实测 |
| Layout | BSND / TND | 引擎的 batch stride 与线性寻址 | 待实测 |
| State 布局 | V-first / K-first | 两种 tiling key 与两套 VF 路径 | 待实测 |
| State dtype | FP32 / BF16 | bank 数（3 / 4）与 bank 跨度 | 待实测 |
| V | 128 / 256 | 引擎 / RegBase 兼容路径分派 | 待实测 |
| Gate | 预计算 log gate / raw gate（± safe gate、± dt_bias） | `Decay` 两条分支 | 待实测 |
| Beta | 原始 / sigmoid（± `allow_neg_eigval`） | beta 分支 | 待实测 |
| QK norm | on / off | `NormalizePair` | 待实测 |
| 序列 | 单 token / `TOKEN_TILE-1` / `TOKEN_TILE` / 跨 tile / 长序列 | 预取握手与 tile 回绕 | 待实测 |
| 索引 | packed 1D / speculative 2D + `num_accepted_tokens` | `StateSlot` / `InitialSlot` | 待实测 |
| 并发 | `HV % HK == 0`（含 GVA 1:3） | key head 映射 | 待实测 |
| 输出 | inplace / 非 inplace final state | `stateOutput` 选择与 `CopyStateOut` | 待实测 |
| 空任务 | 空序列 / padding tail | 空请求跳过与 flag 闭环 | 待实测 |

## 3. 设计项追溯

| 设计项 | 代码位置 | 状态 |
| --- | --- | --- |
| 模板入口与 A6 分派 | `op_kernel/recurrent_kda.cpp` | 已实现，待编译 |
| 逐 token VF 与两遍读写 | `op_kernel/arch92/recurrent_kda_engine.h` | 已实现，待编译 |
| V=256 RegBase 兼容路径 | `op_kernel/arch92/recurrent_kda_legacy.h` + `arch92/recurrent_kda_regbase.h` | 已实现，待编译 |
| bank 数与预取握手 | 同上（`StateBanks` / `Process`） | 已实现，待实测 |
| 初态搬入 RAW 依赖 | 同上（`Process` 的 `SetFlag<HardEvent::MTE2_V>`） | 已实现，待实测 |
| state 非对齐搬入/搬出 | 同上（`CopyStateIn` / `CopyStateOut` 统一 `DataCopyPad`） | 已实现，待实测 |
| 静态 UB 账本 | 同上（`UbBytes`）+ `op_host/recurrent_kda_tiling_processor.h`（`PlanA6Ub`） | 已实现，待实测 |
| V-first 四行展开 | 同上（`RecurrentKdaVFirstTwoPass` / `RecurrentKdaVFirstUpdateRows`） | 已实现，待实测 |
| V=256 每段 <=8 的 host 拦截 | `op_host/recurrent_kda_tiling_processor.h`（`CheckShapeDimAndRelation`） | 已实现，待实测 |
| tiling key 选择 | `op_host/recurrent_kda_tiling.cpp`（`DoLibApiTiling`） | 已实现，待实测 |
| batch stride 填充 | `op_host/recurrent_kda_tiling_processor.h`（`FillTilingShapeData`） | 已实现，待实测 |
| A6 SOC/CMake 配置 | `op_host/recurrent_kda_def.cpp`、`op_host/CMakeLists.txt` | 已实现，待编译 |

## 4. 精度口径

- 与 PR #1096 的 A5 引擎逐指令同序：`1e-6` 归一化 epsilon、softplus 稳定式、sigmoid、
  FMA 与归约顺序、BF16 State 每 token RINT 舍入、out 使用舍入前 FP32 值。
- 精度阈值沿用算子既有 CPU 标杆与 `docs/agents/reference/精度对比与定位.md` 的口径；
  A6 不新增容差。

## 5. 风险与回退

| 风险 | 影响 | 回退 |
| --- | --- | --- |
| A6 实际 UB 小于 300 544 B（FP32 3 bank + 2 KiB 余量） | `PlanA6Ub` 返回失败，算子不可用 | 降低 `TOKEN_TILE`（32→16）并同步 `RKDA_A6_TOKEN_TILE` 后重新核算 |
| A6 工具链不支持静态 `LocalTensor` 构造或 14 参数 VF | 编译失败 | `V=128` 临时全部走 arch92 RegBase legacy（功能可用、性能退化） |
| A6 性能未达预期 | 收益不足 | `TOKEN_TILE` / bank 数为编译期常量，可单独实验 |
