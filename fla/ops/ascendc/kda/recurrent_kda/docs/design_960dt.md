# RecurrentKda ascend960dt（A6）专项设计与优化

> 基线：PR #997（分支 `codex/recurrent-kda-x8-pr`）的 A5 新实现（静态 UB + 资源 Mutex + 逐 token VF）。
> A6 由同一套代码在 `arch92/` 上按架构常量特化；

## 1. 结论速览

| 问题 | 结论 |
| --- | --- |
| 实现基线 | A5 新实现原样同步在 `arch35/`；A6 在 `arch92/` 复用同一算法，只改架构常量 |
| A6 特性 1：三发射（1 长 + 2 短/拍） | 4 行行块 × 每行 2 段 64 lane = 8 条独立短指令流，天然满足 2 短/拍；**代码无需新增分支**，收益来自编译器排布 |
| A6 特性 2：UB 248KB → 384KB | 窗口 `X` 8→**32**（BF16 / FP32 V-first），FP32 K-first 因 bank 行间距保持 **16**；UB 占用 80.2%/84.4% → **93.0%/84.4%** |
| State bank | 3 → **4**：逐 token 落盘（`ssm_state_indices`）时多一份快照在飞 |
| 理论收益 | 稳态 token 计算不变；tile 级固定开销（DMA 指令、Mutex 对、槽位翻转、循环分支）随 X 减半，**长序列约 3–4% kernel 周期回收**；三发射上限可把短指令槽从 1 条/拍提到 2 条/拍 |
| 未采用 | 5 bank、4 输入槽、4 out 槽、8 行块、2 token/VF（第 6 节给出拒绝理由与数据） |

## 2. A6 硬件特性与利用情况

| 特性 | A5 | A6 | 当前利用方式 | 是否吃满 |
| --- | --- | --- | --- | --- |
| vector 发射 | 双发射（1 长 + 1 短/拍） | **三发射（1 长 + 2 短/拍）** | 4 行 × 2 段 = 8 条独立短指令流；长指令只有 `ReduceSum/Exp/Ln/Div/Sqrt` | 代码侧满足 ILP≥8；是否真同拍需 profiler 确认 |
| UB | 约 248KB 可用 | **384KB** | 4 bank + 窗口 32/16 + 双槽输入/参数/out + guard | 93.0%（BF16 / FP32 V-first）、84.4%（FP32 K-first） |
| State 驻留 | 3 bank 轮转 | **4 bank 轮转** | `cur/next` + 在途快照 + 下一任务初态预取 | 是 |
| 编译期布局 | 静态 Tensor + Mutex | 同左 | 地址全为 `constexpr`，无 TPipe/TQue | 是 |

## 3. 相对 A5 实现做的调整

| 项目 | A5（`arch35/`） | A6（`arch92/`） | 代码位置 |
| --- | --- | --- | --- |
| 窗口 | 8（`RKDA_A5_WINDOW_TOKENS`） | 32（`RKDA_A6_WINDOW_TOKENS`），FP32 K-first 16（`RKDA_A6_WINDOW_TOKENS_KFIRST_FP32`） | `op_kernel/recurrent_kda_struct.h`、`arch92/recurrent_kda_struct.h` |
| State bank | 3 | 4（`RKDA_A6_BANK_COUNT`） | 同上 + `arch92/recurrent_kda.h` 的 `BeginBanks/AdvanceBank/预取 bank` |
| Mutex 份数 | 12（3+3+2+2+2） | 14（4+4+2+2+2） | `arch92/recurrent_kda_struct.h` |
| VF 窗口取模 | `base % 8` | `base % WINDOW`（模板常量，32/16，均为 2 的幂） | `arch92/recurrent_kda_vec.h` |
| 三发射 | 8 条短指令流（双发射只吃 1 短/拍） | 同一份代码，A6 可吃 2 短/拍 | 无额外代码 |
| host 方案 | `PlanA5Ub`：`min(ubSize,248KB)` | `PlanA6Ub`：384KB + 按布局选窗口 | `op_host/recurrent_kda_tiling_processor.h` |
| V=256 legacy 计算 | PR 原样：norm/gate/Cast 走 UB 级 API | 同 A5，已改为全 RegBase（见 §3.1） | `arch35`、`arch92` 的 `recurrent_kda_legacy.h` |

窗口之所以按布局分档：K-first FP32 的 bank 为 `128×136`（32B 行间距避免 Gather/Scatter 撞 bank），
每 bank 比 64KiB 多 4KiB；4 bank 时窗口 32 会把 UB 推到 382,016B（97.2%，余量仅 11,200B），
若平台还保留其它 UB 区就会越界，因此该组合保持窗口 16。

### 3.1 V=256 legacy 路径：与 PR 上 A5 的差异

`RKDALegacy` 只在 **A5/A6 且 V=256** 时被实例化（tiling key = 0），每段长度 ≤ 8；A2/A3 走根目录通用 kernel，不受本节影响。这条路径在 PR 的 A5 里是原样继承的 TPipe 实现，本仓做了两类改造，**arch35（A5）与 arch92（A6）同步**。

**前置事实**：aclnn 与 host 双重拦截 `K == 128`、`V ∈ {128, 256}`，因此 kernel 内 `alignK_ ≡ 128 = TWO_V_LENGTH` 恒成立。

**（1）删除 K≠128 的死分支**

| 位置 | 删除内容 |
| --- | --- |
| `InitLocalBuffers` | `alignK_==128 ? stateInUb : 另开 broadTmp` 的 else |
| `ReduceSumDispatch` | `<=64 / <=128 / else` 三分支，固定 `ReduceSum128` |
| `Compute` | 通用 membase 回退（`MatVecMul → ReduceSum → Sub/Muls → ProcessKQ → ReduceSum`） |
| `CopyInBeta` 尾 | `if (alignK_ != 128) SyncVToS()` |
| `ProcessHead` | `if (alignK_ != 128) beta_ = LoadBeta(...)` |

**（2）计算全部改为 RegBase（不再出现 UB 级 membase 计算）**

| 原实现 | 现实现 |
| --- | --- |
| `NormalizeRows`：逐行 `Mul → ReduceSum → Adds → Sqrt → GetValue → Muls` | `NormScaleVF`：整块 q/k 在寄存器内做 norm，q 顺带 scale；消除每 tile 16 次 V→S 往返 |
| `ApplyGateInKernel`：~40 行 `Add/Muls/Exp/Adds/Div/Ln` + 7 次 barrier | `GateTransformVF`：一次 VF 完成 dt_bias + A_log + safe/raw 变换（safe 用 `RegBase::Sigmoid`，溢出安全） |
| `CopyInGate` 的 `Duplicate/Adds/Cast` | `UnpackRowsVF<gateType,128>`（Load 时 unpack b16 + cast） |
| q/k/v 三条 `Cast(CAST_NONE)` | `UnpackRowsVF<inType,128|128|256>`（v 宽 256，分 4 段） |
| `CopyInState` / 转置分支的 `Adds/Cast` | `UnpackRowsVF<stateType,128>` |
| `Compute` 的状态写回 `Cast(CAST_RINT)` | `PackRowsVF<stateType,128>` |
| `Compute` 的 out 写回 `Cast(CAST_RINT)` | **`PackFlatVF`**（out 一个 V 切片是扁平向量，不能按行块转换） |
| `CopyInBeta` 的 `Adds/Cast` | `UnpackFlatVF<betaType>`（64 lane 分段 + 尾段掩码） |
| `ReadFloat/ExpScalar` 的单元素 `Adds/Duplicate/Exp` | `ScalarCopyVF/ScalarExpVF`（单 lane 寄存器操作） |

**语义不变点**：State 在 legacy 的 UB 内**始终是 FP32**，只在最终写回时 cast 成 `state_dtype`；这与 V=128 静态路径的 bf16 bank（每 token 舍入）不同，因此标杆只在 V=128 时模拟 bf16 存储。DMA 仍走 `DataCopyPad` 与队列 `AllocTensor/EnQue`（属数据搬运，不属于 membase 计算）。

**影响**：per-token 递推成本不变；收益集中在**每 tile 的输入准备阶段**（V=256 一段最多 8 token，固定开销占比高，收益更明显），同时少了每行的标量回读与流水切换。

## 4. UB 预算明细

每 token 的双槽占用（与 A5 同构）：`q/k/v 3×512 + gate 1024 + beta 64 + out 512 = 3136 B`；
固定项：参数双槽 `2×544=1088 B` + guard `2048 B`（含 512B VF 数值 scratch）。

```text
总占用 = banks × (64KiB 或 68KiB) + 3136 × X + 3136
```

### 4.1 A6：BF16（V-first / K-first）与 FP32 V-first —— X = 32

| 元素 | 大小 | 说明 |
| --- | --- | --- |
| State bank ×4 | 262,144 B | 4 × 64KiB，FP32 计算态常驻 |
| q 双槽 | 16,384 B | 512 B/token × 32 |
| k 双槽 | 16,384 B | 同上 |
| v 双槽 | 16,384 B | 同上 |
| gate 双槽 | 32,768 B | 1024 B/token × 32 |
| beta 双槽 | 2,048 B | 32 B/token × 32 |
| 参数双槽 | 1,088 B | A_log 32B + dt_bias 512B，×2 |
| out 双槽 | 16,384 B | 256 B/token × 32（512 B/token 预留） |
| guard | 2,048 B | 含 512 B VF scratch |
| **合计** | **365,632 B** | **93.0%（384KB = 393,216B），余量 27,584 B** |

### 4.2 A6：FP32 K-first —— X = 16

| 元素 | 大小 | 说明 |
| --- | --- | --- |
| State bank ×4 | 278,528 B | 4 × 68KiB（128×136 FP32） |
| q/k/v 双槽 | 24,576 B | 3 × 512 B/token × 16 |
| gate 双槽 | 16,384 B | 1024 B/token × 16 |
| beta 双槽 | 1,024 B | 32 B/token × 16 |
| 参数双槽 | 1,088 B | 不变 |
| out 双槽 | 8,192 B | 512 B/token × 16 |
| guard | 2,048 B | 不变 |
| **合计** | **331,840 B** | **84.4%，余量 61,376 B** |

### 4.3 对比：A5（X=8，3 bank，248KB 预算）

| 元素 | 大小 |
| --- | --- |
| State bank ×3 | 196,608 B |
| q/k/v 双槽 | 12,288 B |
| gate 双槽 | 8,192 B |
| beta 双槽 | 512 B |
| 参数双槽 | 1,088 B |
| out 双槽 | 4,096 B |
| guard | 2,048 B |
| **合计** | **224,832 B（BF16）/ 237,120 B（FP32 K-first）** |

### 4.4 变化带来的量

| 配置 | X=16（上一版） | X=32（本版） | 差 |
| --- | --- | --- | --- |
| BF16 / FP32 V-first | 315,456 B（80.2%） | 365,632 B（93.0%） | +50,176 B |
| FP32 K-first | 331,840 B（84.4%） | 331,840 B（84.4%，保持 16） | 0 |

## 5. 理论收益分析

### 5.1 稳态 token 路径不变

每个 token 的成本由 state 的 `读 + 写`（BF16 32KiB+32KiB，FP32 64KiB+64KiB）与 128 行的向量指令
（decay / `S@k` / outer / `S@q`，约 8 条/行）决定；这些与窗口大小无关。因此扩大窗口**不会**改变
每 token 的 UB 流量与指令数，收益只来自 tile 级固定开销。

### 5.2 tile 级固定开销减半

每个 tile 的固定动作：

| 动作 | 条数 |
| --- | --- |
| q/k/v/gate/beta 输入 DMA | 5 |
| 整 tile out DMA | 1 |
| input / out 各一对 Mutex Lock/Unlock | 4 |
| 槽位翻转、预取分支、tile 循环判断 | ~5–10 条标量 |

按 128 行 × 约 8 条向量指令、约 1 拍/指令估算，单 token ≈ 1.0–1.3K 拍：

| X | tile 拍数（token） | 固定拍数（估） | 固定占 tile |
| --- | --- | --- | --- |
| 16 | 16–21K | 1.0–1.5K | 6–9% |
| 32 | 32–42K | 1.0–1.5K | 3–4.5% |

即长序列下 tile 数减半，约回收 **3–4%** 的 kernel 周期（T=1024 时 64→32 个 tile，T=33 时 3→2 个 tile）。
DMA 总字节不变，减少的是 DMA 指令条数、同步条数与循环开销。

### 5.3 三发射的理论上限

一个 4 行块内：短指令约 40 条（8 载入 + 8 decay + 8 `S@k` 乘加 + 8 outer + 8 `S@q`），
长指令 8 条（4 次 `S@k` 归约 + 4 次 `S@q` 归约）：

| 平台 | 短槽吞吐 | 短槽拍数 | 长槽拍数 | 块拍数（下界） |
| --- | --- | --- | --- | --- |
| A5 双发射 | 1 条/拍 | 40 | 8 | 40 |
| A6 三发射 | 2 条/拍 | 20 | 8 | 20 |

理论上限为块时间减半；实际受 `ReduceSum` 延迟、寄存器依赖链与编译器排布影响，
需用 profiler 观察（见第 7 节）。代码侧已具备 8 条独立短流（4 行 × 2 段），不需要额外改造。

### 5.4 4 bank 的收益

有 `ssm_state_indices`（逐 token 落盘）时，MTE3 每 token 都要写一份 state 快照：

- 3 bank：第 3 个 token 必须复用最早 bank，需等它的 MTE3 完成；
- 4 bank：源/目标之外仍有 2 个 bank，可多留 1 份快照在飞，`Stage1RunIndexedTask` 里
  “重复目标槽”的 `PipeBarrier` 触发次数也下降。

该收益只作用于逐 token 落盘路径，量化需 profiler；对无 indices 的路径无影响（正确性不变）。

## 6. 已评估但未采用的方案

| 方案 | 理论收益 | 不采用原因 |
| --- | --- | --- |
| State bank 4→5 | 再多一份快照 | BF16 可放下（381KB），但 FP32 K-first 需 401KB 超出 384KB；bank 数必须与编译期布局一致，收益仅限落盘路径 |
| 输入槽 2→3/4 | 预取更远 | 当前在 tile 首个 token 就把下一 tile 搬入，16/32 个 token 的计算时间足以覆盖 DMA；实测无增量，只白占 UB |
| out 槽 2→4 | MTE3 更晚回收 | out 每 tile 仅 8KiB，MTE3 被计算完全掩盖 |
| 行块 4→8 | 更多短流填槽 | 寄存器需要 16 State + 16 乘积 + 8 delta，超过向量寄存器预算，必然 spill |
| 2 token / VF | 省一次 state 载入 | 两 token 的 `k/q/gate/beta` 常量必须同时驻留，短指令反而增加，A5 时代已验证净变差 |
| X=64 | tile 数再减半 | BF16 需 428KB、FP32 K-first 需 445KB，均超过 384KB |
| K-first FP32 也开 X=32 | 多 3–4% | 382,016B 占 97.2%，余量仅 11,200B，平台保留区一变就 tiling 失败 |

## 7. 风险

- 窗口 32 时首个 tile 的冷启动 DMA 变大（32 行），但只影响任务首个 tile，且 DMA 字节本身很小
  （每 token 约 2.1KB），对短任务无影响（尾 tile 按有效 token 搬）。
