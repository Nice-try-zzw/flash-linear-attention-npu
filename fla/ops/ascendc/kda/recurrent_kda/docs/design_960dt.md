# RecurrentKda ascend960dt（A6）专项设计与优化

> 本文只讲 A6（`ascend960dt`，`DAV_9201`）相对 A5（`ascend950`，`DAV_3510`）新实现的部分。
> 通用语义（公式、接口、tiling 字段、精度策略）见 [design.md](./design.md)。

## 1. 结论速览

| 问题 | 结论 |
| --- | --- |
| 实现基线 | **A5 新实现**（PR #997 分支 `codex/recurrent-kda-x8-pr`）的静态 UB + 逐 token VF 流水：A5 部分已原样同步到 `arch35/`，A6 由同一套代码在 `arch92/` 上按架构常量特化，不再维护上一版 A6 融合 VF |
| 文件组织 | `arch92/recurrent_kda.h`（任务/预取/写回编排）、`recurrent_kda_vec.h`（单 token VF）、`recurrent_kda_regbase.h`（寄存器辅助）、`recurrent_kda_struct.h`（UB 静态布局与 Mutex 槽）、`recurrent_kda_legacy.h`（V=256 兼容路径） |
| 编译实例 | `KERNEL_MODE` 四档 tiling key：`COMPAT`、`V_FIRST`、`K_FIRST`、`V_FIRST_FUSED`；A6 V=128 按 State 布局与融合位选实例 |
| A6 硬件增量 | ① vector **三发射**（1 长 + 2 短/拍）；② UB **248KB → 384KB** |
| 三发射 | 沿用 A5 新实现的 4 行行块 + `MulAddDst` 折叠：每行 2 段 64 lane，4 行 = 8 条独立短指令流，`ReduceSum` / `Exp` / `Ln` / `Div` 落在长槽 |
| 384KB 用法 | ① 窗口 `X` 从 8 拓到 **16**（每 tile 的 DMA 次数减半、预取更深）；② State bank 从 3 增到 **4**（逐 token 落盘时可多留一份快照在飞） |
| UB 使用率 | V=128 静态布局固定 308.0KB（BF16）/ 324.1KB（FP32 K-first），占 384KB 的 80.2% / 84.4% |
| V=256 | 仍走 `RKDA_TPL_COMPAT` 兼容实例，每段长度必须 ≤ 8 |
| 长序列 | V=128 无长度上限（固定窗口递推，状态跨窗口驻留 UB） |

## 2. 平台与结构差异

| 维度 | A5 | A6 |
| --- | --- | --- |
| 内核选择宏 | `__CCE_AICORE__ == 310` | `__NPU_ARCH__ == 9201` |
| host 平台标识 | `isRegbase = 1` + `isA5 = 1` | `isRegbase = 1` + `isA6 = 1` |
| V=128 kernel | `arch35/`（A5 新实现） | `arch92/`（同一套算法，A6 参数化） |
| V=256 kernel | `arch35/recurrent_kda_legacy.h` | `arch92/recurrent_kda_legacy.h`（同一份通用路径） |
| 窗口 X | 8（`RKDA_A5_WINDOW_TOKENS`） | **16**（`RKDA_A6_WINDOW_TOKENS`） |
| State bank | 3（`RKDA_A5_BANK_COUNT`） | **4**（`RKDA_A6_BANK_COUNT`） |
| UB 预算 | `min(ubSize, 248KB)` | 平台上报的 384KB |
| 编译实例 | `KERNEL_MODE` 四档 | 同左（A6 V=128 走 1/2/3 档） |

V=128 的 kernel 入口分派见 `op_kernel/recurrent_kda.cpp`：`KERNEL_MODE == RKDA_TPL_COMPAT` 走
`RKDALegacy`（V=256 与 A2/A3），其余走静态 UB 的 `RKDAVectorContext`。

## 3. 内存与同步（与 A5 同构）

UB 全部由编译期常量寻址，不使用 `TPipe`/`TQue`/`TBuf`；跨流水依赖由 14 份 Mutex 资源锁保护
（State 访问 4、State 快照 4、输入 2、参数 2、out 2），同流水访存仍用 `LocalMemBar`/`PipeBarrier`。

```text
[0, 4*BANK)           4 个 State bank（K-first FP32 每 bank 128x136，多 4KiB 行间距）
[.., +3*512*X)        q / k / v 双槽（每 token 512B）
[.., +1024*X)         gate 双槽
[.., +64*X)           beta 双槽（每 token 32B 对齐）
[.., +1088)           参数双槽（A_log 32B + dt_bias 512B，×2）
[.., +512*X)          out 双槽
[.., +2048)           guard（含 512B VF 数值 scratch）
```

host 侧在 `PlanA6Ub()` 用同一公式记账并校验：

```text
banks    = 4 * 64KiB  (+4 * 4KiB，仅 K-first FP32)
window   = 3136 * 16  (q/k/v 1536 + gate 1024 + beta 64 + out 512，每 token)
fixed    = 3136       (参数 1088 + guard 2048)
总占用   = banks + window + fixed = 315456B(BF16) / 331840B(FP32 K-first)
```

| state dtype / layout | 占用 | 占 384KB |
| --- | --- | --- |
| BF16（V-first 或 K-first） | 315,456B | 80.2% |
| FP32 K-first | 331,840B | 84.4% |

剩余空间不再分配：静态布局的每个槽都对应 kernel 里的固定偏移，多出来的容量不改变
每 token 的指令量，继续扩大窗口只会拉长首 token 的等待。

## 4. 三个 bank 变四个 bank

A5 的 3 bank 已经够"cur / next / 下一任务初态预取"。A6 加第 4 个 bank 的收益在**逐 token 落盘**
路径（提供 `ssm_state_indices` 且打开 `inplace/output final state`）：

- 3 bank 时旧快照只能留 2 份在飞，第 3 个 token 要复用最早 bank 时必须等它的 MTE3 完成；
- 4 bank 时源/目标之外还有 2 个 bank，MTE3 的 snapshot 多一份在飞窗口，
  `Stage1RunIndexedTask` 里的重复目标槽 `PipeBarrier` 也相应减少。

轮转规则不变：`cur → next` 每 token 前进一格，末 token 时向 `(next + 1) % 4` 预取下一任务初态；
下一任务以该 bank 为 `cur` 重新建立轮转。无 `ssm_state_indices` 时只在末 token 写一次 state，
第 4 个 bank 退化为普通轮转位，不影响正确性。

## 5. 窗口 8 → 16

固定窗口只影响"一个 tile 预取多少 token"，不改变逐 token 的递推语义：

- 每 tile 的 q/k/v/gate/beta 各一次 DMA（`rows = min(16, end-tileStart)`），DMA 指令条数减半；
- 输出按整 tile 批量搬出，MTE3 的块更大、启动次数更少；
- 尾 tile 仍按有效 token 执行（`RecurrentKdaValidTokens`），长序列最后一窗不做无效计算。

窗口在编译期固定为 `RKDA_A6_WINDOW_TOKENS`，不做按 shape 选窗，也不重建 UB 偏移，
保证 host 的 `PlanA6Ub()` 与 kernel 的 `x_` 始终一致。

## 6. 三发射的落点

三发射（1 长 + 2 短）需要同一个 VF 内至少 3 条互不依赖的指令流。A5 新实现已经把
"V 方向相邻 4 行 × 每行前后 2 段 64 lane"编成 8 条短指令流，并把 `Mul + MulAddDst`
折叠进归约（`RKDA_ENABLE_ADD_FOLD_REDUCE` 同一写法）。A6 直接沿用：

- 短指令：`DataCopy`（寄存器↔UB）、`Mul`/`Add`/`Sub`/`Muls`/`Duplicate`/`MulAddDst`；
- 长指令：`ReduceSum`、`Exp`、`Ln`、`Div`、`Sqrt`（raw gate / sigmoid / 归一化）；
- 4 行块让长指令的等待窗口里有其它行的短指令可填，寄存器账在 4 行处达到平衡
  （8 行会逼近向量寄存器上限并触发 spill）。

## 7. 与 A5 的数值一致性

移植不改数学：q/k 归一化仍为 `x / sqrt(sum(x*x) + 1e-6)`（零范数不除），
gate 的 raw/safe 变换、beta sigmoid、K-first 的 Gather/Scatter 布局、BF16 打包舍入都沿用 A5 新实现，
因此 A6 与 A5 在同一输入上的差异只来自 dtype 与调度顺序，不引入新的舍入口径。

## 8. 验证

1. `bash build.sh --pkg --soc=ascend960dt --ops=recurrent_kda -j16`：确认四个 `KERNEL_MODE`
   实例都编译产出，且 `arch92` 没有回退到 `arch35`。
2. 打开 `OP_LOGD` 核对 `windowSize=16`、`stateOutBufferNum=4`、`ubRestBytes=315456/331840`。
3. ATK `-soc=ascend960dt` 跑 V=128 长序列（`executor_recurrent_kda_longseq.py`，T ≥ 9）
   与短序列回归；V=256 只跑每段 ≤ 8 的用例。
4. 专项覆盖：`ssm_state_indices` 逐 token 落盘（4 bank 快照在飞）、尾窗口（T mod 16 ≠ 0）、
   空请求、单 token、跨任务连续执行、K-first FP32（带行间距的 bank 预取/写回）。
