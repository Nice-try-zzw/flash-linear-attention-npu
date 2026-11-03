# RecurrentKda ascend960dt（A6）专项设计与优化实现

本文只描述 A6（`ascend960dt`，`DAV_9201`）相对 A5（`ascend950`，`DAV_3510`）的差异化设计与代码落点。
通用语义、接口与精度策略见 [design.md](./design.md)。

## 1. 背景与现状

`op_kernel/arch92/recurrent_kda.h` 在引入 A6 支持时是 `op_kernel/arch35/recurrent_kda.h` 的逐行副本：

- 每 token 分两段 VF：`DecayMatVecReduce128`（decay + `S@k`）→ `PipeBarrier<PIPE_V>` →
  `ProcessDeltaKQReduce128`（delta + outer + `S@q`）；
- 行级预处理 `NormalizeRows`、`ApplyGateInKernel` 全部用普通 AscendC 向量指令，逐行、逐 op 插 `PipeBarrier`；
- host 侧 tiling 的 UB 候选表与 A5 完全一致，384KB 只体现为 `vStep` 被动变大。

这套实现在 960dt 上会同时浪费两项硬件能力：

| 960dt 能力 | 复用 A5 代码时的浪费点 |
| --- | --- |
| vector 三发射（两条短指令 + 一条长指令并行） | VF 内单行串行，`ReduceSum` 之后立刻依赖 `sum`，等待窗口没有其它独立指令可填；标量链仍按全 lane 发射 |
| 可编程 UB 384KB（A5 为 248KB） | 剩余 UB 只被 `vStep` 吸收，`vStep` 到达 `repeat` 下限后继续放大是超采样，不产生吞吐收益 |

## 2. 平台能力与可利用空间

### 2.1 三发射

960dt 的 vector 单元每个 issue slot 可并行 `1 条长指令 + 2 条短指令`（A5 为双发射）。本算子涉及的指令分类：

- **长指令**（多周期、占用专用资源）：`ReduceSum`、`Exp`、`Ln`、`Div`、`Sqrt`；
- **短指令**（64 lane 逐元素 / 搬运）：`Mul`、`Add`、`Sub`、`Muls`、`Adds`、`Duplicate`、
  `MulAddDst`、`DataCopy<LoadDist::DIST_BRC_B32>`、`DataCopy<StoreDist::DIST_FIRST_ELEMENT_B32>`，
  以及 64 lane 的连续 `DataCopy` 载入 / 回写。

kernel 要真正吃到三发射，必须同时满足：

1. 一个 `__VEC_SCOPE__` 内存在 ≥3 条**互不依赖**的指令流；
2. 长指令落在某条依赖链的等待窗口里，且该窗口内有足够的短指令可发射；
3. 标量（1 lane 有效）计算不要占满 64 lane，避免无意义的发射宽度。

### 2.2 UB 384KB

- A5 UB ≈ 248KB，A6 UB = 384KB（`GetCoreMemSize(UB)` 实际值由平台上报，host 侧不做常量假设）。
- 算子 UB 由三部分构成：固定输入/临时区（`CalcFixedUbBytes`）、按 `vStep` 线性增长的队列+工作区
  （`CalcVStepCoeff`）、以及不以 `vStep` 缩放的行级 scratch（`CalcWorkingUbBytes - CalcFixedUbBytes`）。
- 由此 UB 富余有两条去处：**放大 `vStep`**（减少 `repeat` 次数）和**加深输出 ring**（重叠 MTE3 写回）。
  前者在 `repeat == 1` 后失效，后者是 384KB 相对 A5 的主要增量收益来源。

## 3. 现状瓶颈拆解（A5 复用实现）

| # | 瓶颈 | A5 实现位置（arch35） | 对 960dt 的影响 |
| --- | --- | --- | --- |
| 1 | 每 token 两段 VF + 一次 `PipeBarrier<PIPE_V>` | `Compute` | 每次 VF 退出/进入都会打断三发射的稳态排布，且 barrier 让 MTE3 与 V 无法跨段重叠 |
| 2 | decayed state 回写 UB 再读回 | `DecayMatVecReduce128` / `ProcessDeltaKQReduce128` | 每行状态 UB 网关 4 次（读-写-读-写），其中两条纯为中间结果服务 |
| 3 | 行内串行 | 同上 | `ReduceSum` 后立即 `BRC(v)`/`Sub`/`Mul`，长指令等待期内没有独立指令可填 |
| 4 | 标量链全 lane 发射 | `ProcessDeltaKQReduce128` 的 `useSigmoid` 分支 | beta sigmoid 只需 lane0，却按 64 lane 发射 5 条指令 |
| 5 | `NormalizeRows` 逐行 4 次 barrier + `SyncVToS` + 标量回读 | `NormalizeRows` | 8 token 内每行一次 V→S 同步，scalar 读回延迟直接串在关键路径上 |
| 6 | `ApplyGateInKernel` 每个 op 一次 barrier | `ApplyGateInKernel` | safe 链每行 7 次、unsafe 链每行 4 次 barrier，`dt_bias` 再各加 1 次 |
| 7 | tiling 候选表与 A5 一致 | `FinalizeVStepFromUb` | 384KB 只换来更大的 `vStep`，输出 ring 深度仍停在 A5 也可能取到的值 |

## 4. 设计总览

A6 走**独立实现**：`op_kernel/arch92/recurrent_kda.h` 按 `__NPU_ARCH__ == 9201` 单独编译，
文件内部**不再保留任何"回退 A5"的编译宏开关**——A6 优化路径即本文件的默认路径。
A5 的 `op_kernel/arch35/recurrent_kda.h` 一字未改，两个平台各自演进。

| 主线 | 关键手段 | 对应硬件特性 |
| --- | --- | --- |
| A. token 级融合 VF | 一次 `__VEC_SCOPE__` 完成 decay / `S@k` / delta / outer / `S@q`；状态只回写一次 | 去掉 UB 往返 + 去掉段间 barrier，为三发射腾出长稳态窗口 |
| B. 4 行行块 + 两段切分 | V 方向相邻 4 行（相互独立）× 每行前后两段 64 lane = 8 条短指令流；`ReduceSum`/`Sqrt`/`Div` 长指令与之同拍 | 两条短指令 + 一条长指令的组合并行 |
| C. 384KB 切分 | 候选表扩到 4 项 + `DeepenOutRingForA6` 用剩余 UB 加深 `attn/state` 输出 ring | 384KB 增量 |

同时保留两条"平台无关"的实现分支：

- `alignK_ != TWO_V_LENGTH`（即 `K != 128`，例如 `K=256`）时，`Compute` / `NormalizeRows` 仍走
  通用 AscendC 路径（`MatVecMul` / `ProcessKQ` / `ReduceSumDispatch`），不受 A6 改动影响；
- K-first（`stateVFirst=0`）与 V-first（`stateVFirst=1`）两种 state 布局都兼容，融合 VF 只依赖
  "状态在 UB 内 V-first 存放"这一内部表示。

## 5. 主线 A+B：token 级融合 VF `ProcessTokenFused128`

### 5.1 数据流

单个 token、单个 V-tile（`curSingleV` 行）内：

```text
state[i][:] *= gate[:]                       # i = V 方向行
dot[i]      = sum(state[i][:] * k[:])
delta[i]    = (v[i] - dot[i]) * beta
state[i][:] += delta[i] * k[:]
o[i]        = sum(state[i][:] * q[:])
```

`gate` 在此处已是 `exp(step_log_gate)`（由 `CopyInQKVGate` 末尾的 gate 变换得到），`beta` 已按
`useBetaSigmoid/allowNegEigval` 在寄存器内处理完。

### 5.2 行块与两段切分

`state` 在 UB 内是 V-first `[curSingleV][alignK_]`，且融合路径强制 `alignK_ == TWO_V_LENGTH == 128`，
因此一行的 128 float 恰是两段连续 64 lane（`k[0..63]` / `k[64..127]`），`gate`、`k`、`q` 同样按两段装载。

V 方向第 `4i .. 4i+3` 行之间没有任何数据依赖（4 行共用同一 token 的 `k/q/gate`），因此按 4 行一组
编成 8 条独立短指令流：

```text
[Mul+MulAddDst p0][Mul+MulAddDst p1][Mul+MulAddDst p2][Mul+MulAddDst p3]   8 短
[LONG ReduceSum d0][LONG ReduceSum d1][LONG ReduceSum d2][LONG ReduceSum d3] 4 长
[BRC v0][BRC v1][BRC v2][BRC v3]                                           4 短（长指令等待窗口）
[Sub/Mul(VL1) ×4][Duplicate ×4]                                            8 短
[MulAddDst update ×8]                                                      8 短
[Mul+MulAddDst out ×8]                                                     8 短
[state 回写 ×8]                                                            8 短（与下面的长指令并行）
[LONG ReduceSum ×4][attn 首元素回写 ×4]                                     4 长 + 4 短
```

尾行（`curSingleV % 4 != 0`）退化为单行分支，数值路径完全一致。

为什么要"两段连续"而不是 A5 的 `DIST_DINTLV_B32`（偶/奇交织）：

1. 连续两段与 A5 的 state 行内语义一一对应（第 0 段 = 前 64 列、第 1 段 = 后 64 列），便于逐元素对齐；
2. 连续两段可以直接用 `MulAddDst` 折叠"两段乘加"，奇偶交织同样可以，但交织把"同一行的两半"映射到
   相邻 lane，调试与精度比对时不可读；
3. 与 x8-pr（`codex/recurrent-kda-x8-pr`）A5 优化实现的寄存器映射一致，便于两个分支共用同一套
   VF 写法与后续维护。

`DIST_DINTLV_B32`/`DIST_INTLV_B32` 版本的代码在 A5（arch35）中保留，A6 不再使用。

### 5.3 寄存器与指令排布

| 类别 | 寄存器 | 说明 |
| --- | --- | --- |
| token 常量 | `gate0/gate1`、`k0/k1`、`q0/q1`、`beta`、`one` | 每个 token 只装一次 |
| 4 行状态 | `s00/s01` … `s30/s31` | 更新后直接用于 `S@q`，避免二次读 UB |
| 中间量 | `p0..p3` | 乘积 / `v` 广播 / delta 广播复用，峰值 4 个 |
| 归约结果 | `d0..d3` | `ReduceSum` 结果与 delta 复用 |

峰值 24 个 `RegTensor`（8 token 常量 + 8 状态 + 8 中间），相对 A5 `arch35` 的单行实现（约 10 个）
明显提高，但换来 4 行完全并行；若编译器报寄存器不足，退化手段是把行块从 4 行降到 2 行（见第 12 节风险 3）。

### 5.4 与 A5 两段式的逐行对比（K=128，V=128）

| 指标 | A5 两段式 | A6 融合 VF | 变化 |
| --- | --- | --- | --- |
| 每行向量指令数 | 24（decay 9 + delta 15） | 19 | **-5/行（-21%）** |
| 其中归约折叠 | `Mul`+`Mul`+`Add`+`ReduceSum`（4 条×2 次） | `Mul`+`MulAddDst`+`ReduceSum`（3 条×2 次） | **-2/行** |
| 状态 UB 网关指令 | 4（load/store × 2 段） | 2 段 load + 2 段 store（各 256B） | 指令 +2，但**字节 -50%** |
| 每行状态 UB 字节 | 2 读 2 写 = 2048B | 1 读 1 写 = 1024B | **-50%** |
| 每 token VF 发射次数 | 2 | 1 | **-1** |
| 每 token `PipeBarrier<PIPE_V>` | 1 | 0（合并） | **-1** |
| beta sigmoid 发射宽度 | 64 lane | 1 lane（`MaskPattern::VL1`） | 5 条指令按 1 lane 发射 |
| 指令级并行度 | 1 行（最多 2 段独立） | 4 行（8 段独立） | **×4** |

V=128（`curSingleV=128`）、单 token：向量指令 3072 → 2432，状态 UB 流量 256KB → 128KB。
V=256 时同比翻倍差值。

### 5.5 数值等价性

- `S = exp(g) * S`：寄存器内的 `gate` 与 A5 写回 UB 再读回的数值**逐位相同**（同一份 `exp` 结果，无中间存储舍入）。
- `delta = beta * (v - S@k)`：运算顺序保持 `Sub` → `Mul`，与 A5 的 `Sub(delta, value, delta)` → `Mul(delta, delta, beta)` 一致。
- `S = S + delta⊗k`：仍为"先把 delta 广播到全 lane，再分别乘 `k0/k1`、再加 `s00/s01`"；A6 用
  `MulAddDst(s, delta, k)` 把"乘 + 加"折叠成 1 条指令，**引入 FMA 的单次舍入**，与 A5 的
  "`Mul` 后再 `Add`"相比在末位上可能有 ≤1 ULP 差异。该写法和 A5 通用路径的
  `RKDA_ENABLE_ADD_FOLD_REDUCE`（`MulAddDst`）完全一致，属于本算子已接受的口径；需要严格逐位对齐时，
  把 `MulAddDst` 改回 `Mul`+`Add` 即可（4 行行块结构不变）。
- `o = S@q`：使用更新后的 state（寄存器值），与 A5 从 UB 读回更新后 state 一致。
- `beta` 在 `MaskPattern::VL1` 上做 sigmoid：只有 lane0 参与后续 `Sub/Mul`（后两者也只开 lane0），
  与 A5 的 `pregFull` 版本在 lane0 上结果一致。
- `ReduceSum` 的结果只在 lane0 使用（配合 `VL1` mask 的 `Sub/Mul` 与 `DIST_FIRST_ELEMENT_B32` 存储）。

### 5.6 边界与退化

- 仅当 `alignK_ == TWO_V_LENGTH`（即 `K=128`）时启用；否则走通用路径。
- `curSingleV` 不是 4 的倍数时，末 1~3 行走单行分支（数值路径一致）。
- 状态回写只覆盖 `[0, curSingleV)` 行，与 A5 的写入范围一致（`vStep_` 行内其余行不写）。
- 融合 VF 尾部不额外加 `PipeBarrier<PIPE_V>`，与 A5 `Compute` 尾部写法一致（`Cast`/`DataCopy` 紧随其后）。

## 6. 主线 B：行级预处理 VF 化

### 6.1 `NormalizeRowsVF128`（q/k L2 normalize）

A5：每行 `Mul` → `ReduceSumDispatch` → `Sqrt` → `PipeBarrier` → `SyncVToS` → scalar `GetValue(0)` →
`if (norm > 0)` → `Muls` → `PipeBarrier`，即每行 4 次 barrier 加一次 V→S 同步。

A6：整块收进一个 VF，4 行一组，零 V→S 同步、零标量读回：

```text
行块: 8 段载入 → 4 行 × (Mul + Mul + Add) 12 短
      LONG ReduceSum ×4
VL1:  Adds(1e-6) ×4 → Sqrt ×4 → Div(1,·) ×4 → Duplicate 广播 ×4
全 lane: Mul ×8 → 回写 ×8
```

- 标量链（`Adds`/`Sqrt`/`Div`）收敛到 `pregFirst`（`VL1`），只在 lane0 发射；
- `rstd` 用寄存器内 `Duplicate` 广播，取代"A5 回写 UB → 标量读回 → `Muls` 标量参数"；
- 整块只在末尾保留一次 `PipeBarrier<PIPE_V>`（与原实现末尾 barrier 等价）。

**数值口径**：epsilon 取 `RKDA_L2_NORM_EPSILON = 1e-6`，与 A2/A3、以及 x8-pr（A5 优化分支）的
"`sqrt(sum + 1e-6)`"对齐，也与本文档第 7 节的同步项一致。本仓库的 A5（arch35）当前仍是
"`x / sqrt(sum)` + 范数为 0 时跳过缩放"，两者的相对差异约为 `1e-6 / (sum + 1e-6)`：
`sum` 为 O(1) 的常规输入约 1e-8；`sum ≲ 1e-6`（q/k 元素极小）时差异显著，此时 A6 与 A2/A3 标杆一致、
A5 与标杆分歧。ATK 常规随机用例按 `mixed_tolerance_bm` 判定不受影响，边界口径由
`tests/atk/recurrent_kda/scripts/test_l2norm_epsilon.py` 单独覆盖，见第 12 节风险 8。
A6 不再使用 `max(norm, FLT_MIN)` 的兜底写法，避免"平方和下溢但元素非零"这类退化输入被放大 1e18 倍。
零范数行的结果仍为 0（与 A5 的 `if (norm > 0)` 跳过缩放等效）。

### 6.2 `ApplyGateInKernelVF`（raw gate 转换）

A5：whole-block 指令（`Muls/Exp/Adds/Ln/Div`）之间逐 op `PipeBarrier<PIPE_V>`，`dt_bias` 逐行 `Add` + 每行 barrier。

A6：整块进一个 VF，2 行一组（每行两段 → 4 条短指令流），`dt_bias` 只在 VF 入口装载一次（寄存器常驻），
safe 链与 unsafe 链按 ops 逐项跨行成组发射，`Exp`/`Ln`/`Div` 作为长指令与另一行的 `Muls/Adds` 并行。
只在 VF 末尾保留一次 barrier。

- `realK_ != alignK_` 时仍走 A5 同构的逐行实现（`dt_bias` 长度与 pad 区域不匹配）；
- `dt_bias` 借用 `broadTmpInUb`（`alignK_ == 128` 时它与 `stateInUb` 同址）；gate 变换发生在
  `LoadPrefetchedState` 之前，state 尚未写入，别名安全（与 A5 一致）。

## 7. 与 x8-pr（A5 优化分支）的对齐

参考 `recurrent-kda-x8-pr` 的 `codex/recurrent-kda-x8-pr` 分支（A5 场景的优化与功能扩充）后，
按"能否在 A6 的 TQue 流式架构里兑现"分类处理：

| x8-pr 优化点 | A6 处理 | 说明 |
| --- | --- | --- |
| `MulAddDst` 折叠归约（`FusedSum`） | **已同步** | 5.2/5.4：每行归约 4 条 → 3 条指令 |
| 4 行行块 + 两段连续切分（`UpdateRows4`） | **已同步** | 5.2：8 条独立短指令流，便于三发射 |
| 标量参数（strength/dt_bias）寄存器常驻 | **已同步** | 6.2：`dt_bias` 入口装载一次 |
| L2 归一化 epsilon 与 A2/A3 对齐 | **已同步** | 6.1：`1e-6`，去掉 `FLT_MIN` 兜底 |
| 只在 VF 末尾同步，去掉 V→S 标量回读 | **已同步** | 6.1/6.2 |
| 3 bank state 轮转（`currentBank/nextBank`） | **不适用** | A6 的 state 在 UB 内按 V-tile 常驻整个任务，逐 token 原地更新，本来就没有 state 的 GM 往返；
第 3 个 bank 的作用是"预取下一任务初始 state"，本架构由 `stateInQueue_`（`INPUT_BUFFER_NUM=2`）完成 |
| 静态 UB + `Mutex` 显式调度（`InitSocState`） | **不适用** | A6 仍是 `TPipe` + `TQue` 模型：输入队列双缓冲、输出 ring 由 tiling 控制；
换成静态 UB 属于整算子重写，风险大于收益 |
| 整 tile 输出批量写回（`CollectOutputBlock` 轮转聚合） | **未采用** | A6 的 `curSingleV` 随 `vStep` 变化（8~128），聚合需要 64 lane 对齐；
当前每行一条 `DIST_FIRST_ELEMENT_B32`（1 条指令/行），已与长指令并行发射 |
| tiling key 模板化编译期分发（`recurrent_kda_tiling_key.h`） | **未采用** | A6 融合路径只在 beta 的 `useSigmoid/allowNeg` 上做运行期二分派发（3 个组合），
每个 token 一次，相对 VF 主体开销可忽略；引入 `ASCENDC_TPL` 会同时改动 A5 共享的 kernel 入口 |
| DMA 描述符缓存（`queryCopy_` 等成员化） | **不适用** | A6 的 q/k/v/gate 搬运粒度为"每 tile 8 token 一次"，描述符本来只在 `ProcessHead` 内构造一次，
不存在 x8-pr 里"逐 token 构造 `DataCopyExtParams`"的开销 |

其中的"功能扩充"（`ssmStateStride` 元数据步长、`accepted tokens` 截断、通用递推序列、K-first/V-first
两种布局）在 A6 分支里已具备同一套语义，本次不需要额外移植。

## 8. 主线 C：384KB UB 切分

### 8.1 候选表扩展

`FinalizeVStepFromUb` 的 profile 候选由 3 项扩为 4 项：

| 下标 | `stateOutBufferNum` | `attnOutBufferNum` | 使用平台 |
| --- | --- | --- | --- |
| 0 | 1 | 1 | A5 + A6 |
| 1 | 1 | 2 | A5 + A6 |
| 2 | 2 | 2 | A5 + A6 |
| 3 | 2 | 1 | 仅 A6 |

A5 仍只评估前 3 项，行为字节级不变；A6 多评估一项，避免"2/1"这类"state 加深、attn 不加深"的
UB 余量组合被跳过。选择仍按 `IsBetterProfile`：先最小 `repeatTime`，再最大 ring 总深度，最后最大 `vStep`。

### 8.2 `DeepenOutRingForA6`

选定 profile 后，用**真实剩余量**再加深 ring：

```text
spareBytes = ubSize - fixedUbBytes - queueCoeff*vStep - scratchBytes
queueCoeff = CalcVStepCoeff(aDk, depth) - CalcComputeUbCoeff(aDk)
scratchBytes = (CalcWorkingUbBytes - CalcFixedUbBytes) + CalcComputeUbCoeff(aDk)*vStep
attn 每级  = 2 * vStep 字节
state 每级 = stateDtypeSize * aDk * vStep 字节
```

先加深 `attnOutBufferNum`（代价极小），再加深 `stateOutBufferNum`，上限 `RKDA_MAX_OUT_BUFFER_NUM = 2`。
加深后 `ubRestBytes` 用更新过的 `queueCoeff` 重算，保证仍然非负。

**为什么上限是 2**：kernel 侧 pending 机制只保留一个待写副本（`hasPendingAttn` / `hasPendingState`
各持一个 offset），深度 2 已能让"上一 token 的 MTE3 写回"与"下一 token 的向量计算"完全重叠；
再深只会占用 UB，需要同时改 kernel 的 TQue 深度与 pending 队列结构才能兑现收益，
因此本版本把剩余 UB 留给后续规划（见 8.4）。

### 8.3 推算示例（K=V=128、FP32 state、`nv=96`）

按公式推算（实际以 `OP_LOGD` 打印的 `vStep/stateOutBufferNum/attnOutBufferNum/ubRestBytes` 为准）：

| 项 | A5（UB≈248KB） | A6（UB=384KB） |
| --- | --- | --- |
| `fixedUbBytes` | 15424 | 15424 |
| `workingUbBytes` | 34880 | 34880 |
| `CalcVStepCoeff`（`state/attn=1/1`，K-first） | 2570 | 2570 |
| 首轮 `vStep`（16 对齐取整后） | 80 | 128 |
| `repeatTime` / 最终 `vStep` | 2 / 64 | **1 / 128** |
| 最终 ring 深度（`state/attn`） | `2/2`（候选表按 ring 总深度胜出） | `1/2`（K-first）；`2/2`（V-first） |
| 加深后剩余 UB | ≈21.7KB | ≈29.1KB |

关键点：

- A6 因为 UB 富余，首轮 `vStep` 就能到 128，`repeatTime` 直接落到 1，**每个 V-tile 循环覆盖整个 V 轴**，
  每 token 少一趟 V-tile 循环、少一次 VF 进入/退出；
- `repeat=1` 下 state tile 每级就是 `stateDtypeSize*aDk*vStep = 64KB`，K-first 的空闲量（≈29KB）不够再加深一级
  state ring，此时只加深便宜的 `attnOut` ring；V-first 因为没有转置 buffer，空闲量 ≈95KB，可把 state ring 也加深到 2；
- 即便 state ring 停在 1，kernel 的 `AllocTensor` 发生在 `Compute` 末尾，上一 token 的 MTE3 仍可与
  下一 token 的向量计算重叠大半；"放弃 repeat=1、换 ring 深度 2"的权衡**缺乏实测依据**，
  故本版本保持"repeat 优先"的既有排序规则（见第 12 节风险）。

### 8.4 剩余 UB 的后续规划（本版本未启用）

`DeepenOutRingForA6` 之后示例配置仍有 ≈29KB 富余（V-first 在加深前约 ≈95KB），预留方向：

1. **下一 tile 的 QKV 预取**：`MAX_MTP * (2*alignK_ + alignV_) * 2B`，`K=V=128` 时约 10.2KB，
   可把 `Cast` 之后的 fp32 输入与下一 tile 的 GM→UB 搬运重叠；
2. **bf16 state 的第二块转置 buffer**：K-first bf16 state 下加深 `stateTransposeBuf_`，
   让"上一个 V-tile 的转置"与"下一个 V-tile 的 `Cast`"并行；
3. **`vStep` 超采样**：仅在 `repeat > 1` 时把 `vStep` 抬到刚好整除 `realV_`，避免尾块；
   当前实现优先保证 `repeat` 最小，不做该优化。

以上三项都需要新增 tiling 字段或改 kernel TQue 结构，故单独评估，不与本次改动混在一起
（不改 `recurrent_kda_struct.h`，避免 ABI 风险）。

## 9. Host 侧平台标识

- `RecurrentKdaCompileInfo` 新增 `bool isA6{false}`；`InitCompileInfo()` 中
  `isA6 = (npuArch == NpuArch::DAV_9201)`。
- 原 `bool isA5{false}` 重命名为 `bool isRegbase{false}`（改名使语义与实现一致：标记表达的是
  "走 arch35/arch92 的 regbase 短序列路径"），取值 `(npuArch == DAV_3510 || npuArch == DAV_9201)`。该路径下
  arch92 与 arch35 共用同一套 regbase UB 布局
  （输入/命名 scratch 的算法、`broadTmp` 与 state 同址、`ubRestBytes` 语义都一致）。
  对应的 `RecurrentKdaTilingContext::isRegbase` 共 7 处使用点（`RuleCalcBlockDim` 分核、
  `RuleFillTilingShapeData` 容量、`CalcFixedUbBytes` 2 处、`CalcWorkingUbBytes`、`CalcComputeUbCoeff`、
  `RuleFinalizeVStepFromUb`）同口径；A5 行为不变。
  若 A6 仍按"非 regbase"走 A2/A3 的命名 scratch 计算 `ubRestBytes`，会与 arch92 kernel 的
  `TPipe` 分配不符，因此 `isRegbase` 必须同时覆盖 A6。
- `BuildProcessorContext()` 透传 `ctx.isA6`，`RecurrentKdaTilingContext` 新增字段
  `uint32_t isA6 = 0`（与 `isRegbase` 同风格）。
- `PrintTilingData()` 增加 `isA6(ascend960dt)` 日志，便于在无真机时核对分支是否命中。
- `UB 预算` 仍从平台读取（`GetCoreMemSize(UB)`），A6 不硬编码 384KB。
- 该标识是**运行期平台判定**，不是编译宏：host 一份代码同时服务 A5/A6，kernel 侧按
  `__NPU_ARCH__` 选文件，二者互不影响。

## 10. 预期收益（理论估算）

| 维度 | 估算 | 依据 |
| --- | --- | --- |
| 每行向量指令数 | 24 → 19（-21%） | 5.4 表 |
| 状态 UB 流量 | -50%（V=128 时 128KB/token） | 5.4 表 |
| 每 token VF 进入/退出 + barrier | -1 次 VF、-1 次 `PipeBarrier` | 5.4 表 |
| 指令级并行度 | 1 行 → 4 行（8 条独立短指令流） | 5.2 |
| q/k normalize | 每行 -4 barrier、-1 次 V→S 同步、-1 次标量读回 | 6.1 |
| raw gate | 每行 barrier 从 5~8 次降到整块 1 次 | 6.2 |
| V-tile 循环次数 | `repeat` 少 1（示例 2→1） | 8.3 |
| 三发射兑现 | 行块独立 + 长指令与短指令重叠，理论上界为 950 的 1.5×；受 `ReduceSum` 依赖链与 UB 带宽限制，实际按 VF 段周期下降 20%~30% 量级预估 | 2.1 / 5.3 |

上述为解析估算，**未在真机上跑过 profiler**，只用于排序优化点与设定验收方向。

## 11. 回归与 A/B 方式

由于不再保留编译宏开关，A/B 只能按"代码版本"对比：

| 对比对象 | 方式 |
| --- | --- |
| A5 基线 | 直接跑 `ascend950` 同一 case，比较输出与耗时 |
| A6 旧实现（两段式 + 逐行预处理） | 用 git 回退到引入 A6 优化前的提交（或临时把 `Compute`/`NormalizeRows`/`ApplyGateInKernel` 的分发改回通用路径），跑同一 case |
| 单独评估某一项 | 临时局部修改（例如把 `MulAddDst` 拆回 `Mul`+`Add`，或把行块从 4 行改到 2 行），不做宏开关 |

tiling 侧的 A6 分支同样无开关，随平台自动生效；需要对比时把 `compileInfo_.isA6` 临时置 0 即回到 A5 候选表。

## 12. 风险与未验证项

1. **无 960dt 环境**：本次只做代码变更与静态自检，未编译、未运行、未做 profiler 采集；
   三发射的实际打包率依赖编译器调度，`ReduceSum` 是否被识别为长指令需用 profiler 确认。
   涉及的 MicroAPI 面为 `CreateMask<float, MaskPattern::ALL/VL1>`、带 mask 的
   `Adds/Muls/Sub/Mul/Exp/Ln/Sqrt/Div/Duplicate/MulAddDst`、`LoadDist::DIST_BRC_B32`、
   `StoreDist::DIST_FIRST_ELEMENT_B32`、以及 64 lane 连续 `DataCopy`。
   其中 `MaskPattern::VL1`、`ReduceSum`、`MulAddDst`、`DataCopy` 已在仓库内 `arch35` 的 VF 代码中使用，
   但**仍需在目标 toolkit 的 9201 头文件里确认可用**再编译。
2. **VF 退出后到普通 V 指令的同步**：`NormalizeRowsVF128`/`ApplyGateInKernelVF` 尾部保留一次
   `PipeBarrier<PIPE_V>`；`ProcessTokenFused128` 维持与 A5 `Compute` 尾部一致的"无 barrier"写法，
   若真机出现状态/输出读到旧值，优先在该处补 barrier 复测。
3. **寄存器压力**：融合 VF 峰值 24 个 `RegTensor`（8 token 常量 + 8 状态 + 8 中间）。
   若编译器报寄存器不足，退化手段依次为：(a) 行块 4 行 → 2 行（流数 8 → 4）；
   (b) 把 `q0/q1` 推迟到 `S@q` 之前再装载（少两个长生命周期寄存器，代价是 2 条载入指令）。
4. **`MulAddDst` 的 FMA 舍入**：见 5.5，与 A5 可能差 ≤1 ULP；这是本算子通用路径已有的口径。
5. **`V=256` 形状**：融合 VF 的行数为 `curSingleV`，`K=128` 前提不受 V 影响，但 `vStep` 与
   `repeat` 的取值会随 UB 变化，需按 8.3 重新核算。
6. **`repeat` 与 ring 深度的取舍**：`IsBetterProfile` 仍按"`repeat` 优先"排序，A6 上可能落在
   `repeat=1 + state ring 1` 的组合（见 8.3）。若真机显示 MTE3 写回成为瓶颈，可改为"A6 优先
   保证 `stateOutBufferNum=2`，`repeat` 允许 +1"，该策略变更只影响 tiling，不动 kernel。
7. **连续两段 vs 奇偶交织**：A6 改成连续两段后，每行多 2 条 256B 载入/回写指令（少 2 条 lane 置换）。
   若真机显示 `DIST_DINTLV_B32`/`DIST_INTLV_B32` 反而更省周期，可把 5.2 的装载/回写换回交织版本，
   行块结构与 `MulAddDst` 折叠不受影响。
8. **A6 归一化 epsilon 与当前 A5 的差异**：A6 用 `sqrt(sum + 1e-6)`（对齐 A2/A3 与 x8-pr 的 A5 优化），
   而本仓库 A5（arch35）仍是 `sqrt(sum)`；相对差异约 `1e-6 / (sum + 1e-6)`——常规输入 ~1e-8，
   `sum ≲ 1e-6` 的边界输入显著。若验收要求 A5/A6 在极小输入上逐位一致，
   把 `NormalizeRowsVF128` 中的 `Adds(RKDA_L2_NORM_EPSILON)` 去掉即可（其余结构不变）；
   等价做法是让 A5 同步改用 1e-6（对齐 A2/A3/x8-pr 口径）。

## 13. 代码落点

| 设计点 | 文件 | 位置 |
| --- | --- | --- |
| A6 常量 | `op_kernel/arch92/recurrent_kda.h` | `A6_ROW_BLOCK` / `RKDA_L2_NORM_EPSILON` |
| q/k normalize VF | 同上 | `NormalizeRowsVF128`，`NormalizeRows` 内按 `alignK_` 分发 |
| raw gate VF | 同上 | `ApplyGateInKernelVF`，`ApplyGateInKernel` 内分发 |
| token 级融合 VF | 同上 | `ProcessTokenFused128` / `ProcessTokenFused128Dispatch` |
| `Compute` 分发 | 同上 | `Compute`（`alignK_ == 128` 融合；否则通用路径） |
| 删除 A5 两段式 128 路径 | 同上 | 移除 `DecayMatVecReduce128` / `ProcessDeltaKQReduce128(Dispatch)`（arch35 保留） |
| 候选表 + ring 加深 | `op_host/recurrent_kda_tiling_processor.h` | `RKDA_MAX_OUT_BUFFER_NUM`、`DeepenOutRingForA6`、`FinalizeVStepFromUb` |
| 平台标识 | `op_host/recurrent_kda_tiling.h/.cpp` | `RecurrentKdaCompileInfo::isA6`、`InitCompileInfo`、`BuildProcessorContext`、`PrintTilingData` |

## 14. 验收建议（待真机）

1. 精度：用 `docs/design.md` 第 11 节的验收矩阵，对 `K=128,V=128`、`K=128,V=256`、
   `K-first/V-first`、`bf16/fp32 state`、`safe gate/unsafe gate`、`beta sigmoid 开/关` 组合做覆盖，
   要求与 A5 基线在容差内一致（`MulAddDst` 折叠可能带来 ≤1 ULP 差异，按算子既有容差判定）；
2. 性能：对比 A5 与 A6 的 `vec_ratio`、`mte2/mte3` 与总耗时，重点看 A6 的 `vec_ratio` 是否因三发射下降；
3. 回归：`OP_LOGD` 确认 `isA6=1` 且 `stateOutBufferNum/attnOutBufferNum` 实际加深到 2、
   `ubRestBytes >= 0`；
4. 若寄存器不足或 `MulAddDst` 在 9201 上不可用，按第 12 节风险 3/4 的退化手段复测。
