# recurrent_kda A6 环境配置与验证方案

## 1. 当前结论

A6 实现使用 `ascend960dt`、`Ascend960DT_968B7`、`dav-9201` 和
`__NPU_ARCH__ == 9201`，复用 A5 的 arch35 AIV kernel。`recurrent_kda` 是
`KERNEL_TYPE_AIV_ONLY`，不存在 `chunk_fwd_h` 的 AIC:AIV 1:2 配对和跨核 round
握手，因此不需要移植 1:1 head 映射。

2026-10-21 提交的 camodel 日志只包含 SOC 识别、runtime 初始化、SQE 下发和仿真结束，
没有 `[CMP]`、cosine、max_abs 或 PASS/FAIL 对拍记录；最后收到 `SIGINT` 并终止。
因此该日志不能证明精度通过。由于当前也没有可用仿真环境，本轮不能产出 NPU 精度结论，
也没有数值偏差证据可用于修复 kernel。

## 2. 代码与环境映射

| 层 | A6 配置 |
| --- | --- |
| SOC 短名 | `ascend960dt` |
| soc version | `Ascend960DT_968B7` |
| NPU arch | `dav-9201` / `__NPU_ARCH__ == 9201` |
| kernel | `arch35/recurrent_kda.h`，AIV-only |
| kernel 选项 | `--cce-no-dcache-flush` |
| CANN 下限 | 9.2.0 |
| wheel tier | `a6` |
| OPP 路径 | `op_impl/ai_core/tbe/kernel/ascend960dt/recurrent_kda/` |

host tiling 把 A5 和 arch 号 9201 都置为 `useArch35`，保证 kernel 与 tiling 一致使用
arch35 的序列窗口、UB 预算和分核规则。A6 与 A5 相同，每段序列长度不超过 8。

## 3. 环境准备

### 3.1 真机或 CANN 容器

```bash
source /path/to/cann-9.2.0/set_env.sh
source /path/to/fla_npu_transformer/bin/set_env.bash
python scripts/check_npu_env.py
npu-smi info
```

需要 CANN 9.2.0 或更高版本、Ascend960DT toolkit/ops、匹配的 `torch`/`torch_npu`，
以及可工作的 CMake/GCC/BiSheng/`asc_opc` 编译工具链。

### 3.2 A6 camodel

```bash
export CANN=/path/to/cann-9.2.0
export SIMDIR="$CANN/x86_64-linux/simulator/dav_9201/camodel"
export LD_LIBRARY_PATH="$SIMDIR:$CANN/lib64:$CANN/lib64/plugin/opskernel:$CANN/lib64/plugin/nnengine:$LD_LIBRARY_PATH"
export ASCEND_HOME_PATH="$CANN"
export ASCEND_TOOLKIT_HOME="$CANN"
export ASCEND_OPP_PATH="$CANN/opp"
export ASCEND_TDT_UNINIT=1
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

若 `torch_npu` 的 SOC 表不识别 A6，需要使用 caller-aware
`aclrtGetSocName` shim：来自 `libtorch_npu.so` 的查询返回可识别的 A5 SOC 名，仅用于
前端查表；来自 ACL/GE/NNOP/tiling 的查询返回 `Ascend960DT_968B7`，用于查找 A6 OPP。

```bash
export LD_PRELOAD=/absolute/path/to/shim_soc.so
```

camodel 已知限制：

- 单个 case 可能耗时 15 分钟以上；
- 不提供有代表性的 profiling/周期数结论；
- 输出完成后进程可能在 `torch_npu` 析构阶段异常退出，多 case 必须在同一 Python
  进程内串行执行，并同时记录结果 JSON 和退出码；
- `top.json`、DDR checkpoint 等缺失告警必须结合最终 kernel 完成和 Python 对拍结果判断，
  告警本身不能替代精度结论。

## 4. 构建

```bash
bash build.sh \
  --soc=ascend960dt \
  --pkg \
  --vendor_name=fla_npu \
  --ops=recurrent_kda

./build_out/fla_npu_linux-*.run --install
python scripts/verify_libcust_opapi_md5.py \
  --run-package build_out/fla_npu_linux-*.run
```

验收要求：

- CMake 将 `ascend960dt` 映射到 `arch35`；
- 生成 `ascend960dt` 的 `recurrent_kda` kernel JSON/二进制和 config；
- kernel JSON 的任务类型为 AIV-only；
- run 包安装成功，MD5 一致性检查为 `[OK]`；
- 重启 Python 进程后加载的是本轮 OPP。

## 5. 无 ATK/acl 的 Python 精度矩阵

ATK 不可用时使用
[`scripts/validate_recurrent_kda_a6.py`](../../../../../../scripts/validate_recurrent_kda_a6.py)。
脚本不导入 ATK，也不直接调用 acl API，只使用公开
`fla_npu.ops.ascendc.recurrent_kda`、`torch_npu` 和独立 FP64 PyTorch 参考递推。

### 5.1 Harness 自检

该步骤不依赖 NPU、torch_npu 或 fla_npu：

```bash
python scripts/validate_recurrent_kda_a6.py \
  --suite full \
  --cpu-self-check
```

应输出 12 个 `[CPU-SELF-CHECK PASS]` 和
`[METRIC-SELF-CHECK PASS] gross mismatch rejected`。

### 5.2 A6 运行

```bash
# 单个最小 dense case
python scripts/validate_recurrent_kda_a6.py \
  --suite smoke \
  --case smoke_bsnd_fp32 \
  --device 0 \
  --json outputs/recurrent_kda_a6_smoke.json

# smoke 组（最小 dense + GVA raw gate）
python scripts/validate_recurrent_kda_a6.py \
  --suite smoke \
  --device 0 \
  --json outputs/recurrent_kda_a6_smoke_group.json

# 核心矩阵
python scripts/validate_recurrent_kda_a6.py \
  --suite core \
  --device 0 \
  --json outputs/recurrent_kda_a6_core.json

# 完整矩阵
python scripts/validate_recurrent_kda_a6.py \
  --suite full \
  --device 0 \
  --json outputs/recurrent_kda_a6_full.json
```

单项 PASS 必须同时满足：

- 所有可见输出 finite；
- cosine similarity >= `0.999`；
- relative L2 <= `0.01`；
- max abs error <= `0.1`；
- `rtol=0.01, atol=0.02` 下 matched ratio >= `0.99`；
- 非原位调用不修改输入 state；
- 重复运行场景的输出和 state 完全一致。

脚本逐输出打印 cosine、relative L2、max_abs、matched ratio 和最大误差位置，最终输出：

```text
[PASS] smoke_bsnd_fp32
  [PASS] out: cosine=... rel_l2=... max_abs=... matched=... max_idx=[...]
RESULT: PASS 1/1 cases
```

该独立判据用于 A6 smoke 和回归，不替代具备 ATK 时的正式 `mixed_tolerance_bm` 验收。

### 5.3 覆盖场景

| Suite | 场景 |
| --- | --- |
| smoke | BF16 QKV/FP32 state 最小 dense；GVA raw gate |
| core | 无 `cu_seqlens`、safe gate FP16、`V=256`、BF16/K-first state、TND 空序列、重复 state slot、非连续 QKV/state |
| full | core + speculative INT32 indices、无 final state、`initial_state=None`、同进程确定性 |

完整场景列表：

1. `smoke_bsnd_fp32`
2. `dense_no_cu`
3. `raw_gate_gva_bf16`
4. `safe_gate_fp16_v256`
5. `kfirst_bf16_state`
6. `varlen_empty_sequences`
7. `packed_indices_repeat`
8. `noncontiguous_qkv_state`
9. `speculative_indices_int32`
10. `no_final_state_non_inplace`
11. `initial_state_none`
12. `determinism_gva`

## 6. 正式 ATK（环境恢复后）

具备 ATK 和 acl 后，再补跑：

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -soc=ascend960dt \
  -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -soc=ascend960dt \
  -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -soc=ascend960dt \
  -npu_device_id=0 -scope=mssanitizer
```

## 7. 本轮静态证据

| 检查 | 结果 |
| --- | --- |
| A6 SOC/tier 映射和 wheel 定向单测 | 通过 |
| Python 脚本语法检查 | 通过 |
| 12 场景 CPU reference/harness 自检 | 通过 |
| 明显错误输出拒绝测试 | 通过 |
| 提交日志精度判定 | 无效：无对拍结果且被 `SIGINT` 终止 |
| A6 构建和 Python NPU 矩阵 | 未执行，当前无仿真环境 |

只有生成 `RESULT: PASS N/N cases` 的 JSON/日志后，才能将 A6 精度状态更新为通过。
