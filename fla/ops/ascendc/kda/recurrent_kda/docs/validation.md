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

2026-10-22 的单 case 对拍已定位到 `smoke_bsnd_fp32` 的具体问题：NPU 指标与把
`initial_state` 全零后运行 FP64 参考实现得到的指标完全一致。根因不是 A6 arch35
kernel 算术，而是 Stable-ABI 的 `inplace_final_state=False` 路径使用
`torch.empty_like(initial_state)` 创建 scratch，却没有先复制调用方的
`initial_state`。该路径已改为 `initial_state.clone()`，并增加离线回归
`tests/test_stable_recurrent_kda.py` 检查 scratch 内容、独立存储和输入 state 不变。

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

本次 non-inplace 问题修复位于 wheel 内的 Python Stable-ABI 入口
`fla_npu/ops/ascendc/_stable.py`，只安装 OPP run 包不会更新该文件。需要同时重新构建并
安装 wheel：

```bash
FLA_NPU_SOC=ascend960dt FLA_NPU_OPS=recurrent_kda \
  python scripts/build_wheel.py

python -m pip install --force-reinstall --no-cache-dir --no-deps \
  dist/flash_linear_attention_npu_a6-*.whl
```

重新安装后必须启动新的 Python 进程再运行 camodel 用例。

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

应输出 13 个 `[CPU-SELF-CHECK PASS]` 和
`[METRIC-SELF-CHECK PASS] gross mismatch rejected`。

Stable-ABI scratch 回归：

```bash
python -m unittest tests.test_stable_recurrent_kda
```

应输出 `Ran 2 tests ... OK`，覆盖输入 state 被复制到独立 scratch、原输入不被修改、
scratch 不带 autograd 状态，以及 `initial_state=None` 时创建 FP32 零 state。

### 5.2 A6 运行

推荐使用 camodel 专用入口。它负责加载 CANN 和 wheel 环境、把 camodel 放到
`LD_LIBRARY_PATH` 首位、设置 `ASCEND_TDT_UNINIT=1` 和 `LD_PRELOAD` shim，并在同一
Python 进程内串行执行整组用例：

```bash
bash scripts/run_recurrent_kda_a6_camodel.sh \
  --cann /path/to/cann-9.2.0 \
  --simdir /path/to/cann-9.2.0/x86_64-linux/simulator/dav_9201/camodel \
  --shim /absolute/path/to/shim_soc.so \
  --package-env /path/to/site-packages/fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash \
  --python /path/to/python \
  --suite smoke \
  --case smoke_bsnd_fp32 \
  --device 0
```

也可以使用环境变量：

```bash
export CANN_HOME=/path/to/cann-9.2.0
export SIM_DIR="$CANN_HOME/x86_64-linux/simulator/dav_9201/camodel"
export SHIM_SOC=/absolute/path/to/shim_soc.so
export FLA_NPU_ENV=/path/to/fla_npu_transformer/bin/set_env.bash
export FLA_NPU_PYTHON=/path/to/python

bash scripts/run_recurrent_kda_a6_camodel.sh --suite smoke
```

运行前可用 `--dry-run` 检查最终解析出的 Python 命令。默认单次运行超时为 21600 秒；
JSON 和完整日志写到 `outputs/recurrent_kda_a6_camodel/`。camodel 在全部 case 完成后
析构崩溃时可能返回 139，wrapper 会保留真实退出码并打印 JSON 中的
`PASS/FAIL N/M cases`，精度结果以 JSON 为准。

wrapper 还会检查当前导入的 wheel 中 `_stable.py` 是否包含本次 scratch 修复；若仍指向
旧 wheel，会直接报 `installed wheel is stale`，避免把旧 Python runtime 的结果误认为
新修复仍失败。确认环境后才允许使用 `--skip-package-check`。

不经过 wrapper、已手工配置好环境时，也可以直接运行 Python：

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

完整用例表（所有 case 均为 `K=128`、BF16 QKV）：

| Case | Layout/Shape | Gate/Beta/State | 关键分支 |
| --- | --- | --- | --- |
| `smoke_bsnd_fp32` | BSND，B1/T2/H1/HV1/V128 | FP32/FP32/FP32 V-first | 最小 dense、FP64 golden |
| `smoke_bsnd_fp32_inplace` | BSND，B1/T2/H1/HV1/V128 | FP32/FP32/FP32 V-first | 与最小 dense 同输入，inplace 对照 |
| `dense_no_cu` | BSND，B2/T3/H1/HV1/V128 | FP32/FP32/FP32 | 不传 `cu_seqlens`，多 batch |
| `raw_gate_gva_bf16` | BSND，B1/T4/H2/HV4/V128 | BF16/BF16/FP32 | raw gate、GVA、L2 norm、beta sigmoid + neg |
| `safe_gate_fp16_v256` | BSND，B2/T2/H1/HV2/V256 | FP16/FP16/FP32 | safe gate、FP16、`V=256` |
| `kfirst_bf16_state` | BSND，B1/T4/H2/HV4/V128 | FP32/FP32/BF16 | K-first state |
| `varlen_empty_sequences` | TND，T17，lengths `[1,0,7,0,9]` | BF16/FP32/FP32 | 多个空序列、每段 <=8 |
| `packed_indices_repeat` | BSND，B1/T8/H2/HV4/V128 | FP32/FP32/BF16 | packed INT64 indices、重复 slot、原位 state |
| `noncontiguous_qkv_state` | BSND，B1/T4/H2/HV4/V128 | FP16/BF16/FP32 | Q/K/V feature gap、state head gap |
| `speculative_indices_int32` | BSND，B2/T4/H1/HV2/V128 | FP32/FP32/FP32 | 二维 INT32 indices、accepted tokens |
| `no_final_state_non_inplace` | BSND，B1/T2/H1/HV1/V128 | FP32/FP32/FP32 | 不返回 final state、输入 state 不被修改 |
| `initial_state_none` | BSND，B1/T3/H2/HV4/V128 | BF16/FP32/FP32 | raw+safe gate、wrapper 创建零 state |
| `determinism_gva` | BSND，B2/T8/H2/HV4/V128 | BF16/FP32/FP32 | GVA、beta sigmoid、同进程重复两次 |

单个用例命令：

```bash
bash scripts/run_recurrent_kda_a6_camodel.sh \
  --suite full \
  --case <case-name> \
  --device 0
```

例如仅运行 GVA raw gate：

```bash
bash scripts/run_recurrent_kda_a6_camodel.sh \
  --suite full \
  --case raw_gate_gva_bf16 \
  --device 0
```

一次性运行多个指定 case：重复传入 `--case`。

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
| 13 场景 CPU reference/harness 自检 | 通过 |
| 明显错误输出拒绝测试 | 通过 |
| Stable-ABI non-inplace scratch 继承 initial_state 回归 | 通过 |
| 提交日志精度判定 | 无效：无对拍结果且被 `SIGINT` 终止 |
| A6 构建和 Python NPU 矩阵 | 未执行，当前无仿真环境 |

只有生成 `RESULT: PASS N/N cases` 的 JSON/日志后，才能将 A6 精度状态更新为通过。

## 8. 重装新 wheel 后仍失败时的二分诊断

先在目标 Python 环境确认实际导入的是新 wheel，而不是旧 site-packages：

```bash
python -c "import inspect, fla_npu, fla_npu.ops.ascendc._stable as s; print(fla_npu.__version__); print(fla_npu.__file__); print(s.__file__); print('detach().clone' in inspect.getsource(s.npu_recurrent_kda))"
```

最后一项必须为 `True`。随后运行同输入的 non-inplace 与 inplace 对照：

```bash
# Stable-ABI non-inplace（本次修复路径）
FLA_NPU_STABLE_ABI=stable \
bash scripts/run_recurrent_kda_a6_camodel.sh \
  --suite full --case smoke_bsnd_fp32 --device 0

# 同一输入的 inplace kernel 对照
FLA_NPU_STABLE_ABI=stable \
bash scripts/run_recurrent_kda_a6_camodel.sh \
  --suite full --case smoke_bsnd_fp32_inplace --device 0

# 强制 ctypes reference，排除 Stable-ABI 影响
FLA_NPU_STABLE_ABI=ctypes \
bash scripts/run_recurrent_kda_a6_camodel.sh \
  --suite full --case smoke_bsnd_fp32 --device 0
```

判读：

| stable non-inplace | stable inplace | ctypes non-inplace | 结论 |
| --- | --- | --- | --- |
| PASS | PASS | PASS | Python scratch 修复和 kernel 路径均通过 |
| FAIL | PASS | PASS | 仍是 Python Stable-ABI scratch 问题，检查是否加载新 wheel |
| FAIL | FAIL | FAIL | kernel/host tiling 没有正确读取 `initial_state`，转 DumpTensor 定位 CopyIn |

新脚本的 JSON 会记录 `fla_npu.version/file`、实际 backend、fallback 次数，以及
actual/expected tensor norm。若 actual norm 接近零而 expected norm 正常，优先检查
`PrefetchState` / `LoadPrefetchedState`，而不是继续调整数值容差。
