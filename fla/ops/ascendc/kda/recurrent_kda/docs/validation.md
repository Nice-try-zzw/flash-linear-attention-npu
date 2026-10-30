# RecurrentKda A6 validation

## Implemented path

- SOC: `ascend960dt`
- SOC version: `Ascend960DT_968B7`
- NPU arch: `dav-9201` / `__NPU_ARCH__ == 9201`
- Kernel entry: `op_kernel/arch92/recurrent_kda.h`
- Execution type: `KERNEL_TYPE_AIV_ONLY`
- Compile option: `--cce-no-dcache-flush`

`RecurrentKda` is a Vector-only operator. A6 uses a dedicated `arch92`
entry but shares the regbase implementation with A5 because there is no A6
Cube specialization or AIC/AIV pairing in this operator.

## Build

```bash
bash build.sh \
  --soc=ascend960dt \
  --pkg \
  --vendor_name=fla_npu \
  --ops=recurrent_kda
```

The generated kernel must be under:

```text
op_impl/ai_core/tbe/kernel/ascend960dt/recurrent_kda/
```

The kernel JSON must report `AIV_ONLY`; `arch35` must not be selected for A6.

## Stable-ABI regression

```bash
python -m unittest tests.test_stable_recurrent_kda
```

This covers:

- non-inplace scratch starts from the caller's `initial_state`;
- scratch uses independent storage and does not carry autograd state;
- the caller's state remains unchanged;
- `initial_state=None` creates an FP32 zero state.

## Accuracy cases

Run the existing KDA ATK matrix after installing the A6 wheel:

```bash
bash tests/atk/run_test_cpu.sh \
  -op=recurrent_kda \
  -soc=ascend960dt \
  -npu_device_id=0 \
  -scope=accuracy

bash tests/atk/run_test_cpu.sh \
  -op=recurrent_kda \
  -soc=ascend960dt \
  -npu_device_id=0 \
  -scope=determinism
```

Required coverage:

- BSND and TND;
- fixed length and `cu_seqlens`, including empty sequences;
- raw unsafe gate, raw safe gate and precomputed gate;
- FP32/BF16/FP16 gate and beta;
- V-first/K-first and FP32/BF16 state;
- `V=128` and `V=256`;
- packed and speculative state indices;
- in-place and non-in-place final state;
- non-contiguous Q/K/V and state;
- segments no longer than 8 tokens.

## Current verification boundary

Completed in this change:

- A6 source dispatch, operator registration and regbase tiling branch;
- Stable-ABI scratch fix and its two offline regressions;
- `git diff --check` and Python syntax checks.

Pending in an A6 build environment:

- kernel compilation with CANN 9.2.0 and the A6 toolchain;
- camodel/real-device accuracy and determinism;
- A6 performance profiling.

## A6 triple-issue optimization attempt

### Change under test

- Scope: `K=128` A6 RegBase paths only.
- Hot paths: `DecayMatVecReduce128A6` and `ProcessDeltaKQReduce128A6`.
- Scheduling change: process three independent V rows per group so the A6
  three-issue scheduler receives three independent multiply/update/reduce
  chains.
- Invariants: token order, per-row FP32 arithmetic order, reduction order,
  UB layout, workspace and synchronization are unchanged.
- Fallback: A5/A2/A3 use the original single-row loops; groups with fewer than
  three remaining V rows use the original single-row tail chain.

### Completed offline checks

```bash
python -m unittest tests.test_recurrent_kda_a6_sources tests.test_stable_recurrent_kda
git diff --check
```

Result: 7 tests passed; no whitespace errors. The Bash harness also passes
`bash -n`.

### Required A6 accuracy and performance verification

After installing an A6 build containing this change, run:

```bash
bash tests/atk/run_test_cpu.sh \
  -op=recurrent_kda \
  -soc=ascend960dt \
  -npu_device_id=0 \
  -scope=accuracy

bash tests/atk/run_test_cpu.sh \
  -op=recurrent_kda \
  -soc=ascend960dt \
  -npu_device_id=0 \
  -scope=determinism
```

Performance comparison must use the same confirmed A6 case against the
`A6-dev-recurrent_kda` baseline and this branch. Profile both builds with:

```bash
msprof \
  --application="bash tests/atk/run_test_cpu.sh -op=recurrent_kda -soc=ascend960dt -npu_device_id=0 -scope=performance" \
  --output=<new-output-directory>
```

Use the median `Task Duration(us)` for the `recurrent_kda` kernel from
`op_summary`, with at least five repeated runs per build. Also compare
`SIMD_ExecIPC`, `dcache_refill_count` and scalar `load_store_ratio`. No
performance result is claimed in this branch because no A6 toolchain or model
case was available in the current environment.
