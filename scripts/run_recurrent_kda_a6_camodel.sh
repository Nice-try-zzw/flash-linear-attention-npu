#!/usr/bin/env bash
# Run recurrent_kda A6 precision cases against the dav_9201 camodel runtime.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${FLA_NPU_PYTHON:-${PYTHON:-python3}}"
CANN_HOME="${CANN_HOME:-${ASCEND_HOME_PATH:-/home/z00981887/Ascend/cann}}"
SIM_DIR="${SIM_DIR:-}"
SHIM_SOC="${SHIM_SOC:-}"
FLA_NPU_ENV="${FLA_NPU_ENV:-${FLA_NPU_OPP_ENV:-}}"
DEVICE="${DEVICE:-0}"
SUITE="${SUITE:-smoke}"
TIMEOUT_SECONDS="${CAMODEL_TIMEOUT:-21600}"
OUTPUT_ROOT="${CAMODEL_OUTPUT_ROOT:-${REPO_ROOT}/outputs/recurrent_kda_a6_camodel}"
JSON_PATH=""
DRY_RUN=FALSE
SKIP_PREFLIGHT=FALSE
SKIP_PACKAGE_CHECK=FALSE
CASE_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_recurrent_kda_a6_camodel.sh [options]

Required environment (flags can override):
  CANN_HOME       CANN 9.2.0 root, containing set_env.sh
  SIM_DIR         .../x86_64-linux/simulator/dav_9201/camodel
  SHIM_SOC        caller-aware aclrtGetSocName shim, e.g. shim_soc.so
  FLA_NPU_ENV     installed fla_npu_transformer/bin/set_env.bash (recommended)

Options:
  --suite <smoke|core|full>   Case suite, default smoke.
  --case <name>               Run one named case; may be repeated.
  --device <id>               torch_npu device, default 0.
  --python <path>             Python executable.
  --cann <path>               CANN root.
  --simdir <path>             A6 camodel directory.
  --shim <path>               SOC shim shared object.
  --package-env <path>        fla_npu vendor set_env.bash.
  --timeout <seconds>         Per whole suite timeout, default 21600.
  --output-root <path>        Log/JSON root.
  --json <path>               Exact JSON output path.
  --skip-preflight            Skip the no-NPU CPU harness self-check.
  --skip-package-check        Skip the installed-wheel scratch-fix check.
  --dry-run                   Print the resolved command and exit.
  -h, --help                  Show this help.

Examples:
  bash scripts/run_recurrent_kda_a6_camodel.sh --suite smoke
  bash scripts/run_recurrent_kda_a6_camodel.sh \
    --suite smoke --case smoke_bsnd_fp32
  bash scripts/run_recurrent_kda_a6_camodel.sh --suite core
EOF
}

die() {
  echo "[A6-CAMODEL][ERROR] $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) SUITE="${2:?missing --suite value}"; shift 2 ;;
    --case) CASE_ARGS+=(--case "${2:?missing --case value}"); shift 2 ;;
    --device) DEVICE="${2:?missing --device value}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?missing --python value}"; shift 2 ;;
    --cann) CANN_HOME="${2:?missing --cann value}"; shift 2 ;;
    --simdir) SIM_DIR="${2:?missing --simdir value}"; shift 2 ;;
    --shim) SHIM_SOC="${2:?missing --shim value}"; shift 2 ;;
    --package-env) FLA_NPU_ENV="${2:?missing --package-env value}"; shift 2 ;;
    --timeout) TIMEOUT_SECONDS="${2:?missing --timeout value}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?missing --output-root value}"; shift 2 ;;
    --json) JSON_PATH="${2:?missing --json value}"; shift 2 ;;
    --skip-preflight) SKIP_PREFLIGHT=TRUE; shift ;;
    --skip-package-check) SKIP_PACKAGE_CHECK=TRUE; shift ;;
    --dry-run) DRY_RUN=TRUE; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$SUITE" in
  smoke|core|full) ;;
  *) die "unsupported suite: ${SUITE}" ;;
esac
[[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || die "timeout must be an integer"

if [[ -z "$SIM_DIR" ]]; then
  for candidate in \
    "${CANN_HOME}/x86_64-linux/simulator/dav_9201/camodel" \
    "${CANN_HOME}/../x86_64-linux/simulator/dav_9201/camodel"; do
    if [[ -d "$candidate" ]]; then
      SIM_DIR="$(cd "$candidate" && pwd)"
      break
    fi
  done
fi

if [[ -z "$SHIM_SOC" ]]; then
  for candidate in \
    "${REPO_ROOT}/shim_soc.so" \
    "/home/wnc/a6_camodel_work/shim_soc.so" \
    "/home/z00981887/a6_camodel_work/shim_soc.so"; do
    if [[ -f "$candidate" ]]; then
      SHIM_SOC="$candidate"
      break
    fi
  done
fi

if [[ "$PYTHON_BIN" != */* ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || die "Python executable not found"
[[ -d "$SIM_DIR" ]] || die "camodel directory not found; set SIM_DIR or --simdir"
[[ -f "$SHIM_SOC" ]] || die "SOC shim not found; set SHIM_SOC or --shim"

export CANN_HOME SIM_DIR SHIM_SOC
export ASCEND_HOME_PATH="$CANN_HOME"
export ASCEND_TOOLKIT_HOME="$CANN_HOME"
export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-${CANN_HOME}/opp}"
export ASCEND_AICPU_PATH="${ASCEND_AICPU_PATH:-${CANN_HOME}}"
export TOOLCHAIN_HOME="${TOOLCHAIN_HOME:-${CANN_HOME}/toolkit}"
export ASCEND_TDT_UNINIT=1
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-1}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-1}"
export ASCEND_HOST_LOG_LEVEL="${ASCEND_HOST_LOG_LEVEL:-1}"

if [[ -f "${CANN_HOME}/set_env.sh" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CANN_HOME}/set_env.sh"
  set -u
fi
if [[ -n "$FLA_NPU_ENV" ]]; then
  [[ -f "$FLA_NPU_ENV" ]] || die "fla_npu environment file not found: ${FLA_NPU_ENV}"
  set +u
  # shellcheck source=/dev/null
  source "$FLA_NPU_ENV"
  set -u
fi

export LD_LIBRARY_PATH="${SIM_DIR}:${CANN_HOME}/lib64:${CANN_HOME}/lib64/plugin/opskernel:${CANN_HOME}/lib64/plugin/nnengine:${CANN_HOME}/devlib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${CANN_HOME}/python/site-packages:${CANN_HOME}/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH:-}"
export LD_PRELOAD="${SHIM_SOC}${LD_PRELOAD:+:${LD_PRELOAD}}"
export FLA_NPU_SOC=ascend960dt
export FLA_NPU_CAMODEL_MODE=1
export FLA_NPU_CAMODEL_SIM_DIR="$SIM_DIR"
export FLA_NPU_CAMODEL_SHIM="$SHIM_SOC"
ulimit -n 65535 2>/dev/null || true

mkdir -p "$OUTPUT_ROOT"
if [[ -z "$JSON_PATH" ]]; then
  JSON_PATH="${OUTPUT_ROOT}/recurrent_kda_a6_${SUITE}_$(date +%Y%m%d_%H%M%S).json"
fi
mkdir -p "$(dirname "$JSON_PATH")"
LOG_PATH="${JSON_PATH%.json}.log"

RUN_ARGS=(
  "${REPO_ROOT}/scripts/validate_recurrent_kda_a6.py"
  --suite "$SUITE"
  --device "$DEVICE"
  --json "$JSON_PATH"
)
if [[ ${#CASE_ARGS[@]} -gt 0 ]]; then
  RUN_ARGS+=("${CASE_ARGS[@]}")
fi

echo "[A6-CAMODEL] CANN_HOME=${CANN_HOME}"
echo "[A6-CAMODEL] SIM_DIR=${SIM_DIR}"
echo "[A6-CAMODEL] SHIM_SOC=${SHIM_SOC}"
echo "[A6-CAMODEL] PYTHON_BIN=${PYTHON_BIN}"
echo "[A6-CAMODEL] suite=${SUITE} device=${DEVICE} timeout=${TIMEOUT_SECONDS}s"
echo "[A6-CAMODEL] JSON=${JSON_PATH}"
echo "[A6-CAMODEL] LOG=${LOG_PATH}"

if [[ "$DRY_RUN" == "TRUE" ]]; then
  printf '[A6-CAMODEL] COMMAND='
  printf '%q ' "$PYTHON_BIN" -u "${RUN_ARGS[@]}"
  printf '\n'
  exit 0
fi

if [[ "$SKIP_PREFLIGHT" != "TRUE" ]]; then
  "$PYTHON_BIN" "${REPO_ROOT}/scripts/validate_recurrent_kda_a6.py" \
    --suite "$SUITE" "${CASE_ARGS[@]}" --cpu-self-check
fi

if [[ "$SKIP_PACKAGE_CHECK" != "TRUE" ]]; then
  "$PYTHON_BIN" -c '
import inspect
import fla_npu
from fla_npu.ops.ascendc import _stable

source = inspect.getsource(_stable.npu_recurrent_kda)
assert "initial_state.detach().clone()" in source, (
    f"installed wheel at {fla_npu.__file__} predates the recurrent_kda "
    "non-inplace scratch-state fix"
)
print("[A6-CAMODEL] installed fla_npu", getattr(fla_npu, "__version__", "unknown"), fla_npu.__file__)
print("[A6-CAMODEL] stable source", _stable.__file__)
print("[A6-CAMODEL] SCRATCH_FIX_LOADED")
' || die "installed wheel is stale; rebuild and reinstall the a6 wheel"
fi

"$PYTHON_BIN" -c 'import torch, torch_npu; print("[A6-CAMODEL] torch", torch.__version__, "torch_npu", torch_npu.__version__)'

set +e
timeout --signal=INT --kill-after=60s "$TIMEOUT_SECONDS" \
  "$PYTHON_BIN" -u "${RUN_ARGS[@]}" 2>&1 | tee "$LOG_PATH"
RUN_RC=${PIPESTATUS[0]}
set -e

echo "[A6-CAMODEL] EXIT_CODE=${RUN_RC}"
if [[ -f "$JSON_PATH" ]]; then
  "$PYTHON_BIN" - "$JSON_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as file:
    result = json.load(file)
print(
    "[A6-CAMODEL] JSON_RESULT="
    f"{'PASS' if result.get('passed') else 'FAIL'} "
    f"{sum(bool(case.get('passed')) for case in result.get('cases', []))}/"
    f"{len(result.get('cases', []))} cases"
)
PY
else
  echo "[A6-CAMODEL] JSON_RESULT=MISSING"
fi

# Camodel may segfault during torch_npu teardown after all cases completed.
# Preserve its real exit code; the JSON is the precision authority.
exit "$RUN_RC"
