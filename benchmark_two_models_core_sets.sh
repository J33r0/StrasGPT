#!/usr/bin/env sh
set -eu

# Run benchmark_ws_vs_worksharing.sh for two models across predefined CPU sets.
#
# Predefined sets:
#   0-2, 1-6, 7, 0-7
#
# Each model is benchmarked with step counts 17 and 57.
#
# Example:
#   ./benchmark_two_models_core_sets.sh \
#     --model-a ~/model_zoo/Qwen3-0.6B \
#     --model-b ~/model_zoo/Llama3.2-1B \
#     --prompt-file test/prompts/goldilocks_8.txt \
#     --repeats 3 \
#     --out-dir bench_dual_models

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
BENCH_SCRIPT="$SCRIPT_DIR/benchmark_ws_vs_worksharing.sh"
BIN="$SCRIPT_DIR/strasgpt"
MODEL_A=""
MODEL_B=""
PROMPT_FILE=""
STEPS="17"
STEP_COUNTS="17 57"
REPEATS="15"
THREADS_MIN="1"
OUT_DIR="bench_dual_models"
EXTRA_ARGS=""

# Keep these predefined unless you intentionally edit this file.
CORE_SETS="7 3-6 0-2 0-7"

usage() {
  cat <<EOF
Usage: $0 --model-a <dir> --model-b <dir> --prompt-file <file> [options]

Required:
  --model-a <dir>          First model directory
  --model-b <dir>          Second model directory
  --prompt-file <file>     Prompt file path

Optional:
  --bench-script <path>    Benchmark launcher (default: ./benchmark_ws_vs_worksharing.sh)
  --bin <path>             Binary path (default: ./strasgpt)
  --repeats <n>            Repeats per (mode,thread) (default: 3)
  --threads-min <n>        Min thread count (default: 1)
  --out-dir <dir>          Output directory root (default: bench_dual_models)
  --extra-args "..."       Extra args passed as-is to strasgpt
  --help                   Show this help

CPU sets used:
  0-2, 3-6, 7, 0-7
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model-a) MODEL_A="$2"; shift 2 ;;
    --model-b) MODEL_B="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --bench-script) BENCH_SCRIPT="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --threads-min) THREADS_MIN="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [ -z "$MODEL_A" ] || [ -z "$MODEL_B" ] || [ -z "$PROMPT_FILE" ]; then
  echo "Error: --model-a, --model-b and --prompt-file are required." >&2
  usage
  exit 1
fi

if [ ! -f "$BENCH_SCRIPT" ]; then
  echo "Error: benchmark script not found: $BENCH_SCRIPT" >&2
  exit 1
fi

if [ ! -x "$BIN" ]; then
  echo "Error: binary not found or not executable: $BIN" >&2
  exit 1
fi

if [ ! -d "$MODEL_A" ]; then
  echo "Error: model directory not found: $MODEL_A" >&2
  exit 1
fi

if [ ! -d "$MODEL_B" ]; then
  echo "Error: model directory not found: $MODEL_B" >&2
  exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
  echo "Error: prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

count_cpu_list() {
  # Count CPUs in list syntax like "0-2,4,6-7".
  list="$1"
  count=0
  oldifs="$IFS"
  IFS=','
  for part in $list; do
    case "$part" in
      *-*)
        start=${part%-*}
        end=${part#*-}
        count=$((count + end - start + 1))
        ;;
      *)
        count=$((count + 1))
        ;;
    esac
  done
  IFS="$oldifs"
  echo "$count"
}

safe_name() {
  # Keep path component readable and filesystem-friendly.
  echo "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

run_model() {
  model_path="$1"
  model_label="$2"

  model_dir="$RUN_ROOT/${model_label}"
  mkdir -p "$model_dir"

  for cpu_set in $CORE_SETS; do
    tmax="$(count_cpu_list "$cpu_set")"

    if [ "$tmax" -lt "$THREADS_MIN" ]; then
      echo "[WARN] Skipping model=$model_label cpu_set=$cpu_set because threads-max ($tmax) < threads-min ($THREADS_MIN)"
      continue
    fi

    cpu_tag="$(echo "$cpu_set" | tr ',' '_' | tr '-' '_')"
    out_subdir="$model_dir/cpu_${cpu_tag}"
    mkdir -p "$out_subdir"

    for steps in $STEP_COUNTS; do
      step_dir="$out_subdir/steps_${steps}"
      mkdir -p "$step_dir"

      echo "[RUN] model=$model_label steps=$steps cpu_set=$cpu_set threads=${THREADS_MIN}..${tmax} repeats=$REPEATS"

      if [ -n "$EXTRA_ARGS" ]; then
        if ! sh "$BENCH_SCRIPT" \
          --bin "$BIN" \
          --model "$model_path" \
          --prompt-file "$PROMPT_FILE" \
          --steps "$steps" \
          --threads-min "$THREADS_MIN" \
          --threads-max "$tmax" \
          --repeats "$REPEATS" \
          --out-dir "$step_dir" \
          --cpu-list "$cpu_set" \
          --extra-args "$EXTRA_ARGS"; then
          RUN_FAILED=1
        fi
      else
        if ! sh "$BENCH_SCRIPT" \
          --bin "$BIN" \
          --model "$model_path" \
          --prompt-file "$PROMPT_FILE" \
          --steps "$steps" \
          --threads-min "$THREADS_MIN" \
          --threads-max "$tmax" \
          --repeats "$REPEATS" \
          --out-dir "$step_dir" \
          --cpu-list "$cpu_set"; then
          RUN_FAILED=1
        fi
      fi
    done
  done
}

mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$OUT_DIR/two_models_core_sets_${TS}"
mkdir -p "$RUN_ROOT"

RUN_FAILED=0

run_model "$MODEL_A" "$(safe_name "$(basename "$MODEL_A")")"
run_model "$MODEL_B" "$(safe_name "$(basename "$MODEL_B")")"

echo
echo "Done."
echo "- Root output: $RUN_ROOT"
echo "- CPU sets   : $CORE_SETS"

if [ "$RUN_FAILED" -ne 0 ]; then
  echo "One or more benchmark batches failed. Check logs under: $RUN_ROOT" >&2
  exit 1
fi
