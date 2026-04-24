#!/usr/bin/env sh
set -eu

# Benchmark StrasGPT with WS disabled/enabled across thread counts.
# Designed to run on Linux and Android shells (toybox-compatible tools).
#
# Example:
#   ./benchmark_ws_vs_worksharing.sh \
#     --bin ./strasgpt \
#     --model ~/model_zoo/Qwen3-0.6B \
#     --prompt-file test/prompts/goldilocks_8.txt \
#     --steps 17 \
#     --threads-min 1 \
#     --threads-max 8 \
#     --repeats 3
#     --out-dir bench_results \
#     --cpu-list "0-3" \

BIN="./strasgpt"
MODEL=""
PROMPT_FILE=""
STEPS="17"
THREADS_MIN="1"
THREADS_MAX="8"
REPEATS="3"
OUT_DIR="bench_results"
EXTRA_ARGS=""
CPU_LIST=""

usage() {
  cat <<EOF
Usage: $0 --model <dir> --prompt-file <file> [options]

Required:
  --model <dir>            Model directory
  --prompt-file <file>     Prompt file path

Optional:
  --bin <path>             Binary path (default: ./strasgpt)
  --steps <n>              -n value (default: 17)
  --threads-min <n>        Min thread count (default: 1)
  --threads-max <n>        Max thread count (default: 8)
  --repeats <n>            Repeats per (mode,thread) (default: 3)
  --out-dir <dir>          Output directory (default: bench_results)
  --cpu-list <list>        CPU list (e.g. 0-3 or 4,5,6). Script auto-adapts
                            to taskset flavor (-c or bitmask mode).
  --extra-args "..."       Extra args passed as-is to strasgpt
  --help                   Show this help

Modes tested:
  ws_off: STRASGPT_WS_ENABLE=0
  ws_on : STRASGPT_WS_ENABLE=1
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --bin) BIN="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --threads-min) THREADS_MIN="$2"; shift 2 ;;
    --threads-max) THREADS_MAX="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --cpu-list) CPU_LIST="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [ -z "$MODEL" ] || [ -z "$PROMPT_FILE" ]; then
  echo "Error: --model and --prompt-file are required." >&2
  usage
  exit 1
fi

if [ ! -x "$BIN" ]; then
  echo "Error: binary not found or not executable: $BIN" >&2
  exit 1
fi

if ! "$BIN" --help >/dev/null 2>&1; then
  echo "Error: binary cannot run on this machine (wrong architecture or missing runtime): $BIN" >&2
  exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
  echo "Error: prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
CSV="$OUT_DIR/ws_vs_worksharing_${TS}.csv"
SUMMARY="$OUT_DIR/ws_vs_worksharing_${TS}_summary.txt"

printf "timestamp,mode,threads,repeat,exit_code,prefill_s,prefill_tps,decode_s,decode_tps,log_file\n" > "$CSV"

extract_prefill_s() {
  echo "$1" | sed -n 's/.*prefill):[[:space:]]*[0-9][0-9]* tokens in[[:space:]]*\([0-9.][0-9.]*\) s.*/\1/p' | tail -n 1
}

extract_prefill_tps() {
  echo "$1" | sed -n 's/.*prefill):.*(\([0-9.][0-9.]*\) token\/s).*/\1/p' | tail -n 1
}

extract_decode_s() {
  echo "$1" | sed -n 's/.*decode):[[:space:]]*[0-9][0-9]* tokens in[[:space:]]*\([0-9.][0-9.]*\) s.*/\1/p' | tail -n 1
}

extract_decode_tps() {
  echo "$1" | sed -n 's/.*decode):.*(\([0-9.][0-9.]*\) token\/s).*/\1/p' | tail -n 1
}

cpulist_to_mask() {
  # Convert list like "0-2,4,6-7" to a hex bitmask (without 0x prefix).
  list="$1"
  mask=0
  oldifs="$IFS"
  IFS=','
  for part in $list; do
    case "$part" in
      *-*)
        start=${part%-*}
        end=${part#*-}
        i=$start
        while [ "$i" -le "$end" ]; do
          mask=$((mask | (1 << i)))
          i=$((i + 1))
        done
        ;;
      *)
        mask=$((mask | (1 << part)))
        ;;
    esac
  done
  IFS="$oldifs"
  printf "%x\n" "$mask"
}

run_one() {
  mode="$1"
  threads="$2"
  rep="$3"

  run_ts="$(date +%Y%m%d_%H%M%S)"
  log_file="$OUT_DIR/run_${run_ts}_${mode}_t${threads}_r${rep}.log"

  echo "[RUN] mode=$mode threads=$threads repeat=$rep"

  cmd_exit=0
  taskset_mask=""
  if [ -n "$CPU_LIST" ]; then
    taskset_mask="$(cpulist_to_mask "$CPU_LIST")"
  fi

  if [ "$mode" = "ws_on" ]; then
    if [ -n "$CPU_LIST" ]; then
      if taskset -c "$CPU_LIST" sh -c ':' >/dev/null 2>&1; then
        if ! env STRASGPT_WS_ENABLE=1 taskset -c "$CPU_LIST" "$BIN" -m "$MODEL" -f "$PROMPT_FILE" -n "$STEPS" -t "$threads" $EXTRA_ARGS > "$log_file" 2>&1; then
          cmd_exit=1
        fi
      elif env STRASGPT_WS_ENABLE=1 taskset "$taskset_mask" "$BIN" -m "$MODEL" -f "$PROMPT_FILE" -n "$STEPS" -t "$threads" $EXTRA_ARGS > "$log_file" 2>&1; then
        :
      elif ! env STRASGPT_WS_ENABLE=1 taskset "0x$taskset_mask" "$BIN" -m "$MODEL" -f "$PROMPT_FILE" -n "$STEPS" -t "$threads" $EXTRA_ARGS > "$log_file" 2>&1; then
        cmd_exit=1
      fi
    elif ! STRASGPT_WS_ENABLE=1 "$BIN" -m "$MODEL" -f "$PROMPT_FILE" -n "$STEPS" -t "$threads" $EXTRA_ARGS > "$log_file" 2>&1; then
      cmd_exit=1
    fi
  else
    if [ -n "$CPU_LIST" ]; then
      if taskset -c "$CPU_LIST" sh -c ':' >/dev/null 2>&1; then
        if ! env STRASGPT_WS_ENABLE=0 taskset -c "$CPU_LIST" "$BIN" -m "$MODEL" -f "$PROMPT_FILE" -n "$STEPS" -t "$threads" $EXTRA_ARGS > "$log_file" 2>&1; then
          cmd_exit=1
        fi
      elif env STRASGPT_WS_ENABLE=0 taskset "$taskset_mask" "$BIN" -m "$MODEL" -f "$PROMPT_FILE" -n "$STEPS" -t "$threads" $EXTRA_ARGS > "$log_file" 2>&1; then
        :
      elif ! env STRASGPT_WS_ENABLE=0 taskset "0x$taskset_mask" "$BIN" -m "$MODEL" -f "$PROMPT_FILE" -n "$STEPS" -t "$threads" $EXTRA_ARGS > "$log_file" 2>&1; then
        cmd_exit=1
      fi
    elif ! STRASGPT_WS_ENABLE=0 "$BIN" -m "$MODEL" -f "$PROMPT_FILE" -n "$STEPS" -t "$threads" $EXTRA_ARGS > "$log_file" 2>&1; then
      cmd_exit=1
    fi
  fi

  # Capture process status from log by checking if performance lines exist.
  out="$(cat "$log_file")"
  prefill_s="$(extract_prefill_s "$out")"
  prefill_tps="$(extract_prefill_tps "$out")"
  decode_s="$(extract_decode_s "$out")"
  decode_tps="$(extract_decode_tps "$out")"

  exit_code="$cmd_exit"
  if [ "$cmd_exit" -ne 0 ] || [ -z "$prefill_tps" ] || [ -z "$decode_tps" ]; then
    exit_code="1"
  fi

  [ -n "$prefill_s" ] || prefill_s="NA"
  [ -n "$prefill_tps" ] || prefill_tps="NA"
  [ -n "$decode_s" ] || decode_s="NA"
  [ -n "$decode_tps" ] || decode_tps="NA"

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$(date +%Y-%m-%dT%H:%M:%S)" "$mode" "$threads" "$rep" "$exit_code" \
    "$prefill_s" "$prefill_tps" "$decode_s" "$decode_tps" "$log_file" >> "$CSV"
}

t="${THREADS_MIN}"
while [ "$t" -le "$THREADS_MAX" ]; do
  r="1"
  while [ "$r" -le "$REPEATS" ]; do
    run_one "ws_off" "$t" "$r"
    run_one "ws_on" "$t" "$r"
    r=$((r + 1))
  done
  t=$((t + 1))
done

{
  echo "=== Summary (average prefill/decode token/s by thread) ==="
  echo "CSV: $CSV"
  echo
  awk -F',' -v tmin="$THREADS_MIN" -v tmax="$THREADS_MAX" '
    NR==1 { next }
    $5=="0" {
      key=$2":"$3
      if ($7!="NA") {
        prefill_sum[key]+=$7
        prefill_cnt[key]+=1
      }
      if ($9!="NA") {
        decode_sum[key]+=$9
        decode_cnt[key]+=1
      }
    }
    END {
      print "threads,ws_off_prefill_tps,ws_on_prefill_tps,prefill_speedup,prefill_speedup_pct,ws_off_decode_tps,ws_on_decode_tps,decode_speedup,decode_speedup_pct"
      row_count=0
      sum_prefill_speedup=0
      sum_decode_speedup=0
      for (t=tmin; t<=tmax; t++) {
        k0="ws_off:" t
        k1="ws_on:" t
        if (prefill_cnt[k0]>0 || prefill_cnt[k1]>0 || decode_cnt[k0]>0 || decode_cnt[k1]>0) {
          prefill_off=(prefill_cnt[k0]>0)?prefill_sum[k0]/prefill_cnt[k0]:0
          prefill_on=(prefill_cnt[k1]>0)?prefill_sum[k1]/prefill_cnt[k1]:0
          prefill_speedup=(prefill_off>0)?prefill_on/prefill_off:0
          prefill_speedup_pct=(prefill_off>0)?((prefill_on-prefill_off)/prefill_off)*100:0

          decode_off=(decode_cnt[k0]>0)?decode_sum[k0]/decode_cnt[k0]:0
          decode_on=(decode_cnt[k1]>0)?decode_sum[k1]/decode_cnt[k1]:0
          decode_speedup=(decode_off>0)?decode_on/decode_off:0
          decode_speedup_pct=(decode_off>0)?((decode_on-decode_off)/decode_off)*100:0

          printf "%d,%.6f,%.6f,%.6f,%+.2f%%,%.6f,%.6f,%.6f,%+.2f%%\n", \
            t, prefill_off, prefill_on, prefill_speedup, prefill_speedup_pct, \
            decode_off, decode_on, decode_speedup, decode_speedup_pct

          row_count+=1
          sum_prefill_speedup+=prefill_speedup
          sum_decode_speedup+=decode_speedup
        }
      }

      # Mean row removed - not needed
    }
  ' "$CSV"
} | tee "$SUMMARY"

echo
echo "Done."
echo "- Raw CSV: $CSV"
echo "- Summary : $SUMMARY"
echo "- Logs    : $OUT_DIR/run_*"
