#!/bin/sh
# Check elastic loops on a real GPU and model.
#
#   tests/elastic_gpu_check.sh MODEL.gguf [extra ds4 flags]
#
# Extra flags are passed to every run, e.g. --ssd-streaming on a 24 GB GPU.
# Set DS4_CHECK_CPU=1 to also compare against the CPU backend (needs the whole
# model in RAM, e.g. a DGX Spark).  Set DS4_CHECK_BASELINE=/path/to/ds4 built
# from main to check that the stock path is unchanged by this branch.
# Writes logs and dumps to ./elastic-check/.
set -u
MODEL=${1:?usage: $0 MODEL.gguf [extra ds4 flags]}
shift
EXTRA="$*"
OUT=elastic-check
BLOCK=${DS4_CHECK_BLOCK:-20:28}
TOKENS=${DS4_CHECK_TOKENS:-48}
PROMPT=${DS4_CHECK_PROMPT:-"Explain in three sentences why the sky is blue."}
mkdir -p "$OUT"
cmp_py=$(dirname "$0")/compare_logprobs.py
status=0

BIN=./ds4
run() {   # name backend flags...
    name=$1; backend=$2; shift 2
    printf '== %-14s ' "$name"
    if $BIN -m "$MODEL" "$backend" --nothink --ctx 4096 -n "$TOKENS" \
            -p "$PROMPT" --dump-logprobs "$OUT/$name.json" $EXTRA "$@" \
            > "$OUT/$name.log" 2>&1; then
        echo "ok"
    else
        echo "FAILED (see $OUT/$name.log)"; tail -5 "$OUT/$name.log"; status=1
    fi
}

text() {  # greedy continuation of a dump, for eyeballing
    python3 -c 'import json,sys; print("".join(s["selected"] for s in json.load(open(sys.argv[1]))["steps"]))' "$1" 2>/dev/null
}

run cuda-stock  --cuda
run cuda-stock2 --cuda
run cuda-loops1 --cuda --loop-block "$BLOCK" --loops 1
run cuda-loops2 --cuda --loop-block "$BLOCK" --loops 2
run cuda-loops3 --cuda --loop-block "$BLOCK" --loops 3

echo
echo "-- determinism baseline (should be IDENTICAL; if not, CUDA itself is nondeterministic here)"
python3 "$cmp_py" "$OUT/cuda-stock.json" "$OUT/cuda-stock2.json"
echo "-- loops 1 must equal stock"
python3 "$cmp_py" "$OUT/cuda-stock.json" "$OUT/cuda-loops1.json" || status=1
echo "-- loops 2 and 3 must differ from stock and from each other"
python3 "$cmp_py" "$OUT/cuda-stock.json" "$OUT/cuda-loops2.json" && status=1
python3 "$cmp_py" "$OUT/cuda-loops2.json" "$OUT/cuda-loops3.json" && status=1

if [ -n "${DS4_CHECK_BASELINE:-}" ]; then
    BIN=$DS4_CHECK_BASELINE run cuda-baseline --cuda
    BIN=./ds4
    echo "-- stock on this branch must equal the baseline binary"
    python3 "$cmp_py" "$OUT/cuda-baseline.json" "$OUT/cuda-stock.json" || status=1
fi

if [ "${DS4_CHECK_CPU:-0}" = 1 ]; then
    run cpu-stock  --cpu
    run cpu-loops3 --cpu --loop-block "$BLOCK" --loops 3
    echo "-- CPU vs CUDA: the loops-3 gap should be the same order as the stock gap."
    echo "   A much larger gap means the GPU rollback of provisional passes leaks state."
    python3 "$cmp_py" "$OUT/cpu-stock.json" "$OUT/cuda-stock.json"
    python3 "$cmp_py" "$OUT/cpu-loops3.json" "$OUT/cuda-loops3.json"
fi

echo
for n in cuda-stock cuda-loops2 cuda-loops3; do
    echo "-- $n:"; text "$OUT/$n.json"; echo
done
[ $status = 0 ] && echo "elastic GPU check: PASS" || echo "elastic GPU check: FAIL"
exit $status
