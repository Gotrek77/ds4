#!/bin/sh
# Sweep elastic-loop schedules and measure what the model tolerates.
#
#   tests/elastic_sweep.sh MODEL.gguf [extra ds4 flags]
#
# For every recurrent block and loop count it records:
#   ppl       perplexity of a fixed text (teacher forced, through the loops)
#   eos       step at which a free greedy answer ends (-1: never within N)
#   distinct2 fraction of distinct token bigrams in that answer (low = looping)
#
# Results go to elastic-sweep/results.csv, sorted into elastic-sweep/summary.txt.
#
# Environment:
#   DS4_SWEEP_BACKEND  --cuda (default), --metal or --cpu
#   DS4_SWEEP_TEXT     perplexity text (default README.md)
#   DS4_SWEEP_TOKENS   scored tokens (default 512)
#   DS4_SWEEP_STARTS   block starts (default "4 12 20 28 35")
#   DS4_SWEEP_WIDTHS   block widths (default "2 4 8")
#   DS4_SWEEP_LOOPS    loop counts (default "2 3")
#   DS4_SWEEP_PROMPT   free-generation prompt
#   DS4_SWEEP_GEN      free-generation tokens (default 96)
# Extra flags go to every run, e.g. --ssd-streaming --ssd-streaming-cache-experts 1000.
set -u
MODEL=${1:?usage: $0 MODEL.gguf [extra ds4 flags]}
shift
EXTRA="$*"
BACKEND=${DS4_SWEEP_BACKEND:---cuda}
TEXT=${DS4_SWEEP_TEXT:-README.md}
TOKENS=${DS4_SWEEP_TOKENS:-512}
STARTS=${DS4_SWEEP_STARTS:-4 12 20 28 35}
WIDTHS=${DS4_SWEEP_WIDTHS:-2 4 8}
LOOPS=${DS4_SWEEP_LOOPS:-2 3}
PROMPT=${DS4_SWEEP_PROMPT:-"Ciao! Presentati in due frasi."}
GEN=${DS4_SWEEP_GEN:-96}
N_LAYER=43
OUT=elastic-sweep
CMP=$(dirname "$0")/compare_logprobs.py
mkdir -p "$OUT"
CSV=$OUT/results.csv
[ -f "$CSV" ] || echo "block,loops,layer_passes,ppl,eos,distinct2,seconds" > "$CSV"

measure() {  # block loopflags...
    block=$1; shift
    # A schedule is its block and loop count; rows already measured are kept.
    if grep -q "^$block,$LOOPS_NOW," "$CSV"; then echo "== $block x$LOOPS_NOW (cached)"; return; fi
    name="$block-x$LOOPS_NOW"
    name=$(echo "$name" | tr ':' '_')
    printf '== %-14s ' "$block x$LOOPS_NOW"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    ppl=$(./ds4 -m "$MODEL" $BACKEND --ctx 4096 -n "$TOKENS" \
              --perplexity-file "$TEXT" $EXTRA "$@" 2> "$OUT/$name.ppl.log" |
          sed -n 's/.* ppl=\([0-9.]*\).*/\1/p')
    # shellcheck disable=SC2086
    ./ds4 -m "$MODEL" $BACKEND --ctx 4096 --nothink -n "$GEN" -p "$PROMPT" \
          --dump-logprobs "$OUT/$name.json" $EXTRA "$@" > "$OUT/$name.gen.log" 2>&1
    gen=$(python3 "$CMP" --stats "$OUT/$name.json" 2>/dev/null || echo "? ?")
    t1=$(date +%s)
    eos=${gen% *}; d2=${gen#* }
    echo "ppl=${ppl:-FAILED} eos=$eos distinct2=$d2 ($((t1 - t0))s)"
    echo "$block,$LOOPS_NOW,$PASSES,${ppl:-},$eos,$d2,$((t1 - t0))" >> "$CSV"
}

LOOPS_NOW=1; PASSES=$N_LAYER
measure stock

for w in $WIDTHS; do
    for s in $STARTS; do
        e=$((s + w))
        [ $e -le $N_LAYER ] || continue
        for n in $LOOPS; do
            LOOPS_NOW=$n; PASSES=$((N_LAYER + (n - 1) * w))
            measure "$s:$e" --loop-block "$s:$e" --loops "$n"
        done
    done
done

# Rank by perplexity relative to stock; healthy runs end their answer.
python3 - "$CSV" > "$OUT/summary.txt" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
base = next((float(r["ppl"]) for r in rows if r["block"] == "stock" and r["ppl"]), None)
def key(r):
    try: return float(r["ppl"])
    except ValueError: return float("inf")
print(f"{'block':>8} {'loops':>5} {'passes':>6} {'ppl':>9} {'vs stock':>9} {'eos':>4} {'dist2':>6}")
for r in sorted(rows, key=key):
    rel = f"{key(r) / base:8.3f}x" if base and r["ppl"] else "        ?"
    print(f"{r['block']:>8} {r['loops']:>5} {r['layer_passes']:>6} {r['ppl'] or '?':>9} {rel} {r['eos']:>4} {r['distinct2']:>6}")
PY
echo
cat "$OUT/summary.txt"
