# Elastic loops, weight-space programming, multi-level decoding

Three extensions of the DeepSeek V4 CPU reference path. They share one idea:
keep the stock model untouched and add a narrow seam where the experiment
plugs in.

| Extension | Seam | Public API |
|---|---|---|
| Elastic looped transformer | layer schedule of one decode position | `ds4_loop_config`, `ds4_session_set_loops()` |
| Weight-space programming | bytes of the private model mapping | `ds4_wsp_*()` |
| Coarse-to-fine decoding | what a decode step emits and feeds back | `ds4_mlc_*()`, `ds4_session_step()` |

The CPU backend (`--cpu`) runs all three: its weights are plain pointers into
the GGUF mapping and every layer pass is an explicit C call. The DeepSeek V4
graph (CUDA, Metal) runs elastic loops in LAST mode on a single GPU, with or
without SSD streaming. Anything the graph does not support is refused with an
error rather than half-supported. The remaining ports are described at the
end.

## 1. Elastic looped transformer

```
embd -> [0, a) -> ( [a, b) )^N -> [b, L) -> output head
         input     recurrent        output
```

`forward_token_looped_cpu()` runs the input block once, the recurrent block
`num_loops` times feeding its output back to its input, and the output block
once. The hyper-connection residual stream (`n_hc x n_embd`) has the same shape
at every layer boundary, so looping needs no adapter. With `num_loops == 1` the
session keeps calling the original `forward_token_raw_swa_cpu_decode_scratch()`,
so a stock session runs exactly the code it ran before.

Each pass of a recurrent layer writes a KV row for the same position, so the
schedule must decide which pass owns the cache:

- **`DS4_LOOP_KV_LAST`**: one cache per layer, as in the stock model. Each
  non-final pass saves a `ds4_layer_cache_mark`, runs (so it attends to the
  history plus its own row), and restores the mark. A pass changes the cache in
  three ways only: it appends a raw row (sliding the window when full), updates
  the compressor states, and may append a compressed row. The mark stores the
  three counters, the raw row the slide drops, and the compressor states, which
  is enough to undo the pass exactly. This mode needs no extra memory, and
  `num_loops` can change at any token.
- **`DS4_LOOP_KV_PER_LOOP`**: loop `k > 0` owns a private cache for each
  recurrent layer (`ds4_loop_state.extra`), allocated with the same
  `layer_cache_init()` as the stock cache. Every pass attends to the history
  produced by the same pass. This costs `(N-1) * (b-a)` extra layer caches.

Elastic policy (`ds4_session_set_loops()`): the main cache always holds one
row per position and layer, so any schedule can continue from it. Changing the
depth or the block live keeps the checkpoint, so the history mixes the depths
it was computed at. Only per-loop levels can be missing history. The state
counts `synced` levels: decoding with `N` loops lowers it to `N`, and raising
`num_loops` above it invalidates the checkpoint so the next sync rebuilds.
Callers that want an exact history for the new schedule call
`ds4_session_invalidate()`.

The stock layer-major prefill implements only the stock schedule. A looped
session therefore rebuilds one position at a time through the same decode
function. Disk/snapshot payloads describe stock state, so looped sessions
refuse to save them, and stock payloads cannot be loaded into looped sessions.

```sh
./ds4 --cuda --loop-block 20:28 --loops 3 -p "..."      # LAST mode
./ds4 --cpu  --loop-block 20:28 --loops 3 --loop-kv per-loop -p "..."
# REPL: /loops 1   /loops 4
```

### On the graph backends

`metal_graph_encode_token_raw_swa()` (in-memory weights) and the SSD-streaming
decode walk the same tape as the CPU (`loop_tape_layer()`). A provisional
pass leaves less state behind on the graph than on the CPU:

- raw rows live in a ring at `pos % raw_cap`, so the final pass rewrites the
  row a provisional pass wrote;
- compressed and indexer rows are appended at host counters, so restoring the
  counters makes the final pass rewrite the same rows;
- the compressor frontiers are the only bytes to save and restore. They are
  copied into one scratch set with device copies queued between the layer
  kernels, so nothing synchronizes.

CUDA decode-island graphs are keyed by layer and buffer addresses, so a
repeated layer replays the island it would have replayed anyway. Batched
prefill and native multi-session batches encode the stock layer sequence:
looped sessions prefill through the decode tape one token at a time, and
batches containing a looped session are serialized. Streaming readahead
follows the tape, so the first recurrent layer is prefetched again when the
block wraps.

LAST-mode prefill is inherently sequential, not merely unoptimized: loop `k`
of token `t` must attend to the final rows of earlier tokens, which exist only
after those tokens have run every loop. A layer-major batch cannot respect
that dependency. PER_LOOP would batch, because each loop attends only to its
own loop's rows. With SSD streaming on CUDA, the expert cache has no automatic
budget. Stock sessions grow it as a side effect of wide prefill requests,
which looped sessions never issue. Pass `--ssd-streaming-cache-experts N` so
experts are reused across tokens; the engine warns when it is missing.

Refused on the graph: PER_LOOP KV, GLM/Qwen/V4.1 graphs, multi-GPU placements,
tensor parallel, distributed sessions, and MTP/DSpark speculation.

## 2. Weight-Space Programming

Every tensor is a row-major matrix. `cols` is GGUF `dim[0]` (the matvec input,
contiguous in memory), and `rows` is everything else, so expert `e` of
`[cols, out, n_expert]` owns rows `[e*out, (e+1)*out)`. Masking columns of
`ffn_down` ablates hidden neurons, and masking rows of `ffn_up`/`ffn_gate` does
the same from the other side.

On the CPU backend the model is mapped `MAP_PRIVATE`. An edit `mprotect`s the
touched pages writable, which makes them copy-on-write, and then protects them
again. The GGUF file is never written. `ds4_wsp_revert()` re-reads the dirty
span from the file, because map offset equals file offset. Every edit
funnels through `wsp_edit_rows()`: decode a row, apply a callback, encode it.

- F32, F16, BF16, Q8_0: every operation. Q8_0 blocks are requantized only
  if their values changed, so a narrow edit never perturbs the rest of a row.
- Other block types (K-quants, IQ2, MXFP4): only zeroing whole rows, which
  is exact for every GGML block format (scale 0 gives value 0).

```c
int t = ds4_wsp_find(e, "blk.12.ffn_down_shexp.weight");
ds4_wsp_scale(e, t, DS4_WSP_COLS, 100, 16, 0.0f, err, sizeof err);   /* mask */
ds4_wsp_load_matrix(e, t, closed_form, rows, cols, err, sizeof err);
ds4_wsp_revert_all(e, err, sizeof err);
```

Patch scripts (`--wsp FILE`, `ds4_wsp_apply_script()`) are line oriented:

```
mask   blk.10.ffn_down_shexp.weight cols 0-15,77
scale  blk.3.attn_output_b.weight rows 128-255 0.5
load   blk.20.ffn_gate_inp.weight router.f32
revert *
```

`gguf-tools/wsp.py compile patch.json -o patch.wsp` compiles a JSON
description, with `{L}` layer templates, into a script. It also writes
closed-form matrices: identity, scaled identity, a projector that removes
directions, or any `.npy`.

Live sessions keep the KV state they computed with the old weights. Sessions
over edited weights refuse disk payload saves.

## 3. Coarse-to-fine multi-level decoding

A `ds4_mlc` holds `L` codebooks over the hidden space (`n_embd`). Decoding is
residual quantization from the coarsest level down:

```
r0 = h
c_l = sample_k(<codebook_l[k], r_l> / T)
r_{l+1} = r_l - codebook_l[c_l]
concept = sum_l codebook_l[c_l]
```

`h` is the normalized head input of the last position
(`ds4_session_hidden()`). A step can stop at fewer levels to get a coarser
concept. `ds4_session_step()` makes the decode loop polymorphic. With no
codebook it samples and evaluates a token. With a codebook it decodes the
codes and evaluates the concept vector as the next input embedding
(`ds4_session_eval_embedding()`). The `token_tag` argument is still recorded,
because hash-routed MoE layers choose experts by token id.

```c
ds4_step_options o = {.temperature = 0.7f, .rng = &rng, .mlc = mlc,
                      .mlc_levels = 2, .concept_tag = tag};
ds4_step st;
ds4_session_step(s, &o, &st, err, sizeof err);   /* st.kind, st.codes[] */
```

Codebook file (`gguf-tools/wsp.py codebook out.ds4c level0.npy ...`):
`"DS4C"`, `u32 version=1`, `u32 levels`, `u32 dim`, then for each level
`u32 K` followed by `K*dim` f32 values, all little-endian.

A stock DeepSeek model was not trained to read concept embeddings. The API
provides the decoding loop and the data path, while the codebooks (and any
fine-tuning that makes the model use them) come from outside.

## Porting to the graph backends

- PER_LOOP on the graph: extra per-layer raw/compressed KV tensors for each
  loop level, selected by the tape's loop index.
- WSP: graph backends map the GGUF `MAP_SHARED`, so host edits would write the
  file. Metal wraps the mapping as no-copy buffers and would need a private
  copy of the edited tensors plus a buffer rebind. CUDA keeps device copies of
  weight ranges plus derived artifacts (Q8 to F16/F32 caches, repacked
  experts), so an edit must update the device range and rebuild or invalidate
  every artifact derived from it. `wsp_mark_dirty()` already records the
  touched span.
- Concepts: the graph backends already accept embedding rows for vision
  spans. A concept step is one such row.

## Tests

`make test-elastic-wsp` runs without a model. It covers mark/rollback against
stock passes for ratio-0/4/128 layers, the loop policy, the WSP codecs,
editing and reverting a real private mapping (and checking the file is never
written), patch scripts, and codebook decoding.

`make test-elastic-model` (Linux, about 4 GB of RAM, 15-20 minutes on 4
cores) runs the real Flash CPU layer code on a synthetic model. The model has
the exact Flash shape, passes the production weight binder, and has random
weights shared by all 43 layers. It checks, bit-exactly:

- `num_loops == 1` against the stock decoder;
- LAST mode against a reference that rolls back by deep-copying caches,
  including the final KV state of every layer;
- PER_LOOP against an unrolled stack with a cache per loop;
- WSP row masks and scales through the output head, with revert restoring
  identical logits;
- session sync/eval with loops, the elastic policy, and a concept step
  followed by a token step.

## Finding schedules the model tolerates

A stock model was not trained to loop, so every block and depth needs
measuring. `tests/elastic_sweep.sh MODEL.gguf` runs each schedule twice. The
first run computes perplexity of a fixed text, teacher-forced through the
loops. The second runs free greedy generation and records whether the answer
ends (`eos`) and how much it repeats (`distinct2`). Results land in
`elastic-sweep/summary.txt`, ranked against stock. Perplexity alone misses
the typical failure: an answer that starts well and never stops. That is why
the second run checks free generation. Rerunning the script skips schedules
already in `elastic-sweep/results.csv`.
