# Elastic loops, weight-space programming, multi-level decoding

Three extensions of the DeepSeek V4 CPU reference path. They share one idea:
keep the stock model untouched and add a narrow seam where the experiment
plugs in.

| Extension | Seam | Public API |
|---|---|---|
| Elastic looped transformer | layer schedule of one decode position | `ds4_loop_config`, `ds4_session_set_loops()` |
| Weight-space programming | bytes of the private model mapping | `ds4_wsp_*()` |
| Coarse-to-fine decoding | what a decode step emits and feeds back | `ds4_mlc_*()`, `ds4_session_step()` |

All three run on the CPU backend (`--cpu`), where weights are plain pointers
into the GGUF mapping and every layer pass is an explicit C call. The graph
backends (Metal, CUDA, ROCm) keep their exact behavior and refuse these
requests with an error rather than half-supporting them. Porting needs one
hook per backend, described at the end.

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
./ds4 --cpu --loop-block 20:28 --loops 3 --loop-kv last -p "..."
# REPL: /loops 1   /loops 4
```

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

- Loops: the graph decode builds a per-layer command list. A looped
  schedule repeats layers `[a,b)` and, in LAST mode, needs a device-side
  mark/restore of the raw ring and compressor states. PER_LOOP needs extra
  per-layer KV tensors.
- WSP: Metal wraps the shared mapping as no-copy buffers, so the engine would
  need a private copy of the edited tensors plus a buffer rebind. CUDA/ROCm
  need a re-upload of the touched span. `wsp_mark_dirty()` already records
  exactly that span.
- Concepts: the graph backends already accept embedding rows for vision
  spans. A concept step is one such row.

## Tests

`make test-elastic-wsp` runs without a model. It covers mark/rollback against
stock passes for ratio-0/4/128 layers, the loop policy, the WSP codecs,
editing and reverting a real private mapping (and checking the file is never
written), patch scripts, and codebook decoding.
