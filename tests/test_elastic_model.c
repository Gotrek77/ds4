/* End-to-end checks of elastic loops, WSP and concept steps through the real
 * DeepSeek V4 Flash CPU layer code, without the 80 GB model.
 *
 * The synthetic model has the exact Flash shape and passes the production
 * weight binder and validator.  All 43 layers alias the same random tensors
 * (one copy per compression class), so the whole model fits in ~4 GB: a
 * weight-tied stack, which is also the natural shape of a looped model.  The
 * data lives in a memfd mapped MAP_PRIVATE, like the CPU backend maps a GGUF,
 * so WSP copy-on-write and revert run their production paths.
 *
 * Every looped result is compared bit-exactly against an independent
 * reference: KV rollback against deep cache copies, per-loop caches against an
 * unrolled stack of separate caches. */
#include "../ds4.c"
#include <assert.h>
#include <sys/mman.h>

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

/* ---- synthetic model ------------------------------------------------- */

static uint64_t rs = 0x243f6a8885a308d3ull;
static inline uint64_t rnext(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return rs;
}
static inline float runi(void) {   /* [-1, 1) */
    return (float)((rnext() >> 40) & 0xffffff) / 8388608.0f - 1.0f;
}

typedef struct {
    char key[96];
    uint64_t off, bytes;
} blob;

static struct {
    uint8_t *w;          /* writable view while building */
    uint64_t size, used;
    blob blobs[256];
    int n_blobs;
    ds4_tensor t[2048];
    int n_t;
} B;

static uint64_t tensor_bytes_of(uint32_t type, uint64_t elems) {
    uint64_t b = 0;
    CHECK(tensor_nbytes(type, elems, &b));
    return b;
}

static void fill_blob(uint8_t *p, uint32_t type, uint64_t elems, uint64_t in_dim, const char *suffix) {
    const float s = 1.0f / sqrtf((float)in_dim);
    switch (type) {
    case DS4_TENSOR_F32: {
        float *f = (float *)p;
        const bool norm = strstr(suffix, "norm") != NULL;
        for (uint64_t i = 0; i < elems; i++) f[i] = norm ? 1.0f : 0.05f * runi();
        break;
    }
    case DS4_TENSOR_F16: {
        uint16_t *h = (uint16_t *)p;
        const float scale = !strcmp(suffix, "token_embd.weight") ? 1.0f : s;
        for (uint64_t i = 0; i < elems; i++) h[i] = f32_to_f16(scale * runi());
        break;
    }
    case DS4_TENSOR_Q8_0: {
        const uint16_t d = f32_to_f16(s * 1.7f / 127.0f);
        for (uint64_t b = 0; b < elems / 32; b++) {
            uint8_t *q = p + b * 34;
            memcpy(q, &d, 2);
            for (int i = 0; i < 32; i += 8) {
                const uint64_t r = rnext();
                memcpy(q + 2 + i, &r, 8);
            }
        }
        break;
    }
    case DS4_TENSOR_IQ2_XXS: {
        const uint16_t d = f32_to_f16(s / 40.0f);
        for (uint64_t b = 0; b < elems / 256; b++) {
            uint8_t *q = p + b * 66;
            memcpy(q, &d, 2);
            for (int i = 2; i < 66; i += 8) {
                uint64_t r = rnext();
                memcpy(q + i, &r, i + 8 <= 66 ? 8 : 66 - i);
            }
        }
        break;
    }
    case DS4_TENSOR_Q2_K: {
        const uint16_t d = f32_to_f16(s / 12.0f), dmin = f32_to_f16(s / 24.0f);
        for (uint64_t b = 0; b < elems / 256; b++) {
            uint8_t *q = p + b * 84;
            for (int i = 0; i < 80; i += 8) {
                uint64_t r = rnext();
                memcpy(q + i, &r, 8);
            }
            memcpy(q + 80, &d, 2);
            memcpy(q + 82, &dmin, 2);
        }
        break;
    }
    case DS4_TENSOR_I32: {   /* ffn_gate_tid2eid: 6 distinct experts per token */
        int32_t *v = (int32_t *)p;
        for (uint64_t i = 0; i < elems; i++) {
            const uint64_t tok = i / DS4_N_EXPERT_USED, j = i % DS4_N_EXPERT_USED;
            v[i] = (int32_t)((tok * 7 + j * 37) % DS4_N_EXPERT);
        }
        break;
    }
    default:
        CHECK(!"unsupported synthetic type");
    }
}

/* Tensors with the same key share one data region. */
static void add_tensor(const char *name, const char *key, uint32_t type,
                       uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    const uint64_t elems = d0 * (ndim > 1 ? d1 : 1) * (ndim > 2 ? d2 : 1);
    const uint64_t bytes = tensor_bytes_of(type, elems);
    blob *bl = NULL;
    for (int i = 0; i < B.n_blobs; i++) if (!strcmp(B.blobs[i].key, key)) bl = &B.blobs[i];
    if (!bl) {
        CHECK(B.n_blobs < 256);
        bl = &B.blobs[B.n_blobs++];
        snprintf(bl->key, sizeof(bl->key), "%s", key);
        bl->off = (B.used + 63) & ~63ull;
        bl->bytes = bytes;
        B.used = bl->off + bytes;
        CHECK(B.used <= B.size);
        const char *suffix = strrchr(key, '|') ? strrchr(key, '|') + 1 : key;
        fill_blob(B.w + bl->off, type, elems, d0, suffix);
    }
    CHECK(bl->bytes == bytes && B.n_t < 2048);
    ds4_tensor *t = &B.t[B.n_t++];
    t->name = (ds4_str){strdup(name), strlen(name)};
    t->ndim = ndim;
    t->dim[0] = d0; t->dim[1] = d1; t->dim[2] = d2;
    t->type = type;
    t->elements = elems;
    t->bytes = bytes;
    t->abs_offset = t->rel_offset = bl->off;
}

static void add_layer_tensor(uint32_t il, uint32_t ratio, const char *suffix, uint32_t type,
                             uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2, bool per_ratio) {
    char name[128], key[96];
    snprintf(name, sizeof(name), "blk.%u.%s", il, suffix);
    if (per_ratio) snprintf(key, sizeof(key), "r%u|%s", ratio, suffix);
    else snprintf(key, sizeof(key), "%s", suffix);
    add_tensor(name, key, type, ndim, d0, d1, d2);
}

static ds4_engine E;

static void build_model(void) {
    B.size = 5ull << 30;
    const int fd = memfd_create("ds4-synthetic", 0);
    CHECK(fd >= 0 && ftruncate(fd, (off_t)B.size) == 0);
    B.w = mmap(NULL, B.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    CHECK(B.w != MAP_FAILED);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++)
        g_ds4_compress_ratios[il] = ds4_expected_layer_compress_ratio(il);

    const uint64_t E_ = DS4_N_EMBD, HC = DS4_N_HC, V = DS4_N_VOCAB;
    const uint64_t hc_dim = E_ * HC, hc_mix = 2 * HC + HC * HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t HD = DS4_N_HEAD_DIM, LQ = DS4_N_LORA_Q, FF = DS4_N_FF_EXP, NE = DS4_N_EXPERT;

    add_tensor("token_embd.weight", "token_embd.weight", DS4_TENSOR_F16, 2, E_, V, 0);
    add_tensor("output_hc_base.weight", "output_hc_base", DS4_TENSOR_F32, 1, HC, 0, 0);
    add_tensor("output_hc_fn.weight", "output_hc_fn", DS4_TENSOR_F16, 2, hc_dim, HC, 0);
    add_tensor("output_hc_scale.weight", "output_hc_scale", DS4_TENSOR_F32, 1, 1, 0, 0);
    add_tensor("output_norm.weight", "output_norm", DS4_TENSOR_F32, 1, E_, 0, 0);
    add_tensor("output.weight", "output.weight", DS4_TENSOR_Q8_0, 2, E_, V, 0);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t r = ds4_layer_compress_ratio(il);
#define L(SUF, T, N, A, B_, C) add_layer_tensor(il, r, SUF, T, N, A, B_, C, false)
#define LR(SUF, T, N, A, B_, C) add_layer_tensor(il, r, SUF, T, N, A, B_, C, true)
        L("hc_attn_fn.weight", DS4_TENSOR_F16, 2, hc_dim, hc_mix, 0);
        L("hc_attn_scale.weight", DS4_TENSOR_F32, 1, 3, 0, 0);
        L("hc_attn_base.weight", DS4_TENSOR_F32, 1, hc_mix, 0, 0);
        L("attn_norm.weight", DS4_TENSOR_F32, 1, E_, 0, 0);
        L("attn_q_a.weight", DS4_TENSOR_Q8_0, 2, E_, LQ, 0);
        L("attn_q_a_norm.weight", DS4_TENSOR_F32, 1, LQ, 0, 0);
        L("attn_q_b.weight", DS4_TENSOR_Q8_0, 2, LQ, q_dim, 0);
        L("attn_kv.weight", DS4_TENSOR_Q8_0, 2, E_, HD, 0);
        L("attn_kv_a_norm.weight", DS4_TENSOR_F32, 1, HD, 0, 0);
        L("attn_sinks.weight", DS4_TENSOR_F32, 1, DS4_N_HEAD, 0, 0);
        L("attn_output_a.weight", DS4_TENSOR_Q8_0, 2, HD * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low, 0);
        L("attn_output_b.weight", DS4_TENSOR_Q8_0, 2, out_low, E_, 0);
        if (r != 0) {
            const uint64_t cw = (r == 4 ? 2u : 1u) * HD;
            LR("attn_compressor_ape.weight", DS4_TENSOR_F16, 2, cw, r, 0);
            LR("attn_compressor_kv.weight", DS4_TENSOR_F16, 2, E_, cw, 0);
            LR("attn_compressor_gate.weight", DS4_TENSOR_F16, 2, E_, cw, 0);
            LR("attn_compressor_norm.weight", DS4_TENSOR_F32, 1, HD, 0, 0);
        }
        if (r == 4) {
            const uint64_t iq = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t iw = 2u * DS4_N_INDEXER_HEAD_DIM;
            LR("indexer.attn_q_b.weight", DS4_TENSOR_F16, 2, LQ, iq, 0);
            LR("indexer.proj.weight", DS4_TENSOR_F16, 2, E_, DS4_N_INDEXER_HEAD, 0);
            LR("indexer_compressor_ape.weight", DS4_TENSOR_F16, 2, iw, r, 0);
            LR("indexer_compressor_kv.weight", DS4_TENSOR_F16, 2, E_, iw, 0);
            LR("indexer_compressor_gate.weight", DS4_TENSOR_F16, 2, E_, iw, 0);
            LR("indexer_compressor_norm.weight", DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0);
        }
        L("hc_ffn_fn.weight", DS4_TENSOR_F16, 2, hc_dim, hc_mix, 0);
        L("hc_ffn_scale.weight", DS4_TENSOR_F32, 1, 3, 0, 0);
        L("hc_ffn_base.weight", DS4_TENSOR_F32, 1, hc_mix, 0, 0);
        L("ffn_norm.weight", DS4_TENSOR_F32, 1, E_, 0, 0);
        L("ffn_gate_inp.weight", DS4_TENSOR_F16, 2, E_, NE, 0);
        L("ffn_gate_exps.weight", DS4_TENSOR_IQ2_XXS, 3, E_, FF, NE);
        L("ffn_up_exps.weight", DS4_TENSOR_IQ2_XXS, 3, E_, FF, NE);
        L("ffn_down_exps.weight", DS4_TENSOR_Q2_K, 3, FF, E_, NE);
        L("ffn_gate_shexp.weight", DS4_TENSOR_Q8_0, 2, E_, FF, 0);
        L("ffn_up_shexp.weight", DS4_TENSOR_Q8_0, 2, E_, FF, 0);
        L("ffn_down_shexp.weight", DS4_TENSOR_Q8_0, 2, FF, E_, 0);
        if (il < DS4_N_HASH_LAYER) L("ffn_gate_tid2eid.weight", DS4_TENSOR_I32, 2, DS4_N_EXPERT_USED, V, 0);
#undef L
#undef LR
    }

    /* Map it the way model_open() maps a GGUF for the CPU backend. */
    munmap(B.w, B.size);
    B.size = (B.used + 4095) & ~4095ull;
    CHECK(ftruncate(fd, (off_t)B.size) == 0);
    ds4_model *m = &E.model;
    m->fd = fd;
    m->ngram_fd = -1;
    m->map = mmap(NULL, B.size, PROT_READ, MAP_PRIVATE, fd, 0);
    CHECK(m->map != MAP_FAILED);
    m->size = m->file_size = B.size;
    m->private_map = true;
    m->n_tensors = (uint64_t)B.n_t;
    m->tensors = B.t;
    E.backend = DS4_BACKEND_CPU;
    weights_bind(&E.weights, m, false, 0, 0, true, false);
    fprintf(stderr, "synthetic Flash model: %d tensors, %.2f GiB of data\n",
            B.n_t, (double)B.used / (1024.0 * 1024.0 * 1024.0));
}

/* ---- references ------------------------------------------------------ */

static void layer_cache_clone(ds4_layer_cache *dst, const ds4_layer_cache *src) {
    const uint32_t ratio = src->compress_ratio;
    const uint64_t attn = layer_cache_state_floats(ratio, DS4_N_HEAD_DIM);
    const uint64_t index = ratio == 4 ? layer_cache_state_floats(ratio, DS4_N_INDEXER_HEAD_DIM) : 0;
    *dst = *src;
#define DUP(F, N) dst->F = src->F ? memcpy(xmalloc((N) * sizeof(float)), src->F, (N) * sizeof(float)) : NULL
    DUP(raw_kv, (uint64_t)src->cap_raw * DS4_N_HEAD_DIM);
    DUP(attn_comp_kv, (uint64_t)src->comp_cap * DS4_N_HEAD_DIM);
    DUP(attn_state_kv, attn);
    DUP(attn_state_score, attn);
    DUP(index_comp_kv, (uint64_t)src->comp_cap * DS4_N_INDEXER_HEAD_DIM);
    DUP(index_state_kv, index);
    DUP(index_state_score, index);
#undef DUP
}

static bool layer_cache_equal(const ds4_layer_cache *a, const ds4_layer_cache *b) {
    const uint32_t ratio = a->compress_ratio;
    const uint64_t attn = layer_cache_state_floats(ratio, DS4_N_HEAD_DIM);
    const uint64_t index = ratio == 4 ? layer_cache_state_floats(ratio, DS4_N_INDEXER_HEAD_DIM) : 0;
    if (a->n_raw != b->n_raw || a->n_comp != b->n_comp || a->n_index_comp != b->n_index_comp)
        return false;
#define EQ(F, N) ((N) == 0 || memcmp(a->F, b->F, (N) * sizeof(float)) == 0)
    return EQ(raw_kv, (uint64_t)a->n_raw * DS4_N_HEAD_DIM) &&
           EQ(attn_comp_kv, (uint64_t)a->n_comp * DS4_N_HEAD_DIM) &&
           EQ(attn_state_kv, attn) && EQ(attn_state_score, attn) &&
           EQ(index_comp_kv, (uint64_t)a->n_index_comp * DS4_N_INDEXER_HEAD_DIM) &&
           EQ(index_state_kv, index) && EQ(index_state_score, index);
#undef EQ
}

/* Looped decode written independently of forward_token_looped_cpu():
 * LAST mode undoes provisional passes by restoring a deep copy of the layer
 * cache; PER_LOOP mode uses a whole separate KV cache per loop. */
static void forward_reference(float *logits, ds4_kv_cache *caches, const ds4_loop_config *cfg,
                              int token, uint32_t pos, ds4_cpu_decode_scratch *sc) {
    float *cur = sc->cur, *next = sc->next;
    embed_token_f16(&E.model, &E.weights, token, sc->plain);
    hc_from_plain_embedding(cur, sc->plain, DS4_N_EMBD, DS4_N_HC);
    uint32_t order[DS4_MAX_LAYER * 8], loop_of[DS4_MAX_LAYER * 8], n = 0;
    for (uint32_t il = 0; il < cfg->rec_start; il++) { order[n] = il; loop_of[n++] = 0; }
    for (uint32_t k = 0; k < cfg->num_loops; k++)
        for (uint32_t il = cfg->rec_start; il < cfg->rec_end; il++) { order[n] = il; loop_of[n++] = k; }
    for (uint32_t il = cfg->rec_end; il < DS4_N_LAYER; il++) { order[n] = il; loop_of[n++] = 0; }

    for (uint32_t i = 0; i < n; i++) {
        const uint32_t il = order[i], k = loop_of[i];
        const bool per_loop = cfg->kv_mode == DS4_LOOP_KV_PER_LOOP;
        const bool rec = il >= cfg->rec_start && il < cfg->rec_end;
        ds4_layer_cache *lc = &caches[per_loop && rec ? k : 0].layer[il];
        const bool provisional = !per_loop && rec && k + 1 < cfg->num_loops;
        ds4_layer_cache saved;
        if (provisional) layer_cache_clone(&saved, lc);
        layer_forward_raw_swa_one(next, &E.model, &E.weights.layer[il], lc, cur, il, pos, token,
                                  NULL, 0.0f, 0.0f, sc);
        if (provisional) {
            layer_cache_free(lc);
            *lc = saved;
        }
        float *t = cur; cur = next; next = t;
    }
    output_logits_one_decode_scratch(logits, &E.model, &E.weights, cur, sc);
}

/* ---- tests ------------------------------------------------------------ */

enum { CTX = 512, RAW_CAP = 16 };
static const int prompt[] = {1, 517, 42, 9001, 77, 123, 4096, 65000, 3, 18, 2718, 31415,
                             27, 1000, 12, 99, 5, 7, 11, 13, 17, 19, 23, 29};
#define NPROMPT ((int)(sizeof(prompt) / sizeof(prompt[0])))

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int argmax(const float *x, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
    return b;
}

static bool all_finite(const float *x, int n) {
    for (int i = 0; i < n; i++) if (!isfinite(x[i])) return false;
    return true;
}

/* Logits after each prompt token for one schedule, through the engine code. */
static float *run_engine(const ds4_loop_config *cfg, bool stock_fn, int ntok) {
    const uint32_t V = DS4_N_VOCAB;
    float *out = xmalloc((size_t)ntok * V * sizeof(float));
    ds4_kv_cache cache;
    kv_cache_init(&cache, CTX, RAW_CAP);
    ds4_cpu_decode_scratch sc;
    cpu_decode_scratch_init(&sc, CTX);
    ds4_loop_state ls = {.cfg = *cfg, .levels = 1, .synced = 1, .ctx_size = CTX};
    if (ls.cfg.num_loops == 0) ls.cfg.num_loops = 1;
    loop_state_reserve(&ls, 0);
    /* Per-loop caches use the default raw capacity; match the reference. */
    for (uint32_t i = 0; ls.extra && i < (ls.levels - 1) * loop_state_rec_layers(&ls); i++) {
        layer_cache_free(&ls.extra[i]);
        layer_cache_init(&ls.extra[i], ls.cfg.rec_start + i % loop_state_rec_layers(&ls), CTX, RAW_CAP);
    }
    for (int t = 0; t < ntok; t++) {
        if (stock_fn)
            forward_token_raw_swa_cpu_decode_scratch(out + (size_t)t * V, &E.model, &E.weights, &cache,
                                                     prompt[t], (uint32_t)t, NULL, 0, 0, &sc);
        else
            forward_token_looped_cpu(out + (size_t)t * V, &E.model, &E.weights, &cache, &ls,
                                     prompt[t], NULL, (uint32_t)t, NULL, 0, 0, &sc);
    }
    loop_state_free(&ls);
    cpu_decode_scratch_free(&sc);
    kv_cache_free(&cache);
    return out;
}

static float *run_reference(const ds4_loop_config *cfg, int ntok, ds4_kv_cache *keep) {
    const uint32_t V = DS4_N_VOCAB;
    float *out = xmalloc((size_t)ntok * V * sizeof(float));
    ds4_kv_cache *caches = xcalloc(cfg->num_loops, sizeof(*caches));
    for (uint32_t k = 0; k < cfg->num_loops; k++) kv_cache_init(&caches[k], CTX, RAW_CAP);
    ds4_cpu_decode_scratch sc;
    cpu_decode_scratch_init(&sc, CTX);
    for (int t = 0; t < ntok; t++)
        forward_reference(out + (size_t)t * V, caches, cfg, prompt[t], (uint32_t)t, &sc);
    cpu_decode_scratch_free(&sc);
    if (keep) *keep = caches[0];
    else kv_cache_free(&caches[0]);
    for (uint32_t k = 1; k < cfg->num_loops; k++) kv_cache_free(&caches[k]);
    free(caches);
    return out;
}

static bool same(const float *a, const float *b, int ntok) {
    return memcmp(a, b, (size_t)ntok * DS4_N_VOCAB * sizeof(float)) == 0;
}

static float max_abs_diff(const float *a, const float *b, int ntok) {
    float d = 0.0f;
    for (size_t i = 0; i < (size_t)ntok * DS4_N_VOCAB; i++) d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

int main(void) {
    double t0 = now();
    build_model();
    fprintf(stderr, "built in %.1fs\n", now() - t0);

    /* 1. The looped decoder with a stock schedule is the stock decoder. */
    t0 = now();
    ds4_loop_config stock = {0};
    float *ref = run_engine(&stock, true, NPROMPT);
    fprintf(stderr, "stock decode: %.2fs/token\n", (now() - t0) / NPROMPT);
    CHECK(all_finite(ref, NPROMPT * (int)DS4_N_VOCAB));
    float *x = run_engine(&stock, false, NPROMPT);
    CHECK(same(ref, x, NPROMPT));
    free(x);
    ds4_loop_config one = {.rec_start = 4, .rec_end = 9, .num_loops = 1, .kv_mode = DS4_LOOP_KV_PER_LOOP};
    x = run_engine(&one, false, NPROMPT);
    CHECK(same(ref, x, NPROMPT));
    free(x);
    /* The reference itself agrees with stock when not looped. */
    one.num_loops = 1;
    x = run_reference(&one, NPROMPT, NULL);
    CHECK(same(ref, x, NPROMPT));
    free(x);
    fprintf(stderr, "ok: num_loops=1 is bit-identical to the stock decoder\n");

    /* 2. LAST mode: mark/restore == deep-copy rollback, bit for bit.  The
     * block [4,9) mixes ratio-4 (indexer) and ratio-128 layers; 24 tokens
     * through a 16-row window exercise the slide and ratio-4 compression. */
    ds4_loop_config last = {.rec_start = 4, .rec_end = 9, .num_loops = 3, .kv_mode = DS4_LOOP_KV_LAST};
    ds4_kv_cache ref_cache;
    float *lr = run_reference(&last, NPROMPT, &ref_cache);
    float *le = run_engine(&last, false, NPROMPT);
    CHECK(all_finite(le, NPROMPT * (int)DS4_N_VOCAB));
    CHECK(same(lr, le, NPROMPT));
    CHECK(!same(ref, le, NPROMPT));
    fprintf(stderr, "ok: LAST loops=3 matches deep-copy reference; differs from stock by %.3g\n",
            (double)max_abs_diff(ref, le, NPROMPT));

    /* Final cache state also matches (re-run keeping the engine cache). */
    {
        ds4_kv_cache cache;
        kv_cache_init(&cache, CTX, RAW_CAP);
        ds4_cpu_decode_scratch sc;
        cpu_decode_scratch_init(&sc, CTX);
        ds4_loop_state ls = {.cfg = last, .levels = 1, .synced = 1, .ctx_size = CTX};
        float *lg = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
        for (int t = 0; t < NPROMPT; t++)
            forward_token_looped_cpu(lg, &E.model, &E.weights, &cache, &ls, prompt[t], NULL,
                                     (uint32_t)t, NULL, 0, 0, &sc);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++)
            CHECK(layer_cache_equal(&cache.layer[il], &ref_cache.layer[il]));
        CHECK(cache.layer[4].n_index_comp > 0 && cache.layer[4].n_raw == RAW_CAP);
        free(lg);
        loop_state_free(&ls);
        cpu_decode_scratch_free(&sc);
        kv_cache_free(&cache);
        kv_cache_free(&ref_cache);
        fprintf(stderr, "ok: LAST final KV caches identical to the reference (all 43 layers)\n");
    }
    free(lr);

    /* 3. PER_LOOP == unrolled stack with a private cache per loop. */
    ds4_loop_config per = {.rec_start = 10, .rec_end = 14, .num_loops = 3, .kv_mode = DS4_LOOP_KV_PER_LOOP};
    float *pr = run_reference(&per, NPROMPT, NULL);
    float *pe = run_engine(&per, false, NPROMPT);
    CHECK(same(pr, pe, NPROMPT));
    per.kv_mode = DS4_LOOP_KV_LAST;
    float *pl = run_engine(&per, false, NPROMPT);
    CHECK(!same(pe, pl, NPROMPT));
    fprintf(stderr, "ok: PER_LOOP loops=3 matches unrolled reference; LAST vs PER_LOOP differ by %.3g\n",
            (double)max_abs_diff(pe, pl, NPROMPT));
    free(pr); free(pe); free(pl); free(le);

    /* 4. WSP through real inference. */
    {
        char err[512];
        const int n = 8;
        const float *base = ref + (size_t)(n - 1) * DS4_N_VOCAB;
        const int top = argmax(base, (int)DS4_N_VOCAB);
        const int out_t = ds4_wsp_find(&E, "output.weight");
        CHECK(out_t >= 0);
        /* Row t of the output head produces logit t: masking it zeroes it. */
        CHECK(ds4_wsp_scale(&E, out_t, DS4_WSP_ROWS, (uint64_t)top, 1, 0.0f, err, sizeof(err)) == 0);
        float *w1 = run_engine(&stock, true, n);
        const float *l1 = w1 + (size_t)(n - 1) * DS4_N_VOCAB;
        CHECK(l1[top] == 0.0f);
        for (int i = 0; i < (int)DS4_N_VOCAB; i++) if (i != top) CHECK(l1[i] == base[i]);
        /* Scaling a row by 2 doubles that logit (Q8_0 requantization). */
        CHECK(ds4_wsp_revert_all(&E, err, sizeof(err)) == 0);
        CHECK(ds4_wsp_scale(&E, out_t, DS4_WSP_ROWS, (uint64_t)top, 1, 2.0f, err, sizeof(err)) == 0);
        float *w2 = run_engine(&stock, true, n);
        const float r = w2[(size_t)(n - 1) * DS4_N_VOCAB + top] / base[top];
        CHECK(fabsf(r - 2.0f) < 0.02f);
        /* Ablate shared-expert neurons in one layer (Q8_0 columns, aliased
         * across all layers here) and zero routed expert rows (Q2_K). */
        CHECK(ds4_wsp_revert_all(&E, err, sizeof(err)) == 0);
        const int down = ds4_wsp_find(&E, "blk.10.ffn_down_shexp.weight");
        const int exps = ds4_wsp_find(&E, "blk.10.ffn_down_exps.weight");
        CHECK(ds4_wsp_scale(&E, down, DS4_WSP_COLS, 0, 512, 0.0f, err, sizeof(err)) == 0);
        CHECK(ds4_wsp_scale(&E, exps, DS4_WSP_ROWS, 0, 4096 * 32, 0.0f, err, sizeof(err)) == 0);
        float *w3 = run_engine(&stock, true, n);
        CHECK(all_finite(w3, n * (int)DS4_N_VOCAB) && !same(ref, w3, n));
        const uint64_t dirty = ds4_wsp_dirty_bytes(&E);
        /* Revert restores bit-identical inference. */
        CHECK(ds4_wsp_revert_all(&E, err, sizeof(err)) == 0 && ds4_wsp_dirty_bytes(&E) == 0);
        float *w4 = run_engine(&stock, true, n);
        CHECK(same(ref, w4, n));
        fprintf(stderr, "ok: WSP mask/scale change real logits as predicted; revert of %.1f MiB "
                        "restores bit-identical output\n", (double)dirty / (1024.0 * 1024.0));
        free(w1); free(w2); free(w3); free(w4);
    }

    /* 5. Session API: looped sync/eval, elastic changes, concept steps. */
    {
        char err[512];
        ds4_session *s = xcalloc(1, sizeof(*s));
        s->engine = &E;
        s->ctx_size = CTX;
        s->prefill_cap = CTX;
        kv_cache_init(&s->cpu_cache, CTX, 0);
        cpu_decode_scratch_init(&s->cpu_scratch, CTX);
        s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
        s->sample_probs = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
        s->loop = (ds4_loop_state){.cfg = {.num_loops = 1}, .levels = 1, .synced = 1, .ctx_size = CTX};

        ds4_tokens p = {0};
        for (int i = 0; i < 6; i++) ds4_tokens_push(&p, prompt[i]);
        /* Stock sync: layer-major prefill; the hidden state is captured. */
        CHECK(ds4_session_sync(s, &p, err, sizeof(err)) == 0);
        float *hidden = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
        CHECK(ds4_session_hidden(s, hidden, (int)DS4_N_EMBD) == (int)DS4_N_EMBD);
        CHECK(all_finite(hidden, (int)DS4_N_EMBD));

        /* Switch to LAST loops live: checkpoint kept (elastic). */
        ds4_loop_config cfg = {.rec_start = 4, .rec_end = 9, .num_loops = 3};
        CHECK(ds4_session_set_loops(s, &cfg, err, sizeof(err)) == 0);
        CHECK(s->checkpoint_valid && s->checkpoint.len == 6);
        CHECK(ds4_session_eval(s, prompt[6], err, sizeof(err)) == 0);
        CHECK(all_finite(s->logits, (int)DS4_N_VOCAB));

        /* A non-prefix prompt makes a looped session rebuild position by
         * position: it must equal the direct looped decode of that prompt. */
        ds4_tokens q = {0};
        for (int i = 0; i < 10; i++) ds4_tokens_push(&q, prompt[i]);
        q.v[0] = 2;
        CHECK(ds4_session_sync(s, &q, err, sizeof(err)) == 0);
        {
            ds4_kv_cache cache;
            kv_cache_init(&cache, CTX, 0);
            ds4_cpu_decode_scratch sc;
            cpu_decode_scratch_init(&sc, CTX);
            ds4_loop_state ls = {.cfg = cfg, .levels = 1, .synced = 1, .ctx_size = CTX};
            float *lg = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
            for (int t = 0; t < q.len; t++)
                forward_token_looped_cpu(lg, &E.model, &E.weights, &cache, &ls, q.v[t], NULL,
                                         (uint32_t)t, NULL, 0, 0, &sc);
            CHECK(!memcmp(lg, s->logits, (size_t)DS4_N_VOCAB * sizeof(float)));
            free(lg);
            loop_state_free(&ls);
            cpu_decode_scratch_free(&sc);
            kv_cache_free(&cache);
        }
        /* Disk payloads are refused for looped state. */
        FILE *fp = tmpfile();
        CHECK(ds4_session_save_payload(s, fp, err, sizeof(err)) != 0 && strstr(err, "looped"));
        fclose(fp);

        /* PER_LOOP with history needs a rebuild, then decodes. */
        cfg.kv_mode = DS4_LOOP_KV_PER_LOOP;
        CHECK(ds4_session_set_loops(s, &cfg, err, sizeof(err)) == 0);
        CHECK(!s->checkpoint_valid);
        CHECK(ds4_session_sync(s, &q, err, sizeof(err)) == 0 && s->checkpoint.len == q.len);
        CHECK(s->loop.synced == 3);
        cfg.num_loops = 2;                  /* lowering: live */
        CHECK(ds4_session_set_loops(s, &cfg, err, sizeof(err)) == 0 && s->checkpoint_valid);
        CHECK(ds4_session_eval(s, 77, err, sizeof(err)) == 0 && s->loop.synced == 2);
        cfg.num_loops = 3;                  /* raising past synced: rebuild */
        CHECK(ds4_session_set_loops(s, &cfg, err, sizeof(err)) == 0 && !s->checkpoint_valid);

        /* Concept steps: residual codebooks over the real hidden size. */
        cfg.num_loops = 1;
        CHECK(ds4_session_set_loops(s, &cfg, err, sizeof(err)) == 0);
        CHECK(ds4_session_sync(s, &p, err, sizeof(err)) == 0);
        char path[] = "/tmp/ds4-mlc-model.XXXXXX";
        int fd = mkstemp(path);
        CHECK(fd >= 0);
        fp = fdopen(fd, "wb");
        const uint32_t hdr[3] = {1, 2, (uint32_t)DS4_N_EMBD}, ks[2] = {16, 64};
        fwrite("DS4C", 1, 4, fp);
        fwrite(hdr, 4, 3, fp);
        for (int l = 0; l < 2; l++) {
            fwrite(&ks[l], 4, 1, fp);
            for (uint64_t i = 0; i < (uint64_t)ks[l] * DS4_N_EMBD; i++) {
                const float v = runi() * (l == 0 ? 0.05f : 0.01f);
                fwrite(&v, 4, 1, fp);
            }
        }
        fclose(fp);
        ds4_mlc *mlc = NULL;
        CHECK(ds4_mlc_open(&mlc, path, err, sizeof(err)) == 0);
        unlink(path);
        uint64_t rng = 7;
        ds4_step_options so = {.temperature = 0.0f, .rng = &rng, .mlc = mlc, .concept_tag = 1};
        ds4_step st;
        const int pos0 = s->checkpoint.len;
        CHECK(ds4_session_step(s, &so, &st, err, sizeof(err)) == 0);
        CHECK(st.kind == DS4_STEP_CONCEPT && st.levels == 2 && s->checkpoint.len == pos0 + 1);
        CHECK(st.codes[0] >= 0 && st.codes[0] < 16 && st.codes[1] >= 0 && st.codes[1] < 64);
        CHECK(all_finite(s->logits, (int)DS4_N_VOCAB) && s->has_concepts);
        so.mlc = NULL;                      /* then an ordinary token step */
        CHECK(ds4_session_step(s, &so, &st, err, sizeof(err)) == 0);
        CHECK(st.kind == DS4_STEP_TOKEN && s->checkpoint.len == pos0 + 2);
        fp = tmpfile();
        CHECK(ds4_session_save_payload(s, fp, err, sizeof(err)) != 0 && strstr(err, "concept"));
        fclose(fp);
        fprintf(stderr, "ok: session sync/eval with loops, elastic policy, concept step "
                        "then token step\n");
        ds4_mlc_free(mlc);
        ds4_tokens_free(&p);
        ds4_tokens_free(&q);
        free(hidden);
    }

    free(ref);
    puts("test_elastic_model: ok");
    return 0;
}
