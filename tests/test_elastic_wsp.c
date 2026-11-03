/* Elastic loops, weight-space programming and multi-level decoding, tested
 * without a model: the private helpers are exercised directly on synthetic
 * caches, a real private file mapping, and synthetic codebooks. */
#include "../ds4.c"
#include <assert.h>

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static float frand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)((rng_state >> 40) & 0xffffff) / (float)0xffffff * 2.0f - 1.0f;
}

static void fill(float *x, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) x[i] = frand();
}

/* ---- KV mark / rollback ---------------------------------------------- */

static void layer_cache_copy(ds4_layer_cache *dst, const ds4_layer_cache *src) {
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

/* Equal as seen by attention: live rows, counts and compressor states. */
static bool layer_cache_same(const ds4_layer_cache *a, const ds4_layer_cache *b) {
    const uint32_t ratio = a->compress_ratio;
    const uint64_t attn = layer_cache_state_floats(ratio, DS4_N_HEAD_DIM);
    const uint64_t index = ratio == 4 ? layer_cache_state_floats(ratio, DS4_N_INDEXER_HEAD_DIM) : 0;
    if (a->n_raw != b->n_raw || a->n_comp != b->n_comp || a->n_index_comp != b->n_index_comp)
        return false;
#define SAME(F, N) ((N) == 0 || memcmp(a->F, b->F, (N) * sizeof(float)) == 0)
    return SAME(raw_kv, (uint64_t)a->n_raw * DS4_N_HEAD_DIM) &&
           SAME(attn_comp_kv, (uint64_t)a->n_comp * DS4_N_HEAD_DIM) &&
           SAME(attn_state_kv, attn) && SAME(attn_state_score, attn) &&
           SAME(index_comp_kv, (uint64_t)a->n_index_comp * DS4_N_INDEXER_HEAD_DIM) &&
           SAME(index_state_kv, index) && SAME(index_state_score, index);
#undef SAME
}

/* What one decode pass does to a layer cache. */
static void fake_pass(ds4_layer_cache *lc, bool emit_comp) {
    float kv[4096];
    fill(kv, DS4_N_HEAD_DIM);
    kv_cache_push_raw(lc, kv);
    const uint32_t ratio = lc->compress_ratio;
    if (!ratio) return;
    fill(lc->attn_state_kv, layer_cache_state_floats(ratio, DS4_N_HEAD_DIM));
    fill(lc->attn_state_score, layer_cache_state_floats(ratio, DS4_N_HEAD_DIM));
    if (emit_comp) kv_cache_push_comp(lc->attn_comp_kv, &lc->n_comp, lc->comp_cap, DS4_N_HEAD_DIM, kv);
    if (ratio == 4) {
        fill(lc->index_state_kv, layer_cache_state_floats(ratio, DS4_N_INDEXER_HEAD_DIM));
        fill(lc->index_state_score, layer_cache_state_floats(ratio, DS4_N_INDEXER_HEAD_DIM));
        if (emit_comp) kv_cache_push_comp(lc->index_comp_kv, &lc->n_index_comp, lc->comp_cap,
                                          DS4_N_INDEXER_HEAD_DIM, kv);
    }
}

static void test_mark_rollback(void) {
    const uint32_t ratios[] = {0, 4, 128};
    ds4_layer_cache_mark mark = {0};
    for (size_t r = 0; r < sizeof(ratios) / sizeof(ratios[0]); r++) {
        g_ds4_compress_ratios[0] = ratios[r];
        for (uint32_t prefill = 0; prefill < 12; prefill++) {
            ds4_layer_cache lc, before;
            layer_cache_init(&lc, 0, 1024, 8);
            for (uint32_t i = 0; i < prefill; i++) fake_pass(&lc, i % 3 == 2);
            layer_cache_copy(&before, &lc);

            /* Several provisional passes must leave no trace... */
            for (int k = 0; k < 3; k++) {
                layer_cache_mark_save(&mark, &lc);
                fake_pass(&lc, k == 1);
                assert(lc.n_raw >= 1);
                layer_cache_mark_restore(&mark, &lc);
                assert(layer_cache_same(&lc, &before));
            }
            /* ...and the committing pass behaves like a stock pass. */
            const uint64_t saved = rng_state;
            fake_pass(&lc, true);
            rng_state = saved;
            fake_pass(&before, true);
            assert(layer_cache_same(&lc, &before));

            layer_cache_free(&lc);
            layer_cache_free(&before);
        }
    }
    layer_cache_mark_free(&mark);
    g_ds4_compress_ratios[0] = 0;
}

/* ---- loop scheduling policy ------------------------------------------ */

static ds4_engine fake_engine;

static ds4_session *fake_session(int tokens) {
    ds4_session *s = xcalloc(1, sizeof(*s));
    s->engine = &fake_engine;
    s->ctx_size = 256;
    s->loop.cfg.num_loops = 1;
    s->loop.levels = 1;
    s->loop.synced = 1;
    s->loop.ctx_size = 256;
    for (int i = 0; i < tokens; i++) token_vec_push(&s->checkpoint, i);
    s->checkpoint_valid = true;
    return s;
}

static void fake_session_free(ds4_session *s) {
    loop_state_free(&s->loop);
    token_vec_free(&s->checkpoint);
    free(s);
}

static void test_loop_policy(void) {
    fake_engine.backend = DS4_BACKEND_CPU;
    char err[256];
    ds4_loop_config bad = {.rec_start = 4, .rec_end = 4, .num_loops = 2};
    assert(ds4_loop_config_check(&bad, 43, err, sizeof(err)) != 0);
    bad.rec_end = 44;
    assert(ds4_loop_config_check(&bad, 43, err, sizeof(err)) != 0);
    bad = (ds4_loop_config){.rec_start = 0, .rec_end = 4, .num_loops = DS4_LOOP_MAX + 1};
    assert(ds4_loop_config_check(&bad, 43, err, sizeof(err)) != 0);
    ds4_loop_config stock = {0};
    assert(ds4_loop_config_check(&stock, 43, err, sizeof(err)) == 0);

    /* LAST mode is elastic: any change keeps a live checkpoint. */
    ds4_session *s = fake_session(5);
    ds4_loop_config last = {.rec_start = 10, .rec_end = 20, .num_loops = 4};
    assert(ds4_session_set_loops(s, &last, err, sizeof(err)) == 0);
    assert(s->checkpoint_valid && s->loop.levels == 1 && s->loop.extra == NULL);
    last.num_loops = 2;
    assert(ds4_session_set_loops(s, &last, err, sizeof(err)) == 0);
    assert(s->checkpoint_valid);
    fake_session_free(s);

    /* PER_LOOP: a fresh session allocates and syncs every level... */
    s = fake_session(0);
    ds4_loop_config per = {.rec_start = 10, .rec_end = 13, .num_loops = 3,
                           .kv_mode = DS4_LOOP_KV_PER_LOOP};
    assert(ds4_session_set_loops(s, &per, err, sizeof(err)) == 0);
    assert(s->loop.levels == 3 && s->loop.synced == 3);
    ds4_kv_cache cache;
    memset(&cache, 0, sizeof(cache));
    assert(loop_layer_cache(&cache, &s->loop, 0, 11) == &cache.layer[11]);
    assert(loop_layer_cache(&cache, &s->loop, 1, 10) == &s->loop.extra[0]);
    assert(loop_layer_cache(&cache, &s->loop, 2, 12) == &s->loop.extra[3 + 2]);
    for (int i = 0; i < 4; i++) {
        token_vec_push(&s->checkpoint, i);
        loop_state_note_position(&s->loop);
    }
    assert(s->loop.synced == 3);

    /* ...lowering is free, but the dropped level misses the next position... */
    per.num_loops = 2;
    assert(ds4_session_set_loops(s, &per, err, sizeof(err)) == 0);
    assert(s->checkpoint_valid && s->loop.levels == 3);
    token_vec_push(&s->checkpoint, 9);
    loop_state_note_position(&s->loop);
    assert(s->loop.synced == 2);

    /* ...so raising again needs a rebuild. */
    per.num_loops = 3;
    assert(ds4_session_set_loops(s, &per, err, sizeof(err)) == 0);
    assert(!s->checkpoint_valid);
    loop_state_reset(&s->loop);
    assert(s->loop.synced == 3);

    /* A new range drops the per-loop caches of the old one. */
    per.rec_start = 30;
    per.rec_end = 31;
    assert(ds4_session_set_loops(s, &per, err, sizeof(err)) == 0);
    assert(s->loop.levels == 3 && s->loop.extra[0].cap_raw != 0);
    stock.num_loops = 1;
    assert(ds4_session_set_loops(s, &stock, err, sizeof(err)) == 0);
    assert(!loop_state_active(&s->loop) && s->loop.extra == NULL);
    fake_session_free(s);

    /* Non-CPU sessions accept only the stock schedule. */
    fake_engine.backend = DS4_BACKEND_METAL;
    s = fake_session(0);
    assert(ds4_session_set_loops(s, &stock, err, sizeof(err)) == 0);
    assert(ds4_session_set_loops(s, &last, err, sizeof(err)) != 0);
    fake_session_free(s);
    fake_engine.backend = DS4_BACKEND_CPU;
}

/* ---- WSP codecs ------------------------------------------------------- */

static void test_wsp_codecs(void) {
    float in[64], out[64];
    fill(in, 64);
    uint8_t buf[64 * 4];

    wsp_row_encode(DS4_TENSOR_F32, buf, in, 64);
    wsp_row_decode(DS4_TENSOR_F32, buf, out, 64);
    assert(!memcmp(in, out, sizeof(in)));

    wsp_row_encode(DS4_TENSOR_F16, buf, in, 64);
    wsp_row_decode(DS4_TENSOR_F16, buf, out, 64);
    for (int i = 0; i < 64; i++) assert(fabsf(in[i] - out[i]) < 1e-3f);

    wsp_row_encode(DS4_TENSOR_BF16, buf, in, 64);
    wsp_row_decode(DS4_TENSOR_BF16, buf, out, 64);
    for (int i = 0; i < 64; i++) assert(fabsf(in[i] - out[i]) < 1e-2f);
    assert(f32_to_bf16_bits(1.0f) == 0x3f80);

    /* Q8_0: blocks whose values did not change keep their exact bytes. */
    memset(buf, 0, sizeof(buf));
    for (int b = 0; b < 2; b++) {
        const uint16_t d = f32_to_f16(0.01f * (b + 1));
        memcpy(buf + b * 34, &d, 2);
        for (int i = 0; i < 32; i++) buf[b * 34 + 2 + i] = (uint8_t)(int8_t)(i - 16 + b);
    }
    uint8_t orig[68];
    memcpy(orig, buf, sizeof(orig));
    wsp_row_decode(DS4_TENSOR_Q8_0, buf, out, 64);
    wsp_row_encode(DS4_TENSOR_Q8_0, buf, out, 64);
    assert(!memcmp(buf, orig, sizeof(orig)));
    out[40] = 0.0f;                     /* touch only the second block */
    wsp_row_encode(DS4_TENSOR_Q8_0, buf, out, 64);
    assert(!memcmp(buf, orig, 34));
    float back[64];
    wsp_row_decode(DS4_TENSOR_Q8_0, buf, back, 64);
    assert(back[40] == 0.0f);
    /* Requantization error is at most one step of the new scale (amax/127). */
    for (int i = 0; i < 32; i++) assert(back[i] == out[i]);
    for (int i = 32; i < 64; i++) assert(fabsf(back[i] - out[i]) <= 0.32f / 127.0f);
}

/* ---- WSP over a real private mapping ---------------------------------- */

static char tmpdir[] = "/tmp/ds4-wsp-test.XXXXXX";

static void path_in(char *out, size_t cap, const char *name) {
    snprintf(out, cap, "%s/%s", tmpdir, name);
}

/* Two tensors: an F16 4x8 matrix and a Q2_K tensor of one 256-element row
 * per output (only row masking is allowed there). */
static void fake_model(ds4_engine *e, uint8_t *file, uint64_t size) {
    char path[256];
    path_in(path, sizeof(path), "model.bin");
    FILE *fp = fopen(path, "wb");
    assert(fp && fwrite(file, 1, size, fp) == size);
    fclose(fp);

    memset(e, 0, sizeof(*e));
    e->backend = DS4_BACKEND_CPU;
    ds4_model *m = &e->model;
    m->fd = open(path, O_RDONLY);
    assert(m->fd >= 0);
    m->map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, m->fd, 0);
    assert(m->map != MAP_FAILED);
    m->size = m->file_size = size;
    m->private_map = true;
    m->n_tensors = 2;
    m->tensors = xcalloc(2, sizeof(ds4_tensor));

    ds4_tensor *t = &m->tensors[0];
    t->name = (ds4_str){"blk.0.ffn_down.weight", strlen("blk.0.ffn_down.weight")};
    t->ndim = 2;
    t->dim[0] = 8;
    t->dim[1] = 4;
    t->type = DS4_TENSOR_F16;
    t->elements = 32;
    t->bytes = 64;
    t->abs_offset = 0;

    t = &m->tensors[1];
    t->name = (ds4_str){"blk.0.ffn_down_exps.weight", strlen("blk.0.ffn_down_exps.weight")};
    t->ndim = 3;
    t->dim[0] = 256;
    t->dim[1] = 2;
    t->dim[2] = 2;
    t->type = DS4_TENSOR_Q2_K;
    t->elements = 1024;
    t->bytes = 4 * 84;
    t->abs_offset = 4096;
}

static void test_wsp_mapping(void) {
    assert(mkdtemp(tmpdir));
    const uint64_t size = 8192;
    uint8_t *file = xcalloc(1, size);
    for (int i = 0; i < 32; i++) {
        const uint16_t h = f32_to_f16((float)(i + 1));
        memcpy(file + i * 2, &h, 2);
    }
    for (int i = 0; i < 4 * 84; i++) file[4096 + i] = (uint8_t)(i * 7 + 1);

    ds4_engine e;
    fake_model(&e, file, size);
    char err[512];
    const int dense = ds4_wsp_find(&e, "blk.0.ffn_down.weight");
    const int exps = ds4_wsp_find(&e, "blk.0.ffn_down_exps.weight");
    assert(dense == 0 && exps == 1 && ds4_wsp_find(&e, "nope") == -1);

    ds4_wsp_tensor info;
    assert(ds4_wsp_info(&e, dense, &info) == 0);
    assert(!strcmp(info.name, "blk.0.ffn_down.weight") && info.rows == 4 && info.cols == 8);
    assert(info.editable && !info.dirty);
    assert(ds4_wsp_info(&e, exps, &info) == 0 && info.rows == 4 && !info.editable);

    /* Column mask = ablate input features 2..3 of every row. */
    assert(ds4_wsp_scale(&e, dense, DS4_WSP_COLS, 2, 2, 0.0f, err, sizeof(err)) == 0);
    /* Row scale. */
    assert(ds4_wsp_scale(&e, dense, DS4_WSP_ROWS, 1, 1, 2.0f, err, sizeof(err)) == 0);
    float row[8];
    for (uint64_t r = 0; r < 4; r++) {
        assert(ds4_wsp_read_row(&e, dense, r, row, err, sizeof(err)) == 0);
        for (int c = 0; c < 8; c++) {
            float want = (float)(r * 8 + c + 1) * (r == 1 ? 2.0f : 1.0f);
            if (c == 2 || c == 3) want = 0.0f;
            assert(row[c] == want);
        }
    }
    /* Quantized rows: masking works, other edits are refused. */
    assert(ds4_wsp_scale(&e, exps, DS4_WSP_ROWS, 2, 2, 0.0f, err, sizeof(err)) == 0);
    for (int i = 0; i < 2 * 84; i++) assert(e.model.map[4096 + 2 * 84 + i] == 0);
    assert(e.model.map[4096] == file[4096]);
    assert(ds4_wsp_scale(&e, exps, DS4_WSP_ROWS, 0, 1, 0.5f, err, sizeof(err)) != 0);
    assert(ds4_wsp_scale(&e, exps, DS4_WSP_COLS, 0, 1, 0.0f, err, sizeof(err)) != 0);
    assert(ds4_wsp_scale(&e, dense, DS4_WSP_ROWS, 3, 2, 0.0f, err, sizeof(err)) != 0);
    assert(ds4_wsp_info(&e, dense, &info) == 0 && info.dirty);
    assert(ds4_wsp_dirty_bytes(&e) == 64 + 2 * 84);

    /* The file itself is never written. */
    char path[256];
    path_in(path, sizeof(path), "model.bin");
    uint8_t *disk = xmalloc(size);
    FILE *fp = fopen(path, "rb");
    assert(fp && fread(disk, 1, size, fp) == size);
    fclose(fp);
    assert(!memcmp(disk, file, size));

    /* Revert re-reads the original bytes. */
    assert(ds4_wsp_revert_all(&e, err, sizeof(err)) == 0);
    assert(ds4_wsp_dirty_bytes(&e) == 0);
    assert(!memcmp(e.model.map, file, size));

    /* Closed-form load from a script, relative to the script directory. */
    float m[32];
    for (int i = 0; i < 32; i++) m[i] = (i % 9 == 0) ? 1.0f : 0.0f; /* 4x8 "identity" */
    char mpath[256], spath[256];
    path_in(mpath, sizeof(mpath), "eye.f32");
    fp = fopen(mpath, "wb");
    assert(fp && fwrite(m, sizeof(float), 32, fp) == 32);
    fclose(fp);
    path_in(spath, sizeof(spath), "patch.wsp");
    fp = fopen(spath, "w");
    assert(fp);
    fprintf(fp, "# test patch\n"
                "load blk.0.ffn_down.weight eye.f32\n"
                "scale blk.0.ffn_down.weight cols 0,4-5 3\n"
                "mask blk.0.ffn_down_exps.weight rows 0-1\n");
    fclose(fp);
    assert(ds4_wsp_apply_script(&e, spath, err, sizeof(err)) == 0);
    for (uint64_t r = 0; r < 4; r++) {
        assert(ds4_wsp_read_row(&e, dense, r, row, err, sizeof(err)) == 0);
        for (int c = 0; c < 8; c++) {
            float want = m[r * 8 + c];
            if (c == 0 || c == 4 || c == 5) want *= 3.0f;
            assert(row[c] == want);
        }
    }
    for (int i = 0; i < 2 * 84; i++) assert(e.model.map[4096 + i] == 0);

    fp = fopen(spath, "w");
    fprintf(fp, "revert *\nscale blk.0.ffn_down.weight sideways 1 2\n");
    fclose(fp);
    assert(ds4_wsp_apply_script(&e, spath, err, sizeof(err)) != 0);
    assert(strstr(err, "patch.wsp:2:"));
    assert(ds4_wsp_dirty_bytes(&e) == 0);

    /* Graph backends refuse edits instead of ignoring them. */
    e.backend = DS4_BACKEND_METAL;
    assert(ds4_wsp_scale(&e, dense, DS4_WSP_ROWS, 0, 1, 0.0f, err, sizeof(err)) != 0);

    munmap((void *)e.model.map, size);
    close(e.model.fd);
    free(e.model.tensors);
    free(e.wsp_lo);
    free(e.wsp_hi);
    free(disk);
    free(file);
    unlink(mpath);
    unlink(spath);
    unlink(path);
    rmdir(tmpdir);
}

/* ---- multi-level codes ------------------------------------------------ */

static void test_mlc(void) {
    /* Level 0: 4 coarse directions of norm 4; level 1: 8 fine ones of norm 1,
     * so residual decoding must resolve coarse first, then fine. */
    enum { DIM = 16, L0 = 4, L1 = 8 };
    float b0[L0 * DIM] = {0}, b1[L1 * DIM] = {0};
    for (int k = 0; k < L0; k++) b0[k * DIM + k] = 4.0f;
    for (int k = 0; k < L1; k++) b1[k * DIM + 4 + k] = 1.0f;

    char path[] = "/tmp/ds4-mlc-test.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "wb");
    const uint32_t hdr[4] = {0, 1, 2, DIM};
    fwrite("DS4C", 1, 4, fp);
    fwrite(hdr + 1, 4, 3, fp);
    uint32_t k = L0;
    fwrite(&k, 4, 1, fp);
    fwrite(b0, sizeof(float), L0 * DIM, fp);
    k = L1;
    fwrite(&k, 4, 1, fp);
    fwrite(b1, sizeof(float), L1 * DIM, fp);
    fclose(fp);

    ds4_mlc *m = NULL;
    char err[256];
    assert(ds4_mlc_open(&m, path, err, sizeof(err)) == 0);
    assert(ds4_mlc_levels(m) == 2 && ds4_mlc_dim(m) == DIM);

    float h[DIM] = {0}, concept[DIM];
    h[2] = 4.1f;            /* coarse code 2 */
    h[4 + 5] = 0.9f;        /* fine code 5 */
    int codes[DS4_MLC_MAX_LEVELS];
    assert(ds4_mlc_decode(m, h, 0, 0.0f, NULL, codes, concept) == 0);
    assert(codes[0] == 2 && codes[1] == 5);
    assert(concept[2] == 4.0f && concept[9] == 1.0f && concept[0] == 0.0f);

    /* Truncated decoding stops at the coarse concept. */
    assert(ds4_mlc_decode(m, h, 1, 0.0f, NULL, codes, concept) == 0);
    assert(codes[0] == 2 && concept[2] == 4.0f && concept[9] == 0.0f);

    /* Sampling needs an rng and stays within the codebook. */
    assert(ds4_mlc_decode(m, h, 0, 1.0f, NULL, codes, NULL) != 0);
    uint64_t rng = 1;
    for (int i = 0; i < 50; i++) {
        assert(ds4_mlc_decode(m, h, 0, 0.7f, &rng, codes, NULL) == 0);
        assert(codes[0] >= 0 && codes[0] < L0 && codes[1] >= 0 && codes[1] < L1);
    }
    int bad[2] = {L0, 0};
    assert(ds4_mlc_concept(m, bad, 2, concept) != 0);
    ds4_mlc_free(m);

    fp = fopen(path, "wb");
    fwrite("NOPE", 1, 4, fp);
    fclose(fp);
    assert(ds4_mlc_open(&m, path, err, sizeof(err)) != 0 && m == NULL);
    unlink(path);
}

int main(void) {
    test_mark_rollback();
    test_loop_policy();
    test_wsp_codecs();
    test_wsp_mapping();
    test_mlc();
    puts("test_elastic_wsp: ok");
    return 0;
}
