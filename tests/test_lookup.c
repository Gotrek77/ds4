#include "ds4_lookup.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

static void check_tokens(const int *got, const int *want, int n) {
    for (int i = 0; i < n; i++) CHECK(got[i] == want[i]);
}

static void test_too_short(void) {
    const int hist[] = {1, 2};
    int out[4];
    ds4_lookup_result r = ds4_lookup_propose(hist, 2, 3, out, 4);
    CHECK(r.n == 0 && r.match == 0);
    r = ds4_lookup_propose(NULL, 0, 3, out, 4);
    CHECK(r.n == 0);
}

static void test_no_match(void) {
    const int hist[] = {1, 2, 3, 4, 5, 6, 7};
    int out[4];
    const ds4_lookup_result r = ds4_lookup_propose(hist, 7, 8, out, 4);
    CHECK(r.n == 0 && r.match == 0);
}

/* A match below the minimum length is not worth a verify row. */
static void test_short_match_rejected(void) {
    const int hist[] = {9, 5, 6, 40, 41, 42, 1, 5};
    int out[4];
    const ds4_lookup_result r = ds4_lookup_propose(hist, 8, 6, out, 4);
    CHECK(r.match == 0);
}

static void test_basic_continuation(void) {
    /* "a b c d e f" earlier, then "x a b c" now: propose "d e f". */
    const int hist[] = {10, 11, 12, 13, 14, 15, 99, 10, 11};
    int out[3];
    const ds4_lookup_result r = ds4_lookup_propose(hist, 9, 12, out, 3);
    CHECK(r.n == 3 && r.match == 3);
    const int want[] = {13, 14, 15};
    check_tokens(out, want, 3);
}

/* The longest suffix wins even when a shorter one is more recent. */
static void test_longest_then_most_recent(void) {
    const int hist[] = {7, 1, 2, 3, 50, 51,   /* 7 1 2 3 -> 50: length 4 */
                        8, 1, 2, 3, 60, 61,   /* 8 1 2 3 -> 60: length 3 */
                        7, 1, 2};
    int out[2];
    ds4_lookup_result r = ds4_lookup_propose(hist, 15, 3, out, 2);
    CHECK(r.match == 4);
    CHECK(out[0] == 50 && out[1] == 51);

    /* Equal lengths: the most recent occurrence is proposed. */
    const int tie[] = {1, 2, 3, 50, 9, 1, 2, 3, 60, 9, 1, 2};
    r = ds4_lookup_propose(tie, 12, 3, out, 1);
    CHECK(r.match == 4);
    CHECK(out[0] == 60);
}

/* A repeating pattern continues from its own period past the history end. */
static void test_periodic_overlap(void) {
    const int hist[] = {4, 5, 4, 5, 4, 5};
    int out[5];
    const ds4_lookup_result r = ds4_lookup_propose(hist, 6, 4, out, 5);
    CHECK(r.n == 5);
    CHECK(r.match >= DS4_LOOKUP_MIN_MATCH);
    const int want[] = {5, 4, 5, 4, 5};
    check_tokens(out, want, 5);
}

static void test_long_match_caps_and_is_strong(void) {
    enum { N = 64 };
    int hist[2 * N];
    for (int i = 0; i < N; i++) hist[i] = 100 + i;
    for (int i = 0; i < N - 1; i++) hist[N + i] = 100 + i;
    int out[5];
    const ds4_lookup_result r = ds4_lookup_propose(hist, 2 * N - 1, 100 + N - 1, out, 5);
    CHECK(r.match == DS4_LOOKUP_MAX_MATCH);
    CHECK(r.match >= DS4_LOOKUP_STRONG_MATCH);
    /* The earlier copy ends at index N - 1; it continues with the copy. */
    const int want[] = {100, 101, 102, 103, 104};
    check_tokens(out, want, 5);
}

/* The pending token is part of the searched history as well as the suffix. */
static void test_pending_token_in_history(void) {
    const int hist[] = {1, 2, 3};
    int out[3];
    const ds4_lookup_result r = ds4_lookup_propose(hist, 3, 1, out, 3);
    CHECK(r.match == 0);
    const int hist2[] = {1, 2, 3, 1, 2};
    const ds4_lookup_result r2 = ds4_lookup_propose(hist2, 5, 3, out, 3);
    CHECK(r2.match == 3);
    const int want[] = {1, 2, 3};
    check_tokens(out, want, 3);
}

/* Compare against an obviously correct quadratic search on random data. */
static void test_matches_reference(void) {
    unsigned seed = 12345;
    for (int iter = 0; iter < 2000; iter++) {
        int hist[96];
        const int len = 4 + (int)((seed = seed * 1103515245u + 12345u) >> 16) % 92;
        const int alphabet = 2 + (int)((seed = seed * 1103515245u + 12345u) >> 16) % 5;
        for (int i = 0; i < len; i++)
            hist[i] = (int)((seed = seed * 1103515245u + 12345u) >> 16) % alphabet;
        const int next = (int)((seed = seed * 1103515245u + 12345u) >> 16) % alphabet;
        int all[97];
        memcpy(all, hist, sizeof(int) * (size_t)len);
        all[len] = next;
        const int total = len + 1;
        int best = 0, best_end = -1;
        for (int end = 0; end <= total - 2; end++) {
            int m = 0;
            while (m < DS4_LOOKUP_MAX_MATCH && end - m >= 0 && all[end - m] == all[total - 1 - m]) m++;
            if (m >= best && m > 0) { best = m; best_end = end; }
        }
        int out[6] = {0};
        const ds4_lookup_result r = ds4_lookup_propose(hist, len, next, out, 6);
        if (best < DS4_LOOKUP_MIN_MATCH) {
            CHECK(r.match == 0);
            continue;
        }
        CHECK(r.match == best);
        int ext[97 + 6];
        memcpy(ext, all, sizeof(int) * (size_t)total);
        for (int k = 0; k < 6; k++) ext[total + k] = ext[best_end + 1 + k];
        check_tokens(out, ext + total, 6);
    }
}

int main(void) {
    test_too_short();
    test_no_match();
    test_short_match_rejected();
    test_basic_continuation();
    test_longest_then_most_recent();
    test_periodic_overlap();
    test_long_match_caps_and_is_strong();
    test_pending_token_in_history();
    test_matches_reference();
    if (failures) {
        fprintf(stderr, "test_lookup: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_lookup: ok\n");
    return 0;
}
