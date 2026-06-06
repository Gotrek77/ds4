/* ds4-srm-lib-test.c — Test the SRM library API (single TU with .c) */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#define SRM_LIBRARY_MODE
#include "ds4-srm.c"

int main(void) {
    int pass = 0, fail = 0;
    srm_kb *kb = srm_kb_create();

    /* Test srm_assert_fact */
    if (!srm_assert_fact(kb, "fact: umano(socrate)")) {
        fprintf(stderr, "FAIL: assert fact\n"); fail++;
    } else pass++;

    /* Test srm_assert_rule */
    if (!srm_assert_rule(kb, "rule: mortale(X) ← umano(X)")) {
        fprintf(stderr, "FAIL: assert rule\n"); fail++;
    } else pass++;

    /* Test query via parse_line */
    srm_literal q;
    srm_rule dummy_r;
    srm_literal dummy_f;
    char err[256];
    int qt = srm_parse_line("?mortale(socrate)", &dummy_f, &dummy_r, &q, err, sizeof(err));
    if (qt == 2) pass++; else { fprintf(stderr, "FAIL: parse query\n"); fail++; }
    srm_query_result qr = srm_answer_query(kb, &q);
    if (qr.success) pass++; else { fprintf(stderr, "FAIL: query\n"); fail++; }
    srm_literal_free(&q);
    srm_query_result_free(&qr);

    /* Test forward chaining */
    int added = srm_forward_chain(kb);
    if (added >= 0) pass++; else { fprintf(stderr, "FAIL: forward_chain\n"); fail++; }

    /* Test save/load */
    srm_kb_save(kb, "/tmp/test_srm.kb");
    srm_kb_clear(kb);
    srm_kb_load(kb, "/tmp/test_srm.kb");
    srm_literal q2;
    srm_parse_line("?mortale(socrate)", &dummy_f, &dummy_r, &q2, err, sizeof(err));
    srm_query_result qr2 = srm_answer_query(kb, &q2);
    if (qr2.success) pass++; else { fprintf(stderr, "FAIL: load query\n"); fail++; }
    srm_literal_free(&q2);
    srm_query_result_free(&qr2);

    /* Test prove */
    srm_literal goal;
    srm_parse_line("?mortale(socrate)", &dummy_f, &dummy_r, &goal, err, sizeof(err));
    srm_proof p = srm_prove(kb, &goal, 10);
    if (p.success) pass++; else { fprintf(stderr, "FAIL: prove\n"); fail++; }
    srm_literal_free(&goal);
    srm_proof_free(&p);

    /* Test clear */
    srm_kb_clear(kb);
    srm_literal q3;
    srm_parse_line("?umano(socrate)", &dummy_f, &dummy_r, &q3, err, sizeof(err));
    srm_query_result qr3 = srm_answer_query(kb, &q3);
    if (!qr3.success) pass++; else { fprintf(stderr, "FAIL: clear\n"); fail++; }
    srm_literal_free(&q3);
    srm_query_result_free(&qr3);

    srm_kb_destroy(kb);
    printf("Library test: %d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
