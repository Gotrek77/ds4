/* ds4-srm-test.c — Test suite for the Symbolic Reasoning Module
 *
 * Tests are self-contained: no framework, just PASSED/FAILED output.
 * Run with: ./ds4-srm-test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* Include the implementation directly so we can test internal functions */
#define SRM_TEST_MODE
#include "ds4-srm.c"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, cond) do {                                   \
    if (!(cond)) {                                              \
        fprintf(stderr, "FAILED: %s at %s:%d\n", name, __FILE__, __LINE__); \
        tests_failed++;                                         \
    } else {                                                    \
        tests_passed++;                                         \
    }                                                           \
} while(0)

static void test_term_creation(void) {
    srm_term a = srm_term_atom("socrate");
    TEST("atom type", a.type == SRM_ATOM);
    TEST("atom name", !strcmp(a.name, "socrate"));
    srm_term_free(&a);

    srm_term v = srm_term_var("X");
    TEST("var type", v.type == SRM_VAR);
    TEST("var name", !strcmp(v.name, "X"));
    srm_term_free(&v);

    srm_term i = srm_term_int(42);
    TEST("int type", i.type == SRM_INT);
    TEST("int value", i.ival == 42);
    srm_term_free(&i);

    srm_term s = srm_term_string("hello");
    TEST("string type", s.type == SRM_STRING);
    TEST("string value", !strcmp(s.sval, "hello"));
    srm_term_free(&s);
}

static void test_literal_creation(void) {
    srm_term args[2];
    args[0] = srm_term_atom("pluto");
    args[1] = srm_term_atom("paperino");
    srm_literal l = srm_literal_make("genitore", args, 2);
    srm_term_free(&args[0]); srm_term_free(&args[1]);
    TEST("literal predicate", !strcmp(l.predicate, "genitore"));
    TEST("literal arity", l.arity == 2);
    TEST("literal arg0", !strcmp(l.args[0].name, "pluto"));
    TEST("literal arg1", !strcmp(l.args[1].name, "paperino"));
    srm_literal_free(&l);
}

static void test_kb_add_fact(void) {
    srm_kb kb;
    srm_kb_init(&kb);
    srm_term args[1];
    args[0] = srm_term_atom("socrate");
    srm_literal f = srm_literal_make("umano", args, 1);
    srm_term_free(&args[0]);
    srm_kb_add_fact(&kb, &f);
    srm_literal_free(&f);
    TEST("kb nfacts after add", kb.nfacts == 1);
    TEST("kb fact predicate", !strcmp(kb.facts[0].predicate, "umano"));
    TEST("kb fact arg0", !strcmp(kb.facts[0].args[0].name, "socrate"));
    srm_kb_free(&kb);
}

static void test_unification_simple(void) {
    srm_term t1 = srm_term_atom("socrate");
    srm_term t2 = srm_term_atom("socrate");
    srm_subst s;
    srm_subst_init(&s);
    TEST("unify same atoms", srm_unify(&t1, &t2, &s));
    srm_subst_free(&s);

    srm_term t3 = srm_term_atom("pluto");
    srm_subst_init(&s);
    TEST("unify different atoms", !srm_unify(&t1, &t3, &s));
    srm_subst_free(&s);
    srm_term_free(&t1); srm_term_free(&t2); srm_term_free(&t3);
}

static void test_unification_var(void) {
    srm_term var = srm_term_var("X");
    srm_term val = srm_term_atom("socrate");
    srm_subst s;
    srm_subst_init(&s);
    TEST("unify var with atom", srm_unify(&var, &val, &s));
    TEST("subst size after bind", s.n == 1);
    TEST("subst var name", !strcmp(s.b[0].var, "X"));
    TEST("subst term type", s.b[0].term.type == SRM_ATOM);
    TEST("subst term name", !strcmp(s.b[0].term.name, "socrate"));
    srm_subst_free(&s);
    srm_term_free(&var); srm_term_free(&val);
}

static void test_unification_literals(void) {
    srm_term args1[1], args2[1];
    args1[0] = srm_term_atom("socrate");
    args2[0] = srm_term_atom("socrate");
    srm_literal l1 = srm_literal_make("umano", args1, 1);
    srm_literal l2 = srm_literal_make("umano", args2, 1);
    srm_term_free(&args1[0]); srm_term_free(&args2[0]);

    srm_subst s;
    srm_subst_init(&s);
    TEST("unify same literals", srm_unify_literals(&l1, &l2, &s));
    srm_subst_free(&s);

    srm_term args3[1];
    args3[0] = srm_term_atom("pluto");
    srm_literal l3 = srm_literal_make("umano", args3, 1);
    srm_term_free(&args3[0]);
    srm_subst_init(&s);
    TEST("unify diff args", !srm_unify_literals(&l1, &l3, &s));
    srm_subst_free(&s);

    srm_literal_free(&l1); srm_literal_free(&l2); srm_literal_free(&l3);
}

static void test_parse_fact(void) {
    srm_literal fact;
    srm_rule rule;
    srm_literal query;
    char err[256];
    int type = srm_parse_line("umano(socrate).", &fact, &rule, &query, err, sizeof(err));
    TEST("parse fact type", type == 0);
    TEST("parse fact predicate", !strcmp(fact.predicate, "umano"));
    TEST("parse fact arity", fact.arity == 1);
    TEST("parse fact arg0", !strcmp(fact.args[0].name, "socrate"));
    srm_literal_free(&fact);
}

static void test_parse_rule(void) {
    srm_literal fact;
    srm_rule rule;
    srm_literal query;
    char err[256];
    int type = srm_parse_line("mortale(X) <- umano(X).", &fact, &rule, &query, err, sizeof(err));
    TEST("parse rule type", type == 1);
    TEST("parse rule head", !strcmp(rule.head.predicate, "mortale"));
    TEST("parse rule head arity", rule.head.arity == 1);
    TEST("parse rule head arg0 type", rule.head.args[0].type == SRM_VAR);
    TEST("parse rule head arg0 name", !strcmp(rule.head.args[0].name, "X"));
    TEST("parse rule body len", rule.body_len >= 1);
    TEST("parse rule body pred", !strcmp(rule.body[0].predicate, "umano"));
    srm_rule_free(&rule);
}

static void test_parse_query(void) {
    srm_literal fact;
    srm_rule rule;
    srm_literal query;
    char err[256];
    int type = srm_parse_line("? mortale(socrate).", &fact, &rule, &query, err, sizeof(err));
    TEST("parse query type", type == 2);
    TEST("parse query predicate", !strcmp(query.predicate, "mortale"));
    TEST("parse query arity", query.arity == 1);
    srm_literal_free(&query);
}

static void test_forward_chain_simple(void) {
    srm_kb kb;
    srm_kb_init(&kb);

    /* Add fact */
    srm_term args[1];
    args[0] = srm_term_atom("socrate");
    srm_literal f = srm_literal_make("umano", args, 1);
    srm_term_free(&args[0]);
    srm_kb_add_fact(&kb, &f);
    srm_literal_free(&f);

    /* Add rule: mortale(X) ← umano(X) */
    srm_term hargs[1];
    hargs[0] = srm_term_var("X");
    srm_literal head = srm_literal_make("mortale", hargs, 1);
    srm_term_free(&hargs[0]);

    srm_term bargs[1];
    bargs[0] = srm_term_var("X");
    srm_literal body_lit = srm_literal_make("umano", bargs, 1);
    srm_term_free(&bargs[0]);

    srm_rule rule;
    rule.head = head;
    rule.body = xmalloc(sizeof(srm_literal) * 1);
    rule.body[0] = body_lit;
    rule.body_len = 1;
    srm_kb_add_rule(&kb, &rule);

    /* Run forward chaining */
    int added = srm_forward_chain(&kb);
    TEST("forward chain added", added >= 1);
    TEST("forward chain total facts", kb.nfacts >= 2);

    /* Check that mortale(socrate) is now a fact */
    bool found = false;
    for (int i = 0; i < kb.nfacts; i++) {
        if (!strcmp(kb.facts[i].predicate, "mortale") &&
            kb.facts[i].arity == 1 &&
            kb.facts[i].args[0].type == SRM_ATOM &&
            !strcmp(kb.facts[i].args[0].name, "socrate"))
        {
            found = true;
            break;
        }
    }
    TEST("forward chain derived mortale(socrate)", found);

    srm_kb_free(&kb);
}

static void test_backward_prove(void) {
    srm_kb kb;
    srm_kb_init(&kb);

    /* Add fact */
    srm_term args[1];
    args[0] = srm_term_atom("socrate");
    srm_literal f = srm_literal_make("umano", args, 1);
    srm_term_free(&args[0]);
    srm_kb_add_fact(&kb, &f);
    srm_literal_free(&f);

    /* Add rule: mortale(X) ← umano(X) */
    srm_term hargs[1];
    hargs[0] = srm_term_var("X");
    srm_literal head = srm_literal_make("mortale", hargs, 1);
    srm_term_free(&hargs[0]);
    srm_term bargs[1];
    bargs[0] = srm_term_var("X");
    srm_literal body_lit = srm_literal_make("umano", bargs, 1);
    srm_term_free(&bargs[0]);
    srm_rule rule;
    rule.head = head;
    rule.body = xmalloc(sizeof(srm_literal) * 1);
    rule.body[0] = body_lit;
    rule.body_len = 1;
    srm_kb_add_rule(&kb, &rule);

    /* Prove mortale(socrate) */
    srm_term gargs[1];
    gargs[0] = srm_term_atom("socrate");
    srm_literal goal = srm_literal_make("mortale", gargs, 1);
    srm_term_free(&gargs[0]);
    srm_proof pv = srm_prove(&kb, &goal, 10);
    TEST("backward prove mortale(socrate)", pv.success);
    srm_proof_free(&pv);
    srm_literal_free(&goal);
    srm_kb_free(&kb);
}

static void test_backward_prove_fail(void) {
    srm_kb kb;
    srm_kb_init(&kb);

    srm_term args[1];
    args[0] = srm_term_atom("socrate");
    srm_literal f = srm_literal_make("umano", args, 1);
    srm_term_free(&args[0]);
    srm_kb_add_fact(&kb, &f);
    srm_literal_free(&f);

    /* Prove immortale(socrate) — no rule should match */
    srm_term gargs[1];
    gargs[0] = srm_term_atom("socrate");
    srm_literal goal = srm_literal_make("immortale", gargs, 1);
    srm_term_free(&gargs[0]);
    srm_proof pv = srm_prove(&kb, &goal, 10);
    TEST("backward prove nonexistent", !pv.success);
    srm_proof_free(&pv);
    srm_literal_free(&goal);
    srm_kb_free(&kb);
}

static void test_query_true(void) {
    srm_kb kb;
    srm_kb_init(&kb);
    srm_term args[1];
    args[0] = srm_term_atom("socrate");
    srm_literal f = srm_literal_make("umano", args, 1);
    srm_term_free(&args[0]);
    srm_kb_add_fact(&kb, &f);
    srm_literal_free(&f);

    srm_term qargs[1];
    qargs[0] = srm_term_atom("socrate");
    srm_literal q = srm_literal_make("umano", qargs, 1);
    srm_term_free(&qargs[0]);
    srm_query_result qr = srm_answer_query(&kb, &q);
    TEST("query true fact", qr.success);
    srm_query_result_free(&qr);
    srm_literal_free(&q);
    srm_kb_free(&kb);
}

static void test_query_false(void) {
    srm_kb kb;
    srm_kb_init(&kb);
    srm_term args[1];
    args[0] = srm_term_atom("socrate");
    srm_literal f = srm_literal_make("umano", args, 1);
    srm_term_free(&args[0]);
    srm_kb_add_fact(&kb, &f);
    srm_literal_free(&f);

    srm_term qargs[1];
    qargs[0] = srm_term_atom("pluto");
    srm_literal q = srm_literal_make("umano", qargs, 1);
    srm_term_free(&qargs[0]);
    srm_query_result qr = srm_answer_query(&kb, &q);
    TEST("query false fact", !qr.success);
    srm_query_result_free(&qr);
    srm_literal_free(&q);
    srm_kb_free(&kb);
}

static void test_query_variable(void) {
    srm_kb kb;
    srm_kb_init(&kb);
    srm_term args1[1], args2[1];
    args1[0] = srm_term_atom("socrate");
    args2[0] = srm_term_atom("pluto");
    srm_literal f1 = srm_literal_make("umano", args1, 1);
    srm_literal f2 = srm_literal_make("umano", args2, 1);
    srm_term_free(&args1[0]); srm_term_free(&args2[0]);
    srm_kb_add_fact(&kb, &f1);
    srm_kb_add_fact(&kb, &f2);
    srm_literal_free(&f1); srm_literal_free(&f2);

    srm_term qargs[1];
    qargs[0] = srm_term_var("X");
    srm_literal q = srm_literal_make("umano", qargs, 1);
    srm_term_free(&qargs[0]);
    srm_query_result qr = srm_answer_query(&kb, &q);
    TEST("query with variable", qr.success);
    srm_query_result_free(&qr);
    srm_literal_free(&q);
    srm_kb_free(&kb);
}

static void test_csp_simple(void) {
    srm_csp csp;
    srm_csp_init(&csp);
    srm_csp_add_var(&csp, "A", 1, 9);
    srm_csp_add_var(&csp, "B", 1, 9);

    const char *ops[10], *lefts[10], *rights[10];
    int rhss[10], nops = 0;
    ops[nops] = "="; lefts[nops] = "A"; rights[nops] = "B"; rhss[nops] = 10; nops++;
    ops[nops] = ">"; lefts[nops] = "A"; rights[nops] = "B"; rhss[nops] = 0; nops++;

    srm_buf solutions;
    srm_buf_init(&solutions);
    bool found = srm_csp_solve(&csp, 0, ops, lefts, rights, rhss, nops, &solutions, 5);
    TEST("csp found solutions", found);
    TEST("csp solutions non-empty", solutions.len > 0);
    srm_buf_free(&solutions);
}

static void test_persistence(void) {
    srm_kb kb;
    srm_kb_init(&kb);
    srm_term args[1];
    args[0] = srm_term_atom("socrate");
    srm_literal f = srm_literal_make("umano", args, 1);
    srm_term_free(&args[0]);
    srm_kb_add_fact(&kb, &f);
    srm_literal_free(&f);

    srm_kb_save(&kb, "/tmp/ds4-srm-test.kb");

    srm_kb kb2;
    srm_kb_init(&kb2);
    srm_kb_load(&kb2, "/tmp/ds4-srm-test.kb");
    TEST("loaded fact count", kb2.nfacts == 1);
    TEST("loaded fact predicate", !strcmp(kb2.facts[0].predicate, "umano"));
    TEST("loaded fact arg0", !strcmp(kb2.facts[0].args[0].name, "socrate"));
    srm_kb_free(&kb);
    srm_kb_free(&kb2);
    remove("/tmp/ds4-srm-test.kb");
}

static void test_cli_assert_query(void) {
    /* Test the CLI path: use main() with arguments */
    /* We'll test programmatically by calling internal functions */

    srm_kb kb;
    srm_kb_init(&kb);

    /* Assert fact via internal API (simulates --assert "fact: ...") */
    char err[256];
    srm_literal fact;
    srm_rule rule;
    srm_literal query;
    int type = srm_parse_line("umano(socrate).", &fact, &rule, &query, err, sizeof(err));
    TEST("cli assert parse", type == 0);
    srm_kb_add_fact(&kb, &fact);
    srm_literal_free(&fact);

    /* Query */
    srm_parser_init(&((srm_parser){0}), "? umano(socrate).");
    /* Use parse_line directly */
    srm_literal qfact, qquery;
    srm_rule qrule;
    char qerr[256];
    int qtype = srm_parse_line("? umano(socrate).", &qfact, &qrule, &qquery, qerr, sizeof(qerr));
    TEST("cli query parse", qtype == 2);

    srm_query_result qr = srm_answer_query(&kb, &qquery);
    TEST("cli query result", qr.success);
    srm_query_result_free(&qr);
    srm_literal_free(&qquery);
    srm_kb_free(&kb);
}

static void test_forward_multi_hop(void) {
    /* Test multi-hop reasoning:
     *   genitore(pluto, paperino).
     *   genitore(paperino, paperina).
     *   nonno(X,Y) ← genitore(X,Z) ∧ genitore(Z,Y).
     * Query: ? nonno(pluto, paperina).
     */
    srm_kb kb;
    srm_kb_init(&kb);

    srm_term a1[2];
    a1[0] = srm_term_atom("pluto");
    a1[1] = srm_term_atom("paperino");
    srm_literal f1 = srm_literal_make("genitore", a1, 2);
    srm_term_free(&a1[0]); srm_term_free(&a1[1]);
    srm_kb_add_fact(&kb, &f1);
    srm_literal_free(&f1);

    srm_term a2[2];
    a2[0] = srm_term_atom("paperino");
    a2[1] = srm_term_atom("paperina");
    srm_literal f2 = srm_literal_make("genitore", a2, 2);
    srm_term_free(&a2[0]); srm_term_free(&a2[1]);
    srm_kb_add_fact(&kb, &f2);
    srm_literal_free(&f2);

    /* Rule: nonno(X,Y) ← genitore(X,Z) ∧ genitore(Z,Y) */
    srm_term hargs[2];
    hargs[0] = srm_term_var("X");
    hargs[1] = srm_term_var("Y");
    srm_literal head = srm_literal_make("nonno", hargs, 2);
    srm_term_free(&hargs[0]); srm_term_free(&hargs[1]);

    srm_term b1args[2];
    b1args[0] = srm_term_var("X");
    b1args[1] = srm_term_var("Z");
    srm_literal b1 = srm_literal_make("genitore", b1args, 2);
    srm_term_free(&b1args[0]); srm_term_free(&b1args[1]);

    srm_term b2args[2];
    b2args[0] = srm_term_var("Z");
    b2args[1] = srm_term_var("Y");
    srm_literal b2 = srm_literal_make("genitore", b2args, 2);
    srm_term_free(&b2args[0]); srm_term_free(&b2args[1]);

    srm_rule rule;
    rule.head = head;
    rule.body = xmalloc(sizeof(srm_literal) * 2);
    rule.body[0] = b1;
    rule.body[1] = b2;
    rule.body_len = 2;
    srm_kb_add_rule(&kb, &rule);

    /* Forward chain */
    int added = srm_forward_chain(&kb);
    TEST("forward multi-hop added", added >= 1);

    /* Query nonno(pluto, paperina) */
    srm_term qargs[2];
    qargs[0] = srm_term_atom("pluto");
    qargs[1] = srm_term_atom("paperina");
    srm_literal q = srm_literal_make("nonno", qargs, 2);
    srm_term_free(&qargs[0]); srm_term_free(&qargs[1]);
    srm_query_result qr = srm_answer_query(&kb, &q);
    TEST("forward multi-hop query", qr.success);
    srm_query_result_free(&qr);
    srm_literal_free(&q);
    srm_kb_free(&kb);
}

static void test_backward_multi_hop(void) {
    /* Same as above but using backward chaining (prove) */
    srm_kb kb;
    srm_kb_init(&kb);

    srm_term a1[2];
    a1[0] = srm_term_atom("pluto");
    a1[1] = srm_term_atom("paperino");
    srm_literal f1 = srm_literal_make("genitore", a1, 2);
    srm_term_free(&a1[0]); srm_term_free(&a1[1]);
    srm_kb_add_fact(&kb, &f1);
    srm_literal_free(&f1);

    srm_term a2[2];
    a2[0] = srm_term_atom("paperino");
    a2[1] = srm_term_atom("paperina");
    srm_literal f2 = srm_literal_make("genitore", a2, 2);
    srm_term_free(&a2[0]); srm_term_free(&a2[1]);
    srm_kb_add_fact(&kb, &f2);
    srm_literal_free(&f2);

    srm_term hargs[2];
    hargs[0] = srm_term_var("X");
    hargs[1] = srm_term_var("Y");
    srm_literal head = srm_literal_make("nonno", hargs, 2);
    srm_term_free(&hargs[0]); srm_term_free(&hargs[1]);
    srm_term b1args[2];
    b1args[0] = srm_term_var("X");
    b1args[1] = srm_term_var("Z");
    srm_literal b1 = srm_literal_make("genitore", b1args, 2);
    srm_term_free(&b1args[0]); srm_term_free(&b1args[1]);
    srm_term b2args[2];
    b2args[0] = srm_term_var("Z");
    b2args[1] = srm_term_var("Y");
    srm_literal b2 = srm_literal_make("genitore", b2args, 2);
    srm_term_free(&b2args[0]); srm_term_free(&b2args[1]);
    srm_rule rule;
    rule.head = head;
    rule.body = xmalloc(sizeof(srm_literal) * 2);
    rule.body[0] = b1;
    rule.body[1] = b2;
    rule.body_len = 2;
    srm_kb_add_rule(&kb, &rule);

    /* Prove nonno(pluto, paperina) */
    srm_term gargs[2];
    gargs[0] = srm_term_atom("pluto");
    gargs[1] = srm_term_atom("paperina");
    srm_literal goal = srm_literal_make("nonno", gargs, 2);
    srm_term_free(&gargs[0]); srm_term_free(&gargs[1]);
    srm_proof pv = srm_prove(&kb, &goal, 10);
    TEST("backward multi-hop prove", pv.success);
    srm_proof_free(&pv);
    srm_literal_free(&goal);
    srm_kb_free(&kb);
}

int main(void) {
    printf("ds4-srm test suite\n");
    printf("===================\n\n");

    test_term_creation();
    test_literal_creation();
    test_kb_add_fact();
    test_unification_simple();
    test_unification_var();
    test_unification_literals();
    test_parse_fact();
    test_parse_rule();
    test_parse_query();
    test_forward_chain_simple();
    test_backward_prove();
    test_backward_prove_fail();
    test_query_true();
    test_query_false();
    test_query_variable();
    test_csp_simple();
    test_persistence();
    test_cli_assert_query();
    test_forward_multi_hop();
    test_backward_multi_hop();

    printf("\n===================\n");
    printf("Tests: %d passed, %d failed\n", tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
