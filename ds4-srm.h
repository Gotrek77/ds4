/* ds4-srm.h — Symbolic Reasoning Module public API
 *
 * Provides a lightweight forward/backward chaining engine for Datalog-style
 * logical reasoning. Can be used as a library (in-process) or as a standalone
 * CLI tool.
 *
 * Copyright (C) 2025 DwarfStar authors. See LICENSE for terms.
 */

#ifndef DS4_SRM_H
#define DS4_SRM_H

#ifdef SRM_LIBRARY_MODE
#  define SRM_STATIC
#else
#  define SRM_STATIC static
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Term types
 * -------------------------------------------------------------------------- */

typedef enum {
    SRM_ATOM    = 0,   /* symbolic constant (e.g. tizio, caio) */
    SRM_INT     = 1,   /* integer constant */
    SRM_VAR     = 2,   /* logic variable (e.g. X, Y) */
    SRM_STRING  = 3,   /* quoted string constant */
} srm_term_type;

typedef struct {
    srm_term_type type;
    union {
        char *name;     /* for ATOM, VAR */
        int   ival;     /* for INT */
        char *sval;     /* for STRING */
    };
} srm_term;

/* ---------------------------------------------------------------------------
 * Literal: predicate(args...)
 * -------------------------------------------------------------------------- */

typedef struct {
    char      *predicate;
    srm_term  *args;
    int        arity;
} srm_literal;

/* ---------------------------------------------------------------------------
 * Rule: head ← body1 ∧ body2 ∧ ...
 * -------------------------------------------------------------------------- */

typedef struct {
    srm_literal head;
    srm_literal *body;
    int         body_len;
} srm_rule;

/* ---------------------------------------------------------------------------
 * Substitution
 * -------------------------------------------------------------------------- */

#define SRM_MAX_SUBST 256

typedef struct {
    char *var;
    srm_term term;
} srm_binding;

typedef struct {
    srm_binding *b;
    int n, cap;
} srm_subst;

/* ---------------------------------------------------------------------------
 * Knowledge Base
 * -------------------------------------------------------------------------- */

typedef struct {
    srm_literal *facts;
    int nfacts, fact_cap;
    srm_rule *rules;
    int nrules, rule_cap;
} srm_kb;

/* ---------------------------------------------------------------------------
 * Dynamic string buffer (opaque, use srm_buf_* functions)
 * -------------------------------------------------------------------------- */

typedef struct {
    char *buf;
    size_t len, cap;
} srm_buf;

/* ---------------------------------------------------------------------------
 * Proof result (backward chaining)
 * -------------------------------------------------------------------------- */

typedef struct {
    bool    success;
    srm_subst subst;   /* final substitution */
    srm_buf trace;     /* proof trace text (opaque) */
} srm_proof;

/* ---------------------------------------------------------------------------
 * Query result
 * -------------------------------------------------------------------------- */

typedef struct {
    bool success;
    srm_buf result_text;  /* human-readable result text (opaque) */
} srm_query_result;

/* ---------------------------------------------------------------------------
 * KB lifecycle
 * -------------------------------------------------------------------------- */

/* Allocate and initialize a new KB. Returns NULL on OOM. */
srm_kb *srm_kb_create(void);

/* Free a KB and all its contents. */
void srm_kb_destroy(srm_kb *kb);

/* Remove all facts and rules from a KB without freeing the KB itself. */
void srm_kb_clear(srm_kb *kb);

/* ---------------------------------------------------------------------------
 * Assertion (add facts and rules)
 * -------------------------------------------------------------------------- */

/* Parse a fact line ("fact: ...") and add it to the KB.
 * Returns true on success, false on parse error. */
bool srm_assert_fact(srm_kb *kb, const char *text);

/* Parse a rule line ("rule: ...") and add it to the KB.
 * Returns true on success, false on parse error. */
bool srm_assert_rule(srm_kb *kb, const char *text);

/* Low-level: add a pre-parsed literal as a fact. */
void srm_kb_add_fact(srm_kb *kb, const srm_literal *fact);

/* Low-level: add a pre-parsed rule. */
void srm_kb_add_rule(srm_kb *kb, srm_rule *rule);

/* ---------------------------------------------------------------------------
 * Forward chaining
 * -------------------------------------------------------------------------- */

/* Run forward chaining to derive all conclusions from rules and facts.
 * Returns number of new facts added. */
int srm_forward_chain(srm_kb *kb);

/* ---------------------------------------------------------------------------
 * Query / backward proving
 * -------------------------------------------------------------------------- */

/* Answer a query literal against the KB. Returns all solutions. */
srm_query_result srm_answer_query(srm_kb *kb, const srm_literal *query);

/* Prove a single goal using backward chaining. Returns a proof structure. */
srm_proof srm_prove(srm_kb *kb, const srm_literal *goal, int max_depth);

/* Free a query result (including its result_text buffer). */
void srm_query_result_free(srm_query_result *qr);

/* Free a proof structure (including subst and trace). */
void srm_proof_free(srm_proof *p);

/* ---------------------------------------------------------------------------
 * Parsing helpers
 * -------------------------------------------------------------------------- */

/* Parse a full line. Returns type:
 *   0 = fact  (fills out_fact)
 *   1 = rule  (fills out_rule)
 *   2 = query (fills out_query)
 *  -1 = error (fills err)
 */
int srm_parse_line(const char *line, srm_literal *out_fact,
                   srm_rule *out_rule, srm_literal *out_query,
                   char *err, size_t err_len);

/* Free a literal (predicate, args). */
void srm_literal_free(srm_literal *l);

/* Free a rule (head + body). */
void srm_rule_free(srm_rule *r);

/* ---------------------------------------------------------------------------
 * Persistence (save/load KB to/from file)
 * -------------------------------------------------------------------------- */

/* Save KB to a file in Datalog text format.
 * Path "-" means stdout. */
void srm_kb_save(srm_kb *kb, const char *path);

/* Load KB from a file in Datalog text format.
 * Path "-" means stdin. Appends to existing KB. */
void srm_kb_load(srm_kb *kb, const char *path);

/* ---------------------------------------------------------------------------
 * Buffer utilities (for constructing result strings)
 * -------------------------------------------------------------------------- */

void srm_buf_init(srm_buf *b);
void srm_buf_free(srm_buf *b);
void srm_buf_puts(srm_buf *b, const char *s);
void srm_buf_putf(srm_buf *b, const char *fmt, ...);
char *srm_buf_take(srm_buf *b);  /* null-terminate and return buffer */

#ifdef __cplusplus
}
#endif

#endif /* DS4_SRM_H */
