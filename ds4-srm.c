/* ds4-srm.c — Symbolic Reasoning Module
 *
 * Standalone binary + embeddable library (include ds4-srm.h for the API).
 * All internal functions are static; public API is exposed at the bottom.
 *
 * Copyright (C) 2025 DwarfStar authors. See LICENSE for terms.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <errno.h>

#include "ds4-srm.h"


/* ---------------------------------------------------------------------------
 * Memory helpers (ds4 style)
 * -------------------------------------------------------------------------- */

#define xmalloc(sz)    xmalloc_(sz, __FILE__, __LINE__)
#define xrealloc(p,sz) xrealloc_(p, sz, __FILE__, __LINE__)
#define xstrdup(s)     xstrdup_(s, __FILE__, __LINE__)

static void *xmalloc_(size_t sz, const char *file, int line) {
    void *p = malloc(sz);
    if (!p) { fprintf(stderr, "OOM at %s:%d\n", file, line); exit(1); }
    memset(p, 0, sz);
    return p;
}

static void *xrealloc_(void *p, size_t sz, const char *file, int line) {
    void *n = realloc(p, sz);
    if (!n) { fprintf(stderr, "OOM at %s:%d\n", file, line); exit(1); }
    return n;
}

static char *xstrdup_(const char *s, const char *file, int line) {
    size_t n = strlen(s) + 1;
    char *d = xmalloc_(n, file, line);
    memcpy(d, s, n);
    return d;
}

/* ---------------------------------------------------------------------------
 * Dynamic string buffer (ds4 style)
 * -------------------------------------------------------------------------- */


void srm_buf_init(srm_buf *b) {
    memset(b, 0, sizeof(*b));
}

void srm_buf_free(srm_buf *b) {
    free(b->buf);
    b->buf = NULL;
    b->len = b->cap = 0;
}

static void srm_buf_ensure(srm_buf *b, size_t n) {
    if (b->len + n > b->cap) {
        b->cap = b->cap ? b->cap * 2 : 256;
        while (b->cap < b->len + n) b->cap *= 2;
        b->buf = xrealloc(b->buf, b->cap);
    }
}

void srm_buf_puts(srm_buf *b, const char *s) {
    size_t n = strlen(s);
    srm_buf_ensure(b, n + 1);
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
}

void srm_buf_putf(srm_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    size_t n = 128;
    for (;;) {
        srm_buf_ensure(b, n);
        va_list ap2;
        va_copy(ap2, ap);
        int r = vsnprintf(b->buf + b->len, n, fmt, ap2);
        va_end(ap2);
        if (r >= 0 && (size_t)r < n) { b->len += r; break; }
        n = (size_t)(r >= 0 ? r + 1 : n * 2);
    }
    va_end(ap);
}

char *srm_buf_take(srm_buf *b) {
    srm_buf_ensure(b, 1);
    b->buf[b->len] = '\0';
    char *r = b->buf;
    b->buf = NULL; b->len = b->cap = 0;
    return r;
}

/* ---------------------------------------------------------------------------
 * Term / literal / rule types
 * -------------------------------------------------------------------------- */



static srm_term srm_term_atom(const char *name) {
    srm_term t; t.type = SRM_ATOM; t.name = xstrdup(name); return t;
}

static srm_term srm_term_var(const char *name) {
    srm_term t; t.type = SRM_VAR; t.name = xstrdup(name); return t;
}

static srm_term srm_term_int(int v) {
    srm_term t; t.type = SRM_INT; t.ival = v; return t;
}

static srm_term srm_term_string(const char *s) {
    srm_term t; t.type = SRM_STRING; t.sval = xstrdup(s); return t;
}

static void srm_term_free(srm_term *t) {
    if (t->type == SRM_ATOM || t->type == SRM_VAR) free(t->name);
    if (t->type == SRM_STRING) free(t->sval);
}

static srm_term srm_term_copy(const srm_term *t) {
    srm_term r = {0};
    if (t->type == SRM_ATOM || t->type == SRM_VAR) r.name = xstrdup(t->name);
    else if (t->type == SRM_INT) r.ival = t->ival;
    else if (t->type == SRM_STRING) r.sval = xstrdup(t->sval);
    r.type = t->type;
    return r;
}

/* literal: predicate(args...) */

static srm_literal srm_literal_make(const char *pred, srm_term *args, int arity) {
    srm_literal l;
    l.predicate = xstrdup(pred);
    l.args = xmalloc(sizeof(srm_term) * arity);
    for (int i = 0; i < arity; i++) l.args[i] = srm_term_copy(&args[i]);
    l.arity = arity;
    return l;
}

void srm_literal_free(srm_literal *l) {
    free(l->predicate);
    for (int i = 0; i < l->arity; i++) srm_term_free(&l->args[i]);
    free(l->args);
}

srm_literal srm_literal_copy(const srm_literal *l) {
    return srm_literal_make(l->predicate, l->args, l->arity);
}

/* rule: head ← body1 ∧ body2 ∧ ... */

void srm_rule_free(srm_rule *r) {
    srm_literal_free(&r->head);
    for (int i = 0; i < r->body_len; i++) srm_literal_free(&r->body[i]);
    free(r->body);
}

/* ---------------------------------------------------------------------------
 * Knowledge Base
 * -------------------------------------------------------------------------- */


void srm_kb_init(srm_kb *kb) {
    memset(kb, 0, sizeof(*kb));
}

void srm_kb_free(srm_kb *kb) {
    for (int i = 0; i < kb->nfacts; i++) srm_literal_free(&kb->facts[i]);
    free(kb->facts);
    for (int i = 0; i < kb->nrules; i++) srm_rule_free(&kb->rules[i]);
    free(kb->rules);
}

void srm_kb_add_fact(srm_kb *kb, const srm_literal *fact) {
    if (kb->nfacts >= kb->fact_cap) {
        kb->fact_cap = kb->fact_cap ? kb->fact_cap * 2 : 64;
        kb->facts = xrealloc(kb->facts, sizeof(srm_literal) * kb->fact_cap);
    }
    kb->facts[kb->nfacts++] = srm_literal_copy(fact);
}

void srm_kb_add_rule(srm_kb *kb, srm_rule *rule) {
    if (kb->nrules >= kb->rule_cap) {
        kb->rule_cap = kb->rule_cap ? kb->rule_cap * 2 : 64;
        kb->rules = xrealloc(kb->rules, sizeof(srm_rule) * kb->rule_cap);
    }
    kb->rules[kb->nrules] = *rule; /* move, no copy */
    kb->nrules++;
    /* zero out source so caller doesn't double-free */
    memset(rule, 0, sizeof(*rule));
}

/* ---------------------------------------------------------------------------
 * Unification
 * -------------------------------------------------------------------------- */

/* A substitution is a mapping variable -> term. We store as parallel arrays. */
#define SRM_MAX_SUBST 256



void srm_subst_init(srm_subst *s) {
    memset(s, 0, sizeof(*s));
}

static void srm_subst_clear(srm_subst *s) {
    for (int i = 0; i < s->n; i++) {
        free(s->b[i].var);
        srm_term_free(&s->b[i].term);
    }
    s->n = 0;
}

void srm_subst_free(srm_subst *s) {
    srm_subst_clear(s);
    free(s->b);
    s->b = NULL; s->cap = 0;
}

static void srm_subst_set(srm_subst *s, const char *var, srm_term term) {
    for (int i = 0; i < s->n; i++) {
        if (!strcmp(s->b[i].var, var)) {
            srm_term_free(&s->b[i].term);
            s->b[i].term = srm_term_copy(&term);
            return;
        }
    }
    if (s->n >= s->cap) {
        s->cap = s->cap ? s->cap * 2 : 16;
        s->b = xrealloc(s->b, sizeof(srm_binding) * s->cap);
    }
    s->b[s->n].var = xstrdup(var);
    s->b[s->n].term = srm_term_copy(&term);
    s->n++;
}

/* Deep copy a substitution */
static srm_subst srm_subst_copy(const srm_subst *s) {
    srm_subst c;
    srm_subst_init(&c);
    for (int i = 0; i < s->n; i++)
        srm_subst_set(&c, s->b[i].var, s->b[i].term);
    return c;
}

/* Look up a variable in the substitution. Returns NULL if not bound. */
static srm_term *srm_subst_lookup(srm_subst *s, const char *var) {
    for (int i = 0; i < s->n; i++)
        if (!strcmp(s->b[i].var, var)) return &s->b[i].term;
    return NULL;
}

/* Compare two substitutions for equality (same variable→term bindings) */
static bool srm_subst_equal(const srm_subst *a, const srm_subst *b) {
    if (a->n != b->n) return false;
    for (int i = 0; i < a->n; i++) {
        srm_term *bt = srm_subst_lookup((srm_subst*)b, a->b[i].var);
        if (!bt) return false;
        if (a->b[i].term.type != bt->type) return false;
        if (a->b[i].term.type == SRM_ATOM && strcmp(a->b[i].term.name, bt->name)) return false;
        if (a->b[i].term.type == SRM_INT && a->b[i].term.ival != bt->ival) return false;
    }
    return true;
}

/* Apply substitution to a term (makes a copy). */
srm_term srm_subst_apply(const srm_subst *s, const srm_term *t) {
    if (t->type == SRM_VAR) {
        srm_term *found = srm_subst_lookup((srm_subst*)s, t->name);
        if (found) return srm_term_copy(found);
    }
    return srm_term_copy(t);
}

/* Apply substitution to a literal (makes a copy). */
srm_literal srm_literal_apply(const srm_subst *s, const srm_literal *l) {
    srm_term *args = xmalloc(sizeof(srm_term) * l->arity);
    for (int i = 0; i < l->arity; i++) args[i] = srm_subst_apply(s, &l->args[i]);
    srm_literal result = srm_literal_make(l->predicate, args, l->arity);
    for (int i = 0; i < l->arity; i++) srm_term_free(&args[i]);
    free(args);
    return result;
}

/* Check if two terms unify under a given substitution.
 * Returns true and extends subst with new bindings. */
static bool srm_unify(const srm_term *t1, const srm_term *t2, srm_subst *subst) {
    srm_term t1e = srm_subst_apply(subst, t1);
    srm_term t2e = srm_subst_apply(subst, t2);

    bool result = false;
    /* If either is a variable after applying subst, bind it */
    if (t1e.type == SRM_VAR) {
        /* occurs check: skip for now (simple) */
        srm_subst_set(subst, t1e.name, t2e);
        result = true;
    } else if (t2e.type == SRM_VAR) {
        srm_subst_set(subst, t2e.name, t1e);
        result = true;
    } else if (t1e.type == t2e.type) {
        if (t1e.type == SRM_ATOM && !strcmp(t1e.name, t2e.name)) result = true;
        else if (t1e.type == SRM_INT && t1e.ival == t2e.ival) result = true;
        else if (t1e.type == SRM_STRING && !strcmp(t1e.sval, t2e.sval)) result = true;
        else result = false;
    } else {
        result = false;
    }

    srm_term_free(&t1e);
    srm_term_free(&t2e);
    return result;
}

/* Check if two literals unify. Extends subst. */
static bool srm_unify_literals(const srm_literal *l1, const srm_literal *l2,
                               srm_subst *subst)
{
    if (strcmp(l1->predicate, l2->predicate)) return false;
    if (l1->arity != l2->arity) return false;
    srm_subst save;
    srm_subst_init(&save);
    /* Deep copy of current subst */
    for (int i = 0; i < subst->n; i++)
        srm_subst_set(&save, subst->b[i].var, subst->b[i].term);
    save.n = subst->n;

    for (int i = 0; i < l1->arity; i++) {
        if (!srm_unify(&l1->args[i], &l2->args[i], subst)) {
            /* Rollback: free subst completely and restore from save */
            srm_subst_free(subst);
            subst->b = NULL;
            subst->cap = 0;
            subst->n = 0;
            for (int j = 0; j < save.n; j++)
                srm_subst_set(subst, save.b[j].var, save.b[j].term);
            subst->n = save.n;
            srm_subst_free(&save);
            return false;
        }
    }
    srm_subst_free(&save);
    return true;
}

/* ---------------------------------------------------------------------------
 * Forward chaining
 * -------------------------------------------------------------------------- */

/* Try to match a body literal against KB facts, extending subst.
 * Used by forward chaining backtracking. */
static bool srm_match_one_fact(srm_kb *kb, const srm_literal *lit,
                               srm_subst *subst, int *out_fact_idx)
{
    for (int fi = 0; fi < kb->nfacts; fi++) {
        srm_subst s;
        srm_subst_init(&s);
        for (int i = 0; i < subst->n; i++)
            srm_subst_set(&s, subst->b[i].var, subst->b[i].term);

        if (srm_unify_literals(lit, &kb->facts[fi], &s)) {
            for (int i = 0; i < s.n; i++)
                srm_subst_set(subst, s.b[i].var, s.b[i].term);
            srm_subst_free(&s);
            *out_fact_idx = fi;
            return true;
        }
        srm_subst_free(&s);
    }
    return false;
}

/* Backtracking matcher: given a list of body literals and a KB, find all
 * substitutions that satisfy ALL body literals simultaneously. For each
 * valid substitution, apply to head and add as fact.
 * Returns number of new facts added. */
static int srm_forward_match_rule(srm_kb *kb, srm_rule *r, int body_idx,
                                  srm_subst *subst)
{
    if (body_idx >= r->body_len) {
        /* All body literals matched. Apply substitution to head. */
        srm_literal derived = srm_literal_apply(subst, &r->head);
        /* Check if already in KB */
        for (int fi = 0; fi < kb->nfacts; fi++) {
            srm_subst s;
            srm_subst_init(&s);
            if (srm_unify_literals(&derived, &kb->facts[fi], &s)) {
                srm_subst_free(&s);
                srm_literal_free(&derived);
                return 0; /* already known */
            }
            srm_subst_free(&s);
        }
        srm_kb_add_fact(kb, &derived);
        srm_literal_free(&derived);
        return 1;
    }

    int added = 0;
    srm_subst saved;
    srm_subst_init(&saved);

    for (int fi_idx = 0; fi_idx < kb->nfacts; fi_idx++) {
        /* Save current subst */
        srm_subst_clear(&saved);
        for (int i = 0; i < subst->n; i++)
            srm_subst_set(&saved, subst->b[i].var, subst->b[i].term);
        saved.n = subst->n;

        /* Try to match body[body_idx] with facts[fi_idx] */
        srm_subst s;
        srm_subst_init(&s);
        for (int i = 0; i < subst->n; i++)
            srm_subst_set(&s, subst->b[i].var, subst->b[i].term);

        if (srm_unify_literals(&r->body[body_idx], &kb->facts[fi_idx], &s)) {
            /* Merge s into subst */
            for (int i = 0; i < s.n; i++)
                srm_subst_set(subst, s.b[i].var, s.b[i].term);
            srm_subst_free(&s);

            added += srm_forward_match_rule(kb, r, body_idx + 1, subst);

            /* Restore subst */
            srm_subst_clear(subst);
            for (int i = 0; i < saved.n; i++)
                srm_subst_set(subst, saved.b[i].var, saved.b[i].term);
            subst->n = saved.n;
        } else {
            srm_subst_free(&s);
        }
    }

    srm_subst_free(&saved);
    return added;
}

/* Add all conclusions derivable from rules using current facts.
 * Returns number of new facts added. */
int srm_forward_chain(srm_kb *kb) {
    int added = 0;
    int changed;
    do {
        changed = 0;
        for (int ri = 0; ri < kb->nrules; ri++) {
            srm_rule *r = &kb->rules[ri];
            if (r->body_len == 0) {
                /* Fact rule: add head if not already present */
                bool found = false;
                for (int fi = 0; fi < kb->nfacts; fi++) {
                    srm_subst s;
                    srm_subst_init(&s);
                    if (srm_unify_literals(&r->head, &kb->facts[fi], &s)) {
                        found = true;
                        srm_subst_free(&s);
                        break;
                    }
                    srm_subst_free(&s);
                }
                if (!found) {
                    srm_kb_add_fact(kb, &r->head);
                    changed++;
                }
                continue;
            }

            /* For rules with body, use backtracking matcher */
            srm_subst subst;
            srm_subst_init(&subst);
            changed += srm_forward_match_rule(kb, r, 0, &subst);
            srm_subst_free(&subst);
        }
        added += changed;
    } while (changed > 0 && added < 10000);
    return added;
}

/* ---------------------------------------------------------------------------
 * Backward chaining (Prolog-style)
 * -------------------------------------------------------------------------- */

/* Result of a backward proof attempt */

void srm_proof_init(srm_proof *p) {
    p->success = false;
    srm_subst_init(&p->subst);
    srm_buf_init(&p->trace);
}

void srm_proof_free(srm_proof *p) {
    srm_subst_free(&p->subst);
    srm_buf_free(&p->trace);
}

/* Try to prove a single literal goal using backward chaining.
 * depth is current recursion depth; max_depth limits it. */
static bool srm_backward_prove(srm_kb *kb, const srm_literal *goal,
                               srm_subst *subst, int depth, int max_depth,
                               srm_buf *trace)
{
    if (depth > max_depth) return false;

    /* 1. Try matching against known facts */
    for (int fi = 0; fi < kb->nfacts; fi++) {
        srm_subst s;
        srm_subst_init(&s);
        /* Copy current subst into s */
        for (int i = 0; i < subst->n; i++)
            srm_subst_set(&s, subst->b[i].var, subst->b[i].term);

        if (srm_unify_literals(goal, &kb->facts[fi], &s)) {
            /* Success: goal is a known fact under this substitution */
            srm_buf_putf(trace, "  [fact] %s(", goal->predicate);
            for (int j = 0; j < goal->arity; j++) {
                srm_term a = srm_subst_apply(&s, &goal->args[j]);
                if (a.type == SRM_ATOM) srm_buf_puts(trace, a.name);
                else if (a.type == SRM_INT) srm_buf_putf(trace, "%d", a.ival);
                else srm_buf_puts(trace, "?");
                srm_term_free(&a);
                if (j + 1 < goal->arity) srm_buf_puts(trace, ", ");
            }
            srm_buf_puts(trace, ")\n");
            /* Merge s into subst */
            for (int i = 0; i < s.n; i++)
                srm_subst_set(subst, s.b[i].var, s.b[i].term);
            srm_subst_free(&s);
            return true;
        }
        srm_subst_free(&s);
    }

    /* 2. Try matching against rule heads (backward) */
    for (int ri = 0; ri < kb->nrules; ri++) {
        srm_rule *r = &kb->rules[ri];
        srm_subst s;
        srm_subst_init(&s);
        for (int i = 0; i < subst->n; i++)
            srm_subst_set(&s, subst->b[i].var, subst->b[i].term);

        if (srm_unify_literals(goal, &r->head, &s)) {
            /* Rule head matches. Now try to prove body. */
            bool ok = true;
            for (int bi = 0; bi < r->body_len && ok; bi++) {
                srm_literal body_goal = srm_literal_apply(&s, &r->body[bi]);
                ok = srm_backward_prove(kb, &body_goal, &s, depth + 1, max_depth, trace);
                srm_literal_free(&body_goal);
            }
            if (ok) {
                srm_buf_putf(trace, "  [rule] %s(", r->head.predicate);
                for (int j = 0; j < r->head.arity; j++) {
                    srm_term a = srm_subst_apply(&s, &r->head.args[j]);
                    if (a.type == SRM_ATOM) srm_buf_puts(trace, a.name);
                    else if (a.type == SRM_INT) srm_buf_putf(trace, "%d", a.ival);
                    else srm_buf_puts(trace, "?");
                    srm_term_free(&a);
                    if (j + 1 < r->head.arity) srm_buf_puts(trace, ", ");
                }
                srm_buf_puts(trace, ") ← ");
                for (int bi = 0; bi < r->body_len; bi++) {
                    srm_literal b = srm_literal_apply(&s, &r->body[bi]);
                    srm_buf_puts(trace, b.predicate);
                    srm_buf_puts(trace, "(");
                    for (int j = 0; j < b.arity; j++) {
                        srm_term a = srm_subst_apply(&s, &b.args[j]);
                        if (a.type == SRM_ATOM) srm_buf_puts(trace, a.name);
                        else if (a.type == SRM_INT) srm_buf_putf(trace, "%d", a.ival);
                        else srm_buf_puts(trace, "?");
                        srm_term_free(&a);
                        if (j + 1 < b.arity) srm_buf_puts(trace, ", ");
                    }
                    srm_buf_puts(trace, ")");
                    srm_literal_free(&b);
                    if (bi + 1 < r->body_len) srm_buf_puts(trace, " ∧ ");
                }
                srm_buf_puts(trace, "\n");
                /* Merge s into subst */
                for (int i = 0; i < s.n; i++)
                    srm_subst_set(subst, s.b[i].var, s.b[i].term);
                srm_subst_free(&s);
                return true;
            }
        }
        srm_subst_free(&s);
    }

    return false;
}

/* Prove a goal and return a proof structure */
srm_proof srm_prove(srm_kb *kb, const srm_literal *goal, int max_depth) {
    srm_proof p;
    srm_proof_init(&p);
    srm_subst subst;
    srm_subst_init(&subst);

    srm_buf_putf(&p.trace, "Goal: %s(", goal->predicate);
    for (int j = 0; j < goal->arity; j++) {
        srm_term a = srm_subst_apply(&subst, &goal->args[j]);
        if (a.type == SRM_ATOM) srm_buf_puts(&p.trace, a.name);
        else if (a.type == SRM_INT) srm_buf_putf(&p.trace, "%d", a.ival);
        else srm_buf_puts(&p.trace, "?");
        srm_term_free(&a);
        if (j + 1 < goal->arity) srm_buf_puts(&p.trace, ", ");
    }
    srm_buf_puts(&p.trace, ")\n");

    p.success = srm_backward_prove(kb, goal, &subst, 0, max_depth, &p.trace);
    p.subst = subst;
    return p;
}

/* ---------------------------------------------------------------------------
 * Simple parser (Datalog-like)
 * -------------------------------------------------------------------------- */

/* We parse a line like:
 *   umano(socrate).               → fact
 *   mortale(X) ← umano(X).        → rule
 *   ? mortale(socrate).           → query
 *   constrain(A, int, 1, 9).      → constraint (for CSP)
 *   A + B = 10.                   → arithmetic constraint
 *   A > B.                        → comparison
 */

/* Tokenizer for a single line */
typedef struct {
    const char *s;
    int pos;
} srm_parser;

static void srm_parser_init(srm_parser *p, const char *s) {
    p->s = s;
    p->pos = 0;
}

static void srm_parser_skip_spaces(srm_parser *p) {
    while (p->s[p->pos] && isspace(p->s[p->pos])) p->pos++;
}

static char srm_parser_peek(srm_parser *p) {
    srm_parser_skip_spaces(p);
    return p->s[p->pos];
}

static bool srm_parser_eof(srm_parser *p) {
    srm_parser_skip_spaces(p);
    return p->s[p->pos] == '\0' || p->s[p->pos] == '#';
}

/* Read a token: atom (lowercase start or quoted), variable (uppercase start),
 * integer digits, or symbol like ←, ∧, (, ), =, >, <, +, -, ,, ., ? */
typedef struct {
    char *text;
    int len;
} srm_token;

static void srm_token_free(srm_token *t) {
    free(t->text);
}

/* Read next token */
static srm_token srm_parser_next(srm_parser *p) {
    srm_token tok = {0};
    srm_parser_skip_spaces(p);
    if (p->s[p->pos] == '\0') return tok;

    char c = p->s[p->pos];

    /* Single-character symbols */
    if (c == '(' || c == ')' || c == ',' || c == '.' ||
        c == '?' || c == '=' || c == '>' || c == '<' ||
        c == '+' || c == '-' || c == '/' || c == '[' || c == ']')
    {
        tok.text = xmalloc(2);
        tok.text[0] = c; tok.text[1] = '\0';
        tok.len = 1;
        p->pos++;
        return tok;
    }

    /* ← (0x2190), ∧ (0x2227), ¬ (0x00AC) or other multi-byte UTF-8 symbols */
    if ((unsigned char)c >= 0x80) {
        /* Skip all continuation bytes (>= 0x80) as a single symbol */
        int start = p->pos;
        while (p->s[p->pos] && ((unsigned char)p->s[p->pos] >= 0x80))
            p->pos++;
        /* If we consumed something, return it as a symbol */
        if (p->pos > start) {
            tok.text = xmalloc(p->pos - start + 1);
            memcpy(tok.text, p->s + start, p->pos - start);
            tok.text[p->pos - start] = '\0';
            tok.len = p->pos - start;
            return tok;
        }
    }

    /* Integer */
    if (c >= '0' && c <= '9') {
        int start = p->pos;
        while (p->s[p->pos] && (p->s[p->pos] >= '0' && p->s[p->pos] <= '9')) p->pos++;
        tok.text = xmalloc(p->pos - start + 1);
        memcpy(tok.text, p->s + start, p->pos - start);
        tok.text[p->pos - start] = '\0';
        tok.len = p->pos - start;
        return tok;
    }

    /* Quoted string "..." */
    if (c == '"') {
        p->pos++; /* skip opening quote */
        int start = p->pos;
        while (p->s[p->pos] && p->s[p->pos] != '"') p->pos++;
        tok.text = xmalloc(p->pos - start + 1);
        memcpy(tok.text, p->s + start, p->pos - start);
        tok.text[p->pos - start] = '\0';
        tok.len = p->pos - start;
        if (p->s[p->pos] == '"') p->pos++; /* skip closing quote */
        return tok;
    }

    /* Variable (uppercase start) or atom (lowercase start or other) */
    int start = p->pos;
    while (p->s[p->pos] && (isalpha(p->s[p->pos]) || p->s[p->pos] == '_' ||
           (p->s[p->pos] >= '0' && p->s[p->pos] <= '9')))
        p->pos++;
    tok.text = xmalloc(p->pos - start + 1);
    memcpy(tok.text, p->s + start, p->pos - start);
    tok.text[p->pos - start] = '\0';
    tok.len = p->pos - start;
    return tok;
}

/* Check if a string is a variable (starts with uppercase) */
static bool srm_is_var(const char *s) {
    if (!s || !s[0]) return false;
    return isupper((unsigned char)s[0]) || s[0] == '?';
}

/* Parse a literal: predicate(args...) */
static srm_literal srm_parse_literal(srm_parser *p, char *err, size_t err_len) {
    (void)err_len;
    srm_literal lit = {0};

    /* Read predicate name */
    srm_token tok = srm_parser_next(p);
    if (!tok.text) { err[0] = '\0'; return lit; }
    lit.predicate = tok.text; /* take ownership */

    /* Read '(' */
    srm_token skip = srm_parser_next(p); /* skip ( */
    free(skip.text);
    /* Read args until ')' */
    srm_term args[16];
    int arity = 0;
    while (srm_parser_peek(p) != ')' && !srm_parser_eof(p)) {
        srm_token atok = srm_parser_next(p);
        if (!atok.text) break;
        if (srm_is_var(atok.text)) {
            args[arity++] = srm_term_var(atok.text);
            free(atok.text);
        } else {
            /* Check if integer */
            char *end = NULL;
            long ival = strtol(atok.text, &end, 10);
            if (end && *end == '\0' && ival >= 0) {
                args[arity++] = srm_term_int((int)ival);
                free(atok.text);
            } else {
                args[arity++] = srm_term_atom(atok.text);
                free(atok.text);
            }
        }
        /* skip comma */
        if (srm_parser_peek(p) == ',') {
            srm_token sc = srm_parser_next(p);
            free(sc.text);
        }
    }
    /* skip ')' */
    srm_token close = srm_parser_next(p);
    free(close.text);

    lit.args = xmalloc(sizeof(srm_term) * arity);
    for (int i = 0; i < arity; i++) lit.args[i] = srm_term_copy(&args[i]);
    lit.arity = arity;
    for (int i = 0; i < arity; i++) srm_term_free(&args[i]);
    return lit;
}

/* Parse a line. Returns type: 0=fact, 1=rule, 2=query, -1=error */
/* Fills fact/rule/query pointers accordingly. */
int srm_parse_line(const char *line, srm_literal *out_fact,
                          srm_rule *out_rule, srm_literal *out_query,
                          char *err, size_t err_len)
{
    srm_parser p;
    srm_parser_init(&p, line);

    /* Check for query: starts with '?' */
    if (srm_parser_peek(&p) == '?') {
        srm_token qtok = srm_parser_next(&p); /* skip ? */
        free(qtok.text);
        srm_literal q = srm_parse_literal(&p, err, err_len);
        if (!q.predicate) { err[0] = '\0'; return -1; }
        *out_query = q;
        return 2;
    }

    /* Check for constraint: starts with "constrain" or is arithmetic */
    /* For now, parse as literal and detect constraints later */
    srm_literal head = srm_parse_literal(&p, err, err_len);
    if (!head.predicate) { err[0] = '\0'; return -1; }

    /* Check for rule: ← after head */
    srm_parser_skip_spaces(&p);
    int save = p.pos;
    srm_token next = srm_parser_next(&p);
    bool is_rule = false;
    if (next.text) {
        /* Check for ← (UTF-8), <- (ASCII), :- (Prolog), or ← (literal) */
        if (next.text[0] == 0xE2) {
            /* UTF-8 multi-byte symbol (← or ∧) */
            is_rule = true;
        }
        else if (next.text[0] == '<' && next.text[1] == '-') {
            /* "<-", two chars in same token (unlikely but handle) */
            is_rule = true;
        }
        else if (next.text[0] == '<' && next.text[1] == '\0') {
            /* "<" as separate token; consume following "-" if present */
            if (p.s[p.pos] == '-') { is_rule = true; p.pos++; }
        }
        else if (next.text[0] == ':' && p.s[p.pos] == '-') {
            /* ":-" Prolog style */
            is_rule = true;
            p.pos++;
        }
        free(next.text);
    }
    if (!is_rule) {
        /* Just a fact */
        *out_fact = head;
        return 0;
    }

    /* Parse body literals separated by ∧ (or comma as conjunction) */
    srm_literal body[16];
    int body_len = 0;
    while (!srm_parser_eof(&p) && srm_parser_peek(&p) != '.') {
        body[body_len] = srm_parse_literal(&p, err, err_len);
        if (!body[body_len].predicate) break;
        body_len++;
        /* skip ∧ or comma */
        srm_parser_skip_spaces(&p);
        if ((unsigned char)srm_parser_peek(&p) >= 0x80 || srm_parser_peek(&p) == ',') {
            srm_token sep = srm_parser_next(&p);
            free(sep.text);
        }
    }

    out_rule->head = head;
    out_rule->body = xmalloc(sizeof(srm_literal) * body_len);
    for (int i = 0; i < body_len; i++) {
        out_rule->body[i] = srm_literal_copy(&body[i]);
        srm_literal_free(&body[i]);
    }
    out_rule->body_len = body_len;
    return 1;
}

/* ---------------------------------------------------------------------------
 * KB query
 * -------------------------------------------------------------------------- */

/* Query result */

void srm_query_result_init(srm_query_result *qr) {
    qr->success = false;
    srm_buf_init(&qr->result_text);
}

void srm_query_result_free(srm_query_result *qr) {
    srm_buf_free(&qr->result_text);
}

/* Answer a query literal against the KB. Returns true if query is
 * satisfiable, and prints bindings in result_text. */
srm_query_result srm_answer_query(srm_kb *kb, const srm_literal *query) {
    srm_query_result qr;
    srm_query_result_init(&qr);

    /* Check if query has variables */
    bool has_vars = false;
    for (int i = 0; i < query->arity; i++) {
        if (query->args[i].type == SRM_VAR) { has_vars = true; break; }
    }

    if (!has_vars) {
        /* Simple true/false query */
        for (int fi = 0; fi < kb->nfacts; fi++) {
            srm_subst s;
            srm_subst_init(&s);
            if (srm_unify_literals(query, &kb->facts[fi], &s)) {
                qr.success = true;
                srm_buf_puts(&qr.result_text, "VERO");
                srm_subst_free(&s);
                goto done;
            }
            srm_subst_free(&s);
        }
        /* Also try backward chaining */
        srm_proof pv = srm_prove(kb, query, 10);
        if (pv.success) {
            qr.success = true;
            srm_buf_puts(&qr.result_text, "VERO (dimostrato)");
            srm_buf_puts(&qr.result_text, "\nTraccia:\n");
            srm_buf_puts(&qr.result_text, pv.trace.buf);
            srm_proof_free(&pv);
            goto done;
        }
        srm_proof_free(&pv);
        srm_buf_puts(&qr.result_text, "FALSO");
        goto done;
    }

    /* Query with variables: find all solutions (deduplicated) */
    #define SRM_MAX_SOLS 256
    srm_subst sols[SRM_MAX_SOLS];
    int nsols = 0;
    for (int fi = 0; fi < kb->nfacts; fi++) {
        srm_subst s;
        srm_subst_init(&s);
        if (srm_unify_literals(query, &kb->facts[fi], &s)) {
            /* Check for duplicate */
            bool dup = false;
            for (int si = 0; si < nsols; si++)
                if (srm_subst_equal(&sols[si], &s)) { dup = true; break; }
            if (!dup && nsols < SRM_MAX_SOLS)
                sols[nsols++] = srm_subst_copy(&s);
        }
        srm_subst_free(&s);
    }

    /* Also try backward chaining for variable queries */
    srm_proof pv = srm_prove(kb, query, 10);
    if (pv.success) {
        /* Check for duplicate */
        bool dup = false;
        for (int si = 0; si < nsols; si++)
            if (srm_subst_equal(&sols[si], &pv.subst)) { dup = true; break; }
        if (!dup && nsols < SRM_MAX_SOLS)
            sols[nsols++] = srm_subst_copy(&pv.subst);
    }
    srm_proof_free(&pv);

    if (nsols > 0) {
        qr.success = true;
        srm_buf_puts(&qr.result_text, "Soluzioni:\n");
        for (int si = 0; si < nsols; si++) {
            for (int vi = 0; vi < sols[si].n; vi++) {
                srm_buf_putf(&qr.result_text, "  %s = ", sols[si].b[vi].var);
                srm_term *t = &sols[si].b[vi].term;
                if (t->type == SRM_ATOM) srm_buf_puts(&qr.result_text, t->name);
                else if (t->type == SRM_INT) srm_buf_putf(&qr.result_text, "%d", t->ival);
                else srm_buf_puts(&qr.result_text, "?");
            }
            srm_buf_puts(&qr.result_text, "\n");
        }
    } else {
        srm_buf_puts(&qr.result_text, "nessuna soluzione");
    }

    for (int si = 0; si < nsols; si++) srm_subst_free(&sols[si]);

done:
    return qr;
}

/* ---------------------------------------------------------------------------
 * Simple Constraint Solver (CSP)
 * -------------------------------------------------------------------------- */

/* For now: very simple backtracking over finite integer domains.
 * Constraints are parsed as literals. We handle:
 *   constrain(VAR, int, LO, HI).
 *   A + B = C, A > B, A < B, A = B
 */

#define SRM_MAX_CSP_VARS 16

typedef struct {
    char name[64];
    int lo, hi;
    int value;
    bool assigned;
} srm_csp_var;

typedef struct {
    srm_csp_var vars[SRM_MAX_CSP_VARS];
    int nvars;
} srm_csp;

static void srm_csp_init(srm_csp *csp) {
    memset(csp, 0, sizeof(*csp));
}

/* Add a variable */
static int srm_csp_add_var(srm_csp *csp, const char *name, int lo, int hi) {
    if (csp->nvars >= SRM_MAX_CSP_VARS) return -1;
    srm_csp_var *v = &csp->vars[csp->nvars];
    strncpy(v->name, name, sizeof(v->name)-1);
    v->lo = lo; v->hi = hi;
    v->assigned = false;
    v->value = lo;
    return csp->nvars++;
}

/* Find variable by name */
static srm_csp_var *srm_csp_find(srm_csp *csp, const char *name) {
    for (int i = 0; i < csp->nvars; i++)
        if (!strcmp(csp->vars[i].name, name)) return &csp->vars[i];
    return NULL;
}

/* Evaluate a binary constraint */
static bool srm_csp_check(srm_csp *csp, const char *op,
                          const char *a, const char *b, int rhs)
{
    srm_csp_var *va = srm_csp_find(csp, a);
    srm_csp_var *vb = srm_csp_find(csp, b);
    if (!va || !vb) return false;
    if (!va->assigned || !vb->assigned) return true; /* not yet assigned */
    int av = va->value, bv = vb->value;
    if (!strcmp(op, "=")) return av + bv == rhs;
    if (!strcmp(op, ">")) return av > bv;
    if (!strcmp(op, "<")) return av < bv;
    if (!strcmp(op, ">=")) return av >= bv;
    if (!strcmp(op, "<=")) return av <= bv;
    return false;
}

/* Solve CSP by backtracking */
static bool srm_csp_solve(srm_csp *csp, int var_idx,
                          const char **ops, const char **lefts,
                          const char **rights, int *rhss, int nops,
                          srm_buf *solutions, int max_solutions)
{
    if (var_idx >= csp->nvars) {
        /* All variables assigned; check constraints */
        for (int i = 0; i < nops; i++) {
            if (!srm_csp_check(csp, ops[i], lefts[i], rights[i], rhss[i]))
                return false;
        }
        /* Valid solution */
        if (solutions->len > 0) srm_buf_puts(solutions, " | ");
        for (int i = 0; i < csp->nvars; i++) {
            if (i > 0) srm_buf_puts(solutions, ", ");
            srm_buf_putf(solutions, "%s=%d", csp->vars[i].name, csp->vars[i].value);
        }
        return true;
    }

    srm_csp_var *v = &csp->vars[var_idx];
    int count = 0;
    for (int val = v->lo; val <= v->hi; val++) {
        v->value = val;
        v->assigned = true;
        if (srm_csp_solve(csp, var_idx + 1, ops, lefts, rights, rhss, nops,
                          solutions, max_solutions))
            count++;
        if (max_solutions > 0 && count >= max_solutions) break;
    }
    v->assigned = false;
    return count > 0;
}

/* Parse a constraint line and add to CSP */
static void srm_csp_add_constraint(srm_csp *csp, const char *line,
                                   const char **ops, const char **lefts,
                                   const char **rights, int *rhss, int *nops,
                                   int max_ops)
{
    srm_parser p;
    srm_parser_init(&p, line);

    /* Check for constrain(VAR, int, LO, HI) */
    srm_token tok = srm_parser_next(&p);
    if (!tok.text) return;
    if (!strcmp(tok.text, "constrain")) {
        free(tok.text);
        srm_parser_next(&p); /* skip ( */
        srm_token var = srm_parser_next(&p);
        srm_parser_next(&p); /* skip int */
        srm_token lo = srm_parser_next(&p);
        srm_parser_next(&p); /* skip , */
        srm_token hi = srm_parser_next(&p);
        if (var.text && lo.text && hi.text) {
            int lov = atoi(lo.text);
            int hiv = atoi(hi.text);
            srm_csp_add_var(csp, var.text, lov, hiv);
        }
        free(var.text); free(lo.text); free(hi.text);
        return;
    }
    free(tok.text);

    /* Parse arithmetic: A + B = N  or  A > B etc */
    /* We re-parse from start */
    srm_parser_init(&p, line);
    srm_token a = srm_parser_next(&p);
    srm_token op = srm_parser_next(&p);
    if (!strcmp(op.text, "+")) {
        srm_token b = srm_parser_next(&p);
        srm_token eq = srm_parser_next(&p);
        srm_token rhs = srm_parser_next(&p);
        if (a.text && b.text && rhs.text && eq.text && !strcmp(eq.text, "=")) {
            if (*nops < max_ops) {
                ops[*nops] = xstrdup("=");
                lefts[*nops] = xstrdup(a.text);
                rights[*nops] = xstrdup(b.text);
                rhss[*nops] = atoi(rhs.text);
                (*nops)++;
            }
        }
        free(a.text); free(b.text); free(rhs.text);
    } else if (!strcmp(op.text, ">") || !strcmp(op.text, "<") ||
               !strcmp(op.text, ">=") || !strcmp(op.text, "<=") ||
               !strcmp(op.text, "="))
    {
        srm_token b = srm_parser_next(&p);
        if (a.text && b.text) {
            if (*nops < max_ops) {
                ops[*nops] = xstrdup(op.text);
                lefts[*nops] = xstrdup(a.text);
                rights[*nops] = xstrdup(b.text);
                rhss[*nops] = 0;
                (*nops)++;
            }
        }
        free(a.text); free(b.text);
    } else {
        /* unknown constraint, skip */
    }
    free(op.text);
}

/* ---------------------------------------------------------------------------
 * Persistence (save/load)
 * -------------------------------------------------------------------------- */

void srm_kb_save(srm_kb *kb, const char *path) {
    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno)); return; }

    /* Write facts */
    for (int i = 0; i < kb->nfacts; i++) {
        srm_literal *f = &kb->facts[i];
        fprintf(fp, "%s(", f->predicate);
        for (int j = 0; j < f->arity; j++) {
            srm_term *a = &f->args[j];
            if (a->type == SRM_ATOM) fprintf(fp, "%s", a->name);
            else if (a->type == SRM_INT) fprintf(fp, "%d", a->ival);
            else if (a->type == SRM_VAR) fprintf(fp, "%s", a->name);
            else fprintf(fp, "\"%s\"", a->sval);
            if (j + 1 < f->arity) fprintf(fp, ", ");
        }
        fprintf(fp, ").\n");
    }

    /* Write rules */
    for (int i = 0; i < kb->nrules; i++) {
        srm_rule *r = &kb->rules[i];
        fprintf(fp, "%s(", r->head.predicate);
        for (int j = 0; j < r->head.arity; j++) {
            srm_term *a = &r->head.args[j];
            if (a->type == SRM_ATOM) fprintf(fp, "%s", a->name);
            else if (a->type == SRM_INT) fprintf(fp, "%d", a->ival);
            else if (a->type == SRM_VAR) fprintf(fp, "%s", a->name);
            else fprintf(fp, "\"%s\"", a->sval);
            if (j + 1 < r->head.arity) fprintf(fp, ", ");
        }
        fprintf(fp, ")");
        if (r->body_len > 0) {
            fprintf(fp, " ← ");
            for (int j = 0; j < r->body_len; j++) {
                srm_literal *b = &r->body[j];
                fprintf(fp, "%s(", b->predicate);
                for (int k = 0; k < b->arity; k++) {
                    srm_term *a = &b->args[k];
                    if (a->type == SRM_ATOM) fprintf(fp, "%s", a->name);
                    else if (a->type == SRM_INT) fprintf(fp, "%d", a->ival);
                    else if (a->type == SRM_VAR) fprintf(fp, "%s", a->name);
                    else fprintf(fp, "\"%s\"", a->sval);
                    if (k + 1 < b->arity) fprintf(fp, ", ");
                }
                fprintf(fp, ")");
                if (j + 1 < r->body_len) fprintf(fp, " ∧ ");
            }
        }
        fprintf(fp, ".\n");
    }
    fclose(fp);
}

void srm_kb_load(srm_kb *kb, const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno)); return; }

    char line[4096];
    while (fgets(line, sizeof(line), fp)) {
        /* Strip trailing newline */
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[--len] = '\0';

        /* Skip empty lines and comments */
        char *trim = line;
        while (*trim && isspace(*trim)) trim++;
        if (*trim == '\0' || *trim == '#') continue;

        srm_literal fact;
        srm_rule rule;
        srm_literal query;
        char err[256];
        int type = srm_parse_line(trim, &fact, &rule, &query, err, sizeof(err));
        if (type == 0) {
            srm_kb_add_fact(kb, &fact);
            srm_literal_free(&fact);
        } else if (type == 1) {
            srm_kb_add_rule(kb, &rule);
        }
        /* queries and errors are ignored on load */
    }
    fclose(fp);
}

/* ---------------------------------------------------------------------------
 * Main CLI entry point
 * -------------------------------------------------------------------------- */

#ifdef SRM_LIBRARY_MODE
/* ---------------------------------------------------------------------------
 * Public API wrappers (library mode only)
 * -------------------------------------------------------------------------- */

srm_kb *srm_kb_create(void) {
    srm_kb *kb = (srm_kb *)malloc(sizeof(srm_kb));
    if (!kb) return NULL;
    srm_kb_init(kb);
    return kb;
}

void srm_kb_destroy(srm_kb *kb) {
    if (!kb) return;
    srm_kb_free(kb);
    free(kb);
}

void srm_kb_clear(srm_kb *kb) {
    if (!kb) return;
    srm_kb_free(kb);
    srm_kb_init(kb);
}

bool srm_assert_fact(srm_kb *kb, const char *text) {
    if (!kb || !text) return false;
    if (strncmp(text, "fact: ", 6) == 0) text += 6;
    srm_literal fact, query;
    srm_rule rule;
    char err[256];
    int type = srm_parse_line(text, &fact, &rule, &query, err, sizeof(err));
    if (type == 0) {
        srm_kb_add_fact(kb, &fact);
        srm_forward_chain(kb);
        srm_literal_free(&fact);
        return true;
    }
    return false;
}

bool srm_assert_rule(srm_kb *kb, const char *text) {
    if (!kb || !text) return false;
    if (strncmp(text, "rule: ", 6) == 0) text += 6;
    srm_literal fact, query;
    srm_rule rule;
    char err[256];
    int type = srm_parse_line(text, &fact, &rule, &query, err, sizeof(err));
    if (type == 1) {
        srm_kb_add_rule(kb, &rule);
        srm_forward_chain(kb);
        srm_rule_free(&rule);
        return true;
    }
    return false;
}

#endif /* SRM_LIBRARY_MODE */

static void print_help(void) {
    printf("ds4-srm — Symbolic Reasoning Module (standalone tool)\n");
    printf("Usage: ds4-srm [options]\n");
    printf("Options:\n");
    printf("  --assert <text>     Add fact or rule (e.g. \"fact: umano(socrate)\")\n");
    printf("  --query <text>      Query the KB (e.g. \"mortale(socrate)?\")\n");
    printf("  --prove <text>      Prove a goal with backward chaining\n");
    printf("  --solve <text>      Solve constraints (e.g. \"A+B=10, A>B\")\n");
    printf("  --eval <text>       Evaluate logical expression\n");
    printf("  --save <path>       Save KB to file\n");
    printf("  --load <path>       Load KB from file\n");
    printf("  --clear             Reset KB\n");
    printf("  --json              Use JSON I/O mode (stdin/stdout)\n");
    printf("  --help              Show this help\n");
}

/* JSON output helpers */
static void json_puts(const char *s, srm_buf *out) {
    srm_buf_puts(out, "\"");
    for (const char *p = s; *p; p++) {
        if (*p == '"') srm_buf_puts(out, "\\\"");
        else if (*p == '\\') srm_buf_puts(out, "\\\\");
        else if (*p == '\n') srm_buf_puts(out, "\\n");
        else if (*p == '\t') srm_buf_puts(out, "\\t");
        else srm_buf_putf(out, "%c", *p);
    }
    srm_buf_puts(out, "\"");
}

static void json_result(srm_buf *out, bool ok, const char *result_text) {
    srm_buf_puts(out, "{\"status\":\"");
    srm_buf_puts(out, ok ? "ok" : "error");
    srm_buf_puts(out, "\",\"result\":");
    json_puts(result_text, out);
    srm_buf_puts(out, "}\n");
}

#if !defined(SRM_TEST_MODE) && !defined(SRM_LIBRARY_MODE)
int main(int argc, char **argv) {
    srm_kb kb;
    srm_kb_init(&kb);
    bool json_mode = false;
    const char *load_path = NULL;
    bool do_clear = false;

    /* Collect actions to perform */
    typedef struct {
        enum { ACT_ASSERT, ACT_QUERY, ACT_PROVE, ACT_SOLVE, ACT_EVAL, ACT_SAVE, ACT_LOAD, ACT_CLEAR } type;
        char *text;
    } action;
    action actions[32];
    int nactions = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) { print_help(); return 0; }
        if (!strcmp(argv[i], "--json")) { json_mode = true; continue; }
        if (!strcmp(argv[i], "--clear")) {
            action a = { .type = ACT_CLEAR };
            actions[nactions++] = a;
            continue;
        }
        if (!strcmp(argv[i], "--save") && i+1 < argc) {
            action a = { .type = ACT_SAVE, .text = xstrdup(argv[++i]) };
            actions[nactions++] = a;
            continue;
        }
        if (!strcmp(argv[i], "--load") && i+1 < argc) {
            action a = { .type = ACT_LOAD, .text = xstrdup(argv[++i]) };
            actions[nactions++] = a;
            continue;
        }
        if (!strcmp(argv[i], "--assert") && i+1 < argc) {
            action a = { .type = ACT_ASSERT, .text = xstrdup(argv[++i]) };
            actions[nactions++] = a;
            continue;
        }
        if (!strcmp(argv[i], "--query") && i+1 < argc) {
            action a = { .type = ACT_QUERY, .text = xstrdup(argv[++i]) };
            actions[nactions++] = a;
            continue;
        }
        if (!strcmp(argv[i], "--prove") && i+1 < argc) {
            action a = { .type = ACT_PROVE, .text = xstrdup(argv[++i]) };
            actions[nactions++] = a;
            continue;
        }
        if (!strcmp(argv[i], "--solve") && i+1 < argc) {
            action a = { .type = ACT_SOLVE, .text = xstrdup(argv[++i]) };
            actions[nactions++] = a;
            continue;
        }
        if (!strcmp(argv[i], "--eval") && i+1 < argc) {
            action a = { .type = ACT_EVAL, .text = xstrdup(argv[++i]) };
            actions[nactions++] = a;
            continue;
        }
        fprintf(stderr, "Unknown option: %s\n", argv[i]);
        return 1;
    }

    /* Process actions */
    srm_buf output;
    srm_buf_init(&output);

    for (int ai = 0; ai < nactions; ai++) {
        action *a = &actions[ai];

        if (a->type == ACT_LOAD) {
            srm_kb_free(&kb);
            srm_kb_init(&kb);
            srm_kb_load(&kb, a->text);
            if (!json_mode)
                printf("Loaded KB from %s (%d facts, %d rules)\n",
                       a->text, kb.nfacts, kb.nrules);
            continue;
        }

        if (a->type == ACT_CLEAR) {
            srm_kb_free(&kb);
            srm_kb_init(&kb);
            if (!json_mode) printf("KB cleared.\n");
            continue;
        }

        if (a->type == ACT_SAVE) {
            srm_kb_save(&kb, a->text);
            if (!json_mode)
                printf("Saved KB to %s (%d facts, %d rules)\n",
                       a->text, kb.nfacts, kb.nrules);
            continue;
        }

        if (a->type == ACT_ASSERT) {
            char err[256];
            const char *text = a->text;

            /* Parse "fact: <content>" or "rule: <content>" */
            if (strncmp(text, "fact: ", 6) == 0) {
                text += 6;
                srm_literal fact;
                srm_rule rule;
                srm_literal query;
                int type = srm_parse_line(text, &fact, &rule, &query, err, sizeof(err));
                if (type == 0) {
                    srm_kb_add_fact(&kb, &fact);
                    srm_forward_chain(&kb);
                    if (!json_mode) {
                        printf("Added fact: %s(", fact.predicate);
                        for (int j = 0; j < fact.arity; j++) {
                            srm_term *a = &fact.args[j];
                            if (a->type == SRM_ATOM) printf("%s", a->name);
                            else if (a->type == SRM_INT) printf("%d", a->ival);
                            else printf("?");
                            if (j+1 < fact.arity) printf(", ");
                        }
                        printf(")\n");
                    }
                    srm_literal_free(&fact);
                } else {
                    if (!json_mode) printf("Parse error: %s\n", err);
                }
            } else if (strncmp(text, "rule: ", 6) == 0) {
                text += 6;
                srm_literal fact;
                srm_rule rule;
                srm_literal query;
                int type = srm_parse_line(text, &fact, &rule, &query, err, sizeof(err));
                if (type == 1) {
                    if (!json_mode) {
                        printf("Added rule: %s(", rule.head.predicate);
                        for (int j = 0; j < rule.head.arity; j++) {
                            srm_term *a = &rule.head.args[j];
                            if (a->type == SRM_ATOM) printf("%s", a->name);
                            else if (a->type == SRM_VAR) printf("%s", a->name);
                            else if (a->type == SRM_INT) printf("%d", a->ival);
                            else printf("?");
                            if (j+1 < rule.head.arity) printf(", ");
                        }
                        printf(") ← ...\n");
                    }
                    srm_kb_add_rule(&kb, &rule);
                    srm_forward_chain(&kb);
                    srm_rule_free(&rule);
                } else {
                    if (!json_mode) printf("Parse error: %s\n", err);
                }
            } else {
                if (!json_mode) printf("Use 'fact: ' or 'rule: ' prefix\n");
            }
            continue;
        }

        if (a->type == ACT_QUERY) {
            const char *text = a->text;
            srm_literal fact, query;
            srm_rule rule;
            char err[256];
            int type = srm_parse_line(text, &fact, &rule, &query, err, sizeof(err));
            if (type == 2 && query.predicate) {
                srm_query_result qr = srm_answer_query(&kb, &query);
                if (json_mode) {
                    json_result(&output, true, qr.result_text.buf);
                } else {
                    printf("Query: %s(", query.predicate);
                    for (int j = 0; j < query.arity; j++) {
                        srm_term *a = &query.args[j];
                        if (a->type == SRM_ATOM) printf("%s", a->name);
                        else if (a->type == SRM_VAR) printf("%s", a->name);
                        else if (a->type == SRM_INT) printf("%d", a->ival);
                        else printf("?");
                        if (j+1 < query.arity) printf(", ");
                    }
                    printf(")\n%s\n", qr.result_text.buf);
                }
                srm_literal_free(&query);
                srm_query_result_free(&qr);
            } else {
                if (json_mode)
                    json_result(&output, false, "parse error");
                else
                    printf("Query parse error\n");
            }
            continue;
        }

        if (a->type == ACT_PROVE) {
            const char *text = a->text;
            srm_literal fact, query;
            srm_rule rule;
            char err[256];
            int type = srm_parse_line(text, &fact, &rule, &query, err, sizeof(err));
            if (type == 2 && query.predicate) {
                srm_proof pv = srm_prove(&kb, &query, 20);
                if (json_mode) {
                    srm_buf jout;
                    srm_buf_init(&jout);
                    void srm_buf_putf(&jout, "%s\n%s", pv.success ? "PROVATO" : "NON PROVATO",
                                 pv.trace.buf);
                    json_result(&output, true, jout.buf);
                    srm_buf_free(&jout);
                } else {
                    printf("Prove: %s(", query.predicate);
                    for (int j = 0; j < query.arity; j++) {
                        srm_term *a = &query.args[j];
                        if (a->type == SRM_ATOM) printf("%s", a->name);
                        else if (a->type == SRM_VAR) printf("%s", a->name);
                        else if (a->type == SRM_INT) printf("%d", a->ival);
                        else printf("?");
                        if (j+1 < query.arity) printf(", ");
                    }
                    printf(")\n%s\n%s\n", pv.success ? "PROVATO" : "NON PROVATO",
                           pv.trace.buf);
                }
                srm_literal_free(&query);
                srm_proof_free(&pv);
            } else {
                if (json_mode) json_result(&output, false, "parse error");
                else printf("Prove parse error\n");
            }
            continue;
        }

        if (a->type == ACT_SOLVE) {
            const char *text = a->text;
            srm_csp csp;
            srm_csp_init(&csp);
            const char *ops[32], *lefts[32], *rights[32];
            int rhss[32], nops = 0;

            /* Split text by commas or newlines and process each part */
            char buf[4096];
            strncpy(buf, text, sizeof(buf)-1);
            buf[sizeof(buf)-1] = '\0';
            char *save = NULL;
            char *part = strtok_r(buf, ",", &save);
            while (part) {
                while (*part && isspace(*part)) part++;
                if (*part) srm_csp_add_constraint(&csp, part, ops, lefts, rights, rhss, &nops, 32);
                part = strtok_r(NULL, ",", &save);
            }

            srm_buf solutions;
            srm_buf_init(&solutions);
            bool found = srm_csp_solve(&csp, 0, ops, lefts, rights, rhss, nops, &solutions, 10);

            if (json_mode) {
                json_result(&output, found, solutions.buf ? solutions.buf : "nessuna soluzione");
            } else {
                if (found) printf("Soluzioni: %s\n", solutions.buf ? solutions.buf : "");
                else printf("Nessuna soluzione\n");
            }
            srm_buf_free(&solutions);
            for (int i = 0; i < nops; i++) {
                free((void*)ops[i]); free((void*)lefts[i]); free((void*)rights[i]);
            }
            continue;
        }

        if (a->type == ACT_EVAL) {
            /* Simple evaluation: check if the expression is valid */
            const char *text = a->text;
            /* For now: just parse as query and check if it's provable */
            srm_literal fact, query;
            srm_rule rule;
            char err[256];
            int type = srm_parse_line(text, &fact, &rule, &query, err, sizeof(err));
            if (type == 2 && query.predicate) {
                srm_proof pv = srm_prove(&kb, &query, 20);
                if (json_mode) {
                    json_result(&output, true, pv.success ? "valido" : "non valido");
                } else {
                    printf("%s\n", pv.success ? "valido" : "non valido");
                }
                srm_proof_free(&pv);
            } else {
                if (json_mode) json_result(&output, false, "parse error");
                else printf("Eval parse error\n");
            }
            continue;
        }
    }

    /* Output JSON if requested */
    if (json_mode && output.len > 0) {
        printf("%s", output.buf);
    }

    srm_buf_free(&output);
    for (int i = 0; i < nactions; i++) {
        if (actions[i].text) free(actions[i].text);
    }
    srm_kb_free(&kb);
    return 0;
}
#endif /* !SRM_TEST_MODE && !SRM_LIBRARY_MODE */
