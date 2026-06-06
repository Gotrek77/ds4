/* srm_tool.c — In-process SRM tool for ds4-agent.
 *
 * This single TU includes ds4-srm.c in library mode and provides
 * a C API that the agent's tool dispatcher calls directly, avoiding
 * subprocess overhead.
 *
 * Build: add this file to ds4-agent's compilation and link with -lm.
 *   cc -O2 -I. ds4_agent.c srm_tool.c -o ds4-agent -lm
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

/* ============================================================================
 * SRM library (single-TU inclusion)
 * ============================================================================ */
#define SRM_LIBRARY_MODE
#include "ds4-srm.c"

/* ============================================================================
 * Persistent KB (one per process)
 * ============================================================================ */
static srm_kb srm_kb_global;
static bool  srm_kb_initialized = false;

static void srm_ensure_kb(void) {
    if (!srm_kb_initialized) {
        srm_kb_init(&srm_kb_global);
        srm_kb_initialized = true;
    }
}

/* ============================================================================
 * JSON output helpers (mirror the standalone program's json_result)
 * ============================================================================ */
static void json_puts_escaped(const char *s, srm_buf *out) {
    srm_buf_puts(out, "\"");
    for (const char *p = s; *p; p++) {
        switch (*p) {
            case '"':  srm_buf_puts(out, "\\\""); break;
            case '\\': srm_buf_puts(out, "\\\\"); break;
            case '\n': srm_buf_puts(out, "\\n");  break;
            case '\t': srm_buf_puts(out, "\\t");  break;
            default:
                if ((unsigned char)*p < 0x20) {
                    char hex[8];
                    snprintf(hex, sizeof(hex), "\\u%04x", (unsigned char)*p);
                    srm_buf_puts(out, hex);
                } else {
                    char c[2] = { *p, '\0' };
                    srm_buf_puts(out, c);
                }
                break;
        }
    }
    srm_buf_puts(out, "\"");
}

/* ============================================================================
 * Public API — called by agent's tool dispatcher
 * ============================================================================ */

/* Helper: check if two literals are identical (same predicate, same args). */
static bool srm_literal_equal(const srm_literal *a, const srm_literal *b) {
    if (!a->predicate || !b->predicate) return false;
    if (strcmp(a->predicate, b->predicate) != 0) return false;
    if (a->arity != b->arity) return false;
    for (int i = 0; i < a->arity; i++) {
        if (a->args[i].type != b->args[i].type) return false;
        if (a->args[i].type == SRM_INT && a->args[i].ival != b->args[i].ival) return false;
        if ((a->args[i].type == SRM_ATOM || a->args[i].type == SRM_VAR) &&
            strcmp(a->args[i].name, b->args[i].name) != 0) return false;
        if (a->args[i].type == SRM_STRING &&
            strcmp(a->args[i].sval, b->args[i].sval) != 0) return false;
    }
    return true;
}

/* srm_tool_exec(action, statement) — executes one SRM operation in-process
 * and returns a JSON string (caller must free).
 *
 * Actions: "assert", "query", "solve", "eval", "save", "load", "reset"
 */
char *srm_tool_exec(const char *action, const char *statement) {
    srm_buf out;
    srm_buf_init(&out);

    if (!action || !action[0]) {
        json_result(&out, false, "missing action");
        return srm_buf_take(&out);
    }
    if (!statement) statement = "";

    srm_ensure_kb();

    if (!strcmp(action, "reset")) {
        srm_kb_free(&srm_kb_global);
        srm_kb_init(&srm_kb_global);
        json_result(&out, true, "KB reset");
        return srm_buf_take(&out);
    }

    if (!strcmp(action, "assert")) {
        srm_literal fact = {0};
        srm_rule rule = {0};
        srm_literal query = {0};
        char err[256] = {0};
        int type = srm_parse_line(statement, &fact, &rule, &query, err, sizeof(err));
        if (type == 0 && fact.predicate) {
            srm_kb_add_fact(&srm_kb_global, &fact);
            json_result(&out, true, "fact asserted");
        } else if (type == 1 && rule.head.predicate) {
            srm_kb_add_rule(&srm_kb_global, &rule);
            json_result(&out, true, "rule asserted");
        } else {
            json_result(&out, false, err[0] ? err : "parse error");
        }
        srm_literal_free(&fact);
        srm_rule_free(&rule);
        srm_literal_free(&query);
        return srm_buf_take(&out);
    }

    if (!strcmp(action, "query")) {
        srm_literal fact = {0}, query = {0};
        srm_rule rule = {0};
        char err[256] = {0};
        int type = srm_parse_line(statement, &fact, &rule, &query, err, sizeof(err));
        if (type == 2 && query.predicate) {
            srm_query_result qr = srm_answer_query(&srm_kb_global, &query);
            json_result(&out, qr.success, qr.success ? qr.result_text.buf : "not proven");
            srm_literal_free(&query);
            srm_query_result_free(&qr);
        } else {
            json_result(&out, false, err[0] ? err : "parse error");
        }
        srm_literal_free(&fact);
        srm_rule_free(&rule);
        return srm_buf_take(&out);
    }

    if (!strcmp(action, "solve")) {
        /* Forward chain to derive all conclusions */
        int added = srm_forward_chain(&srm_kb_global);
        char msg[64];
        snprintf(msg, sizeof(msg), "forward chain: %d new facts", added);
        json_result(&out, true, msg);
        return srm_buf_take(&out);
    }

    if (!strcmp(action, "prove")) {
        srm_literal fact = {0}, goal = {0};
        srm_rule rule = {0};
        char err[256] = {0};
        int type = srm_parse_line(statement, &fact, &rule, &goal, err, sizeof(err));
        if (type == 2 && goal.predicate) {
            srm_proof p = srm_prove(&srm_kb_global, &goal, 20);
            srm_buf j;
            srm_buf_init(&j);
            srm_buf_putf(&j, "%s\n%s", p.success ? "PROVATO" : "NON PROVATO", p.trace.buf);
            json_result(&out, true, j.buf);
            srm_buf_free(&j);
            srm_literal_free(&goal);
            srm_proof_free(&p);
        } else {
            json_result(&out, false, err[0] ? err : "parse error");
        }
        srm_literal_free(&fact);
        srm_rule_free(&rule);
        return srm_buf_take(&out);
    }

    if (!strcmp(action, "eval")) {
        /* Evaluate a literal: check if it's a fact or provable */
        srm_literal fact = {0}, lit = {0};
        srm_rule rule = {0};
        char err[256] = {0};
        int type = srm_parse_line(statement, &fact, &rule, &lit, err, sizeof(err));
        if (type == 0 && fact.predicate) {
            /* Check if fact exists in KB */
            bool found = false;
            for (int i = 0; i < srm_kb_global.nfacts; i++) {
                if (srm_literal_equal(&srm_kb_global.facts[i], &fact)) {
                    found = true;
                    break;
                }
            }
            json_result(&out, found, found ? "true" : "false");
            srm_literal_free(&fact);
        } else {
            json_result(&out, false, err[0] ? err : "parse error");
        }
        srm_rule_free(&rule);
        srm_literal_free(&lit);
        return srm_buf_take(&out);
    }

    if (!strcmp(action, "save")) {
        srm_kb_save(&srm_kb_global, statement);
        char msg[128];
        snprintf(msg, sizeof(msg), "saved %d facts, %d rules to %s",
                 srm_kb_global.nfacts, srm_kb_global.nrules, statement);
        json_result(&out, true, msg);
        return srm_buf_take(&out);
    }

    if (!strcmp(action, "load")) {
        srm_kb_load(&srm_kb_global, statement);
        char msg[128];
        snprintf(msg, sizeof(msg), "loaded %d facts, %d rules from %s",
                 srm_kb_global.nfacts, srm_kb_global.nrules, statement);
        json_result(&out, true, msg);
        return srm_buf_take(&out);
    }

    /* Unknown action */
    srm_buf_puts(&out, "{\"status\":\"error\",\"result\":\"unknown action: ");
    json_puts_escaped(action, &out);
    srm_buf_puts(&out, "\"}\n");
    return srm_buf_take(&out);
}

/* ============================================================================
 * Lifecycle hooks — called by ds4-agent at startup / exit
 * ============================================================================ */

static void resolve_home_path(const char *in, char *out, size_t out_max) {
    if (in[0] == '~' && in[1] == '/') {
        const char *home = getenv("HOME");
        if (!home || !home[0]) home = ".";
        snprintf(out, out_max, "%s%s", home, in + 1);
    } else {
        snprintf(out, out_max, "%s", in);
    }
}

/* Load KB from a path on agent startup.  Returns true if a file was loaded. */
bool srm_tool_load_kb(const char *path) {
    if (!path || !path[0]) return false;
    srm_ensure_kb();
    char resolved[4096];
    resolve_home_path(path, resolved, sizeof(resolved));
    FILE *fp = fopen(resolved, "r");
    if (!fp) return false;
    fclose(fp);
    srm_kb_load(&srm_kb_global, resolved);
    return srm_kb_global.nfacts > 0 || srm_kb_global.nrules > 0;
}

/* Save KB to a path on agent exit / compaction.  Returns true on success. */
bool srm_tool_save_kb(const char *path) {
    if (!path || !path[0]) return false;
    srm_ensure_kb();
    char resolved[4096];
    resolve_home_path(path, resolved, sizeof(resolved));
    srm_kb_save(&srm_kb_global, resolved);
    return true;
}
