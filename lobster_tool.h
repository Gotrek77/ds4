#ifndef LOBSTER_TOOL_H
#define LOBSTER_TOOL_H

#include <stddef.h>
#include <stdbool.h>

/* Execute a Lobster DataLog program.
 *
 * program:  DataLog source code (facts + rules + queries).
 * err:      caller-provided buffer for error messages.
 * err_len:  size of err buffer.
 *
 * Returns a malloc'd string with the query results, or NULL on failure
 * (err is set).  Caller must free the returned string.
 */
char *lobster_tool_exec(const char *program, char *err, size_t err_len);

#endif /* LOBSTER_TOOL_H */
