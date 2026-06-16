#include "lobster_tool.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef LOBSTER_SCLI_PATH
#define LOBSTER_SCLI_PATH "scli"
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/* Write the program to a temp file, run scli on it, capture stdout+stderr. */
char *lobster_tool_exec(const char *program, char *err, size_t err_len) {
    if (!program || !program[0]) {
        snprintf(err, err_len, "empty program");
        return NULL;
    }

    /* Temp file for the DataLog source. */
    char tmp_path[] = "/tmp/lobster_XXXXXX";
    int fd = mkstemp(tmp_path);
    if (fd < 0) {
        snprintf(err, err_len, "mkstemp: %s", strerror(errno));
        return NULL;
    }

    size_t plen = strlen(program);
    ssize_t wr = write(fd, program, plen);
    if (wr < 0 || (size_t)wr != plen) {
        snprintf(err, err_len, "write temp file: %s", strerror(errno));
        close(fd);
        unlink(tmp_path);
        return NULL;
    }
    close(fd);

    /* Rename to .scl extension so scli recognises the file type. */
    char scl_path[PATH_MAX];
    snprintf(scl_path, sizeof(scl_path), "%s.scl", tmp_path);
    if (rename(tmp_path, scl_path) != 0) {
        snprintf(err, err_len, "rename: %s", strerror(errno));
        unlink(tmp_path);
        return NULL;
    }

    /* Run scli. */
    int pipe_stdout[2], pipe_stderr[2];
    if (pipe(pipe_stdout) < 0 || pipe(pipe_stderr) < 0) {
        snprintf(err, err_len, "pipe: %s", strerror(errno));
        unlink(scl_path);
        return NULL;
    }

    pid_t pid = fork();
    if (pid < 0) {
        snprintf(err, err_len, "fork: %s", strerror(errno));
        close(pipe_stdout[0]); close(pipe_stdout[1]);
        close(pipe_stderr[0]); close(pipe_stderr[1]);
        unlink(scl_path);
        return NULL;
    }

    if (pid == 0) {
        /* Child: redirect stdout/stderr to pipes. */
        close(pipe_stdout[0]);
        close(pipe_stderr[0]);
        dup2(pipe_stdout[1], STDOUT_FILENO);
        dup2(pipe_stderr[1], STDERR_FILENO);
        close(pipe_stdout[1]);
        close(pipe_stderr[1]);

        execlp(LOBSTER_SCLI_PATH, "scli", scl_path, "-p", "unit", (char *)NULL);
        /* If execlp fails, report to stderr before exiting. */
        char msg[256];
        snprintf(msg, sizeof(msg), "exec scli: %s", strerror(errno));
        (void)!write(STDERR_FILENO, msg, strlen(msg));
        _exit(127);
    }

    /* Parent: read child output. */
    close(pipe_stdout[1]);
    close(pipe_stderr[1]);

    /* Read stdout. */
    size_t out_cap = 65536;
    size_t out_len = 0;
    char *out = malloc(out_cap);
    if (!out) {
        snprintf(err, err_len, "malloc");
        close(pipe_stdout[0]);
        close(pipe_stderr[0]);
        unlink(scl_path);
        return NULL;
    }
    out[0] = '\0';

    ssize_t n;
    while ((n = read(pipe_stdout[0], out + out_len, out_cap - out_len - 1)) > 0) {
        out_len += (size_t)n;
        if (out_cap - out_len < 4096) {
            out_cap *= 2;
            char *tmp = realloc(out, out_cap);
            if (!tmp) { free(out); snprintf(err, err_len, "realloc"); goto cleanup; }
            out = tmp;
        }
    }

    /* Read stderr into a separate buffer for diagnostics. */
    size_t err_cap = 8192;
    size_t err_len2 = 0;
    char *err_buf = malloc(err_cap);
    if (!err_buf) { free(out); snprintf(err, err_len, "malloc"); goto cleanup; }
    err_buf[0] = '\0';
    while ((n = read(pipe_stderr[0], err_buf + err_len2, err_cap - err_len2 - 1)) > 0) {
        err_len2 += (size_t)n;
        if (err_cap - err_len2 < 4096) {
            err_cap *= 2;
            char *tmp = realloc(err_buf, err_cap);
            if (!tmp) { free(out); free(err_buf); snprintf(err, err_len, "realloc"); goto cleanup; }
            err_buf = tmp;
        }
    }

    close(pipe_stdout[0]);
    close(pipe_stderr[0]);
    out[out_len] = '\0';
    err_buf[err_len2] = '\0';

    /* Wait for child and check exit status. */
    int wstatus;
    waitpid(pid, &wstatus, 0);
    if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) {
        /* Failure: include stderr in error message. */
        snprintf(err, err_len, "scli exited with status %d%s%s",
                 WEXITSTATUS(wstatus),
                 err_buf[0] ? ": " : "",
                 err_buf[0] ? err_buf : "");
        free(out);
        free(err_buf);
        unlink(scl_path);
        return NULL;
    }

    free(err_buf);
    unlink(scl_path);

    /* Trim trailing newlines. */
    size_t olen = strlen(out);
    while (olen > 0 && (out[olen-1] == '\n' || out[olen-1] == '\r')) out[--olen] = '\0';
    return out;

cleanup:
    close(pipe_stdout[0]);
    close(pipe_stderr[0]);
    unlink(scl_path);
    return NULL;
}
