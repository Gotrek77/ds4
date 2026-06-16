#include "lobster_tool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main() {
    const char *program =
        "// Water jug problem: 3 bottles 12L, 7L, 5L. Initial 12 full.\n"
        "// Goal: any bottle contains 1 liter.\n"
        "\n"
        "type state(i32, i32, i32)\n"
        "type pour(i32, i32, i32, i32, i32)\n"
        "\n"
        "rel cap(0, 12)\n"
        "rel cap(1, 7)\n"
        "rel cap(2, 5)\n"
        "\n"
        "// Initial state: bottle0=12, bottle1=0, bottle2=0\n"
        "rel state(12, 0, 0)\n"
        "\n"
        "// Pour from i to j when src > 0 and dst < cap[j]\n"
        "// Two cases: src <= cap[j]-dst => pour = src, else pour = cap[j]-dst\n"
        "\n"
        "// Case 1: src <= cap[j]-dst  => pour = src\n"
        "rel pour(0, 1, A, B, A) :- state(A,B,C), A>0, B<7, A <= 7-B\n"
        "rel pour(0, 2, A, C, A) :- state(A,B,C), A>0, C<5, A <= 5-C\n"
        "rel pour(1, 0, B, A, B) :- state(A,B,C), B>0, A<12, B <= 12-A\n"
        "rel pour(1, 2, B, C, B) :- state(A,B,C), B>0, C<5, B <= 5-C\n"
        "rel pour(2, 0, C, A, C) :- state(A,B,C), C>0, A<12, C <= 12-A\n"
        "rel pour(2, 1, C, B, C) :- state(A,B,C), C>0, B<7, C <= 7-B\n"
        "\n"
        "// Case 2: src > cap[j]-dst => pour = cap[j]-dst\n"
        "rel pour(0, 1, A, B, 7-B) :- state(A,B,C), A>0, B<7, A > 7-B\n"
        "rel pour(0, 2, A, C, 5-C) :- state(A,B,C), A>0, C<5, A > 5-C\n"
        "rel pour(1, 0, B, A, 12-A) :- state(A,B,C), B>0, A<12, B > 12-A\n"
        "rel pour(1, 2, B, C, 5-C) :- state(A,B,C), B>0, C<5, B > 5-C\n"
        "rel pour(2, 0, C, A, 12-A) :- state(A,B,C), C>0, A<12, C > 12-A\n"
        "rel pour(2, 1, C, B, 7-B) :- state(A,B,C), C>0, B<7, C > 7-B\n"
        "\n"
        "// New state after pour\n"
        "rel state(A-P, B+P, C) :- pour(0,1,A,B,P), state(A,B,C)   // 12->7\n"
        "rel state(A-P, B, C+P) :- pour(0,2,A,C,P), state(A,B,C)   // 12->5\n"
        "rel state(A, B-P, C+P) :- pour(1,2,B,C,P), state(A,B,C)   // 7->5\n"
        "rel state(A+P, B-P, C) :- pour(1,0,B,A,P), state(A,B,C)   // 7->12\n"
        "rel state(A+P, B, C-P) :- pour(2,0,C,A,P), state(A,B,C)   // 5->12\n"
        "rel state(A, B+P, C-P) :- pour(2,1,C,B,P), state(A,B,C)   // 5->7\n"
        "\n"
        "// Goal: any bottle has 1 liter\n"
        "rel goal() :- state(1,_,_)\n"
        "rel goal() :- state(_,1,_)\n"
        "rel goal() :- state(_,_,1)\n"
        "\n"
        "query goal\n";

    char err[256] = {0};
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    char *result = lobster_tool_exec(program, err, sizeof(err));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    if (result) {
        printf("Lobster result: %s\n", result);
        free(result);
    } else {
        printf("Error: %s\n", err);
    }
    printf("Time: %.6f seconds\n", elapsed);
    return 0;
}
