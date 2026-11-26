/* Prompt-lookup drafting.
 *
 * Agents and coding sessions spend much of their output reproducing text that
 * is already in the context: code being edited, commands being repeated,
 * quoted documents.  Those continuations cost nothing to draft: find the most
 * recent earlier occurrence of the longest suffix of the history and propose
 * the tokens that followed it.  The target verifies them like any other
 * draft, so the output distribution is unchanged.
 *
 * The history is a span followed by one pending token (the token about to be
 * fed), so callers never copy a long session history to append it.  A match
 * may overlap the suffix it matched; the proposal then continues from its own
 * period, which is what repeats a list marker, an indent or a fence.
 *
 * The scan is linear in the history and touches one int per position; even a
 * million-token session costs well under a decode step. */
#ifndef DS4_LOOKUP_H
#define DS4_LOOKUP_H

/* Shortest suffix match worth proposing at all. */
#define DS4_LOOKUP_MIN_MATCH 3
/* A match this long is trusted on its own, without any drafter. */
#define DS4_LOOKUP_STRONG_MATCH 8
/* Longer matches carry no more information; stop the scan there. */
#define DS4_LOOKUP_MAX_MATCH 16
/* Tokens a weaker match must share with the neural drafter's proposal. */
#define DS4_LOOKUP_AGREE 2

typedef struct {
    int n;     /* proposed tokens written to out */
    int match; /* suffix length that selected them; 0 means no proposal */
} ds4_lookup_result;

static inline int ds4_lookup_at(const int *hist, int len, int next, int i) {
    return i < len ? hist[i] : next;
}

/* Propose up to cap tokens continuing hist[0..len) + next. */
static inline ds4_lookup_result ds4_lookup_propose(const int *hist, int len, int next,
                                                   int *out, int cap) {
    ds4_lookup_result r = {0, 0};
    const int total = len + 1;
    if (!out || cap <= 0 || len < 0 || (len > 0 && !hist) || total < DS4_LOOKUP_MIN_MATCH + 1)
        return r;
    int best = 0, best_end = -1;
    /* end is the index of the last token of a candidate occurrence; the
     * suffix itself ends at total - 1 and is skipped.  Scanning backwards
     * makes the first longest match the most recent one. */
    for (int end = total - 2; end >= 0; end--) {
        if (ds4_lookup_at(hist, len, next, end) != next) continue;
        int m = 1;
        while (m < DS4_LOOKUP_MAX_MATCH && end - m >= 0 &&
               ds4_lookup_at(hist, len, next, end - m) ==
               ds4_lookup_at(hist, len, next, total - 1 - m))
            m++;
        if (m > best) {
            best = m;
            best_end = end;
            if (m == DS4_LOOKUP_MAX_MATCH) break;
        }
    }
    if (best < DS4_LOOKUP_MIN_MATCH) return r;
    for (int k = 0; k < cap; k++) {
        const int i = best_end + 1 + k;
        /* Past the end of the history the continuation is the proposal
         * itself, shifted by the match period (i - total < k). */
        out[k] = i < total ? ds4_lookup_at(hist, len, next, i) : out[i - total];
    }
    r.n = cap;
    r.match = best;
    return r;
}

#endif
