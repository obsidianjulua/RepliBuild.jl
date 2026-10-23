#include "conflict_trap.h"

/* beta's types: another `impl`, `Mode` and `Kind` under the same bare names. */

double ct_vocab_score(beta::impl v)
{
    if (v.n_vocab < 0)
        throw 2;
    return v.scores * v.n_vocab;
}

beta::Kind ct_kind(int x)
{
    if (x < 0)
        throw 3;
    return x ? beta::Kind::K_Q : beta::Kind::K_P;
}

int ct_beta_mode(beta::Mode m)
{
    if (m == beta::MODE_R)
        throw 4;
    return static_cast<int>(m) + 7;
}
