#include "s.h"

int pair(int a, int b) RB_NOEXCEPT { return a * 10 + b; }
int pair_sum(struct pair p) RB_NOEXCEPT { return p.a + p.b; }
struct pair make_pair(int a, int b) RB_NOEXCEPT { struct pair p = { a, b }; return p; }

int triple(int x) RB_NOEXCEPT { return x * 3; }
int triple_sum(const struct triple *t) RB_NOEXCEPT { return t->x + t->y + t->z; }

int flags(int lo, int hi) RB_NOEXCEPT { return lo + hi; }
struct flags make_flags(void) RB_NOEXCEPT { struct flags f = { 1, 2 }; return f; }

int num(int i) RB_NOEXCEPT { return -i; }
union num make_num(void) RB_NOEXCEPT { union num n; n.i = 4; return n; }

int packed_s(int c, int v) RB_NOEXCEPT { return c + v; }
struct packed_s make_packed(void) RB_NOEXCEPT { struct packed_s p = { 1, 2 }; return p; }
