#include "slice_test.h"

/* Deliberately the same name as slice_a.c's static — after LLVM.link! one of
 * them gets a uniquified name (hidden_counter.1 or similar). The promotion
 * map must record whatever name the linker actually assigned. */
static long hidden_counter = 0;

long st_b_bump(long delta) { hidden_counter += delta; return hidden_counter; }
long st_b_get(void)        { return hidden_counter; }

/* Hidden-visibility, external-linkage helper called from slice_a.c — the
 * LUAI_FUNC class. Defined in a DIFFERENT TU from its caller on purpose: it
 * cannot be internalized before the link, and noinline keeps the call edge in
 * st_scaled's body after O2. */
ST_INTERNAL long st_hidden_scale(long x) { return 3 * x + 1; }
