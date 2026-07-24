#include "slice_test.h"

/* Deliberately the same name as slice_a.c's static — after LLVM.link! one of
 * them gets a uniquified name (hidden_counter.1 or similar). The promotion
 * map must record whatever name the linker actually assigned. */
static long hidden_counter = 0;

long st_b_bump(long delta) { hidden_counter += delta; return hidden_counter; }
long st_b_get(void)        { return hidden_counter; }
