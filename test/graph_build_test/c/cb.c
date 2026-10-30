/* cb: a C shared library linked against ca. Both calls reach ca's one counter. */
#include "ca.h"

#ifndef CB_K
#error "cb's own define is missing"
#endif

int cb_twice(int x) { return ca_next(x) + ca_next(x) + CB_K; }
