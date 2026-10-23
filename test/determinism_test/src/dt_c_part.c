/* The C unit: its DWARF spells the shared types the C way. */
#include "dt_shared.h"

int dt_g_zeta = 3;
int dt_g_alpha = 1;
int dt_g_mid = 2;

int dt_run_with(dt_progress_cb cb, void *user) { return cb ? cb(1, user) : 0; }

int dt_run_raw(int (*cb)(int)) { return cb ? cb(2) : 0; }

bool dt_traits_quantized(const dt_traits *t) { return t->is_quantized; }

/* C11 anonymous member: `b->dm` reaches into the union. */
int dt_block_qs(const dt_block *b) { return b->qs + b->dm; }
