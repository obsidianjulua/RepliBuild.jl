#include <setjmp.h>
#include <stdarg.h>
#include "slice_test.h"

/* Mutable static → MUST be promoted (single-copy-of-state invariant). */
static long hidden_counter = 0;

/* Static functions, address-taken through the table → survive O2, promoted. */
static long op_double(long x) { return 2 * x; }
static long op_negate(long x) { return -x; }
static long op_square(long x) { return x * x; }

typedef long (*st_op_fn)(long);

/* CONST static → must NOT be promoted (slices may embed constants). */
static st_op_fn const OP_TABLE[3] = { op_double, op_negate, op_square };

/* MUTABLE static fn-ptr slot → promoted. */
static st_op_fn current_op = op_double;

long st_bump(long delta) { hidden_counter += delta; return hidden_counter; }
long st_get_count(void)  { return hidden_counter; }

long st_apply(int op, long x) { return OP_TABLE[op % 3](x); }
void st_set_op(int op)        { current_op = OP_TABLE[op % 3]; }
long st_call_op(long x)       { return current_op(x); }

/* Public entry over a hidden-visibility helper from the other TU. Its slice
 * must `declare` the PROMOTED name — an un-promoted `st_hidden_scale` is not
 * in the dynsym and deadlocks the JIT. */
long st_scaled(long x) { return st_hidden_scale(x); }

/* Reads a hidden-visibility CONST table defined in the other TU. The index is
 * opaque to the optimizer, so the load survives and the slice must bind the
 * table by symbol. */
long st_table_at(int i) { return ST_HIDDEN_TABLE[i & 3]; }

long st_sum(int n, ...) {
    va_list ap;
    long s = 0;
    va_start(ap, n);
    for (int i = 0; i < n; i++) s += va_arg(ap, long);
    va_end(ap);
    return s;
}

/* Mutable static jmp_buf → promoted; exercises setjmp survival. */
static jmp_buf st_env;

long st_guarded_div(long a, long b) {
    if (setjmp(st_env)) return -1;
    if (b == 0) longjmp(st_env, 1);
    return a / b;
}
