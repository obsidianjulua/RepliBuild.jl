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

/* Two DISTINCT C symbols that sanitise to the SAME Julia name: the wrapper
 * collapses `_+` and rstrips a trailing `_`, so both become `st_collide`.
 * They differ in arity, so both survive the signature dedup and both need a
 * Tier-1 slice — which only works if the slice CONSTANT is keyed on the
 * mangled symbol. Keyed on the Julia name they share one const and whichever
 * loses gets bound to the other's slice module; `Base.llvmcall` resolves the
 * const at codegen, so the loser fails on its FIRST CALL, long after wrap.
 * (Live instance: lua's `luaL_checkversion_` vs the `luaL_checkversion` macro
 * shim.) Different return shapes so a mis-bind shows up as a wrong VALUE, not
 * only as a missing-entry error. */
long st_collide_(long x)         { return x + 1000; }
long st__collide(long x, long y) { return x * 100 + y; }

/* Address-significant internal constant: no `unnamed_addr`, because the code
 * below compares its ADDRESS rather than its contents. Embedding it into a
 * slice would hand the JIT a second copy at a different address, and
 * `st_is_sentinel` would then answer differently depending on which tier ran
 * — the cJSON divergence class rotated from value identity onto address
 * identity, and just as silent. The Slicer must refuse both functions.
 * (OP_TABLE above is the contrast: address never taken, so LLVM marks it
 * `unnamed_addr` and embedding it is sound.) */
static const long ST_SENTINEL = -1;

const long *st_sentinel(void)          { return &ST_SENTINEL; }
long st_is_sentinel(const long *p)     { return p == &ST_SENTINEL; }

/* A pointer return is `is_c_lto_safe`, so the pre-pass accepts and slices this
 * — but the emitter maps `const char *` to Cstring, which `lto_shape_ok`
 * refuses, so the call site stays a ccall. Acceptance is therefore strictly
 * weaker than emission, and a writer keyed on acceptance strands this slice on
 * disk: reachable by nothing, shipped inside the package. That is the shape of
 * all 19 orphans in the Hub lua wrapper. */
const char *st_name(int i) {
    static const char *const NAMES[3] = { "double", "negate", "square" };
    return NAMES[i % 3];
}

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
