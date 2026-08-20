#ifndef SLICE_TEST_H
#define SLICE_TEST_H

/* Fixture for the static-promotion pass (llvmcall slicing contract, M1).
 * Every category of internal-linkage symbol the pass must classify appears
 * here; test_static_promotion.jl asserts the exact promotion decisions. */

long st_bump(long delta);       /* mutates static hidden_counter (slice_a.c)  */
long st_get_count(void);        /* reads it back                              */
long st_apply(int op, long x);  /* dispatch through static CONST fn-ptr table */
void st_set_op(int op);         /* writes static MUTABLE fn-ptr slot          */
long st_call_op(long x);        /* calls through the mutable slot             */
long st_sum(int n, ...);        /* varargs — must survive promotion/build     */

/* Distinct symbols, one Julia name after sanitisation (`_+` collapse +
 * trailing-`_` rstrip): the slice-const collision class. Different arity, so
 * both survive the dedup and both must get their own `_SLICE_` constant. */
long st_collide_(long x);
long st__collide(long x, long y);

/* Sliceable (pointer return) but NOT emittable (Cstring fails lto_shape_ok):
 * the acceptance-vs-emission gap that leaves orphan slices on disk. */
const char *st_name(int i);

/* Reach an internal constant whose ADDRESS is observable (no `unnamed_addr`).
 * Embedding duplicates it at a new address; the Slicer must refuse instead. */
const long *st_sentinel(void);
long st_is_sentinel(const long *p);
long st_guarded_div(long a, long b); /* setjmp/longjmp over a static jmp_buf  */

long st_b_bump(long delta);     /* slice_b.c: same-named static counter —     */
long st_b_get(void);            /* exercises the post-link rename             */

/* Lua's LUAI_FUNC shape: EXTERNAL linkage (the linker resolves it across TUs)
 * but HIDDEN visibility (it never reaches the .so's dynamic symbol table).
 * `default<O2>` has no internalize pass, so such a function survives as a
 * `define hidden` — invisible to both the internal-linkage promotion rule and
 * to dlsym. A slice that calls one then `declare`s a name ORC cannot resolve
 * and DEADLOCKS the JIT on first call. Promotion must therefore key on
 * "not exportable from the dynsym", not on internal linkage alone.
 * noinline keeps the call edge alive so a slice actually has to bind it. */
#define ST_INTERNAL __attribute__((visibility("hidden"), noinline))

ST_INTERNAL long st_hidden_scale(long x);  /* defined in slice_b.c           */
long st_scaled(long x);                    /* public API; calls it (slice_a) */

/* Lua's LUAI_DDEF class: a CONST global that is external-linkage + hidden.
 * Internal constants are exempt from promotion because the Slicer embeds them,
 * but this one is not local linkage — the Slicer reads it as "external
 * constant → declare" and binds it by symbol, so it must reach the dynsym.
 * (lua's `luaT_typenames_` is precisely this.) */
__attribute__((visibility("hidden"))) extern const long ST_HIDDEN_TABLE[4];
long st_table_at(int i);                   /* indexes it (slice_a)           */

/* Scale + fan-out (slice_scale.c). st_sc_hub reaches every function in that TU
 * through two layers, so a slicer that followed reachability would emit the
 * whole module for it; declarations-only must emit one `define` plus `declare`s.
 * Only the functions the tests target directly are declared here — the 64
 * leaves and 8 mids are deliberately header-less. They are external-linkage and
 * `default<O2>` has no internalize pass, so they survive to the dynsym and
 * create the fan-out; they are call-graph, not API. */
long st_sc_hub(long v);   /* = 192v + 8416 (see slice_scale.c for the algebra) */
long st_sc_mid_0(long v); /* one layer down: 8 leaves                          */
long st_sc_leaf_00(long v);

#endif
