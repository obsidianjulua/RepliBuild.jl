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
long st_guarded_div(long a, long b); /* setjmp/longjmp over a static jmp_buf  */

long st_b_bump(long delta);     /* slice_b.c: same-named static counter —     */
long st_b_get(void);            /* exercises the post-link rename             */

#endif
