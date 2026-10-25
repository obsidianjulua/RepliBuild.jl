#ifndef ZCOLLIDE_H
#define ZCOLLIDE_H

#include <stddef.h>

/* Fixture for test_tier1_alias_binding.jl.
 *
 * Every public symbol here is named after a zlib export, so on Linux every
 * one of them is also defined by the `libz.so.1` Julia loads at startup, and
 * the test loads a decoy library with the same names first for the rest
 * (`decoy/zdecoy.c`). A Tier 1 slice of a function that calls another public
 * function of this library used to `declare` the callee by its public name,
 * which ORC resolves process-wide: the earlier library won. Each function
 * returns a sentinel no other definition of the name returns, so the test can
 * tell whose code ran. */

/* The leaves return `zc_seed` (4242) rather than a literal: a literal is
 * propagated into every caller at -O2 (IPSCCP, noinline notwithstanding), and
 * a caller that no longer calls has nothing to declare. Hidden, like zlib's own
 * internal state under -DHAVE_HIDDEN: an exported mutable global is reached
 * through the GOT, and ELF lets an earlier-loaded definition interpose it for
 * the library's native code too, which is not what this fixture is about.
 * Promotion renames it to `__rb_zcollide_zc_seed`, the name the slices bind. */
extern int zc_seed __attribute__((visibility("hidden")));
void zc_set_seed(int s);

int gzclose_w(void *file);             /* leaf, noinline: zc_seed              */
int gzclose(void *file);               /* calls gzclose_w: zc_seed + 1         */
int gzclose_r(void *file);             /* calls gzclose: zc_seed + 101         */

unsigned long adler32_z(unsigned long adler, const unsigned char *buf, size_t len);
                                       /* leaf, noinline: adler + len + zc_seed */
unsigned long adler32(unsigned long adler, const unsigned char *buf, unsigned int len);
                                       /* calls adler32_z, + 1                 */

extern int z_verbose;                  /* public mutable state                 */
int zc_verbosity(void);                /* reads it                             */
void zc_set_verbosity(int v);          /* writes it                            */

extern const long z_errmsg[4];         /* public constant table                */
long zc_table_at(int i);               /* reads it                             */

#endif
