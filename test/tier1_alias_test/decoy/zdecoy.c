/* The decoy: every symbol of zcollide.h again, with other values. The test
 * compiles this to a shared library and dlopens it RTLD_GLOBAL before the
 * wrapper loads its own, which is what Julia's bundled libz does to the Hub's
 * zlib. A slice bound through the process reaches these; one bound through the
 * library's `__rb_` aliases never can. */
#include <stddef.h>

static int zc_seed_decoy = -4242;
void zc_set_seed(int s) { zc_seed_decoy = s; }
int gzclose_w(void *file) { (void)file; return -1000 + zc_seed_decoy * 0; }
int gzclose(void *file) { (void)file; return -2000; }
int gzclose_r(void *file) { (void)file; return -3000; }
unsigned long adler32_z(unsigned long adler, const unsigned char *buf, size_t len)
{
    (void)adler; (void)buf; (void)len;
    return 0xBADUL;
}
unsigned long adler32(unsigned long adler, const unsigned char *buf, unsigned int len)
{
    (void)adler; (void)buf; (void)len;
    return 0xBADUL;
}
int z_verbose = -70;
int zc_verbosity(void) { return z_verbose; }
void zc_set_verbosity(int v) { z_verbose = v; }
const long z_errmsg[4] = {-1, -2, -3, -4};
long zc_table_at(int i) { (void)i; return -9; }
