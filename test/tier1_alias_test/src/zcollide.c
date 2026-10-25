/* See include/zcollide.h. Leaves are noinline so the public-calls-public edge
 * survives -O2 and the caller's slice has to declare the callee. */
#include "zcollide.h"

#define ZC_LEAF __attribute__((noinline))

int zc_seed = 4242;
void zc_set_seed(int s) { zc_seed = s; }

ZC_LEAF int gzclose_w(void *file) { (void)file; return zc_seed; }
int gzclose(void *file) { return gzclose_w(file) + 1; }
int gzclose_r(void *file) { return gzclose(file) + 100; }

ZC_LEAF unsigned long adler32_z(unsigned long adler, const unsigned char *buf, size_t len)
{
    (void)buf;
    return adler + (unsigned long)len + (unsigned long)zc_seed;
}
unsigned long adler32(unsigned long adler, const unsigned char *buf, unsigned int len)
{
    return adler32_z(adler, buf, len) + 1;
}

int z_verbose = 70;
int zc_verbosity(void) { return z_verbose; }
void zc_set_verbosity(int v) { z_verbose = v; }

const long z_errmsg[4] = {11, 13, 17, 19};
long zc_table_at(int i) { return z_errmsg[i & 3]; }
