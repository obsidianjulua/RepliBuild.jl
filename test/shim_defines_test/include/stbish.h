/* A single-header library in the stb style: declarations always, the
   implementation only where STBISH_IMPLEMENTATION is defined, plus a static
   inline function and a value macro that never become symbols. */
#ifndef STBISH_H
#define STBISH_H

#define STBISH_VERSION 7

typedef struct { int lo, hi; } StbishRange;

int stbish_add(int a, int b);
StbishRange stbish_range(int lo, int hi);
int stbish_span(StbishRange r);

static inline int stbish_twice(int v) { return 2 * v; }

#endif

#ifdef STBISH_IMPLEMENTATION
/* [wrap] shim_defines writes a bare NAME as 1, as -DNAME would. */
#if STBISH_IMPLEMENTATION != 1
#error "STBISH_IMPLEMENTATION is not 1"
#endif
static int stbish_scale = 1;
int stbish_add(int a, int b) { return (a + b) * stbish_scale; }
StbishRange stbish_range(int lo, int hi) { StbishRange r = { lo, hi }; return r; }
int stbish_span(StbishRange r) { return r.hi - r.lo; }
#endif
