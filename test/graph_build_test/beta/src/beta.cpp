// beta: a shared library linked against alpha, with the object target
// beta_helpers folded in.
#include "beta.h"
#include "alpha.h"  // on beta's include path as the build system resolved it

#ifndef BETA_OFFSET
#error "beta's own define is missing"
#endif
#ifdef ALPHA_FACTOR
#error "alpha's define reached beta"
#endif

int beta_helper(int x);  // defined by the object target beta_helpers

extern "C" int beta_combine(int x) { return alpha_scale(x) + beta_helper(x) + BETA_OFFSET; }
