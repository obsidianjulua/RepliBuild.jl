// beta_helpers: an OBJECT target. It is compiled under its own group and its
// code lands in beta, which calls it by its C++ name.
#ifndef HELPER_BIAS
#error "the object target's own define is missing"
#endif
#ifdef BETA_OFFSET
#error "beta's define reached the object target"
#endif

int beta_helper(int x) { return x * 10 + HELPER_BIAS; }
