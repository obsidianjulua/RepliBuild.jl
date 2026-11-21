#pragma once
// An implementation header (fmt's format-inl.h without FMT_HEADER_ONLY): it
// declares a class AND defines a function with external linkage, so exactly
// one translation unit may include it.
struct Gadget {
    int g;
    Gadget(int x) : g(x) {}
    int thrice() const { return 3 * g; }
};
int gadget_total(const Gadget &x) { return x.g + 1; }    // not inline
int gadget_count = 0;                                     // nor is this
