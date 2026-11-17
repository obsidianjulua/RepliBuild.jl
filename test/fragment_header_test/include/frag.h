#ifndef FRAG_H
#define FRAG_H

/* Library-free fixture for a header that is half of a pair (2026-11-16, from
   Jolt's Compute/CPU/WrapShaderBegin.h, which opens namespaces and a class
   that only WrapShaderEnd.h closes). A class declared through the pair has
   the OPENING header as its declaration file, so the forced unit included
   that header on its own, and every header after it sat inside its scope. */

namespace fh {

/* Header-only constructors and a member nothing in the .cpp uses: forced. */
struct Plain {
    Plain(int v) : v(v) {}
    int twice() const { return v * 2; }
    int v;
};

struct Other {
    Other(float f) : f(f) {}
    float f;
};

int plain_v(const Plain *p);
float other_f(const Other *o);

}

#endif
