#include "frag.h"
#include "wrap_begin.h"
    int get() const { return k; }
#include "wrap_end.h"

namespace fh {
int plain_v(const Plain *p) { return p->v; }
float other_f(const Other *o) { return o->f; }
int wrapper_k(const wrapped::Wrapper *w) { return w->get(); }
}
