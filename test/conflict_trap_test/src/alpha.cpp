#include "conflict_trap.h"

/* alpha's types, and the records both units share. */

extern "C" {

int ct_impl_sum(alpha::impl v) { return v.n_layer * 10 + v.n_head; }

int ct_holder_sum(Holder h) { return h.i.n_layer * 100 + h.i.n_head * 10 + h.k; }

int ct_alpha_kind(alpha::Kind k) { return static_cast<int>(k) + 40; }

int ct_duo_sum(Duo d) { return d.a + 2 * d.b; }

int ct_mode_value(alpha::Mode m) { return static_cast<int>(m) * 100 + 1; }

int ct_impl_ptr_sum(const alpha::impl *v) { return v->n_layer * 10 + v->n_head; }

}

int ct_duo_twice(Duo d)
{
    if (d.a < 0)
        throw 1;
    return (d.a + d.b) * 2;
}
