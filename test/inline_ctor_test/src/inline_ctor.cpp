#include "inline_ctor.h"

namespace ic {

Priv Priv::make(int s) { return Priv(s); }
int Priv::get() const { return s; }
Defined::Defined(int v) : v(v * 2) {}
Del::Del(int v) : v(v) {}

int plain_value(const Plain *p) { return p->v; }
int dflt_sum(const Dflt *d) { return d->sum; }
float expl_f(const Expl *e) { return e->f; }
int abs_get(const Abs *a) { return a->get() + a->k; }
int priv_get(const Priv *p) { return p->get(); }
int defined_value(const Defined *d) { return d->v; }
int del_value(const Del *d) { return d->v; }

}
