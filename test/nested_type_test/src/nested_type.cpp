#include "nested_type.h"

World::RayResult::RayResult(float fraction, int id) : fraction(fraction), id(id) {}
bool World::RayResult::hit() const { return fraction < 1.0f; }
World::Callback::Info::Info(int code) : code(code), pad(0) {}
World::Mark::Mark(int m) : m(m) {}
int World::cast(const RayResult &r) const { return r.hit() ? r.id + bodies : -1; }

nt::Plain::Plain(int v) : v(v) {}
nt::Box::Lid::Lid(int w) : w(w) {}

int World_Mark(const World::Mark *m) { return m->m; }
int info_code(const World::Callback::Info *i) { return i->code; }
int lid_w(const nt::Box::Lid *l) { return l->w; }
int plain_v(const nt::Plain *p) { return p->v; }
int box_n(const nt::Box *b) { return b->n; }
int callback_n(const World::Callback *c) { return c->n; }
