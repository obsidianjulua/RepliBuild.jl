#include "sret.h"

SelfRef::SelfRef(int x) : self(this), v(x), w(x * 0.5) {}
SelfRef::SelfRef(const SelfRef &o) : self(this), v(o.v), w(o.w) {}
SelfRef::~SelfRef() { self = nullptr; }

SelfRef make_self(int x) { return SelfRef(x); }
int     self_ok(const SelfRef &s) { return s.self == &s ? 1 : 0; }
int     self_value(const SelfRef &s) { return s.v; }

Plain make_plain(int x) { return Plain{x, x + 1, x + 2}; }
