#include "bv.h"

static int g_live = 0, g_tally = 0;
int  live() { return g_live; }
int  tally() { return g_tally; }
void reset_tally() { g_tally = 0; }

// An owner of one heap int: the shape every owning class below shares.
#define OWNER_CTOR(C)  C::C(int v) : p(new int(v)) { ++g_live; }
#define OWNER_COPY(C)  C::C(const C &o) : p(new int(*o.p)) { ++g_live; g_tally += 100; }
#define OWNER_MOVE(C)  C::C(C &&o) noexcept : p(o.p) { o.p = nullptr; g_tally += 10; }
#define OWNER_DTOR(C)  C::~C() { if (p) { delete p; --g_live; } g_tally += 1; }

OWNER_CTOR(Res)  OWNER_COPY(Res)  OWNER_DTOR(Res)
int Res::get() const { return *p; }

OWNER_CTOR(Uniq)  OWNER_MOVE(Uniq)  OWNER_DTOR(Uniq)
int Uniq::get() const { return p ? *p : -1; }

OWNER_CTOR(Pinned)  OWNER_DTOR(Pinned)
int Pinned::get() const { return *p; }

OWNER_CTOR(Guarded)  OWNER_COPY(Guarded)  OWNER_DTOR(Guarded)
int Guarded::get() const { return *p; }

Plain::~Plain() { g_tally += 1; }

Dflt::Dflt(int x) : v(x) {}
Dflt::~Dflt() { g_tally += 1; }

Holder::Holder(int v) : r(v), id(v) {}
Holder::~Holder() {}

Tracked::Tracked(int x) : v(x), self(&v) {}
Tracked::Tracked(const Tracked &o) : v(o.v), self(&v) { g_tally += 100; }
Tracked::~Tracked() { g_tally += 1; }
bool Tracked::ok() const { return self == &v; }

Poly::Poly(int x) : v(x) {}
Poly::Poly(const Poly &o) : v(o.v) { g_tally += 100; }
Poly::~Poly() { g_tally += 1; }
int Poly::get() const { return v; }

Outer::Inner::Inner(int v) : p(new int(v)) { ++g_live; }
Outer::Inner::Inner(const Inner &o) : p(new int(*o.p)) { ++g_live; g_tally += 100; }
Outer::Inner::~Inner() { if (p) { delete p; --g_live; } g_tally += 1; }

namespace ns {
OWNER_CTOR(NRes)  OWNER_COPY(NRes)  OWNER_DTOR(NRes)
int NRes::get() const { return *p; }

OWNER_CTOR(NUniq)  OWNER_MOVE(NUniq)  OWNER_DTOR(NUniq)
int NUniq::get() const { return p ? *p : -1; }

template <class T> Box<T>::Box(T v) : p(new T(v)) { ++g_live; }
template <class T> Box<T>::Box(const Box &o) : p(new T(*o.p)) { ++g_live; g_tally += 100; }
template <class T> Box<T>::~Box() { if (p) { delete p; --g_live; } g_tally += 1; }
template struct Box<int>;

NRes  make_nres(int v) { return NRes(v); }
NUniq make_nuniq(int v) { return NUniq(v); }
Box<int> make_box(int v) { return Box<int>(v); }
int take_nres(NRes r) { return r.get(); }
int consume_nuniq(NUniq u) { return u.get(); }
int take_box(Box<int> b) { return *b.p; }
}

namespace one { OWNER_CTOR(Twin)  OWNER_COPY(Twin)  OWNER_DTOR(Twin) }
namespace two { OWNER_CTOR(Twin)  OWNER_COPY(Twin)  OWNER_DTOR(Twin) }

Res     make_res(int v) { return Res(v); }
Uniq    make_uniq(int v) { return Uniq(v); }
Pinned  make_pinned(int v) { return Pinned(v); }
Guarded make_guarded(int v) { return Guarded(v); }
Dflt    make_dflt(int v) { return Dflt(v); }
Holder  make_holder(int v) { return Holder(v); }
Tracked make_tracked(int v) { return Tracked(v); }
Poly    make_poly(int v) { return Poly(v); }
Outer::Inner make_inner(int v) { return Outer::Inner(v); }
one::Twin    make_twin(int v) { return one::Twin(v); }

int take_res(Res r) { return r.get(); }
int consume_uniq(Uniq u) { return u.get(); }
int take_pinned(Pinned p) { return p.get(); }
int take_guarded(Guarded g) { return g.get(); }
int take_plain(Plain p) { return p.v; }
int take_dflt(Dflt d) { return d.v; }
int take_holder(Holder h) { return h.id; }
int take_tracked(Tracked t) { return t.ok() ? t.v : -999; }
int take_poly(Poly p) { return p.get(); }
int take_inner(Outer::Inner i) { return *i.p; }
int take_twin(one::Twin t) { return *t.p; }
int take_both(Res a, int k, Uniq b) { return a.get() * k + b.get(); }
