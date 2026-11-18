// A class with a destructor, taken by value through a Tier 2 thunk
// (test_byvalue_class_live.jl, 2026-11-17).
//
// C++ passes such a class as a pointer to a temporary the CALLER makes and
// ends, so the thunk has to make one: with the copy constructor, with the move
// constructor where the class cannot be copied, with its bytes only where the
// copy is trivial, and not at all where none of these exists. Each class here
// is one of those cases, and the counters say what the thunk did:
//
//   live()   owned heap ints alive
//   tally()  +100 per copy constructor, +10 per move constructor, +1 per destructor
#pragma once

int  live();
int  tally();
void reset_tally();

// Copyable owner: the callee gets a copy, the caller keeps its own.
struct Res { int *p; explicit Res(int v); Res(const Res &o); ~Res(); int get() const; };
// Move-only owner: the copy constructor is deleted.
struct Uniq { int *p; explicit Uniq(int v); Uniq(const Uniq &) = delete; Uniq(Uniq &&o) noexcept; ~Uniq(); int get() const; };
// Neither copyable nor movable.
struct Pinned { int *p; explicit Pinned(int v); Pinned(const Pinned &) = delete; ~Pinned(); int get() const; };
// The copy constructor is private (and defined).
class Guarded { int *p; Guarded(const Guarded &o); public: explicit Guarded(int v); ~Guarded(); int get() const; };
// An implicit copy that is trivial.
struct Plain { int v; ~Plain(); };
// A defaulted copy that is trivial: declared, and no symbol on any build.
struct Dflt { int v; explicit Dflt(int x); Dflt(const Dflt &) = default; ~Dflt(); };
// An implicit copy that is not trivial (the member's), which nothing in the
// library uses, so the library has no definition of it.
struct Holder { Res r; int id; explicit Holder(int v); ~Holder(); };
// The object points into itself: its bytes are not a copy of it.
struct Tracked { int v; int *self; explicit Tracked(int x); Tracked(const Tracked &o); ~Tracked(); bool ok() const; };
// The destructor is virtual.
struct Poly { int v; explicit Poly(int x); Poly(const Poly &o); virtual ~Poly(); virtual int get() const; };
// The destructor is implicit and virtual. The library then has its deleting
// variant (D0) and its base-object one (D2), and clang emits no
// complete-object one (D1): it is the same function as D2. D0 runs
// `operator delete` on the object, which for a temporary is the thunk's frame.
struct VKid : Poly { explicit VKid(int x); VKid(const VKid &o); int get() const override; };
// A member aligned to 16: the temporary is an object, and aligned as one.
struct Wide { alignas(16) long long a; long long b; explicit Wide(int x); Wide(const Wide &o); ~Wide(); };
// A nested class: `Outer::Inner` to its functions, `Inner` to a parameter.
struct Outer {
    struct Inner { int *p; explicit Inner(int v); Inner(const Inner &o); ~Inner(); };
    int unused;
};

namespace ns {
// The same two owners in a namespace: `ns::NRes` to its functions, `NRes` to
// a parameter.
struct NRes { int *p; explicit NRes(int v); NRes(const NRes &o); ~NRes(); int get() const; };
struct NUniq { int *p; explicit NUniq(int v); NUniq(const NUniq &) = delete; NUniq(NUniq &&o) noexcept; ~NUniq(); int get() const; };
// A class template's specialization.
template <class T> struct Box { T *p; explicit Box(T v); Box(const Box &o); ~Box(); };
NRes  make_nres(int v);
NUniq make_nuniq(int v);
Box<int> make_box(int v);
int take_nres(NRes r);
int consume_nuniq(NUniq u);
int take_box(Box<int> b);
}

// Two classes of one bare name: a parameter spelled `Twin` names either.
namespace one { struct Twin { int *p; explicit Twin(int v); Twin(const Twin &o); ~Twin(); }; }
namespace two { struct Twin { int *p; explicit Twin(int v); Twin(const Twin &o); ~Twin(); }; }

Res     make_res(int v);
Uniq    make_uniq(int v);
Pinned  make_pinned(int v);
Guarded make_guarded(int v);
Dflt    make_dflt(int v);
Holder  make_holder(int v);
Tracked make_tracked(int v);
Poly    make_poly(int v);
Wide    make_wide(int v);
VKid    make_vkid(int v);
Outer::Inner make_inner(int v);
one::Twin    make_twin(int v);

int take_res(Res r);
int consume_uniq(Uniq u);
int take_pinned(Pinned p);
int take_guarded(Guarded g);
int take_plain(Plain p);
int take_dflt(Dflt d);
int take_holder(Holder h);
int take_tracked(Tracked t);            // t.ok() ? t.v : -999
int take_poly(Poly p);
int take_wide(Wide w);                  // w.a when the temporary is 16-aligned, else -1
int take_vkid(VKid k);                  // k.get(): v + 1000
int take_inner(Outer::Inner i);
int take_twin(one::Twin t);
int take_both(Res a, int k, Uniq b);    // a.get() * k + b.get()
