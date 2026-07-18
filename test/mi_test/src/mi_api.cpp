// test/mi_test/src/mi_api.cpp
// Multiple-inheritance fixture: two polymorphic bases, compiler-laid-out
// (Itanium x86_64: Base1 subobject @0, Base2 subobject @16).
//
// Ground-truth values are chosen so a missing `this` adjustment is
// unmistakable: a Base2 method called with an unadjusted Derived* reads
// Base1's `a` (111) instead of Base2's `b` (222).

#include <cstdint>

class Base1 {
public:
    Base1();
    virtual ~Base1();
    virtual int32_t get_a() const;
    int32_t a;
};

class Base2 {
public:
    Base2();
    virtual ~Base2();
    virtual int32_t get_b() const;
    virtual void set_b(int32_t v);
    int32_t double_b() const;   // non-virtual: direct symbol call, still needs base-relative this
    int32_t b;
};

class Derived : public Base1, public Base2 {
public:
    Derived();
    ~Derived() override;
    int32_t get_b() const override;  // override of the SECOND base's virtual: b + 1000.
                                     // Reached through Base2's vtable group via the
                                     // compiler's _ZThn16_ this-adjusting thunk —
                                     // the definitive virtual-dispatch canary.
    int32_t get_sum() const;    // non-virtual, derived-relative this: a + b + extra
    int32_t extra;
};

Base1::Base1() : a(111) {}
Base1::~Base1() {}
int32_t Base1::get_a() const { return a; }

Base2::Base2() : b(222) {}
Base2::~Base2() {}
int32_t Base2::get_b() const { return b; }
void Base2::set_b(int32_t v) { b = v; }
int32_t Base2::double_b() const { return b * 2; }

Derived::Derived() : extra(3) {}
Derived::~Derived() {}
int32_t Derived::get_b() const { return b + 1000; }
int32_t Derived::get_sum() const { return a + b + extra; }

extern "C" {
// Upcast factory: hands out a Base2* that is REALLY a Derived — the only way
// a wrapper user can prove override dispatch (static Base2::get_b gives b,
// virtual dispatch gives b + 1000).
Base2* make_derived_as_base2() { return new Derived(); }

// Polymorphic deletion through the base pointer: the virtual dtor's deleting
// variant in the vtable adjusts back to the Derived allocation.
void free_base2(Base2* b) { delete b; }
}

// Nested-type member attribution fixture: a class whose members are split by
// a NESTED enum definition. DWARF emits the enum as a child DIE between the
// member DIEs; the line-oriented metadata parser used to re-point its member
// attribution at the enum and never restore the class, silently dropping
// every member after the nested type (found via box2d's b2Shape::m_radius,
// which follows the nested Shape::Type enum). `after` and `kind` are the
// canaries.
class NestedEnumHolder {
public:
    NestedEnumHolder();
    ~NestedEnumHolder();
    enum Kind { K_A = 1, K_B = 2 };
    // The enum-typed member comes FIRST: clang emits the nested enum's DIE
    // (with enumerator children + null terminator) right after the member
    // that references it — placing it BETWEEN this member and the next two,
    // exactly b2Shape's {m_type; nested Type enum; m_radius} shape.
    Kind kind;
    float after;                    // used to vanish from metadata
    int32_t before;                 // this one too
};

NestedEnumHolder::NestedEnumHolder() : kind(K_B), after(2.0f), before(1) {}
NestedEnumHolder::~NestedEnumHolder() {}

extern "C" {

NestedEnumHolder* make_nested_enum_holder() { return new NestedEnumHolder(); }
void free_nested_enum_holder(NestedEnumHolder* p) { delete p; }

Derived* make_derived() { return new Derived(); }
void free_derived(Derived* d) { delete d; }

}
