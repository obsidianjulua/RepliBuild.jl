// test/vi_test/src/vi_api.cpp
// Virtual-inheritance diamond fixture.
//
//   VBase (virtual base: vptr + t)
//   Left  : public virtual VBase   (vptr + l)
//   Right : public virtual VBase   (vptr + r)
//   Diamond : public Left, public Right  (+ d, overrides VBase::tag)
//
// The load-bearing fact: the VBase subobject's offset is NOT static.
//   standalone Left:   [Left vptr, l @8][VBase @16: vptr, t @24]
//   Diamond:           [Left vptr, l @8][Right vptr @16, r @24][d @28: tail-padding reuse][VBase @32: vptr, t @40]
// The offset to VBase from a Left* is 16 in one case and 32 in the other —
// only the vbase-offset entry in the object's vtable knows which. Any
// static-offset upcast is wrong for one of the two.
//
// Values are chosen so every wrong path is unmistakable:
//   t=7, l=100, r=200, d=300; Diamond::tag() returns t + 1000.

#include <cstdint>

class VBase {
public:
    VBase();
    virtual ~VBase();
    virtual int32_t tag() const;
    int32_t t;
};

class Left : public virtual VBase {
public:
    Left();
    virtual ~Left();
    virtual int32_t lval() const;
    int32_t l;
};

class Right : public virtual VBase {
public:
    Right();
    virtual ~Right();
    virtual int32_t rval() const;
    int32_t r;
};

class Diamond : public Left, public Right {
public:
    Diamond();
    ~Diamond() override;
    int32_t tag() const override;   // override of the VIRTUAL BASE's method
    int32_t dval() const;           // non-virtual: l + r + t + d
    int32_t d;
};

VBase::VBase() : t(7) {}
VBase::~VBase() {}
int32_t VBase::tag() const { return t; }

Left::Left() : l(100) {}
Left::~Left() {}
int32_t Left::lval() const { return l; }

Right::Right() : r(200) {}
Right::~Right() {}
int32_t Right::rval() const { return r; }

Diamond::Diamond() : d(300) {}
Diamond::~Diamond() {}
int32_t Diamond::tag() const { return t + 1000; }
int32_t Diamond::dval() const { return l + r + t + d; }

extern "C" {

Left* make_left() { return new Left(); }
Diamond* make_diamond() { return new Diamond(); }
// A Left* that is REALLY a Diamond — same static type as make_left(),
// different dynamic vbase offset. The canary pair.
Left* make_diamond_as_left() { return new Diamond(); }

void free_left(Left* p) { delete p; }        // virtual dtor: deletes Diamond correctly too
void free_diamond(Diamond* p) { delete p; }

}
