#ifndef INLINE_CTOR_H
#define INLINE_CTOR_H

/* Library-free fixture for constructors defined inline in a header and used by
   no translation unit of the library (2026-11-15, from Bullet3's btSphereShape,
   btDefaultMotionState and btRigidBodyConstructionInfo). The library has no
   symbol for such a constructor, so the class could not be built from Julia
   without a hand-written placement-new shim. The build now emits those
   definitions itself (Compiler.generate_inline_constructors): a generated unit
   includes this header and odr-uses each declared-but-undefined constructor
   through placement new. Built with -fstandalone-debug, as Bullet is: under
   constructor homing a class nothing constructs is declaration-only in DWARF
   and the pass cannot see its constructors. */

namespace ic {

/* Header-only constructors, two of them; nothing in the .cpp constructs a Plain. */
struct Plain {
    Plain(int v) : v(v) {}
    Plain(int a, int b) : v(a * b) {}
    int v;
};

/* A default argument: the symbol spells both parameters. */
struct Dflt {
    Dflt(int a, int b = 3) : sum(a + b) {}
    int sum;
};

/* An explicit constructor: placement new is direct-initialization. */
struct Expl {
    explicit Expl(float f) : f(f) {}
    float f;
};

/* Abstract: the unit's guard declines it (named in the wrap report). */
struct Abs {
    Abs(int k) : k(k) {}
    virtual ~Abs() {}
    virtual int get() const = 0;
    int k;
};

/* A private constructor: skipped by the declaration's accessibility. */
class Priv {
    Priv(int s) : s(s) {}
    int s;
public:
    static Priv make(int s);
    int get() const;
};

/* Defined in the .cpp: nothing to force, the wrapper's binding is untouched. */
struct Defined {
    Defined(int v);
    int v;
};

/* A deleted constructor beside a defined one. */
struct Del {
    Del() = delete;
    Del(int v);
    int v;
};

int plain_value(const Plain *p);
int dflt_sum(const Dflt *d);
float expl_f(const Expl *e);
int abs_get(const Abs *a);
int priv_get(const Priv *p);
int defined_value(const Defined *d);
int del_value(const Del *d);

}

#endif
