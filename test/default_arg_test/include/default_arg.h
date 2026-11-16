#ifndef DEFAULT_ARG_H
#define DEFAULT_ARG_H

/* Library-free fixture for C++ default arguments (2026-11-15, from Bullet3:
   btDefaultMotionState(start) had to be called with its second transform
   spelled out, and btHingeConstraint::setLimit(lo, hi) with all five
   arguments). A default exists only in the header, so the build asks the
   compiler: one generated function per call that leaves trailing arguments
   out (Compiler._default_argument_plan). Built with -fstandalone-debug. */

namespace da {

struct Vec {
    float x, y;
};

class Hinge {
public:
    /* An out-of-line constructor with one default. */
    Hinge(const Vec &a, const Vec &b = Vec{1.0f, 2.0f});

    /* Inline, three defaults: setLimit(lo, hi), (lo, hi, soft), (lo, hi, soft, bias). */
    void setLimit(float lo, float hi, float soft = 0.9f, float bias = 0.3f, float relax = 1.0f) {
        m_lo = lo; m_hi = hi; m_soft = soft; m_bias = bias; m_relax = relax;
    }

    /* Out of line, two defaults, beside an overload that takes a double: a
       float argument is an exact match for this one. */
    int step(float dt, int maxSub = 1, float fixed = 0.25f);
    int step(double dt);

    /* count(a) is its own member: the one-argument call is that member's,
       and no function is generated for it. */
    int count(int a, int b = 2) const;
    int count(int a) const;

    /* A by-value record returned through the short call. */
    Vec scaled(float k, float bias = 0.5f) const;

    /* No defaults: every shorter call is a compile error, silently dropped. */
    void none(int a, int b);

    /* A static member: its short call is not built. */
    static int make(int a, int b = 5);

    float lo() const, hi() const, soft() const, bias() const, relax() const;
    float ax() const, bx() const, by() const;
    int last() const;

protected:
    void prot(int a, int b = 1);

private:
    Vec m_a, m_b;
    float m_lo, m_hi, m_soft, m_bias, m_relax;
    int m_last;
};

/* A header-only constructor with a default (btDefaultMotionState's shape):
   the build forces the constructor and its short call. */
struct State {
    State(const Vec &start, const Vec &offset = Vec{7.0f, 8.0f}) : start(start), offset(offset) {}
    Vec start, offset;
};

float state_offset_y(const State *s);
float state_start_x(const State *s);

}

#endif
