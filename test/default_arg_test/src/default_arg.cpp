#include "default_arg.h"

namespace da {

Hinge::Hinge(const Vec &a, const Vec &b)
    : m_a(a), m_b(b), m_lo(0), m_hi(0), m_soft(0), m_bias(0), m_relax(0), m_last(0) {}
int Hinge::step(float dt, int maxSub, float fixed) { m_last = maxSub * 100 + (int)(fixed * 100) + (int)dt; return m_last; }
int Hinge::step(double dt) { m_last = -(int)dt; return m_last; }
int Hinge::count(int a, int b) const { return a * 10 + b; }
int Hinge::count(int a) const { return -a; }
Vec Hinge::scaled(float k, float bias) const { return Vec{m_a.x * k + bias, m_a.y * k + bias}; }
void Hinge::none(int a, int b) { m_last = a + b; }
int Hinge::make(int a, int b) { return a + b; }
void Hinge::prot(int a, int b) { m_last = a - b; }
float Hinge::lo() const { return m_lo; }
float Hinge::hi() const { return m_hi; }
float Hinge::soft() const { return m_soft; }
float Hinge::bias() const { return m_bias; }
float Hinge::relax() const { return m_relax; }
float Hinge::ax() const { return m_a.x; }
float Hinge::bx() const { return m_b.x; }
float Hinge::by() const { return m_b.y; }
int Hinge::last() const { return m_last; }

float state_offset_y(const State *s) { return s->offset.y; }
float state_start_x(const State *s) { return s->start.x; }

}
