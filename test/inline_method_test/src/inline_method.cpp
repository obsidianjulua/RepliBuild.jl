#include "inline_method.h"

namespace im {

Body::Body(float mass) : m_pos{0, 0, 0}, m_vel{0, 0, 0}, m_mass(mass), m_scale(1.0f), m_active(1), m_ids{nullptr, 0} {}
Body::~Body() {}
int Body::outOfLine(int k) { return k * 3; }

float body_mass(const Body *b) { return b->mass(); }
int body_kind(const Body *b) { return b->kind(); }
float body_vel_x(const Body *b) { return b->getVelocity().x; }
void body_set_ids(Body *b, int *data, int n) {
    Arr<int> &a = b->ids();
    a.data = data;
    a.n = n;
}

}
