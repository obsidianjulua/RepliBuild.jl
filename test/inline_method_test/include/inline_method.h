#ifndef INLINE_METHOD_H
#define INLINE_METHOD_H

/* Library-free fixture for member functions defined inline in a class body
   and called by no translation unit of the library (2026-11-15, from Bullet3:
   btRigidBody::applyCentralImpulse, setLinearVelocity, getLinearVelocity,
   btCollisionObject::isActive, btTransform::getOrigin, …). The library has no
   symbol for such a member, so the wrapper had no binding and a caller wrote
   fields at DWARF offsets. The build now takes each one's address in its
   forced unit (Compiler.generate_inline_definitions), so clang emits the
   definition. Built with -fstandalone-debug, as Bullet is. */

namespace im {

struct Vec3 {
    float x, y, z;
};

/* A class template the library instantiates: its members are declared in the
   specialization's DIE, and are not forced. `[types] templates` instantiates
   a specialization's members when a recipe asks for it. */
template <class T> struct Arr {
    T *data;
    int n;
    int size() const { return n; }
    T &at(int i) { return data[i]; }
};

class Body {
public:
    Body(float mass);                       /* defined in the .cpp */

    /* Header-only, called by nothing in the library: the gap. */
    void setVelocity(const Vec3 &v) { m_vel = v; }
    const Vec3 &getVelocity() const { return m_vel; }
    void applyImpulse(const Vec3 &j) {
        m_vel.x += j.x * invMass();
        m_vel.y += j.y * invMass();
        m_vel.z += j.z * invMass();
    }
    bool isActive() const { return m_active != 0; }
    void setActive(bool on) { m_active = on ? 1 : 0; }

    /* A const and a non-const overload of one name: two symbols. */
    Vec3 &position() { return m_pos; }
    const Vec3 &position() const { return m_pos; }

    /* An overload set that differs in a parameter. */
    void scale(int k) { m_scale = (float)k; }
    void scale(float k, float bias) { m_scale = k + bias; }
    float getScale() const { return m_scale; }

    /* always_inline (Bullet's SIMD_FORCE_INLINE): still has an address. */
    inline __attribute__((always_inline)) float invMass() const { return 1.0f / m_mass; }

    /* A static member. */
    static float defaultMass() { return 2.0f; }

    /* An operator. */
    Body &operator+=(const Vec3 &d) {
        m_pos.x += d.x; m_pos.y += d.y; m_pos.z += d.z;
        return *this;
    }

    /* Inline, but the library calls it: already a weak definition. */
    float mass() const { return m_mass; }

    /* Out of line: the library defines it, nothing to force. */
    int outOfLine(int k);

    /* Declared, defined nowhere: taking its address would leave the library
       with an undefined symbol. The unit's IR only declares it; skipped. */
    int declaredOnly(int k);

    /* A member function template: no declaration to force. */
    template <class T> T as() const { return (T)m_mass; }

    /* Virtual and inline: emitted with the vtable, never forced. */
    virtual int kind() const { return 7; }
    virtual ~Body();

    Arr<int> &ids() { return m_ids; }

protected:
    void protectedInline() { m_scale = -1.0f; }

private:
    void privateInline() { m_scale = -2.0f; }

    Vec3 m_pos;
    Vec3 m_vel;
    float m_mass;
    float m_scale;
    int m_active;
    Arr<int> m_ids;
};

float body_mass(const Body *b);
int body_kind(const Body *b);
float body_vel_x(const Body *b);
void body_set_ids(Body *b, int *data, int n);

}

#endif
