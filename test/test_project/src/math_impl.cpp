#include "math_types.h"
#include <cmath>
#include <algorithm>

// Vector2 operations
Vector2 vec2_add(Vector2 a, Vector2 b) {
    return {a.x + b.x, a.y + b.y};
}

Vector2 vec2_scale(Vector2 v, double s) {
    return {v.x * s, v.y * s};
}

double vec2_dot(Vector2 a, Vector2 b) {
    return a.x * b.x + a.y * b.y;
}

double vec2_length(Vector2 v) {
    return std::sqrt(vec2_dot(v, v));
}

Vector2 vec2_normalize(Vector2 v) {
    double len = vec2_length(v);
    return len > 0.0 ? vec2_scale(v, 1.0 / len) : Vector2{0, 0};
}

// Vector3 operations
Vector3 vec3_add(Vector3 a, Vector3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vector3 vec3_sub(Vector3 a, Vector3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vector3 vec3_scale(Vector3 v, double s) {
    return {v.x * s, v.y * s, v.z * s};
}

double vec3_dot(Vector3 a, Vector3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vector3 vec3_cross(Vector3 a, Vector3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

double vec3_length(Vector3 v) {
    return std::sqrt(vec3_dot(v, v));
}

Vector3 vec3_normalize(Vector3 v) {
    double len = vec3_length(v);
    return len > 0.0 ? vec3_scale(v, 1.0 / len) : Vector3{0, 0, 0};
}

// Quaternion operations
Quaternion quat_identity() {
    return {1.0, 0.0, 0.0, 0.0};
}

Quaternion quat_from_axis_angle(Vector3 axis, double angle) {
    double half = angle * 0.5;
    double s = std::sin(half);
    Vector3 n = vec3_normalize(axis);
    return {std::cos(half), n.x * s, n.y * s, n.z * s};
}

Quaternion quat_multiply(Quaternion a, Quaternion b) {
    return {
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    };
}

Vector3 quat_rotate_vector(Quaternion q, Vector3 v) {
    Vector3 u = {q.x, q.y, q.z};
    double s = q.w;

    Vector3 uv = vec3_cross(u, v);
    Vector3 uuv = vec3_cross(u, uv);

    return vec3_add(v, vec3_scale(vec3_add(vec3_scale(uv, 2.0 * s), vec3_scale(uuv, 2.0)), 1.0));
}

Quaternion quat_normalize(Quaternion q) {
    double len = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (len > 0.0) {
        return {q.w / len, q.x / len, q.y / len, q.z / len};
    }
    return quat_identity();
}

// Matrix3x3 operations
Matrix3x3 mat3_identity() {
    Matrix3x3 m;
    for (int i = 0; i < 9; i++) m.data[i] = 0.0;
    m.data[0] = m.data[4] = m.data[8] = 1.0;
    return m;
}

Matrix3x3 mat3_from_quaternion(Quaternion q) {
    Matrix3x3 m;
    double xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

    m.data[0] = 1.0 - 2.0 * (yy + zz);
    m.data[1] = 2.0 * (xy + wz);
    m.data[2] = 2.0 * (xz - wy);
    m.data[3] = 2.0 * (xy - wz);
    m.data[4] = 1.0 - 2.0 * (xx + zz);
    m.data[5] = 2.0 * (yz + wx);
    m.data[6] = 2.0 * (xz + wy);
    m.data[7] = 2.0 * (yz - wx);
    m.data[8] = 1.0 - 2.0 * (xx + yy);

    return m;
}

Matrix3x3 mat3_multiply(Matrix3x3 a, Matrix3x3 b) {
    Matrix3x3 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.data[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; k++) {
                result.data[i * 3 + j] += a.data[i * 3 + k] * b.data[k * 3 + j];
            }
        }
    }
    return result;
}

Vector3 mat3_transform(Matrix3x3 m, Vector3 v) {
    return {
        m.data[0] * v.x + m.data[1] * v.y + m.data[2] * v.z,
        m.data[3] * v.x + m.data[4] * v.y + m.data[5] * v.z,
        m.data[6] * v.x + m.data[7] * v.y + m.data[8] * v.z
    };
}

double mat3_determinant(Matrix3x3 m) {
    return m.data[0] * (m.data[4] * m.data[8] - m.data[5] * m.data[7]) -
           m.data[1] * (m.data[3] * m.data[8] - m.data[5] * m.data[6]) +
           m.data[2] * (m.data[3] * m.data[7] - m.data[4] * m.data[6]);
}

Matrix3x3 mat3_transpose(Matrix3x3 m) {
    Matrix3x3 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.data[i * 3 + j] = m.data[j * 3 + i];
        }
    }
    return result;
}

Matrix3x3 mat3_inverse(Matrix3x3 m) {
    double det = mat3_determinant(m);
    if (std::abs(det) < 1e-10) return mat3_identity();

    Matrix3x3 inv;
    inv.data[0] = (m.data[4] * m.data[8] - m.data[5] * m.data[7]) / det;
    inv.data[1] = (m.data[2] * m.data[7] - m.data[1] * m.data[8]) / det;
    inv.data[2] = (m.data[1] * m.data[5] - m.data[2] * m.data[4]) / det;
    inv.data[3] = (m.data[5] * m.data[6] - m.data[3] * m.data[8]) / det;
    inv.data[4] = (m.data[0] * m.data[8] - m.data[2] * m.data[6]) / det;
    inv.data[5] = (m.data[2] * m.data[3] - m.data[0] * m.data[5]) / det;
    inv.data[6] = (m.data[3] * m.data[7] - m.data[4] * m.data[6]) / det;
    inv.data[7] = (m.data[1] * m.data[6] - m.data[0] * m.data[7]) / det;
    inv.data[8] = (m.data[0] * m.data[4] - m.data[1] * m.data[3]) / det;

    return inv;
}

// Matrix4x4 operations
Matrix4x4 mat4_identity() {
    Matrix4x4 m;
    for (int i = 0; i < 16; i++) m.data[i] = 0.0;
    m.data[0] = m.data[5] = m.data[10] = m.data[15] = 1.0;
    return m;
}

Matrix4x4 mat4_from_transform(Transform t) {
    Matrix3x3 rot = mat3_from_quaternion(t.rotation);
    Matrix4x4 m = mat4_identity();

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m.data[i * 4 + j] = rot.data[i * 3 + j];
        }
    }

    m.data[12] = t.position.x;
    m.data[13] = t.position.y;
    m.data[14] = t.position.z;

    return m;
}

Matrix4x4 mat4_multiply(Matrix4x4 a, Matrix4x4 b) {
    Matrix4x4 result;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.data[i * 4 + j] = 0.0;
            for (int k = 0; k < 4; k++) {
                result.data[i * 4 + j] += a.data[i * 4 + k] * b.data[k * 4 + j];
            }
        }
    }
    return result;
}

Vector4 mat4_transform(Matrix4x4 m, Vector4 v) {
    return {
        m.data[0] * v.x + m.data[1] * v.y + m.data[2] * v.z + m.data[3] * v.w,
        m.data[4] * v.x + m.data[5] * v.y + m.data[6] * v.z + m.data[7] * v.w,
        m.data[8] * v.x + m.data[9] * v.y + m.data[10] * v.z + m.data[11] * v.w,
        m.data[12] * v.x + m.data[13] * v.y + m.data[14] * v.z + m.data[15] * v.w
    };
}

Matrix4x4 mat4_inverse(Matrix4x4 m) {
    // Simplified - assumes affine transform
    return mat4_identity(); // Placeholder
}

// Transform operations
Transform transform_identity() {
    return {{0, 0, 0}, quat_identity(), {1, 1, 1}};
}

Transform transform_create(Vector3 pos, Quaternion rot, Vector3 scale) {
    return {pos, rot, scale};
}

Transform transform_combine(Transform parent, Transform child) {
    Transform result;
    result.rotation = quat_multiply(parent.rotation, child.rotation);
    result.scale = {parent.scale.x * child.scale.x,
                    parent.scale.y * child.scale.y,
                    parent.scale.z * child.scale.z};
    result.position = vec3_add(parent.position,
                               quat_rotate_vector(parent.rotation, child.position));
    return result;
}

Vector3 transform_point(Transform t, Vector3 p) {
    Vector3 scaled = {p.x * t.scale.x, p.y * t.scale.y, p.z * t.scale.z};
    Vector3 rotated = quat_rotate_vector(t.rotation, scaled);
    return vec3_add(t.position, rotated);
}

Vector3 transform_direction(Transform t, Vector3 d) {
    return quat_rotate_vector(t.rotation, d);
}

// AABB operations
AABB aabb_from_points(const Vector3* points, size_t count) {
    if (count == 0) return {{0, 0, 0}, {0, 0, 0}};

    AABB box = {points[0], points[0]};
    for (size_t i = 1; i < count; i++) {
        box.min.x = std::min(box.min.x, points[i].x);
        box.min.y = std::min(box.min.y, points[i].y);
        box.min.z = std::min(box.min.z, points[i].z);
        box.max.x = std::max(box.max.x, points[i].x);
        box.max.y = std::max(box.max.y, points[i].y);
        box.max.z = std::max(box.max.z, points[i].z);
    }
    return box;
}

AABB aabb_combine(AABB a, AABB b) {
    return {
        {std::min(a.min.x, b.min.x), std::min(a.min.y, b.min.y), std::min(a.min.z, b.min.z)},
        {std::max(a.max.x, b.max.x), std::max(a.max.y, b.max.y), std::max(a.max.z, b.max.z)}
    };
}

bool aabb_contains_point(AABB box, Vector3 point) {
    return point.x >= box.min.x && point.x <= box.max.x &&
           point.y >= box.min.y && point.y <= box.max.y &&
           point.z >= box.min.z && point.z <= box.max.z;
}

bool aabb_intersects(AABB a, AABB b) {
    return a.min.x <= b.max.x && a.max.x >= b.min.x &&
           a.min.y <= b.max.y && a.max.y >= b.min.y &&
           a.min.z <= b.max.z && a.max.z >= b.min.z;
}

Vector3 aabb_center(AABB box) {
    return {
        (box.min.x + box.max.x) * 0.5,
        (box.min.y + box.max.y) * 0.5,
        (box.min.z + box.max.z) * 0.5
    };
}

Vector3 aabb_size(AABB box) {
    return vec3_sub(box.max, box.min);
}

// Sphere operations
Sphere sphere_from_points(const Vector3* points, size_t count) {
    if (count == 0) return {{0, 0, 0}, 0.0};

    AABB box = aabb_from_points(points, count);
    Vector3 center = aabb_center(box);

    double max_dist = 0.0;
    for (size_t i = 0; i < count; i++) {
        double dist = vec3_length(vec3_sub(points[i], center));
        max_dist = std::max(max_dist, dist);
    }

    return {center, max_dist};
}

bool sphere_contains_point(Sphere s, Vector3 point) {
    double dist = vec3_length(vec3_sub(point, s.center));
    return dist <= s.radius;
}

bool sphere_intersects_sphere(Sphere a, Sphere b) {
    double dist = vec3_length(vec3_sub(a.center, b.center));
    return dist <= (a.radius + b.radius);
}

// Ray operations
Ray ray_create(Vector3 origin, Vector3 direction) {
    return {origin, vec3_normalize(direction)};
}

bool ray_intersect_aabb(Ray ray, AABB box, RayHit* hit) {
    double tmin = 0.0, tmax = 1e10;

    for (int i = 0; i < 3; i++) {
        double d = (&ray.direction.x)[i];
        double o = (&ray.origin.x)[i];
        double bmin = (&box.min.x)[i];
        double bmax = (&box.max.x)[i];

        if (std::abs(d) < 1e-10) {
            if (o < bmin || o > bmax) return false;
        } else {
            double t1 = (bmin - o) / d;
            double t2 = (bmax - o) / d;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return false;
        }
    }

    if (hit) {
        hit->hit = true;
        hit->distance = tmin;
        hit->point = ray_at(ray, tmin);
    }
    return true;
}

bool ray_intersect_sphere(Ray ray, Sphere sphere, RayHit* hit) {
    Vector3 oc = vec3_sub(ray.origin, sphere.center);
    double b = vec3_dot(oc, ray.direction);
    double c = vec3_dot(oc, oc) - sphere.radius * sphere.radius;
    double discriminant = b * b - c;

    if (discriminant < 0) return false;

    double t = -b - std::sqrt(discriminant);
    if (t < 0) t = -b + std::sqrt(discriminant);
    if (t < 0) return false;

    if (hit) {
        hit->hit = true;
        hit->distance = t;
        hit->point = ray_at(ray, t);
        hit->normal = vec3_normalize(vec3_sub(hit->point, sphere.center));
    }
    return true;
}

Vector3 ray_at(Ray ray, double t) {
    return vec3_add(ray.origin, vec3_scale(ray.direction, t));
}

// Callback storage
static TransformCallback g_transform_callback = nullptr;
static IntersectionTest g_intersection_test = nullptr;

void register_transform_callback(TransformCallback callback) {
    g_transform_callback = callback;
}

void register_intersection_test(IntersectionTest test) {
    g_intersection_test = test;
}
