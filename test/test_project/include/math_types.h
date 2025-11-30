#pragma once

#include <cstdint>
#include <cstddef>

// Complex enum types
enum class VectorSpace {
    R2 = 2,
    R3 = 3,
    R4 = 4
};

enum MatrixType {
    IDENTITY = 0,
    DIAGONAL = 1,
    SYMMETRIC = 2,
    ORTHOGONAL = 3,
    SPARSE = 4
};

// Forward declarations
struct Vector3;
struct Matrix3x3;
struct Quaternion;

// Basic vector types
struct Vector2 {
    double x;
    double y;
};

struct Vector3 {
    double x;
    double y;
    double z;
};

struct Vector4 {
    double x;
    double y;
    double z;
    double w;
};

// Matrix types
struct Matrix2x2 {
    double data[4];
};

struct Matrix3x3 {
    double data[9];
};

struct Matrix4x4 {
    double data[16];
};

// Quaternion for rotations
struct Quaternion {
    double w;
    double x;
    double y;
    double z;
};

// Transform struct with nested types
struct Transform {
    Vector3 position;
    Quaternion rotation;
    Vector3 scale;
};

// Bounding volumes
struct AABB {
    Vector3 min;
    Vector3 max;
};

struct Sphere {
    Vector3 center;
    double radius;
};

// Ray for raycasting
struct Ray {
    Vector3 origin;
    Vector3 direction;
};

// Hit result
struct RayHit {
    bool hit;
    double distance;
    Vector3 point;
    Vector3 normal;
};

// Function pointer types
typedef void (*TransformCallback)(const Transform* transform);
typedef bool (*IntersectionTest)(const Ray* ray, const AABB* box, RayHit* hit);

// API functions
extern "C" {
    // Vector operations
    Vector2 vec2_add(Vector2 a, Vector2 b);
    Vector2 vec2_scale(Vector2 v, double s);
    double vec2_dot(Vector2 a, Vector2 b);
    double vec2_length(Vector2 v);
    Vector2 vec2_normalize(Vector2 v);

    Vector3 vec3_add(Vector3 a, Vector3 b);
    Vector3 vec3_sub(Vector3 a, Vector3 b);
    Vector3 vec3_scale(Vector3 v, double s);
    double vec3_dot(Vector3 a, Vector3 b);
    Vector3 vec3_cross(Vector3 a, Vector3 b);
    double vec3_length(Vector3 v);
    Vector3 vec3_normalize(Vector3 v);

    // Quaternion operations
    Quaternion quat_identity();
    Quaternion quat_from_axis_angle(Vector3 axis, double angle);
    Quaternion quat_multiply(Quaternion a, Quaternion b);
    Vector3 quat_rotate_vector(Quaternion q, Vector3 v);
    Quaternion quat_normalize(Quaternion q);

    // Matrix operations
    Matrix3x3 mat3_identity();
    Matrix3x3 mat3_from_quaternion(Quaternion q);
    Matrix3x3 mat3_multiply(Matrix3x3 a, Matrix3x3 b);
    Vector3 mat3_transform(Matrix3x3 m, Vector3 v);
    double mat3_determinant(Matrix3x3 m);
    Matrix3x3 mat3_transpose(Matrix3x3 m);
    Matrix3x3 mat3_inverse(Matrix3x3 m);

    Matrix4x4 mat4_identity();
    Matrix4x4 mat4_from_transform(Transform t);
    Matrix4x4 mat4_multiply(Matrix4x4 a, Matrix4x4 b);
    Vector4 mat4_transform(Matrix4x4 m, Vector4 v);
    Matrix4x4 mat4_inverse(Matrix4x4 m);

    // Transform operations
    Transform transform_identity();
    Transform transform_create(Vector3 pos, Quaternion rot, Vector3 scale);
    Transform transform_combine(Transform parent, Transform child);
    Vector3 transform_point(Transform t, Vector3 p);
    Vector3 transform_direction(Transform t, Vector3 d);

    // Bounding volume operations
    AABB aabb_from_points(const Vector3* points, size_t count);
    AABB aabb_combine(AABB a, AABB b);
    bool aabb_contains_point(AABB box, Vector3 point);
    bool aabb_intersects(AABB a, AABB b);
    Vector3 aabb_center(AABB box);
    Vector3 aabb_size(AABB box);

    Sphere sphere_from_points(const Vector3* points, size_t count);
    bool sphere_contains_point(Sphere s, Vector3 point);
    bool sphere_intersects_sphere(Sphere a, Sphere b);

    // Ray operations
    Ray ray_create(Vector3 origin, Vector3 direction);
    bool ray_intersect_aabb(Ray ray, AABB box, RayHit* hit);
    bool ray_intersect_sphere(Ray ray, Sphere sphere, RayHit* hit);
    Vector3 ray_at(Ray ray, double t);

    // Callback registration
    void register_transform_callback(TransformCallback callback);
    void register_intersection_test(IntersectionTest test);
}
