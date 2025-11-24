// Simplified math operations without Eigen dependency

struct Vector3d {
    double x, y, z;
};

struct Matrix3d {
    double data[9];  // 3x3 row-major
};

Vector3d vec3_create(double x, double y, double z) {
    return {x, y, z};
}

Vector3d vec3_add(Vector3d a, Vector3d b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

double vec3_dot(Vector3d a, Vector3d b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

Vector3d vec3_cross(Vector3d a, Vector3d b) {
    return {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}

double vec3_norm(Vector3d v) {
    return __builtin_sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

Matrix3d mat3_identity() {
    return {1,0,0, 0,1,0, 0,0,1};
}

Vector3d mat3_mul_vec(Matrix3d m, Vector3d v) {
    return {
        m.data[0]*v.x + m.data[1]*v.y + m.data[2]*v.z,
        m.data[3]*v.x + m.data[4]*v.y + m.data[5]*v.z,
        m.data[6]*v.x + m.data[7]*v.y + m.data[8]*v.z
    };
}
