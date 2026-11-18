// Fixture for test_cpp_globals_live.jl: C++ globals outside the global
// namespace (B19, 2026-11-18). Every global here is exported under a mangled
// symbol except `g_top`, and two of them share the bare name `x`.
#pragma once
#include <string>

extern int g_top;                                    // 9, the global scope
namespace a { extern int x; }                        // 7
namespace b { extern int x; }                        // 8
namespace cfg { extern int depth; extern const double scale; constexpr int LIMIT = 64; }
extern std::string g_string;                         // "global-string"

struct Stat {
    static int count;                                // bumped by bump()
    static const double factor;                      // 2.5
    static constexpr int K = 9;                      // no storage, no symbol
    static int bump();
protected:
    static int prot_;                                // no accessor
};

namespace geo {
class Vec3 {
public:
    double x, y, z;
    Vec3();
    ~Vec3();
    static int live_count;                           // constructions less destructions
};
class Counter {
    static int total_;                               // private by default: no accessor
public:
    static int total();
};
Vec3 *vec3_new();
void vec3_free(Vec3 *v);
}
