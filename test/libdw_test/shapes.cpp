#include <cstdint>
namespace geo {
enum class Kind : std::uint8_t { A = 1, B = 200 };
enum Big : unsigned long long { HUGE = 0xffff000000000000ULL, NEG = 5 };
struct Base { virtual ~Base(); int id; };
struct Vec { float x, y; unsigned flag : 3; unsigned rest : 5; double d[4]; static int count; };
struct Shape : Base { Vec v; Kind k; int (*cb)(int, const char*); int area(int scale = 2) const; static Shape make(); };
Base::~Base() {}
int Vec::count = 3;
int Shape::area(int scale) const { return id * scale; }
Shape Shape::make() { return Shape(); }
int free_fn(const Vec& v, Kind k, long double ld, ...) { return (int)v.x + (int)k; }
template <class T, int N> struct Arr { T data[N]; };
Arr<int, 3> g_arr;
const volatile int g_cv = 4;
typedef union { int i; float f; } U;
U g_u;
}
