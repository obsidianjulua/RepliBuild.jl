// test/aot_test/src/aot_api.cpp — AOT thunk fixture (test_aot_thunks.jl).
//
// Every C++ function here is mangled and not `noexcept`, so `is_ccall_safe`
// routes it to Tier 2. Built with `[compile] aot_thunks = true`, each one is
// called through lib<name>_thunks.<dlext> instead of a JIT. The extern "C"
// factories at the bottom are Tier 3 on purpose: a mixed wrapper emits the
// DISPATCH_TIER table, so the test can tell the two tiers apart.
//
// The by-value records cover the three ways a record crosses on each host:
//   Pair32  8 bytes  — one INTEGER eightbyte (SysV); a register (Win64)
//   Vec2   16 bytes  — two SSE eightbytes (SysV); by reference + sret (Win64)
//   Vec3   24 bytes  — MEMORY on both
//
// The throwing functions cover one return shape each, because on Windows each
// shape is caught by a different libJLCS guard (jlcs_guard_i32 / _void / _f64
// / _ptr / _sret): checked_add, set_total, safe_ratio, make_square, vec2_unit.
// throw_int throws something that is not a std::exception.
//
// No STL type appears, not even inside a body: messages are formatted into a
// char buffer and thrown through runtime_error(const char*), which libc++
// defines out of line. STL thunks on Windows/libc++ 23 are a separate gap.
//
// Constructors and destructors are defined out of line so no inline special
// member reaches the wrap surface.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

struct Pair32 {
    int32_t a;
    int32_t b;
};

struct Vec2 {
    double x;
    double y;
};

struct Vec3 {
    double x;
    double y;
    double z;
};

// ── Plain member functions ──────────────────────────────────────────────────

class Accumulator {
public:
    explicit Accumulator(int32_t start);
    ~Accumulator();

    int32_t add(int32_t v);            // mutates, returns the new total
    int32_t total() const;
    Vec2 as_vec2() const;              // member returning a record by value: {total, -total}

    int32_t checked_add(int32_t v);    // throws on v < 0, total unchanged (i32 return)
    void set_total(int32_t v);         // throws on v < 0, total unchanged (void return)

    int32_t total_;
};

Accumulator::Accumulator(int32_t start) : total_(start) {}
Accumulator::~Accumulator() {}

int32_t Accumulator::add(int32_t v) {
    total_ += v;
    return total_;
}

int32_t Accumulator::total() const { return total_; }

Vec2 Accumulator::as_vec2() const {
    return Vec2{static_cast<double>(total_), -static_cast<double>(total_)};
}

int32_t Accumulator::checked_add(int32_t v) {
    if (v < 0) {
        char msg[96];
        std::snprintf(msg, sizeof msg, "checked_add: negative v=%d", static_cast<int>(v));
        throw std::runtime_error(msg);
    }
    return add(v);
}

void Accumulator::set_total(int32_t v) {
    if (v < 0) {
        char msg[96];
        std::snprintf(msg, sizeof msg, "set_total: negative v=%d", static_cast<int>(v));
        throw std::runtime_error(msg);
    }
    total_ = v;
}

// ── Virtual dispatch ────────────────────────────────────────────────────────
// Shape's own answers are sentinels, so a thunk that called the statically
// named implementation instead of dispatching through the vtable is visible:
// Shape_area on a Square would give -1.0 instead of side * side.

class Shape {
public:
    Shape();
    virtual ~Shape();
    virtual double area() const;
    virtual int32_t sides() const;
    double area_times(double k) const;   // non-virtual; dispatches from C++
};

class Square : public Shape {
public:
    explicit Square(double s);
    ~Square() override;
    double area() const override;
    int32_t sides() const override;
    double side;
};

Shape::Shape() {}
Shape::~Shape() {}
double Shape::area() const { return -1.0; }
int32_t Shape::sides() const { return 0; }
double Shape::area_times(double k) const { return area() * k; }

Square::Square(double s) : side(s) {}
Square::~Square() {}
double Square::area() const { return side * side; }
int32_t Square::sides() const { return 4; }

// ── By-value records ────────────────────────────────────────────────────────

Pair32 pair_swap(Pair32 p) { return Pair32{p.b, p.a}; }

Vec2 vec2_scale(Vec2 v, double k) { return Vec2{v.x * k, v.y * k}; }

double vec2_dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }

Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return Vec3{a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x};
}

// ── Exceptions crossing the thunk, one return shape each ───────────────────

double safe_ratio(double a, double b) {
    if (b == 0.0) {
        char msg[96];
        std::snprintf(msg, sizeof msg, "safe_ratio: zero divisor for a=%g", a);
        throw std::runtime_error(msg);
    }
    return a / b;
}

Square* make_square(double side) {
    if (!(side > 0.0)) {
        char msg[96];
        std::snprintf(msg, sizeof msg, "make_square: side must be positive, got %g", side);
        throw std::runtime_error(msg);
    }
    return new Square(side);
}

Vec2 vec2_unit(Vec2 v) {
    double n = std::sqrt(v.x * v.x + v.y * v.y);
    if (n == 0.0)
        throw std::runtime_error("vec2_unit: zero-length vector");
    return Vec2{v.x / n, v.y / n};
}

int32_t throw_int(int32_t code) {
    if (code != 0)
        throw code;   // not a std::exception
    return 0;
}

// ── Tier 3 factories ────────────────────────────────────────────────────────

extern "C" {

Accumulator* aot_new_accumulator(int32_t start) { return new Accumulator(start); }
void aot_free_accumulator(Accumulator* a) { delete a; }

Shape* aot_new_shape() { return new Shape(); }
Shape* aot_new_square_as_shape(double side) { return new Square(side); }
void aot_free_shape(Shape* s) { delete s; }

}
