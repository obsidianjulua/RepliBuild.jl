#pragma once

#include <cstddef>
#include <string_view>

class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const { return 0.0; }
    virtual double perimeter() const { return 0.0; }
};

class Rectangle : public Shape {
    double width, height;
public:
    Rectangle(double w, double h);
    double area() const override;
    double perimeter() const override;
};

class Circle : public Shape {
    double radius;
public:
    Circle(double r);
    double area() const override;
    double perimeter() const override;
};

// A constructor at GLOBAL scope. There `class` is the bare name, and the
// generator's old "name == class, skip it" rule dropped exactly these, silently,
// while geom::Segment2D's constructor below was emitted (audit F25, 2026-09-26).
// Emitted as `Span_Span`, the same naming as `geom_Segment2D_Segment2D`.
class Span {
public:
    Span(int lo, int hi);
    int width() const;
    int lo, hi;
};

// A constructor taking value-type structs BY VALUE. Constructors are named C4 in
// DWARF and C1/C2 in the symbol table; until 2026-09-26 that join always missed,
// the signature was guessed from the demangled string (`Point2D` → `Any`), and the
// thunk stored garbage. See verify.jl, "constructor by-value struct arguments".
// In a namespace; `Span` above is the global-scope constructor.
namespace geom {
struct Point2D { double x, y; };

class Segment2D {
public:
    Segment2D(Point2D from, Point2D to);
    double length2() const;
    Point2D a, b;
};

// By-value records the Tier-2 thunk used to type as a pointer. `Unit` is an
// empty class: SysV gives it no register, so each argument after one moved a
// register over. `Cell<double>` is a template record, whose spelling was never
// looked up: its bytes went in as a pointer and it came back from RAX while
// the callee wrote XMM0. `std::string_view` has no layout in the metadata (a
// toolchain type), so the wrapper refuses the call. See verify.jl.
struct Unit {};
int unit_scaled(Unit u, int x, Unit v, int y);
Unit unit_make(int x);
template <typename T> struct Cell { T v; };
Cell<double> cell_doubled(Cell<double> c);
std::size_t name_length(std::string_view s);

// Non-trivial for the purposes of calls (a user destructor): returned through a
// hidden pointer at ANY size. The thunk read it from EAX, and the callee stored
// the object through the register holding `n` — a SIGSEGV for n = 42.
struct Ticket { int n; ~Ticket(); };
Ticket ticket_make(int n);

// Packed layouts. `Packed` is 5 bytes with `i` at offset 1; `Pack2` is
// `#pragma pack(2)`, 14 bytes at 0/2/6. Both were emitted at natural alignment
// (8 and 16 bytes), so pointer reads were garbage, and `Pack2` was an opaque
// byte array to the thunk, broken by value too.
struct Packed { char c; int i; } __attribute__((packed));
#pragma pack(push, 2)
struct Pack2 { char c; int i; double d; };
#pragma pack(pop)
Packed packed_make(int a);
double packed_sum(Packed p);
Pack2 pack2_make(int a);
double pack2_sum(Pack2 p);
const Packed *packed_table();          // {1,10}, {2,20}, {3,30}

// An anonymous union embedded in a record. As a `mutable struct` field the union
// was a reference, so `Variant` came out 24 bytes instead of 16 and `d` read back
// as garbage (audit F22, 2026-09-26). Anonymous unions are immutable byte
// regions now, inlined like C.
struct Variant { union { int i; float f; } u; double d; };
Variant variant_make(int a);           // {u.i = a, d = a + 0.25}
double variant_sum(Variant v);         // u.i + d

// A float-array member. DWARF gives the array no byte_size of its own and it was
// recorded as 0, so the struct counted as unsizable and came back as an integer
// byte blob while the callee returned it in XMM registers (audit F21, 2026-09-26).
struct Arr3 { float v[3]; };
Arr3 arr3_make(float a);               // {a, a + 1, a + 2}
double arr3_sum(Arr3 s);
}

extern "C" {
    Shape* create_rectangle(double w, double h);
    Shape* create_circle(double r);
    double get_area(const Shape* s);
    double get_perimeter(const Shape* s);
    void delete_shape(Shape* s);
}
