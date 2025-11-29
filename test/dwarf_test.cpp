// Comprehensive C++ test file for DWARF extraction testing
// Tests: primitives, structs, classes, enums, templates, inheritance, arrays, pointers, references

#include <cstdint>
#include <string>
#include <vector>
#include <map>

// ============================================================================
// ENUMERATIONS
// ============================================================================

// Simple enum
enum Color {
    RED = 0,
    GREEN = 1,
    BLUE = 2
};

// Enum class (C++11)
enum class Status : uint8_t {
    IDLE = 0,
    RUNNING = 10,
    STOPPED = 20,
    ERROR = 255
};

// Enum with explicit underlying type
enum Priority : int32_t {
    LOW = -100,
    MEDIUM = 0,
    HIGH = 100,
    CRITICAL = 1000
};

// ============================================================================
// SIMPLE STRUCTS
// ============================================================================

struct Point2D {
    double x;
    double y;
};

struct Point3D {
    double x;
    double y;
    double z;
};

// Struct with various types
struct DataRecord {
    int32_t id;
    float value;
    bool active;
    char tag;
    uint64_t timestamp;
};

// Nested struct
struct BoundingBox {
    Point2D min;
    Point2D max;
};

// Struct with arrays
struct Grid {
    int rows;
    int cols;
    double data[10];
    int indices[5][5];  // Multi-dimensional array
};

// ============================================================================
// CLASSES
// ============================================================================

class Vector3D {
public:
    double x, y, z;

    Vector3D();
    Vector3D(double x, double y, double z);
    ~Vector3D();

    double length() const;
    Vector3D normalize() const;
    Vector3D operator+(const Vector3D& other) const;
    Vector3D& operator+=(const Vector3D& other);
};

// Class with inheritance
class Shape {
protected:
    Color color;
    int id;

public:
    Shape(Color c, int i);
    virtual ~Shape();
    virtual double area() const = 0;  // Pure virtual
    virtual double perimeter() const = 0;
    Color getColor() const;
};

class Circle : public Shape {
private:
    double radius;
    Point2D center;

public:
    Circle(double r, Point2D c, Color col);
    double area() const override;
    double perimeter() const override;
    double getRadius() const;
};

class Rectangle : public Shape {
private:
    double width;
    double height;
    Point2D origin;

public:
    Rectangle(double w, double h, Point2D o, Color col);
    double area() const override;
    double perimeter() const override;
};

// ============================================================================
// TEMPLATE STRUCTURES (instantiated)
// ============================================================================

template<typename T>
struct Pair {
    T first;
    T second;
};

// Force template instantiation
using IntPair = Pair<int>;
using DoublePair = Pair<double>;

template<typename T, int N>
struct FixedArray {
    T data[N];
    int size;
};

using FloatArray10 = FixedArray<float, 10>;

// ============================================================================
// FUNCTION POINTERS AND CALLBACKS
// ============================================================================

typedef int (*BinaryOp)(int, int);
typedef void (*Callback)(void* userdata, const char* message);

struct EventHandler {
    Callback callback;
    void* userdata;
    int priority;
};

// ============================================================================
// UNIONS
// ============================================================================

union Value {
    int32_t i;
    float f;
    double d;
    void* p;
};

struct Tagged {
    enum { INT, FLOAT, DOUBLE, POINTER } tag;
    Value value;
};

// ============================================================================
// BITFIELDS
// ============================================================================

struct Flags {
    unsigned int enabled : 1;
    unsigned int visible : 1;
    unsigned int locked : 1;
    unsigned int unused : 5;
    unsigned int priority : 8;
};

// ============================================================================
// COMPLEX NESTING
// ============================================================================

struct Scene {
    BoundingBox bounds;
    std::vector<Circle> circles;
    std::vector<Rectangle> rectangles;
    std::map<int, Point3D> landmarks;
    Flags flags;
};

// ============================================================================
// FUNCTION SIGNATURES - PRIMITIVES
// ============================================================================

// Basic types
int add(int a, int b) { return a + b; }
double multiply(double x, double y) { return x * y; }
bool is_positive(int x) { return x > 0; }
void do_nothing() { }
char get_char(const char* str, int index) { return str[index]; }

// Fixed-width types
int8_t add_i8(int8_t a, int8_t b) { return a + b; }
uint32_t combine(uint16_t hi, uint16_t lo) { return (uint32_t(hi) << 16) | lo; }
int64_t mul64(int32_t a, int32_t b) { return int64_t(a) * int64_t(b); }

// ============================================================================
// FUNCTION SIGNATURES - POINTERS
// ============================================================================

// Pointer parameters
int* create_array(int size);
void fill_array(int* arr, int size, int value);
const char* get_name();

// Double pointers
void allocate_matrix(double** matrix, int rows, int cols);

// Void pointers
void* get_opaque_handle();
void process_data(void* data, int size);

// ============================================================================
// FUNCTION SIGNATURES - REFERENCES
// ============================================================================

// Const references
double dot(const Vector3D& a, const Vector3D& b);
void normalize_in_place(Vector3D& v);

// Return by reference
Vector3D& get_global_vector();

// ============================================================================
// FUNCTION SIGNATURES - STRUCTS/CLASSES
// ============================================================================

// Return struct by value
Point2D midpoint(Point2D a, Point2D b);
Vector3D cross(Vector3D a, Vector3D b);
BoundingBox compute_bounds(const Point2D* points, int count);

// Struct parameters
double distance(Point2D a, Point2D b);
Circle create_circle(Point2D center, double radius, Color color);

// ============================================================================
// FUNCTION SIGNATURES - ENUMS
// ============================================================================

Color blend_colors(Color a, Color b);
Status get_status();
void set_priority(Priority p);
bool is_high_priority(Priority p) { return p >= HIGH; }

// ============================================================================
// FUNCTION SIGNATURES - ARRAYS
// ============================================================================

// Array parameters
void process_values(const double values[], int count);
void init_grid(Grid* grid, int rows, int cols);

// Return array (as pointer)
int* generate_sequence(int start, int count);

// ============================================================================
// FUNCTION SIGNATURES - TEMPLATES (instantiated)
// ============================================================================

IntPair make_int_pair(int a, int b) {
    IntPair p;
    p.first = a;
    p.second = b;
    return p;
}

FloatArray10 create_float_array() {
    FloatArray10 arr;
    arr.size = 10;
    return arr;
}

// ============================================================================
// FUNCTION SIGNATURES - FUNCTION POINTERS
// ============================================================================

int apply_binary_op(int a, int b, BinaryOp op);
void register_callback(Callback cb, void* userdata);
BinaryOp get_addition_op();

// ============================================================================
// FUNCTION SIGNATURES - CONST CORRECTNESS
// ============================================================================

const Point2D* get_const_point();
void modify_point(Point2D* p);
const char* get_string() { return "test"; }
int count_chars(const char* str);

// ============================================================================
// FUNCTION SIGNATURES - VOLATILE
// ============================================================================

volatile int* get_volatile_ptr();
void write_volatile(volatile int* ptr, int value);

// ============================================================================
// FUNCTION SIGNATURES - VARIADIC (C-style)
// ============================================================================

void log_message(const char* format, ...);
int sum_ints(int count, ...);

// ============================================================================
// METHOD SIGNATURES
// ============================================================================

// Vector3D methods
Vector3D::Vector3D() : x(0), y(0), z(0) {}
Vector3D::Vector3D(double x, double y, double z) : x(x), y(y), z(z) {}
Vector3D::~Vector3D() {}

double Vector3D::length() const {
    return 0.0;  // Placeholder
}

Vector3D Vector3D::normalize() const {
    return Vector3D(x, y, z);
}

Vector3D Vector3D::operator+(const Vector3D& other) const {
    return Vector3D(x + other.x, y + other.y, z + other.z);
}

Vector3D& Vector3D::operator+=(const Vector3D& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

// Shape methods
Shape::Shape(Color c, int i) : color(c), id(i) {}
Shape::~Shape() {}
Color Shape::getColor() const { return color; }

// Circle methods
Circle::Circle(double r, Point2D c, Color col) : Shape(col, 0), radius(r), center(c) {}
double Circle::area() const { return 3.14159 * radius * radius; }
double Circle::perimeter() const { return 2 * 3.14159 * radius; }
double Circle::getRadius() const { return radius; }

// Rectangle methods
Rectangle::Rectangle(double w, double h, Point2D o, Color col)
    : Shape(col, 0), width(w), height(h), origin(o) {}
double Rectangle::area() const { return width * height; }
double Rectangle::perimeter() const { return 2 * (width + height); }

// ============================================================================
// STATIC AND INLINE FUNCTIONS
// ============================================================================

static int internal_helper(int x) { return x * 2; }
inline int fast_max(int a, int b) { return a > b ? a : b; }

// ============================================================================
// EXTERN "C" FUNCTIONS
// ============================================================================

extern "C" {
    int c_add(int a, int b);
    void c_process(const char* msg);
}

int c_add(int a, int b) { return a + b; }
void c_process(const char* msg) { }

// ============================================================================
// NAMESPACES
// ============================================================================

namespace math {
    double pi() { return 3.14159; }
    double deg_to_rad(double deg) { return deg * 3.14159 / 180.0; }
}

namespace utils {
    int clamp(int value, int min, int max) {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }
}

// ============================================================================
// OVERLOADED FUNCTIONS
// ============================================================================

int abs(int x) { return x < 0 ? -x : x; }
double abs(double x) { return x < 0.0 ? -x : x; }
float abs(float x) { return x < 0.0f ? -x : x; }

// ============================================================================
// IMPLEMENTATION STUBS
// ============================================================================

int* create_array(int size) { return nullptr; }
void fill_array(int* arr, int size, int value) {}
const char* get_name() { return "test"; }
void allocate_matrix(double** matrix, int rows, int cols) {}
void* get_opaque_handle() { return nullptr; }
void process_data(void* data, int size) {}
double dot(const Vector3D& a, const Vector3D& b) { return 0.0; }
void normalize_in_place(Vector3D& v) {}
Vector3D& get_global_vector() { static Vector3D v; return v; }
Point2D midpoint(Point2D a, Point2D b) { Point2D p; return p; }
Vector3D cross(Vector3D a, Vector3D b) { return Vector3D(); }
BoundingBox compute_bounds(const Point2D* points, int count) { BoundingBox b; return b; }
double distance(Point2D a, Point2D b) { return 0.0; }
Circle create_circle(Point2D center, double radius, Color color) { return Circle(radius, center, color); }
Color blend_colors(Color a, Color b) { return RED; }
Status get_status() { return Status::IDLE; }
void set_priority(Priority p) {}
void process_values(const double values[], int count) {}
void init_grid(Grid* grid, int rows, int cols) {}
int* generate_sequence(int start, int count) { return nullptr; }
int apply_binary_op(int a, int b, BinaryOp op) { return op(a, b); }
void register_callback(Callback cb, void* userdata) {}
BinaryOp get_addition_op() { return nullptr; }
const Point2D* get_const_point() { return nullptr; }
void modify_point(Point2D* p) {}
int count_chars(const char* str) { return 0; }
volatile int* get_volatile_ptr() { return nullptr; }
void write_volatile(volatile int* ptr, int value) {}
void log_message(const char* format, ...) {}
int sum_ints(int count, ...) { return 0; }
