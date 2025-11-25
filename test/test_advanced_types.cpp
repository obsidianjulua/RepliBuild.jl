// test_advanced_types.cpp
// Test C++ file to validate enum, array, and function pointer extraction from DWARF

#include <cmath>

// ========== ENUMS ==========

// Basic enum (C-style, unscoped)
enum Color {
    Red = 0,
    Green = 1,
    Blue = 2
};

// Scoped enum with explicit type
enum class Status : unsigned int {
    Idle = 0,
    Running = 100,
    Stopped = 200,
    Error = 999
};

// Enum with negative values
enum class Direction : int {
    North = 1,
    South = -1,
    East = 2,
    West = -2
};

// ========== ARRAYS ==========

// Struct with fixed-size arrays
struct Matrix3x3 {
    double data[9];  // Flattened 3x3 matrix
};

// Struct with multi-dimensional array
struct Grid {
    int cells[4][4];  // 4x4 grid
    double values[3];  // Simple 1D array
};

// ========== FUNCTION POINTERS ==========

// C-style callback
typedef int (*IntCallback)(double x, double y);

// Function pointer in struct
struct Callbacks {
    IntCallback process;
    void (*cleanup)();
};

// ========== TEST FUNCTIONS ==========

// Function returning enum
Color get_primary_color() {
    return Red;
}

// Function taking enum parameter
int color_to_int(Color c) {
    return static_cast<int>(c);
}

// Function with scoped enum
Status check_status(Status s) {
    return s;
}

// Function creating matrix
Matrix3x3 create_identity_matrix() {
    Matrix3x3 m;
    for (int i = 0; i < 9; i++) {
        m.data[i] = (i % 4 == 0) ? 1.0 : 0.0;
    }
    return m;
}

// Function accessing array
double matrix_sum(Matrix3x3 m) {
    double sum = 0.0;
    for (int i = 0; i < 9; i++) {
        sum += m.data[i];
    }
    return sum;
}

// Function with grid
int grid_get(Grid g, int row, int col) {
    if (row < 4 && col < 4) {
        return g.cells[row][col];
    }
    return -1;
}

// Function using callback (function pointer)
int apply_callback(IntCallback cb, double x, double y) {
    if (cb) {
        return cb(x, y);
    }
    return 0;
}

// Sample callback implementation
int add_callback(double x, double y) {
    return static_cast<int>(x + y);
}

// ========== MIXED TYPES ==========

// Struct combining everything
struct ComplexType {
    Color color;               // Enum
    Status status;             // Scoped enum
    double coords[3];          // Array
    IntCallback handler;       // Function pointer
    int matrix[2][3];          // 2D array
};

// Function using complex type
ComplexType create_complex(Color c, Status s, IntCallback cb) {
    ComplexType ct;
    ct.color = c;
    ct.status = s;
    ct.handler = cb;
    ct.coords[0] = 1.0;
    ct.coords[1] = 2.0;
    ct.coords[2] = 3.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            ct.matrix[i][j] = i * 3 + j;
        }
    }
    return ct;
}

// ========== VALIDATION ==========

// Simple test to ensure everything works
int run_tests() {
    // Test enum
    Color c = get_primary_color();
    int c_val = color_to_int(c);

    // Test scoped enum
    Status s = check_status(Status::Running);

    // Test matrix
    Matrix3x3 m = create_identity_matrix();
    double sum = matrix_sum(m);

    // Test grid
    Grid g;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            g.cells[i][j] = i * 4 + j;
        }
    }
    int val = grid_get(g, 2, 2);

    // Test callback
    int result = apply_callback(add_callback, 10.0, 20.0);

    // Test complex type
    ComplexType ct = create_complex(Red, Status::Idle, add_callback);

    return (c_val == 0 && sum == 3.0 && val == 10 && result == 30) ? 0 : 1;
}
