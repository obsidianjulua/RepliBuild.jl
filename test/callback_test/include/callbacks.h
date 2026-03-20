#ifndef CALLBACKS_H
#define CALLBACKS_H

#ifdef __cplusplus
extern "C" {
#endif

// Define function pointer types
typedef int (*BinaryOp)(int, int);
typedef void (*ProgressCallback)(float);

// C++ function that executes a callback and returns the result
int execute_binary_op(BinaryOp op, int a, int b);

// C++ function that simulates work and calls back to Julia multiple times
void simulate_work(int iterations, ProgressCallback cb);

#ifdef __cplusplus
}

// =============================================================================
// C++ exception test functions (NOT extern "C" — uses C++ ABI)
// =============================================================================

// Always throws std::runtime_error
int always_throws(int x);

// Throws std::runtime_error if x < 0, otherwise returns x * 2
int throws_if_negative(int x);

// Throws a non-std::exception type (plain int)
int throws_int(int x);

// Void function that throws
void void_thrower();

// Marked noexcept — should stay on fast ccall path
int safe_multiply(int a, int b) noexcept;

// Throws during iteration of a callback-like loop
int throws_midway(int iterations);

#endif // __cplusplus

#endif // CALLBACKS_H