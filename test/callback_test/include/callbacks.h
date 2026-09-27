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

// noexcept routing (test_exceptions.jl, "bare-name noexcept collision"). The
// routing used to trust a regex returning BARE names, so a throwing function
// that shares a name with a noexcept one went to a plain ccall, and its throw
// aborted the process. `noexcept(false)` matched the same regex.
namespace collide {
struct Quiet { int value(int x) const noexcept; };  // really noexcept
struct Loud  { int value(int x) const; };           // same bare name, THROWS
int checked(int x) noexcept(false);                 // NOT noexcept
}

#endif // __cplusplus

#endif // CALLBACKS_H