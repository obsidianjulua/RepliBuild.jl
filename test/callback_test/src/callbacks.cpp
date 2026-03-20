#include "callbacks.h"
#include <stdexcept>
#include <string>

int execute_binary_op(BinaryOp op, int a, int b) {
    if (!op) return 0;

    // Call the provided function pointer (which could be a Julia JIT function)
    return op(a, b);
}

void simulate_work(int iterations, ProgressCallback cb) {
    if (!cb) return;

    // Loop and call back multiple times
    for (int i = 1; i <= iterations; ++i) {
        float progress = (float)i / iterations;
        cb(progress);
    }
}

// =============================================================================
// C++ exception test functions
// =============================================================================

int always_throws(int x) {
    throw std::runtime_error("always_throws called with x=" + std::to_string(x));
    return x; // unreachable
}

int throws_if_negative(int x) {
    if (x < 0) {
        throw std::runtime_error("negative value: " + std::to_string(x));
    }
    return x * 2;
}

int throws_int(int x) {
    throw 42;
    return x; // unreachable
}

void void_thrower() {
    throw std::runtime_error("void function threw");
}

int safe_multiply(int a, int b) noexcept {
    return a * b;
}

int throws_midway(int iterations) {
    int sum = 0;
    for (int i = 0; i < iterations; ++i) {
        if (i == iterations / 2) {
            throw std::runtime_error("threw at iteration " + std::to_string(i));
        }
        sum += i;
    }
    return sum;
}