#include "callbacks.h"

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