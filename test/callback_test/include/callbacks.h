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
#endif

#endif // CALLBACKS_H