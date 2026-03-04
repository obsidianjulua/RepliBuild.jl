#ifndef COMPLEX_OPS_H
#define COMPLEX_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 1. Bitfields - Impossible to map cleanly in basic C ccall
struct HardwareRegister {
    uint32_t enable : 1;
    uint32_t mode : 3;
    uint32_t reserved : 12;
    uint32_t payload : 16;
};

void init_register(struct HardwareRegister* reg);
uint32_t read_payload(struct HardwareRegister* reg);

// 2. Union - overlapping memory layouts, needs typed accessors
union VariantValue {
    int as_int;
    float as_float;
    double as_double;
};

void set_float_variant(union VariantValue* v, float f);
double get_double_variant(union VariantValue* v);

// 3. Callback (Function Pointer)
typedef int (*BinaryOpCallback)(int, int);
int apply_callback(int x, int y, BinaryOpCallback cb);

#ifdef __cplusplus
}
#endif

#endif // COMPLEX_OPS_H
