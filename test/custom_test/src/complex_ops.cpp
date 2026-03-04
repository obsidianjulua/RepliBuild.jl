#include "complex_ops.h"

extern "C" {

void init_register(struct HardwareRegister* reg) {
    if (reg) {
        reg->enable = 1;
        reg->mode = 5;
        reg->reserved = 0;
        reg->payload = 42000;
    }
}

uint32_t read_payload(struct HardwareRegister* reg) {
    if (reg) {
        return reg->payload;
    }
    return 0;
}

void set_float_variant(union VariantValue* v, float f) {
    if (v) {
        v->as_float = f;
    }
}

double get_double_variant(union VariantValue* v) {
    if (v) {
        return v->as_double;
    }
    return 0.0;
}

int apply_callback(int x, int y, BinaryOpCallback cb) {
    if (cb) {
        return cb(x, y);
    }
    return 0;
}

}
