#include "basics.h"
#include <iostream>
#include <cstdarg>

extern "C" {

int global_int = 42;
const char* global_string = "Hello from Global";

void process_padded(PaddedStruct s) {
    std::cout << "Padded: a=" << (int)s.a << ", b=" << s.b << std::endl;
}

void process_packed(PackedStruct s) {
    std::cout << "Packed: a=" << (int)s.a << ", b=" << s.b << std::endl;
}

void process_union(NumberUnion u) {
    std::cout << "Union: i=" << u.i << ", f=" << u.f << std::endl;
}

PaddedStruct make_padded(char a, int b) {
    PaddedStruct s;
    s.a = a;
    s.b = b;
    return s;
}

PackedStruct make_packed(char a, int b) {
    PackedStruct s;
    s.a = a;
    s.b = b;
    return s;
}

int sum_ints(int count, ...) {
    va_list args;
    va_start(args, count);
    int sum = 0;
    for (int i = 0; i < count; i++) {
        sum += va_arg(args, int);
    }
    va_end(args);
    return sum;
}

}
