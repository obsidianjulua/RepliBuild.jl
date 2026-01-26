#pragma once

#include <stdint.h>

// ============================================================================
// STRESS TEST: Basics (Struct Layouts, Global Variables)
// ============================================================================

extern "C" {

// 1. Struct with padding
// C layout: 1 byte (a), 3 bytes padding, 4 bytes (b) -> size 8
struct PaddedStruct {
    char a;
    int b;
};

// 2. Packed struct
// C layout: 1 byte (a), 4 bytes (b) -> size 5
struct __attribute__((packed)) PackedStruct {
    char a;
    int b;
};

// 3. Union
// C layout: max(size(int), size(float)) = 4 bytes
union NumberUnion {
    int i;
    float f;
};

// 4. Global Variables
extern int global_int;
extern const char* global_string;

// Function to manipulate these
void process_padded(PaddedStruct s);
void process_packed(PackedStruct s);
void process_union(NumberUnion u);

PaddedStruct make_padded(char a, int b);
PackedStruct make_packed(char a, int b);

}
