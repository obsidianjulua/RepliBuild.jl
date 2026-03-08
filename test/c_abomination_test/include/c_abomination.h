#ifndef C_ABOMINATION_H
#define C_ABOMINATION_H

#include <stdint.h>
#include <stddef.h>

// 1. Deeply nested anonymous structs and unions
typedef struct {
    int id;
    union {
        struct {
            float x, y, z;
        };
        double raw_data[3];
        struct {
            uint8_t a, b, c, d;
            union {
                uint32_t flags;
                struct {
                    uint8_t f1 : 1;
                    uint8_t f2 : 3;
                    uint8_t f3 : 4;
                };
            };
        } complex_inner;
    };
    char tag[16];
} NightmareStruct;

// 2. Multi-dimensional arrays of structs
typedef struct {
    NightmareStruct matrix[4][4];
    int dimensions[2];
} NightmareMatrix;

// 3. Obscure function pointer typedefs
// A function taking an int and returning a pointer to a function that takes a float and returns a double
typedef double (*InnerFunc)(float);
typedef InnerFunc (*OuterFunc)(int);

// 4. Struct containing function pointers to itself and anonymous types
struct SelfReferential {
    struct SelfReferential* next;
    void (*process)(struct SelfReferential*);
    
    // Function taking a function pointer
    int (*execute_callback)(int (*cb)(NightmareStruct*), NightmareStruct* data);
};

// 5. Array of pointers to arrays
typedef int (*ArrayPtr)[10];

// API Functions to test

// Pass by value of a massive struct
NightmareStruct create_nightmare(int id, float x, float y, float z);

// Modify by pointer
void mutate_nightmare(NightmareStruct* n);

// Function pointer heavy API
double execute_outer(OuterFunc f, int a, float b);

// Opaque struct simulation
typedef struct OpaqueState OpaqueState;

OpaqueState* init_opaque(void);
void process_opaque(OpaqueState* state, struct SelfReferential* self_ref);
void free_opaque(OpaqueState* state);

#endif // C_ABOMINATION_H
