#ifndef C_ABOMINATION_H
#define C_ABOMINATION_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>   /* FILE — see §8 at the bottom */

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

// 6. Tagged union: a NAMED member whose TYPE is an anonymous union.
// Distinct from the C11 anonymous members above (which have no member name and
// inject their fields into the enclosing scope) — here the member is `u`, and
// only the union TYPE is unnamed. This is the shape that made tomlc17's
// toml_datum_t a 40-byte blob with zero named fields: DWARF carried the whole
// member tree, but an unnamed aggregate DIE was dropped on export, so the
// member typed as `Any` and dragged the entire enclosing struct into a blob.
typedef enum { TAG_NONE = 0, TAG_INT = 1, TAG_DBL = 2, TAG_STR = 3 } ValueTag;

typedef struct {
    ValueTag tag;          // offset 0  (enum, 4 bytes)
    uint32_t flags;        // offset 4
    union {                // offset 8, 16 bytes, align 8
        int64_t i;
        double d;
        const char* s;
        struct { const char* ptr; int len; } str;
    } u;
} TaggedValue;             // 24 bytes total

TaggedValue make_tagged_int(int64_t v);
TaggedValue make_tagged_double(double v);
TaggedValue make_tagged_str(const char* s, int len);
int tagged_is(const TaggedValue* t, ValueTag k);

// 7. ALL-FLOAT anonymous union, passed and returned BY VALUE.
// SysV classifies an eightbyte as SSE only when every field overlapping it is
// float/double, so this union travels in XMM. An opaque region standing in for
// it must therefore use a FLOAT element type — an integer one would claim
// INTEGER class and the value would be read out of the wrong register file.
// FloatBox is 16 bytes: eightbyte 0 = SSE (the union), eightbyte 1 = INTEGER
// (kind), so a wrong region type is a wrong VALUE, not just a wrong type.
typedef struct {
    union { float f; double d; } v;   // offset 0, 8 bytes, align 8
    int kind;                         // offset 8
} FloatBox;                           // 16 bytes

FloatBox floatbox_make(double d, int kind);
double floatbox_get(FloatBox b);
int floatbox_kind(FloatBox b);

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

// 8. A libc type in the API surface: `FILE` resolves through DWARF to
// `struct _IO_FILE`, which is on _INTERNAL_TYPE_BLOCKLIST and is therefore
// never declared by the generator. Before 2026-08-02 the blocklist suppressed
// the DECLARATION but not the USES, so `Ptr{_IO_FILE}` landed in the ccall
// signature and the whole module raised UndefVarError at include — every
// function dead, not just this one. Found live on miniaudio's ma_fopen.
// The reproduction needs FILE** specifically, which is why miniaudio's
// ma_fopen(FILE** ppFile, …) found it: a single FILE* is already degraded to
// Ptr{Cvoid} by the type mapper, but the DOUBLE pointer survives as
// Ptr{Ptr{_IO_FILE}} and reaches the ccall signature intact. Verified by
// disabling the fix and watching only the ** form trip the guard.
long stream_open(FILE** out, const char* path);
long stream_write_tag(FILE* out, const char* tag);
FILE* stream_null(void);

#endif // C_ABOMINATION_H
