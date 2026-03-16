// test/mlir_templates/src/templates.cpp
// C++ support library for JLCS dialect template stress tests.
// All functions are extern "C" for clean symbol resolution in JIT.

#include <cstdint>
#include <cstring>

extern "C" {

// ═══════════════════════════════════════════════════════════════════════════════
// 1. PairIntDouble — Pair<int, double> simulation
//    Layout: { int32_t first (offset 0), [4 pad], double second (offset 8) }
//    Total: 16 bytes
// ═══════════════════════════════════════════════════════════════════════════════

struct PairIntDouble {
    int32_t first;   // offset 0
    double  second;  // offset 8
};

void pair_int_double_ctor(PairIntDouble* p, int32_t a, double b) {
    p->first = a;
    p->second = b;
}

void pair_int_double_dtor(PairIntDouble* p) {
    p->first = -1;
    p->second = -1.0;  // sentinel values for dtor verification
}

PairIntDouble make_pair_int_double(int32_t a, double b) {
    PairIntDouble p;
    p.first = a;
    p.second = b;
    return p;
}

int32_t pair_get_first(PairIntDouble* p) {
    return p->first;
}

double pair_get_second(PairIntDouble* p) {
    return p->second;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. NestedPair — Pair<int, Pair<double, float>> simulation
//    Flattened layout:
//      int32_t first       at offset 0
//      [4 pad]
//      double inner_first  at offset 8
//      float  inner_second at offset 16
//      [4 pad]
//    Total: 24 bytes
// ═══════════════════════════════════════════════════════════════════════════════

struct PairDoubleFloat {
    double first;  // offset 0 within inner
    float  second; // offset 8 within inner
};

struct NestedPair {
    int32_t       first;   // offset 0
    PairDoubleFloat second; // offset 8 (aligned to 8)
};

void nested_pair_init(NestedPair* p, int32_t a, double b, float c) {
    p->first = a;
    p->second.first = b;
    p->second.second = c;
}

double nested_get_inner_first(NestedPair* p) {
    return p->second.first;
}

float nested_get_inner_second(NestedPair* p) {
    return p->second.second;
}

int32_t nested_get_outer_first(NestedPair* p) {
    return p->first;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. PackedTriple — PackedTriple<char, int, short> simulation
//    Layout (packed, no padding):
//      char    a at offset 0 (1 byte)
//      int32_t b at offset 1 (4 bytes)
//      int16_t c at offset 5 (2 bytes)
//    Total: 7 bytes
// ═══════════════════════════════════════════════════════════════════════════════

#pragma pack(push, 1)
struct PackedTriple {
    char    a;  // offset 0
    int32_t b;  // offset 1
    int16_t c;  // offset 5
};
#pragma pack(pop)

PackedTriple make_packed_triple(char a, int32_t b, int16_t c) {
    PackedTriple t;
    t.a = a;
    t.b = b;
    t.c = c;
    return t;
}

int32_t packed_triple_sum(PackedTriple* t) {
    return (int32_t)t->a + t->b + (int32_t)t->c;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. ContainerInt — Hand-rolled vtable for virtual dispatch testing
//    Object layout:
//      void** vptr     at offset 0 (pointer to vtable)
//      int32_t stored  at offset 8
//    Vtable layout (array of function pointers):
//      slot 0: get_value(void* self) -> int32_t
//      slot 1: set_value(void* self, int32_t val) -> void
// ═══════════════════════════════════════════════════════════════════════════════

struct ContainerInt {
    void** vptr;      // offset 0
    int32_t stored;   // offset 8
};

int32_t container_int_get_value(void* self) {
    ContainerInt* c = (ContainerInt*)self;
    return c->stored;
}

void container_int_set_value(void* self, int32_t val) {
    ContainerInt* c = (ContainerInt*)self;
    c->stored = val;
}

// Static vtable
typedef int32_t (*GetValueFn)(void*);
typedef void    (*SetValueFn)(void*, int32_t);

static void* container_int_vtable[] = {
    (void*)container_int_get_value,   // slot 0
    (void*)container_int_set_value    // slot 1
};

void container_int_init(ContainerInt* c, int32_t initial) {
    c->vptr = container_int_vtable;
    c->stored = initial;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. FixedArray4 — FixedArray<double, 4> simulation
//    Layout: { double data[4]; }
//    Offsets: [0, 8, 16, 24]
//    Total: 32 bytes (256 bits — triggers sret in FFECallOp lowering)
// ═══════════════════════════════════════════════════════════════════════════════

struct FixedArray4 {
    double data[4];
};

double fixed_array_sum(FixedArray4* arr) {
    return arr->data[0] + arr->data[1] + arr->data[2] + arr->data[3];
}

void fixed_array_fill(FixedArray4* arr, double val) {
    for (int i = 0; i < 4; i++)
        arr->data[i] = val;
}

FixedArray4 fixed_array_make(double a, double b, double c, double d) {
    FixedArray4 arr;
    arr.data[0] = a;
    arr.data[1] = b;
    arr.data[2] = c;
    arr.data[3] = d;
    return arr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. Scope logging — Tracks construction/destruction order for RAII tests
//    Global log array records events: positive = ctor, negative = dtor
// ═══════════════════════════════════════════════════════════════════════════════

static int32_t scope_log[8];
static int32_t scope_log_idx = 0;

void scope_log_reset() {
    scope_log_idx = 0;
    memset(scope_log, 0, sizeof(scope_log));
}

void scope_ctor_a(void* p) { scope_log[scope_log_idx++] = 1; }
void scope_ctor_b(void* p) { scope_log[scope_log_idx++] = 2; }
void scope_ctor_c(void* p) { scope_log[scope_log_idx++] = 3; }
void scope_dtor_a(void* p) { scope_log[scope_log_idx++] = -1; }
void scope_dtor_b(void* p) { scope_log[scope_log_idx++] = -2; }
void scope_dtor_c(void* p) { scope_log[scope_log_idx++] = -3; }

int32_t* get_scope_log() { return scope_log; }
int32_t  get_scope_log_count() { return scope_log_idx; }

} // extern "C"
