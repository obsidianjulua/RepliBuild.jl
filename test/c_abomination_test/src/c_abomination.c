#include "c_abomination.h"
#include <stdlib.h>
#include <string.h>

NightmareStruct create_nightmare(int id, float x, float y, float z) {
    NightmareStruct n;
    memset(&n, 0, sizeof(NightmareStruct));
    n.id = id;
    n.x = x;
    n.y = y;
    n.z = z;
    strcpy(n.tag, "Nightmare");
    return n;
}

void mutate_nightmare(NightmareStruct* n) {
    if (!n) return;
    n->complex_inner.flags = 0xFFFFFFFF;
    n->id += 1;
}

double execute_outer(OuterFunc f, int a, float b) {
    if (!f) return 0.0;
    InnerFunc inner = f(a);
    if (!inner) return 0.0;
    return inner(b);
}

// Opaque struct simulation
struct OpaqueState {
    int counter;
    double values[100];
};

OpaqueState* init_opaque(void) {
    OpaqueState* state = (OpaqueState*)malloc(sizeof(OpaqueState));
    if (state) {
        state->counter = 0;
        memset(state->values, 0, sizeof(state->values));
    }
    return state;
}

void process_opaque(OpaqueState* state, struct SelfReferential* self_ref) {
    if (state) {
        state->counter++;
    }
    if (self_ref && self_ref->process) {
        self_ref->process(self_ref);
    }
}

void free_opaque(OpaqueState* state) {
    if (state) {
        free(state);
    }
}

// Tagged union with a named member of anonymous union type (see header §6).
TaggedValue make_tagged_int(int64_t v) {
    TaggedValue t;
    memset(&t, 0, sizeof(t));
    t.tag = TAG_INT;
    t.flags = 0xABCD1234u;
    t.u.i = v;
    return t;
}

TaggedValue make_tagged_double(double v) {
    TaggedValue t;
    memset(&t, 0, sizeof(t));
    t.tag = TAG_DBL;
    t.flags = 0u;
    t.u.d = v;
    return t;
}

TaggedValue make_tagged_str(const char* s, int len) {
    TaggedValue t;
    memset(&t, 0, sizeof(t));
    t.tag = TAG_STR;
    t.flags = 1u;
    t.u.str.ptr = s;
    t.u.str.len = len;
    return t;
}

int tagged_is(const TaggedValue* t, ValueTag k) {
    return (t && t->tag == k) ? 1 : 0;
}

// All-float anonymous union by value (see header §7).
FloatBox floatbox_make(double d, int kind) {
    FloatBox b;
    memset(&b, 0, sizeof(b));
    b.v.d = d;
    b.kind = kind;
    return b;
}

double floatbox_get(FloatBox b) {
    return b.v.d;
}

int floatbox_kind(FloatBox b) {
    return b.kind;
}
