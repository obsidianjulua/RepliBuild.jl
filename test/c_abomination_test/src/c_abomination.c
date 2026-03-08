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
