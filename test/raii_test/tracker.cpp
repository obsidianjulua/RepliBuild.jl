// tracker.cpp - Simple C++ test library for RAII dialect ops
// Provides extern "C" constructor/destructor functions that set fields
// so we can verify they were called from JIT-compiled MLIR.

struct Tracker {
    int constructed;
    int destructed;
};

extern "C" void tracker_ctor(Tracker* t) {
    t->constructed = 42;
}

extern "C" void tracker_dtor(Tracker* t) {
    t->destructed = 99;
}

// Constructor with an argument — sets the field to the given value
extern "C" void tracker_ctor_val(Tracker* t, int val) {
    t->constructed = val;
}
