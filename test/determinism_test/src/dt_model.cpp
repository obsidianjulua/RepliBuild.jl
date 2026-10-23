#include "dt_model.h"

struct dt_model::impl {
    int n_layer;
    int n_head;
};

dt_model::dt_model(int n_layer) : pimpl(new impl{n_layer, 2 * n_layer}) {}
dt_model::~dt_model() = default;
int dt_model::layers() const { return pimpl->n_layer; }

void dt_hooks_set(dt_hooks &h, int k) { h.on_step = [k](int a) { return a + k; }; }
int dt_hooks_call(dt_hooks &h, int a) { return h.on_step ? h.on_step(a) : -1; }

dt_shape::~dt_shape() {}

dt_square::dt_square(double s) : side(s) {}
dt_square::~dt_square() {}
double dt_square::area() const { return side * side; }

dt_zeta::dt_zeta() : v(1) {}
dt_zeta::~dt_zeta() {}
dt_alpha::dt_alpha() : v(2) {}
dt_alpha::~dt_alpha() {}
dt_mid::dt_mid() : v(3) {}
dt_mid::~dt_mid() {}

// The C++ view of the shared C types.
extern "C" int dt_traits_size(const dt_traits *t) { return t->size; }
extern "C" int dt_block_dm(const dt_block *b) { return b->data.dm; }
