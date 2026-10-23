#include "dt_model.h"

// Another `impl`, of another layout.
struct dt_vocab::impl {
    double scores;
    int n_vocab;
    long long extra;
};

dt_vocab::dt_vocab(int n_vocab) : pimpl(new impl{0.5, n_vocab, 0}) {}
dt_vocab::~dt_vocab() = default;
int dt_vocab::size() const { return pimpl->n_vocab; }
