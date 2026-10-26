// C++ side of the determinism fixture.
#ifndef DT_MODEL_H
#define DT_MODEL_H

#include <functional>
#include <memory>

#include "dt_shared.h"

// Two pimpl classes: each `impl` is a distinct type with the bare name `impl`.
class dt_model {
public:
    explicit dt_model(int n_layer);
    ~dt_model();
    int layers() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

class dt_vocab {
public:
    explicit dt_vocab(int n_vocab);
    ~dt_vocab();
    int size() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// A project record holding a lambda (through libc++'s std::function, whose
// `__func<(lambda at <path>:<line>:<col>), …>` names the closure type) and an
// anonymous union member, which must stay.
struct dt_hooks {
    std::function<int(int)> on_step;
    union {
        int i;
        float f;
    } u;
};
void dt_hooks_set(dt_hooks &h, int k);
int dt_hooks_call(dt_hooks &h, int a);

// A dynamic class: its vtable home's DWARF carries `__clang_vtable`.
class dt_shape {
public:
    virtual ~dt_shape();
    virtual double area() const = 0;
    int id = 0;
};

class dt_square : public dt_shape {
public:
    explicit dt_square(double s);
    ~dt_square() override;
    double area() const override;
    double side;
};

// More deleting destructors, so the Managed blocks have an order to keep. A
// destructor that is not virtual has no deleting variant, and its class gets
// no Managed type (test_cpp_finalizers.jl).
class dt_zeta  { public: dt_zeta();  virtual ~dt_zeta();  int v; };
class dt_alpha { public: dt_alpha(); virtual ~dt_alpha(); int v; };
class dt_mid   { public: dt_mid();   virtual ~dt_mid();   int v; };

#endif
