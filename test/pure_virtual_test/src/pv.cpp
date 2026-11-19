#include "pv.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static int g_live = 0;
static char g_buf[64];

// ── Animal ──────────────────────────────────────────────────────────────────
Animal::Animal(int legs) : legs_(legs) { label_[0] = 0; ++g_live; }
Animal::~Animal() { --g_live; }
int Animal::legs() const { return legs_; }
const char *Animal::describe() const {
    std::snprintf(g_buf, sizeof g_buf, "%s with %d legs", name(), legs());
    return g_buf;
}
int Animal::secret_plus(int n) const { return secret() + hidden() + n; }

static void set_label(char *label, const char *to) {
    std::strncpy(label, to, 15);
    label[15] = 0;
}

Dog::Dog() : Animal(4) {}
const char *Dog::name() const { return label_[0] ? label_ : "dog"; }
double Dog::speed(double terrain) const { return 10.0 * terrain; }
int Dog::rank(int base, unsigned char boost, bool lead) const { return base + boost + (lead ? 1000 : 0); }
void Dog::rename(const char *to) { set_label(label_, to); }
int Dog::risky(int n) const {
    if (n < 0) throw std::runtime_error("dog: negative");
    return 2 * n;
}
Span Dog::span() const { return Span{1.0, 4.0}; }
int Dog::weigh(Span s) const { return (int)(s.lo + s.hi); }
Blob Dog::blob() const { return Blob{{1.0, 2.0, 3.0, 4.0}}; }
int Dog::secret() const { return 40; }
int Dog::hidden() const { return 2; }

Puppy::Puppy() {}
const char *Puppy::name() const { return label_[0] ? label_ : "puppy"; }
double Puppy::speed(double terrain) const { return 3.0 * terrain; }

Bird::Bird(double w) : Animal(2), wingspan(w) {}
const char *Bird::name() const { return label_[0] ? label_ : "bird"; }
double Bird::speed(double terrain) const { return 30.0 * terrain + wingspan; }
int Bird::rank(int base, unsigned char boost, bool lead) const { return 2 * base + boost + (lead ? 2000 : 0); }
void Bird::rename(const char *to) { set_label(label_, to); }
int Bird::risky(int n) const {
    if (n < 0) throw std::runtime_error("bird: negative");
    return 3 * n;
}
Span Bird::span() const { return Span{2.0, 8.0}; }
int Bird::weigh(Span s) const { return (int)(s.hi - s.lo); }
Blob Bird::blob() const { return Blob{{5.0, 6.0, 7.0, 8.0}}; }
int Bird::secret() const { return 70; }
int Bird::hidden() const { return 7; }

Duck::Duck() : Bird(0.5) {}
const char *Duck::name() const { return label_[0] ? label_ : "duck"; }

Animal *make_animal(int kind) {
    switch (kind) {
    case 0: return new Dog();
    case 1: return new Puppy();
    case 2: return new Bird(1.5);
    default: return new Duck();
    }
}
void free_animal(Animal *a) { delete a; }
int live_animals() { return g_live; }

// ── Listener ────────────────────────────────────────────────────────────────
namespace {
struct Counter : Listener {
    int seen = 0;
    int prio;
    explicit Counter(int p) : prio(p) {}
    void on_event(int code) override { seen += code; }
    int priority() const override { return prio; }
};
struct Printer : Logger {
    int n = 0;
    int log(const char *fmt, ...) override {
        va_list ap;
        va_start(ap, fmt);
        int got = std::vsnprintf(g_buf, sizeof g_buf, fmt, ap);
        va_end(ap);
        ++n;
        return got;
    }
    int lines() const override { return n; }
};
}
Listener *make_counter(int priority) { return new Counter(priority); }
int counter_seen(const Listener *l) { return static_cast<const Counter *>(l)->seen; }
void drop_listener(Listener *l) { delete static_cast<Counter *>(l); }
int dispatch(Listener *l, int code) {
    l->on_event(code);
    return l->priority();
}
Logger *make_logger() { return new Printer(); }
void drop_logger(Logger *l) { delete static_cast<Printer *>(l); }

// A class of this file alone: internal linkage, no part of the interface.
namespace {
struct Step {
    virtual int step(int n) const = 0;
};
struct Twice : Step {
    int step(int n) const override { return 2 * n; }
};
}
int run_step(int n) {
    Twice t;
    const Step &s = t;
    return s.step(n);
}

// ── Token ───────────────────────────────────────────────────────────────────
Token::Token() : id(11) {}
Token::~Token() {}
int Token::value() const { return 1; }
Coin::Coin() {}
int Coin::value() const { return 25; }
Token *make_coin() { return new Coin(); }

// ── Shape ───────────────────────────────────────────────────────────────────
Shape::Shape() : tag(5) {}
Shape::~Shape() {}
Poly::Poly(int n_) : n(n_) {}
int Poly::sides() const { return n; }
Square::Square(int side_) : Poly(4), side(side_) {}
int Square::area() const { return side * side; }

// ── Base ────────────────────────────────────────────────────────────────────
Base::~Base() {}
int Base::tag() const { return 7; }
int Derived::tag() const { return Base::tag() + 1; }
Base *make_derived() { return new Derived(); }
void drop_base(Base *b) { delete b; }

// ── zoo::Keeper ─────────────────────────────────────────────────────────────
namespace zoo {
Keeper::~Keeper() {}
namespace {
// The pointer is kept untyped on purpose: this fixture is about the functions,
// and a record holding a pointer to a class that has a factory struct (`Animal`)
// is another matter (its field would name `Animal` before the wrapper defines it).
struct Fan : Keeper {
    void *pet;
    explicit Fan(Animal *a) : pet(a) {}
    bool feeds(const Animal &a) const override { return &a == pet; }
    Animal *favourite() const override { return static_cast<Animal *>(pet); }
};
}
Keeper *hire(Animal *favourite) { return new Fan(favourite); }
void fire(Keeper *k) { delete k; }
}
