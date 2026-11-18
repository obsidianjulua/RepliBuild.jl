// test/pure_virtual_test — pure virtual member functions, which have no symbol
// and are called through the object's vtable (test_pure_virtual_live.jl).
#pragma once

struct Span { double lo; double hi; };      // 16 bytes: by value it is a record
struct Blob { double v[4]; };               // 32 bytes: returned through a hidden pointer

// An interface with out-of-line members. Its vtable is in the library, and
// every pure slot of it holds `__cxa_pure_virtual`, which ends the process.
class Animal {
public:
    explicit Animal(int legs);
    virtual ~Animal();
    virtual const char *name() const = 0;                                  // a `const char*` out
    virtual int legs() const;                                              // not pure: a symbol
    virtual double speed(double terrain) const = 0;                        // a scalar in and out
    virtual int rank(int base, unsigned char boost, bool lead) const = 0;  // integers narrower than int
    virtual void rename(const char *to) = 0;                               // void, a pointer in
    virtual int risky(int n) const = 0;                                    // overrides throw for n < 0
    virtual Span span() const = 0;                                         // a record out
    virtual int weigh(Span s) const = 0;                                   // a record in
    virtual Blob blob() const = 0;                                         // a record through a hidden pointer
    const char *describe() const;                                          // not virtual; calls name()
    int secret_plus(int n) const;                                          // calls secret() and hidden()
protected:
    virtual int secret() const = 0;                                        // protected
    int legs_;
    char label_[16];
private:
    virtual int hidden() const = 0;                                        // private
};

class Dog : public Animal {
public:
    Dog();
    const char *name() const override;
    double speed(double terrain) const override;
    int rank(int base, unsigned char boost, bool lead) const override;
    void rename(const char *to) override;
    int risky(int n) const override;
    Span span() const override;
    int weigh(Span s) const override;
    Blob blob() const override;
protected:
    int secret() const override;
private:
    int hidden() const override;
};
class Puppy final : public Dog {
public:
    Puppy();
    const char *name() const override;
    double speed(double terrain) const override;
};
class Bird : public Animal {
public:
    explicit Bird(double wingspan);
    const char *name() const override;
    double speed(double terrain) const override;
    int rank(int base, unsigned char boost, bool lead) const override;
    void rename(const char *to) override;
    int risky(int n) const override;
    Span span() const override;
    int weigh(Span s) const override;
    Blob blob() const override;
    double wingspan;
protected:
    int secret() const override;
private:
    int hidden() const override;
};
class Duck : public Bird {
public:
    Duck();
    const char *name() const override;
};

Animal *make_animal(int kind);              // 0 Dog, 1 Puppy, 2 Bird, 3 Duck
void    free_animal(Animal *a);
int     live_animals();

// Nothing but pure virtuals, and no member defined out of line: the library
// has no symbol of this class at all.
struct Listener {
    virtual void on_event(int code) = 0;
    virtual int priority() const = 0;
};
Listener *make_counter(int priority);
int       counter_seen(const Listener *l);  // the sum of the codes it was sent
void      drop_listener(Listener *l);
int       dispatch(Listener *l, int code);  // l->on_event(code); return l->priority();

// A variadic pure virtual: a call through the vtable cannot forward `...`.
struct Logger {
    virtual int log(const char *fmt, ...) = 0;
    virtual int lines() const = 0;
};
Logger *make_logger();
void    drop_logger(Logger *l);

// Calls a pure virtual of a class pv.cpp keeps in an unnamed namespace.
int run_step(int n);

// A pure virtual destructor, defined as C++ requires: it has symbols.
struct Token {
    int id;
    Token();
    virtual ~Token() = 0;
    virtual int value() const;
};
struct Coin : Token {
    Coin();
    int value() const override;
};
Token *make_coin();

// Abstract through inheritance: Poly overrides one of Shape's two pure
// virtuals and declares none of its own.
struct Shape {
    int tag;
    Shape();
    virtual ~Shape();
    virtual int area() const = 0;
    virtual int sides() const = 0;
};
struct Poly : Shape {
    int n;
    explicit Poly(int n_);
    int sides() const override;
};
struct Square : Poly {
    int side;
    explicit Square(int side_);
    int area() const override;
};

// A pure virtual that is also defined: it has a symbol, like any function.
struct Base {
    virtual ~Base();
    virtual int tag() const = 0;
};
struct Derived : Base {
    int tag() const override;
};
Base *make_derived();
void  drop_base(Base *b);

// In a namespace, with a reference in, a `bool` out and a typed pointer out.
namespace zoo {
struct Keeper {
    virtual ~Keeper();
    virtual bool feeds(const Animal &a) const = 0;
    virtual Animal *favourite() const = 0;
};
Keeper *hire(Animal *favourite);
void    fire(Keeper *k);
}
