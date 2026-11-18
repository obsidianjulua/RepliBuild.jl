// test/virtual_class_test — which constructor and destructor a wrapper binds,
// and which virtual methods it dispatches (test_virtual_classes_live.jl).
#pragma once

// ── Constructor and destructor variants ─────────────────────────────────────
//
// A diamond over a virtual base. For Left, Right and Both the complete-object
// constructor (C1) and the base-object one (C2) are two functions: C2 takes a
// VTT after `this` and leaves Root alone. The same for D1 and D2. Root has no
// virtual base, so its C1 and C2 are one function.
struct Root {
    int tag;
    Root();
    virtual ~Root();
    virtual int who() const;
};
struct Left : virtual Root {
    int l;
    Left();
    explicit Left(int l_);
    ~Left() override;               // written out: clang emits D2, D1, D0
    int who() const override;
};
struct Right : virtual Root {
    int r;
    Right();                        // destructor implicit: D1, D0, then D2
};
struct Both : Left, Right {
    int b;
    Both();
    Both(int l_, int b_);
    int who() const override;
    int sum() const;                // tag + l + r + b
};

// An abstract class with a virtual base: the library has its base-object
// constructor only, and nothing can build a whole Sketch.
struct Sketch : virtual Root {
    int s;
    Sketch();
    virtual int area() const = 0;
};
struct Solid : Sketch {
    Solid();
    int area() const override;
};

// An abstract class with NO virtual base: the library has its base-object
// constructor only too, and there it is the constructor.
struct Plain {
    int v;
    explicit Plain(int v_);
    virtual ~Plain();
    virtual int get() const = 0;
};

int   live_roots();                 // Root objects constructed and not yet destroyed
Both *spawn_both();                 // `new Both(4, 5)`

// ── Virtual dispatch ────────────────────────────────────────────────────────
//
// Dog, Puppy, Duck and Mallard add no data member to their bases. `rank` is
// `noexcept`, which is what a plain `ccall` can carry. Bird is defined in
// vclass.cpp and one of its methods in vclass_late.cpp, a unit whose debug
// info therefore only DECLARES Bird.
struct Animal {
    int legs_;
    explicit Animal(int legs);
    virtual ~Animal();
    virtual const char *name() const;
    virtual double speed(double terrain) const;
    virtual int rank() const noexcept;
};
struct Dog : Animal {
    Dog();
    const char *name() const override;
    double speed(double terrain) const override;
    int rank() const noexcept override;
};
struct Puppy : Dog {
    Puppy();
    const char *name() const override;
    double speed(double terrain) const override;
    int rank() const noexcept override;
};
struct Bird : Animal {
    double wingspan;
    Bird();
    const char *name() const override;
    double speed(double terrain) const override;    // defined in vclass_late.cpp
};
struct Swimmer {
    int swim;
    Swimmer();
    virtual ~Swimmer();
    virtual int depth() const;
};
struct Duck : Bird, Swimmer {
    Duck();
    const char *name() const override;
    int depth() const override;     // overrides the SECOND base's method
};
struct Mallard : Duck {
    Mallard();
    const char *name() const override;
    int depth() const override;
    double speed(double terrain) const override;
};

Animal *spawn_animal(int kind);     // 0 Dog, 1 Puppy, 2 Bird, 3 Duck, 4 Mallard
void    drop_animal(Animal *a);
