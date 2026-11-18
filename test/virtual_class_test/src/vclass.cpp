#include "vclass.h"

static int g_live_roots = 0;

Root::Root() : tag(9) { ++g_live_roots; }
Root::~Root() { --g_live_roots; }
int Root::who() const { return 0; }

Left::Left() : l(1) {}
Left::Left(int l_) : l(l_) {}
Left::~Left() {}
int Left::who() const { return 1; }

Right::Right() : r(2) {}

Both::Both() : b(3) {}
Both::Both(int l_, int b_) : Left(l_), b(b_) {}
int Both::who() const { return 3; }
int Both::sum() const { return tag + l + r + b; }

Sketch::Sketch() : s(7) {}
Solid::Solid() {}
int Solid::area() const { return s * 6; }

Plain::Plain(int v_) : v(v_) {}
Plain::~Plain() {}

int live_roots() { return g_live_roots; }
Both *spawn_both() { return new Both(4, 5); }

Animal::Animal(int legs) : legs_(legs) {}
Animal::~Animal() {}
const char *Animal::name() const { return "animal"; }
double Animal::speed(double terrain) const { return 1.0 * terrain; }
int Animal::rank() const noexcept { return 1; }

Dog::Dog() : Animal(4) {}
const char *Dog::name() const { return "dog"; }
double Dog::speed(double terrain) const { return 10.0 * terrain; }
int Dog::rank() const noexcept { return 2; }

Puppy::Puppy() {}
const char *Puppy::name() const { return "puppy"; }
double Puppy::speed(double terrain) const { return 3.0 * terrain; }
int Puppy::rank() const noexcept { return 3; }

Bird::Bird() : Animal(2), wingspan(1.5) {}
const char *Bird::name() const { return "bird"; }

Swimmer::Swimmer() : swim(77) {}
Swimmer::~Swimmer() {}
int Swimmer::depth() const { return 1; }

Duck::Duck() {}
const char *Duck::name() const { return "duck"; }
int Duck::depth() const { return 5; }

Mallard::Mallard() {}
const char *Mallard::name() const { return "mallard"; }
int Mallard::depth() const { return 50; }
double Mallard::speed(double terrain) const { return 7.0 * terrain; }

Animal *spawn_animal(int kind) {
    switch (kind) {
    case 0: return new Dog();
    case 1: return new Puppy();
    case 2: return new Bird();
    case 3: return new Duck();
    default: return new Mallard();
    }
}
void drop_animal(Animal *a) { delete a; }
