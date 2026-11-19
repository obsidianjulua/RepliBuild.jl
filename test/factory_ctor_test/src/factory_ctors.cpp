#include "factory_ctors.h"

static int g_animals = 0;
static int g_zebras = 0;

Animal::Animal(int legs) : legs_(legs) { ++g_animals; }
Animal::~Animal() { --g_animals; }
int Animal::legs() const { return legs_; }

Animal *make_animal(int legs) { return new Animal(legs); }
Animal *make_animal_doubled(int half) { return new Animal(2 * half); }
Animal *make_animal_pair(int front, int back) { return new Animal(front + back); }
void free_animal(Animal *a) { delete a; }

Zebra::Zebra(int stripes) : stripes_(stripes) { ++g_zebras; }
Zebra::~Zebra() { --g_zebras; }
int Zebra::stripes() const { return stripes_; }

Zebra *make_zebra(int stripes) { return new Zebra(stripes); }
// Every band is three stripes.
Zebra *make_zebra_striped(int bands) { return new Zebra(3 * bands); }
void free_zebra(Zebra *z) { delete z; }
int pen_stripes(const Pen *p) { return p->z ? p->z->stripes() * p->n : -1; }

int animals_alive() { return g_animals; }
int zebras_alive() { return g_zebras; }
