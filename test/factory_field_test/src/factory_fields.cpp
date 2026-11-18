#include "factory_fields.h"

static int g_animals = 0;
static int g_birds = 0;

Animal::Animal(int legs) : legs_(legs) { ++g_animals; }
Animal::~Animal() { --g_animals; }
int Animal::legs() const { return legs_; }

Bird::Bird(int wings) : wings_(wings) { ++g_birds; }
Bird::~Bird() { --g_birds; }
int Bird::wings() const { return wings_; }

Animal *make_animal(int legs) { return new Animal(legs); }
// The legs of the caged animal plus the cage's count.
Animal *make_animal_caged(Cage c) { return new Animal((c.pet ? c.pet->legs() : 0) + c.n); }
void free_animal(Animal *a) { delete a; }
// The wings asked for plus the legs of the aviary's prey.
Bird *make_bird(const Aviary *home, int wings) {
    return new Bird(wings + (home && home->prey ? home->prey->legs() : 0));
}
void free_bird(Bird *b) { delete b; }

int cage_legs(const Cage *c) { return c->pet ? c->pet->legs() * c->n : -1; }
int den_legs(const Den *d) {
    int s = 0;
    for (int i = 0; i < d->count; ++i) s += d->pack[i]->legs();
    return s;
}
int ark_legs(const Ark *a) {
    int s = 0;
    for (int i = 0; i < 3; ++i) if (a->pets[i]) s += a->pets[i]->legs();
    return s;
}
int aviary_sum(const Aviary *v) {
    return (v->bird ? v->bird->wings() : 0) + (v->prey ? v->prey->legs() : 0);
}
int animals_alive() { return g_animals; }
int birds_alive() { return g_birds; }
