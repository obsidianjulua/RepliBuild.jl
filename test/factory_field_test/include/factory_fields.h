// Records whose fields point at factory classes (test_factory_field_order_live.jl).
// Animal and Bird each have a factory (`make_*`) and a deleter (`free_*`), so
// the wrapper turns them into owning Julia types; every record below names one
// of them in a field. `make_animal_caged` takes the record holding the
// pointer, and `make_bird` a pointer to a record holding a Bird*: a factory
// struct's constructor whose signature names a record that names the class.
#pragma once

class Animal {
public:
    explicit Animal(int legs);
    virtual ~Animal();
    virtual int legs() const;
private:
    int legs_;
};

class Bird {
public:
    explicit Bird(int wings);
    ~Bird();
    int wings() const;
private:
    int wings_;
};

struct Cage { Animal *pet; int n; };
struct Den { Animal **pack; int count; };
struct Ark { Animal *pets[3]; };
struct Aviary { Bird *bird; Animal *prey; };

Animal *make_animal(int legs);
Animal *make_animal_caged(Cage c);
void free_animal(Animal *a);
Bird *make_bird(const Aviary *home, int wings);
void free_bird(Bird *b);

int cage_legs(const Cage *c);
int den_legs(const Den *d);
int ark_legs(const Ark *a);
int aviary_sum(const Aviary *v);
int animals_alive();
int birds_alive();
