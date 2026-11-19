#pragma once
// Factory classes whose factories share a constructor signature
// (test_factory_ctor_collisions_live.jl).

// Two factories take one `int`: the wrapper's `Animal(::Cint)` was defined
// twice and the second won. A third, with its own signature, stays plain.
class Animal {
public:
    explicit Animal(int legs);
    virtual ~Animal();
    virtual int legs() const;
private:
    int legs_;
};
Animal *make_animal(int legs);
Animal *make_animal_doubled(int half);
Animal *make_animal_pair(int front, int back);
void free_animal(Animal *a);

// A record's field names Zebra, so its type is emitted ahead of the records
// and its constructors after them: the same collision in the other form.
class Zebra {
public:
    explicit Zebra(int stripes);
    ~Zebra();
    int stripes() const;
private:
    int stripes_;
};
struct Pen {
    Zebra *z;
    int n;
};
Zebra *make_zebra(int stripes);
Zebra *make_zebra_striped(int bands);
void free_zebra(Zebra *z);
int pen_stripes(const Pen *p);

int animals_alive();
int zebras_alive();
