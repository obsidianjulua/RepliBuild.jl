#ifndef POINTER_ARG_H
#define POINTER_ARG_H
#include <cstddef>
#include <string>
#include <vector>

/* Library-free fixture for what a Tier 2 call accepts where C++ takes a
   pointer (2026-11-17). Every pointer parameter and every `this` of a wrapper
   was `::Any`, and the argument was boxed as it came: a struct value, a Julia
   function or a `Vector` passed for an object was dereferenced as an address.
   None of these functions is `noexcept`, so each is called through a thunk.
   Everything is kept out of line so the compiler cannot see a caller. */

/* A class with members: `this`, and a pointer to another object. */
struct Vec3 {
    double x, y, z;
    Vec3(double x, double y, double z);
    double length() const;
    double dot(const Vec3 *o) const;
};

/* A bitfield record: a `mutable struct` holding its bytes in Julia. */
struct Flags {
    unsigned ro : 1;
    unsigned hidden : 1;
    unsigned level : 6;
    unsigned id : 24;
    int get_level() const;
};
Flags flags_make(unsigned level, unsigned id);
unsigned flags_id_of(const Flags *f);

/* A class a library deleter owns (`ManagedSquare`), and its base: an owning
   object of the derived class passes where the base's `this` goes. */
struct Shape {
    int id;
    explicit Shape(int id);
    virtual ~Shape();
    int shape_id() const;
};
struct Square : Shape {
    int side;
    Square(int id, int side);
    int area() const;
};
Square *square_new(int id, int side);
void square_delete(Square *s);
int shape_id_of(const Shape *s);

/* A class a factory function makes: a `mutable struct` holding the pointer. */
struct Token {
    int v;
    int value() const;
};
Token *make_token(int v);
void free_token(Token *t);
int token_value_of(const Token *t);

/* A function pointer, and user data. */
typedef int (*unary_t)(int);
int apply_unary(unary_t f, int x);
void *echo_ptr(void *p);

/* Pointers to data. */
int sum_ints(const int *p, std::size_t n);
std::size_t count_byte(const unsigned char *p, std::size_t n, unsigned char b);
void fill_ints(int *out, std::size_t n, int v);

/* STL containers behind a reference, a pointer, and by value. */
std::vector<int> vec_iota(int n);
int vec_sum(const std::vector<int> &v);
int vec_sum_val(std::vector<int> v);
void *str_new(const char *s);              /* a `new std::string`, untyped */
void str_append(std::string &s, const char *x);
std::size_t str_size(const std::string *s);
std::size_t str_size_ref(const std::string &s);
void str_delete(std::string *s);

/* C strings, and an overload set with a `const char*` member. */
std::size_t cstr_len(const char *s);
int overloaded(int x);
long long overloaded(long long x);
double overloaded(double x);
bool overloaded(bool b);
const char *overloaded(const char *s);   /* s + 1 */

#endif
