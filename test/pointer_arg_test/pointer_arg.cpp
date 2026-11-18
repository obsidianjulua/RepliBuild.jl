#include "pointer_arg.h"
#include <cmath>
#include <cstring>

Vec3::Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
double Vec3::length() const { return std::sqrt(x * x + y * y + z * z); }
double Vec3::dot(const Vec3 *o) const { return x * o->x + y * o->y + z * o->z; }

int Flags::get_level() const { return (int)level; }
Flags flags_make(unsigned level, unsigned id) {
    Flags f{};
    f.level = level;
    f.id = id;
    return f;
}
unsigned flags_id_of(const Flags *f) { return f->id; }

Shape::Shape(int id_) : id(id_) {}
Shape::~Shape() {}
int Shape::shape_id() const { return id; }
Square::Square(int id_, int side_) : Shape(id_), side(side_) {}
int Square::area() const { return side * side; }
Square *square_new(int id, int side) { return new Square(id, side); }
void square_delete(Square *s) { delete s; }
int shape_id_of(const Shape *s) { return s->shape_id(); }

int Token::value() const { return v; }
Token *make_token(int v) { return new Token{v}; }
void free_token(Token *t) { delete t; }
int token_value_of(const Token *t) { return t->v; }

int apply_unary(unary_t f, int x) { return f(x); }
void *echo_ptr(void *p) { return p; }

int sum_ints(const int *p, std::size_t n) {
    int s = 0;
    for (std::size_t i = 0; i < n; i++) s += p[i];
    return s;
}
std::size_t count_byte(const unsigned char *p, std::size_t n, unsigned char b) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < n; i++) c += p[i] == b;
    return c;
}
void fill_ints(int *out, std::size_t n, int v) {
    for (std::size_t i = 0; i < n; i++) out[i] = v + (int)i;
}

std::vector<int> vec_iota(int n) {
    std::vector<int> v;
    for (int i = 0; i < n; i++) v.push_back(i);
    return v;
}
int vec_sum(const std::vector<int> &v) {
    int s = 0;
    for (int x : v) s += x;
    return s;
}
int vec_sum_val(std::vector<int> v) {
    int s = 0;
    for (int x : v) s += x;
    return s;
}
void *str_new(const char *s) { return new std::string(s); }
void str_append(std::string &s, const char *x) { s += x; }
std::size_t str_size(const std::string *s) { return s->size(); }
std::size_t str_size_ref(const std::string &s) { return s.size(); }
void str_delete(std::string *s) { delete s; }

std::size_t cstr_len(const char *s) { return std::strlen(s); }
int overloaded(int x) { return x + 1; }
long long overloaded(long long x) { return x + 2; }
double overloaded(double x) { return x + 0.5; }
bool overloaded(bool b) { return !b; }
const char *overloaded(const char *s) { return s + 1; }
