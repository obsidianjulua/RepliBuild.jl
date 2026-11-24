// Records named with a pointer, array or function type in their template
// arguments, and std::vector<bool>, crossed by value. Each is a case the type
// matrix found or the same defect reaches (test_template_arg_records_live.jl).
#include <functional>
#include <memory>
#include <vector>

static int g_cells[3] = {5, 6, 7};
int* cell(int i) { return &g_cells[i]; }

// std::unique_ptr<int[]>: eight bytes, returned through a hidden pointer.
std::unique_ptr<int[]> upa_make(int n) {
    auto a = std::make_unique<int[]>((size_t)n);
    for (int i = 0; i < n; i++) a[i] = i * 3;
    return a;
}
int upa_at(const std::unique_ptr<int[]>& a, int i) { return a[(size_t)i]; }

// std::shared_ptr<int[]>: sixteen bytes, the same.
std::shared_ptr<int[]> spa_make(int n) {
    std::shared_ptr<int[]> a(new int[(size_t)n]);
    for (int i = 0; i < n; i++) a[(size_t)i] = i + 100;
    return a;
}
int spa_at(const std::shared_ptr<int[]>& a, int i) { return a[(size_t)i]; }
long spa_count(const std::shared_ptr<int[]>& a) { return a.use_count(); }

// std::vector<int*>: a container of pointers.
std::vector<int*> vp_make() { return {&g_cells[0], &g_cells[1], &g_cells[2]}; }
int vp_sum(const std::vector<int*>& v) { int s = 0; for (int* p : v) s += *p; return s; }

// std::vector<bool>: bit-packed, and larger than every other std::vector.
std::vector<bool> vb_make(int n) {
    std::vector<bool> v((size_t)n, false);
    for (int i = 0; i < n; i += 2) v[(size_t)i] = true;
    return v;
}
int vb_count(const std::vector<bool>& v) { int s = 0; for (bool b : v) s += b; return s; }
int vb_count_val(std::vector<bool> v) { int s = 0; for (bool b : v) s += b; return s; }
int vb_len(const std::vector<bool>& v) { return (int)v.size(); }
int vb_sizeof() { return (int)sizeof(std::vector<bool>); }
int vi_sizeof() { return (int)sizeof(std::vector<int>); }

// An empty class named with a pointer: C++ passes nothing for it, so `a` and
// `b` are the first two arguments.
template <class T> struct Tag {};
int tag_pick(Tag<int*> t, int a, int b) { (void)t; return a * 10 + b; }

// A class with a destructor, named with a pointer: returned through a hidden
// pointer, and taken as a pointer to a copy the caller makes.
static int g_live = 0;
template <class T> struct Holder {
    T v;
    int serial;
    Holder(T x, int s);
    Holder(const Holder& o);
    ~Holder();
};
template <class T> Holder<T>::Holder(T x, int s) : v(x), serial(s) { g_live++; }
template <class T> Holder<T>::Holder(const Holder& o) : v(o.v), serial(o.serial + 1000) { g_live++; }
template <class T> Holder<T>::~Holder() { g_live--; }
template struct Holder<int*>;
Holder<int*> hold(int* p, int s) { return Holder<int*>(p, s); }
int held_ref(const Holder<int*>& h) { return *h.v + h.serial; }
int held(Holder<int*> h) { return *h.v + h.serial; }
int holders_live() { return g_live; }

// std::function by value: C++ passes a pointer to the caller's object.
static std::function<int(int)> g_twice = [](int x) { return 2 * x; };
const std::function<int(int)>* fn_twice() { return &g_twice; }
int fn_apply(std::function<int(int)> f, int x) { return f(x); }
