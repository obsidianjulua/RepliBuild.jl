// ownership_test: every way a C++ object crosses by value that C++ will not
// copy bytewise, beside two that it will. See test_ownership_unit_live.jl.
#pragma once
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace own {

// Plain data: registers or bytes, nothing to end.
struct Plain { int a; double b; };

// A copy constructor of its own and nothing to end.
struct Copied {
    int v[4];
    Copied();
    Copied(const Copied&);
};

// A class with a destructor. Every live one is counted (`live_count`), so a
// test sees each construction, copy, move and ending.
class Tracked {
public:
    explicit Tracked(int v);
    Tracked(const Tracked&);
    Tracked(Tracked&&) noexcept;
    ~Tracked();
    int value() const;
private:
    int* box_;
};

class Shape {
public:
    Shape();
    std::vector<int> points(int n) const;            // owns memory
    std::map<int, int> index(int n) const;           // points into itself
    std::list<int> chain(int n) const;               // points into itself
    std::unique_ptr<Tracked> boxed(int v) const;     // cannot be copied
    Tracked tracked(int v) const;                    // a destructor, and a layout here
    Copied copied() const;                           // nothing to end
    Plain plain() const;                             // plain data
    long take(std::vector<int> v) const;             // reads its by-value parameter
    long keep(std::vector<int> v);                   // keeps it: moves out of it
    int adopt(std::unique_ptr<Tracked> t);           // takes what cannot be copied
    int see(Tracked t) const;                        // a class by value
    long kept() const;
private:
    std::vector<int> kept_;
};

// Functions no class declares.
std::vector<int> iota(int n);
long sum(const std::vector<int>& v);
int map_last(const std::map<int, int>& m);           // the last key, found from the end
long list_sum(const std::list<int>& l);              // a walk to the end
int live_count();

}  // namespace own

// What a test needs to make an object without a wrapper.
extern "C" {
void own_make_tracked(void* out, int v);
int own_tracked_value(const void* p);
void own_make_points(void* out, int n);
long own_points_sum(const void* p);
}
