#pragma once
#include <vector>
#include <string>
#include <map>

// Return a vector of ints
std::vector<int> make_ints(int n);

// Sum the contents of a vector
int sum_vec(const std::vector<int>& v);

// Return a string
std::string greet(const char* name);

// Get string length through the API
int string_len(const std::string& s);

// Create a map with sequential key-value pairs: {0:0, 1:10, 2:20, ...}
std::map<int, int> make_int_map(int n);

// Look up a key in the map, return -1 if not found
int map_lookup(const std::map<int, int>& m, int key);

// Return the number of entries in the map
int map_size(const std::map<int, int>& m);

// A function template instantiated twice. Its derived Julia names carry the
// template argument (`stl_test_scaled_long`, `stl_test_scaled_double`); the
// fixture's `[wrap.rename]` binds both as `scaled`, an overload set Julia
// dispatches on the first argument (`Integer` against `Float64`).
namespace stl_test {
template <typename T> T scaled(T x, int k);
extern template long scaled<long>(long, int);
extern template double scaled<double>(double, int);
}

// A record template behind a typedef, as clipper2 spells `Point64 =
// Point<int64_t>`: [types] templates lists `std::vector<stl_test::Pt64>` and
// every member symbol spells the element `stl_test::Pt<long>`.
namespace stl_test {
template <typename T>
struct Pt {
    T x;
    T y;
};
using Pt64 = Pt<long>;
}  // namespace stl_test

// Build n points (i, 2i); sum the x coordinates.
std::vector<stl_test::Pt64> make_pts(int n);
long sum_pts_x(const std::vector<stl_test::Pt64>& pts);
