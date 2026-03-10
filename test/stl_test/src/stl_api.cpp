#include "stl_api.h"

std::vector<int> make_ints(int n) {
    std::vector<int> v;
    for (int i = 0; i < n; i++) {
        v.push_back(i + 1);
    }
    return v;
}

int sum_vec(const std::vector<int>& v) {
    int total = 0;
    for (int x : v) {
        total += x;
    }
    return total;
}

std::string greet(const char* name) {
    return std::string("hello ") + name;
}

int string_len(const std::string& s) {
    return static_cast<int>(s.size());
}
