#include "stl_api.h"

// --- std::vector<int> functions ---

std::vector<int> make_int_vector(int n) {
    std::vector<int> v;
    v.reserve(n);
    for (int i = 0; i < n; i++) {
        v.push_back(i);
    }
    return v;
}

int sum_vector(const std::vector<int>& v) {
    int sum = 0;
    for (int x : v) {
        sum += x;
    }
    return sum;
}

void append_to_vector(std::vector<int>& v, int val) {
    v.push_back(val);
}

size_t vector_size(const std::vector<int>& v) {
    return v.size();
}

int vector_get(const std::vector<int>& v, size_t index) {
    return v[index];
}

std::vector<int> double_vector(const std::vector<int>& v) {
    std::vector<int> result;
    result.reserve(v.size());
    for (int x : v) {
        result.push_back(x * 2);
    }
    return result;
}

// --- std::string functions ---

std::string make_greeting(const char* name) {
    return std::string("Hello, ") + name + "!";
}

size_t string_length(const std::string& s) {
    return s.length();
}

const char* string_to_cstr(const std::string& s) {
    return s.c_str();
}

// --- C-linkage functions ---

extern "C" {

int add_numbers(int a, int b) {
    return a + b;
}

double multiply(double a, double b) {
    return a * b;
}

}
