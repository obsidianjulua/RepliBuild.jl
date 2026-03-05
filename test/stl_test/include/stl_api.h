#pragma once
#include <vector>
#include <string>
#include <cstddef>

// =============================================================================
// STL Container Test API
// =============================================================================

// --- std::vector<int> functions ---

// Create a vector with values [0, 1, ..., n-1]
std::vector<int> make_int_vector(int n);

// Sum all elements in a vector
int sum_vector(const std::vector<int>& v);

// Append a value to a vector
void append_to_vector(std::vector<int>& v, int val);

// Get size of a vector
size_t vector_size(const std::vector<int>& v);

// Get element at index
int vector_get(const std::vector<int>& v, size_t index);

// Double every element, return new vector
std::vector<int> double_vector(const std::vector<int>& v);

// --- std::string functions ---

// Create a greeting string
std::string make_greeting(const char* name);

// Get string length
size_t string_length(const std::string& s);

// Get C string pointer
const char* string_to_cstr(const std::string& s);

// --- C-linkage test functions (these should work with Tier 1) ---
extern "C" {
    int add_numbers(int a, int b);
    double multiply(double a, double b);
}
