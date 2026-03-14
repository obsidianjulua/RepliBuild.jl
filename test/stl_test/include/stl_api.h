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
