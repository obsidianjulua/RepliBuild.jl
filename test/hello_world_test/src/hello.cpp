#include <iostream>

extern "C" {
void hello_world() { std::cout << "Hello, World from C++!" << std::endl; }
}
