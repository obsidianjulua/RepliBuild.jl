#include <iostream>
#include <sstream>
#include <string>
void os_write(std::ostream& os, int v) { os << "v=" << v; }
int is_read(std::istream& is) { int v = -1; is >> v; return v; }
std::ostringstream* oss_new() { return new std::ostringstream(); }
std::string oss_text(const std::ostringstream& o) { return o.str(); }
void oss_delete(std::ostringstream* o) { delete o; }
std::ostream& os_cout() { return std::cout; }
