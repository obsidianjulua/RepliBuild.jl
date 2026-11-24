#include <initializer_list>
int il_sum(std::initializer_list<int> l) { int t = 0; for (int x : l) t += x; return t; }
double il_sumd(std::initializer_list<double> l) { double t = 0; for (double x : l) t += x; return t; }
int il_count(const std::initializer_list<int>& l) { return (int)l.size(); }
