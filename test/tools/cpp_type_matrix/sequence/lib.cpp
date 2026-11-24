#include <deque>
#include <list>
#include <bitset>
std::deque<int> dq_make(int n) { std::deque<int> d; for (int i = 0; i < n; i++) d.push_back(i); return d; }
int dq_sum(const std::deque<int>& d) { int t = 0; for (int x : d) t += x; return t; }
std::list<int> list_make(int n) { std::list<int> l; for (int i = 0; i < n; i++) l.push_back(i); return l; }
int list_sum(const std::list<int>& l) { int t = 0; for (int x : l) t += x; return t; }
std::bitset<8> bits_make(int v) { return std::bitset<8>((unsigned long)v); }
int bits_count(std::bitset<8> b) { return (int)b.count(); }
// 1 when walking size() nodes from begin() arrives at end(). A list's end marker
// is inside the list object, so a list moved bytewise to another address fails
// this. Reads only the list's own nodes.
int list_intact(const std::list<int>& l) { auto it = l.begin(); for (size_t i = 0; i < l.size(); i++) ++it; return it == l.end() ? 1 : 0; }
// The same check on a value that never left C++: what the probe compares with.
int list_intact_self() { auto l = list_make(4); return list_intact(l); }
