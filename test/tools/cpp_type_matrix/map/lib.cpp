#include <cstring>
#include <map>
#include <unordered_map>
#include <set>
#include <string>
std::map<int, int> map_make(int n) { std::map<int, int> m; for (int i = 0; i < n; i++) m[i] = i * i; return m; }
int map_get(const std::map<int, int>& m, int k) { auto it = m.find(k); return it == m.end() ? -1 : it->second; }
int map_size_val(std::map<int, int> m) { return (int)m.size(); }
std::map<std::string, int> smap_make() { return {{"one", 1}, {"two", 2}}; }
int smap_get(const std::map<std::string, int>& m, const std::string& k) { auto it = m.find(k); return it == m.end() ? -1 : it->second; }
std::unordered_map<int, int> umap_make(int n) { std::unordered_map<int, int> m; for (int i = 0; i < n; i++) m[i] = i + 10; return m; }
int umap_get(const std::unordered_map<int, int>& m, int k) { auto it = m.find(k); return it == m.end() ? -1 : it->second; }
std::set<int> set_make(int n) { std::set<int> s; for (int i = 0; i < n; i++) s.insert(i * 2); return s; }
int set_has(const std::set<int>& s, int k) { return s.count(k) ? 1 : 0; }
// The largest key, through rbegin(). A map's end marker is inside the map
// object and the tree's root points back at it, so a map moved bytewise to
// another address answers with another key. Reads only the map's own nodes.
int map_last_key(const std::map<int, int>& m) { return m.empty() ? -1 : m.rbegin()->first; }
int set_last(const std::set<int>& s) { return s.empty() ? -1 : *s.rbegin(); }
// 1 when the bucket of the map's first element points at the map's own list
// head, which is inside the map object (libstdc++'s layout: the bucket array,
// its length, then the list head). A map moved bytewise to another address
// fails this. -1 on another C++ library: not checked. Reads no stale memory.
int umap_intact(const std::unordered_map<int, int>& m) {
#ifdef __GLIBCXX__
    if (m.empty()) return 1;
    void** buckets; std::memcpy(&buckets, (const char*)&m, sizeof buckets);
    const void* head = (const char*)&m + 2 * sizeof(void*);
    return buckets[m.bucket(m.begin()->first)] == head ? 1 : 0;
#else
    return -1;
#endif
}
// The same check on a value that never left C++: what the probe compares with.
int map_last_key_self() { auto m = map_make(4); return map_last_key(m); }
int set_last_self() { auto s = set_make(4); return set_last(s); }
int umap_intact_self() { auto m = umap_make(5); return umap_intact(m); }
