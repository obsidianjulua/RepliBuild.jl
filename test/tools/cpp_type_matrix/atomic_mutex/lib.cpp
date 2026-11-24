#include <atomic>
#include <mutex>
static std::atomic<int> g_counter{0};
static std::mutex g_mutex;
std::atomic<int>& at_counter() { return g_counter; }
int at_bump(std::atomic<int>& a) { return ++a; }
std::mutex& mx_global() { return g_mutex; }
int mx_locked_add(std::mutex& m, int a, int b) { std::lock_guard<std::mutex> g(m); return a + b; }
