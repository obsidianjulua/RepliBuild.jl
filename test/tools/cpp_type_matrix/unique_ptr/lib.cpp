#include <memory>
struct UNode { int v; };
static int g_live = 0;
struct UCounted { int v; UCounted(int x) : v(x) { g_live++; } ~UCounted() { g_live--; } };
std::unique_ptr<UNode> up_make(int v) { return std::make_unique<UNode>(UNode{v}); }
int up_read(const std::unique_ptr<UNode>& p) { return p ? p->v : -1; }
int up_take(std::unique_ptr<UNode> p) { return p ? p->v : -1; }
UNode* up_raw(const std::unique_ptr<UNode>& p) { return p.get(); }
std::unique_ptr<UCounted> upc_make(int v) { return std::make_unique<UCounted>(v); }
int upc_live() { return g_live; }
int upc_take(std::unique_ptr<UCounted> p) { return p ? p->v : -1; }
std::unique_ptr<int[]> upa_make(int n) { auto a = std::make_unique<int[]>((size_t)n); for (int i = 0; i < n; i++) a[i] = i; return a; }
