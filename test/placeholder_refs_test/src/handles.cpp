// What a library hands out for types a wrapper has only a placeholder for (a
// smart pointer, a std::function), passed back where a reference to one is
// taken (test_placeholder_references_live.jl).
#include <functional>
#include <memory>

struct Node { int v; };

// A smart pointer returned by value, then read and written by reference.
std::shared_ptr<Node> sp_make(int v) { return std::make_shared<Node>(Node{v}); }
int sp_read(const std::shared_ptr<Node>& p) { return p ? p->v : -1; }
long sp_count(const std::shared_ptr<Node>& p) { return p.use_count(); }
void sp_set(std::shared_ptr<Node>& p, int v) { p = std::make_shared<Node>(Node{v}); }

std::unique_ptr<Node> up_make(int v) { return std::make_unique<Node>(Node{v}); }
int up_read(const std::unique_ptr<Node>& p) { return p ? p->v : -1; }

// A std::function the library owns, handed out by address.
static std::function<int(int)> g_twice = [](int x) { return 2 * x; };
const std::function<int(int)>* fn_twice() { return &g_twice; }
int fn_apply_ref(const std::function<int(int)>& f, int x) { return f(x); }
int fn_empty(const std::function<int(int)>& f) { return f ? 0 : 1; }
// A plain function pointer, to make one from Julia and try it where a
// std::function is taken.
typedef int (*plain_cb)(int);
int fn_plain(plain_cb f, int x) { return f(x); }

// An overload set that differs only in a reference against a pointer: two
// functions, and two Julia methods.
int which(const std::shared_ptr<Node>& p) { (void)p; return 1; }
int which(std::shared_ptr<Node>* p) { (void)p; return 2; }
