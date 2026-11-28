#include "own.hpp"
#include <new>
#include <stdexcept>
#include <utility>

namespace own {

static int g_live = 0;

Copied::Copied() : v{1, 2, 3, 4} {}
Copied::Copied(const Copied& o) { for (int i = 0; i < 4; ++i) v[i] = o.v[i]; }

Tracked::Tracked(int v) : box_(new int(v)) { ++g_live; }
Tracked::Tracked(const Tracked& o) : box_(new int(*o.box_)) { ++g_live; }
Tracked::Tracked(Tracked&& o) noexcept : box_(o.box_) { o.box_ = nullptr; ++g_live; }
Tracked::~Tracked() { delete box_; --g_live; }
int Tracked::value() const { return box_ ? *box_ : -1; }

Token::Token(int id) : id_(new int(id)) { ++g_live; }
Token::Token(Token&& o) noexcept : id_(o.id_) { o.id_ = nullptr; ++g_live; }
Token::~Token() { delete id_; --g_live; }
int Token::id() const { return id_ ? *id_ : -1; }

Shape::Shape() {}
std::vector<int> Shape::points(int n) const { std::vector<int> v; for (int i = 1; i <= n; ++i) v.push_back(i); return v; }
std::map<int, int> Shape::index(int n) const { std::map<int, int> m; for (int i = 1; i <= n; ++i) m[i] = i * i; return m; }
std::list<int> Shape::chain(int n) const { std::list<int> l; for (int i = 1; i <= n; ++i) l.push_back(i); return l; }
std::unique_ptr<Tracked> Shape::boxed(int v) const { return std::unique_ptr<Tracked>(new Tracked(v)); }
Tracked Shape::tracked(int v) const { return Tracked(v); }
Copied Shape::copied() const { return Copied(); }
Plain Shape::plain() const { return Plain{7, 2.5}; }
long Shape::take(std::vector<int> v) const { long s = 0; for (int x : v) s += x; return s; }
long Shape::keep(std::vector<int> v) { kept_ = std::move(v); return (long)kept_.size(); }
int Shape::adopt(std::unique_ptr<Tracked> t) { return t ? t->value() : -1; }
int Shape::see(Tracked t) const { return t.value(); }
long Shape::kept() const { long s = 0; for (int x : kept_) s += x; return s; }
Token Shape::token(int id) const { return Token(id); }
int Shape::redeem(Token t) const { return t.id(); }
long Shape::fail(std::vector<int> v) const { throw std::runtime_error("refused " + std::to_string(v.size())); }
Tracked Shape::broken(int v) const { Tracked t(v); throw std::runtime_error("broken"); }

std::vector<int> iota(int n) { std::vector<int> v; for (int i = 0; i < n; ++i) v.push_back(i); return v; }
long sum(const std::vector<int>& v) { long s = 0; for (int x : v) s += x; return s; }
int map_last(const std::map<int, int>& m) { return m.empty() ? 0 : m.rbegin()->first; }
long list_sum(const std::list<int>& l) { long s = 0; for (int x : l) s += x; return s; }
int live_count() { return g_live; }
void append(std::vector<int>& v, int x) { v.push_back(x); }
int peek(const Tracked& t) { return t.value(); }
int peek(const std::vector<int>& v) { return (int)v.size(); }
int boxed_value(const std::unique_ptr<Tracked>& p) { return p ? p->value() : -1; }

}  // namespace own

extern "C" {
void own_make_tracked(void* out, int v) { ::new (out) own::Tracked(v); }
int own_tracked_value(const void* p) { return static_cast<const own::Tracked*>(p)->value(); }
void own_make_points(void* out, int n) { ::new (out) std::vector<int>(own::iota(n)); }
long own_points_sum(const void* p) { return own::sum(*static_cast<const std::vector<int>*>(p)); }
}
