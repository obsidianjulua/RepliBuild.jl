#include <memory>
#include "lib.hpp"
std::shared_ptr<SNode> sp_make(int v) { return std::make_shared<SNode>(SNode{v}); }
int sp_read(const std::shared_ptr<SNode>& p) { return p ? p->v : -1; }
int sp_take(std::shared_ptr<SNode> p) { return p ? p->v : -1; }
long sp_count(const std::shared_ptr<SNode>& p) { return p.use_count(); }
SNode* sp_raw(const std::shared_ptr<SNode>& p) { return p.get(); }
std::weak_ptr<SNode> wp_of(const std::shared_ptr<SNode>& p) { return p; }
int wp_alive(const std::weak_ptr<SNode>& w) { return w.expired() ? 0 : 1; }
