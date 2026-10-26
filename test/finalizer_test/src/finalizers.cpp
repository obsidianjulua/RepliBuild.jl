#include "finalizers.h"

#include <new>

static int g_rect_dtors = 0;
static int g_rect_frees = 0;
static int g_leaf_dtors = 0;
static int g_leaf_frees = 0;
static int g_widget_frees = 0;
static int g_node_dtors = 0;
static int g_node_frees = 0;
static int g_pooled_deletes = 0;

int rect_dtors() noexcept { return g_rect_dtors; }
int rect_frees() noexcept { return g_rect_frees; }
int leaf_dtors() noexcept { return g_leaf_dtors; }
int leaf_frees() noexcept { return g_leaf_frees; }
int widget_frees() noexcept { return g_widget_frees; }
int node_dtors() noexcept { return g_node_dtors; }
int node_frees() noexcept { return g_node_frees; }
int pooled_deletes() noexcept { return g_pooled_deletes; }

Shape::~Shape() {}

Rect::Rect(double w, double h) : w_(w), h_(h) {}
Rect::~Rect() { ++g_rect_dtors; }
double Rect::area() const { return w_ * h_; }
void Rect::operator delete(void *p) { ++g_rect_frees; ::operator delete(p); }

Rect *make_rect(double w, double h) noexcept { return new Rect(w, h); }
Rect *make_rect_nx(double w, double h) noexcept { return new Rect(w, h); }

Plain::~Plain() {}
Plain *make_plain(int v) noexcept { return new Plain{v}; }
Plain *spawn_plain(int v) noexcept { return new Plain{v}; }

Widget::~Widget() {}
Widget *create_widget(int v) noexcept { return new Widget{v}; }
void destroy_widget(Widget *w) noexcept { ++g_widget_frees; delete w; }

Leaf::Leaf(int v) : v_(v) {}
Leaf::~Leaf() { ++g_leaf_dtors; }
int Leaf::value() const { return v_; }
void Leaf::operator delete(void *p) { ++g_leaf_frees; ::operator delete(p); }

Leaf *spawn_leaf(int v) noexcept { return new Leaf(v); }

Node::Node() {}
Node::~Node() { ++g_node_dtors; }
void Node::FreeMemory() {}
void Node::operator delete(void *p) { ++g_node_frees; ::operator delete(p); }

Node *spawn_node() noexcept { return new Node(); }

Pooled::Pooled() {}
Pooled::~Pooled() {}
void Pooled::DeletePooled(Pooled *p) { ++g_pooled_deletes; delete p; }

Pooled *spawn_pooled() noexcept { return new Pooled(); }
