#pragma once

// What the Julia finalizers free, counted. A deleting destructor runs the
// destructor body, then the class's operator delete, so `*_dtors` counts
// destructor bodies and `*_frees` counts storage given back.
int rect_dtors() noexcept;
int rect_frees() noexcept;
int leaf_dtors() noexcept;
int leaf_frees() noexcept;
int widget_frees() noexcept;
int node_dtors() noexcept;
int node_frees() noexcept;
int pooled_deletes() noexcept;

class Shape {
public:
    virtual ~Shape();
    virtual double area() const = 0;
};

class Rect : public Shape {
public:
    Rect(double w, double h);
    ~Rect() override;
    double area() const override;
    static void operator delete(void *p);
private:
    double w_, h_;
};

// A factory, and one whose name names no class.
Rect *make_rect(double w, double h) noexcept;
Rect *make_rect_nx(double w, double h) noexcept;

// No virtual destructor and no deleter function: nothing may own a Plain.
struct Plain {
    int v;
    ~Plain();
};
Plain *make_plain(int v) noexcept;
Plain *spawn_plain(int v) noexcept;

// No virtual destructor, and the library's deleter function.
struct Widget {
    int v;
    ~Widget();
};
Widget *create_widget(int v) noexcept;
void destroy_widget(Widget *w) noexcept;

// A virtual destructor, handed out by a function that is not a factory.
class Leaf {
public:
    explicit Leaf(int v);
    virtual ~Leaf();
    int value() const;
    static void operator delete(void *p);
private:
    int v_;
};
Leaf *spawn_leaf(int v) noexcept;

// A method whose name says free and whose one parameter is `this`: it frees
// buffers, not the object, so it is never a deleter.
class Node {
public:
    Node();
    virtual ~Node();
    void FreeMemory();
    static void operator delete(void *p);
};
Node *spawn_node() noexcept;

// A static member that deletes: the library's own deleter, preferred to D0.
class Pooled {
public:
    Pooled();
    virtual ~Pooled();
    static void DeletePooled(Pooled *p);
};
Pooled *spawn_pooled() noexcept;
