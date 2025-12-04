// test.cpp - Simple C++ class with virtual methods to test vtable parsing

class Base {
public:
    virtual int foo() { return 42; }
    virtual int bar(int x) { return x * 2; }
    virtual ~Base() {}

    int non_virtual() { return 100; }
};

class Derived : public Base {
public:
    int foo() override { return 99; }
    virtual int baz() { return 123; }
};

int main() {
    Base b;
    Derived d;
    return b.foo() + d.bar(5);
}
