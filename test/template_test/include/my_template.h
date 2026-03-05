#pragma once

template <typename T>
class MyBox {
    T value;
public:
    MyBox(T v) : value(v) {}
    T get() const { return value; }
    void set(T v) { value = v; }
};
