// The class a shared_ptr owns here, in a header as a library's is: the build
// asks the compiler about `std::shared_ptr<SNode>` in a unit of its own, and
// that unit has to see the class.
#pragma once
struct SNode { int v; };
