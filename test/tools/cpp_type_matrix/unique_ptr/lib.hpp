// The classes a unique_ptr owns here, in a header as a library's are. The build
// asks the compiler about `std::unique_ptr<UCounted>` in a unit of its own,
// and that unit has to see the class: a class defined only in a source file
// gets no answer, and a unique_ptr to it crosses as bytes that nothing ends.
#pragma once
struct UNode { int v; };
struct UCounted { int v; UCounted(int x); ~UCounted(); };
