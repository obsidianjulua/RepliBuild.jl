# Call the wrapper

`wrap` writes an ordinary Julia module beside the library it calls. This page
is how the library's functions, types and objects look from Julia.

## Load it

```julia
include("path/to/project/julia/Mathlib.jl")
using .Mathlib

ml_dot(ml_vec2(1.0, 2.0), ml_vec2(3.0, 4.0))
```

After [`register`](use.md), load it by name instead, from any session:

```julia
M = RepliBuild.use("mathlib")       # or:  using Mathlib
```

Three rules:

- **Never edit the generated file.** `wrap` overwrites it. Change the TOML, or
  write your own code in your own module.
- **Keep the files in `julia/` together.** The module finds its library beside
  itself.
- **Load a wrapper at the top level** of a script or the REPL, not inside a
  function that then calls into it. That fails with `method too new to be
  called from this world context`; if you must, call through
  `Base.@invokelatest M.f(x)`.

## Find your way around

```julia
names(Mathlib)            # everything the module exports
?ml_normalize             # a function's Julia signature and its C argument types
methods(ml_dot)           # the argument types it accepts
```

Everything is exported, with one exception: a name that Julia itself already
uses. A C type called `Number` is `Mathlib.Number`, never plain `Number`.
`wrap` prints those names when it writes the module.

**A few names cannot be used as the library spells them**, and are given
another. `wrap` prints each one, with the reason, on a line starting `names:`.

| The library has | The wrapper has | Because |
|:----------------|:----------------|:--------|
| A function `String`, `Ptr`, `Cint` | `String_`, `Ptr_`, `Cint_` | The name is a Julia type |
| A function `eval`, `include`, `ccall` | `eval_`, `include_`, `ccall_` | Julia reserves the name |
| A function or method `nothing`, `LIBRARY_PATH` | `nothing_`, `LIBRARY_PATH_` | The wrapper's own code uses the name |
| A struct `Integer`, `Bool` | `c_Integer`, `c_Bool`, with methods `c_Integer_get` | The wrapper's own code uses the name |
| A value `Symbol` of `enum Kind` | `Kind_Symbol` | The wrapper's own code uses the name |

Other names that Julia also has are kept: a function `length`, `open` or
`string`, or a struct `Task`, is `M.length`, `M.open`, `M.string`, `M.Task`.
To choose a function's name yourself, use [`[wrap.rename]`](toml.md#Names),
with the library's name as the key (`String`, not `String_`).

**A function the wrapper cannot call safely is still defined.** Calling it
raises an error that starts with `ABI Safety Trap` and gives the reason: a
`long double` passed by value, a struct whose layout could not be settled, a
function the library has no debug information for. The rest of the module is
unaffected. To list them:

```julia
filter(l -> occursin("Safety Trap", l), readlines("julia/Mathlib.jl"))
```

## How C types appear

| C | Julia |
|:--|:------|
| `int`, `long`, `size_t`, … | Pass any `Integer`. It is converted, and checked for range. |
| `int32_t`, `uint8_t`, … | `Int32`, `UInt8`, … |
| `char` by value | Pass any `Integer` from -128 to 255. A `char` result is a `Cchar`. |
| `double`, `float` | Pass any `Real` in a C wrapper. In a C++ wrapper pass a `Float64` or `Float32`: write `2.0`, not `2`. |
| `bool` | `Bool` |
| `const char*` argument | A `String` |
| `char*` result | A `String`, or `nothing` for NULL |
| `struct S` by value | The struct `S` |
| `S*` | `Ref(s)`, or a `Ptr{S}` |
| `void*` | `Ptr{Cvoid}` |
| An enum | Its own type, with its values as constants |
| A function pointer | `Ptr{Cvoid}`; make one with `@cfunction` |
| A macro from `[wrap.macros]` | A function: `ML_VERSION()`, `lua_pop(L, 1)` |

Each function's docstring states exactly what it takes.

## Structs

A struct is a Julia struct with the same fields:

```julia
v = ml_vec2(3.0, 4.0)
v.x                        # 3.0
ml_vec2()                  # every field zero
```

Structs are **immutable values**, because C passes them by value. `v.x = 9.0`
is an error. Make a new one instead: `ml_vec2(9.0, v.y)`. Where the wrapper
defines them, `setproperties(v; x = 9.0)` and `setproperty(v, :x, 9.0)` return
a changed copy.

**A struct argument takes the struct. A pointer argument takes a `Ref`.**

```julia
ml_dot(v, v)               # double ml_dot(ml_vec2 a, ml_vec2 b)

r = Ref(v)                 # ml_status ml_normalize(ml_vec2 *v)
ml_normalize(r)
r[]                        # the value the function wrote: ml_vec2(0.6, 0.8)
```

Passing a bare struct where a pointer is expected is a `MethodError`.

**A struct whose layout Julia cannot reproduce** (unusual packing, some
bitfields) has no named fields. It holds the raw bytes, and the wrapper gives
it a getter and a setter per field. `wrap` warns when it does this, and
`names(M)` shows the accessors.

**Unions** hold their bytes and have a getter and a setter per member:
`get_U_m(u)` and `set_U_m!(u, v)` in a C wrapper, `get_m(u)` and
`set_m!(u, v)` in a C++ wrapper. `U()` makes a zeroed one.

**Bitfields** have a getter and a setter per field: `get_f(s)`, and
`set_f(s, v)`, which returns a new struct, in a C wrapper; `set_f!(s, v)`,
which changes it, in a C++ wrapper. A signed field (`int level : 7`) reads as
a signed number, and its setter takes the values the field can hold.

## Enums

```julia
ml_normalize(Ref(ml_vec2(0.0, 0.0))) == ML_EDOM     # true
Int(ML_EDOM)                                         # 1
ml_status(1)                                         # ML_EDOM
```

An enum is its own type, and its values are constants in the module. As in C,
any integer of the right size is a valid value: `ml_status(7)` works, and
prints as `ml_status(0x00000007)` because it has no name. Values combine with
`|`, `&` and `xor`, and an `Int` is accepted where a field or argument wants
the enum.

An enum whose values are bit flags is a plain integer type with named
constants.

When a value's name is also the name of a type, a function or another enum's
value in the same module, it gets its enum's name in front: with a C++ class
`Rect` and an `enum class Kind { Circle, Rect }`, the values are `Circle` and
`Kind_Rect`. A comment above the enum in the wrapper lists these.

## Strings

- A `const char*` argument takes a Julia `String`.
- A `char*` result is copied into a `String`. A NULL result is `nothing`.
- If the caller must free the result, say so in
  [`[wrap.cstring_owned]`](toml.md#Strings-the-caller-must-free), and the
  wrapper frees it after copying.
- Every such function also has a `<name>_ptr` version that returns the raw
  pointer and frees nothing, for when you need the pointer itself.

## Variadic functions

Each combination listed in [`[wrap.varargs]`](toml.md#Variadic-functions) is
its own function, named after its extra argument types:

```julia
ml_log("hello\n")                                 # fixed arguments only
ml_log_Cint("n=%d\n", 7)
ml_log_Cstring_Cdouble("%s=%g\n", "pi", 3.14)
```

## Callbacks

A function-pointer parameter takes a `Ptr{Cvoid}`. Create it with
`@cfunction`; the wrapper function's docstring shows the exact call for that
callback's signature.

```julia
function on_progress(done::Cint, user::Ptr{Cvoid})::Cint
    println("done: ", done)
    return 0
end

cb = @cfunction(on_progress, Cint, (Cint, Ptr{Cvoid}))
run_with_progress(cb, C_NULL)
```

The Julia function must be a named top-level function, not a closure.

## Memory the library allocated

Free it with the library's own function. For a buffer the library got from
`malloc`, use the wrapper's `M.libc_free(ptr)`.

Do not use Julia's `Libc.free` on memory the library allocated. On Windows the
library and Julia use different C runtimes, and freeing across them crashes.
`Libc.free` is for memory you got from `Libc.malloc`.

## Pointers

A function that returns `T*` returns a `Ptr{T}` when the wrapper has the type
`T`, and a `Ptr{Cvoid}` otherwise. A typed pointer says what it points at, not
who owns it: memory the library owns stays the library's.

## Global variables

A global variable has an accessor function that returns its value, and a
`<name>_ptr` accessor that returns its address.

In a C++ wrapper a global inside a namespace or a class is named with its
scope, as a function is: `cfg::depth` is `cfg_depth()`, and a public static
member `geo::Vec3::live_count` is `geo_Vec3_live_count()`. A private or
protected static member has no accessor, and neither does a `constexpr`
member, which has no storage to read. A global whose type is a C++ library
type (`std::string`) has the `_ptr` accessor only.

## Sizes and offsets

```julia
M.struct_size(:ml_vec2)              # 16
M.member_offset(:ml_vec2, :y)        # 8
```

These are not exported; call them with the module's name. They take the
numbers from the compiled library, so use them rather than writing a size or
offset into your code. An unknown name raises an error; `M.STRUCT_SIZES` and
`M.STRUCT_OFFSETS` hold every name they know.

`M.LIBRARY_PATH` is the library file the wrapper calls, and
`M.BUILD_GENERATOR` the RepliBuild version that wrote it.

---

# C++

## Names

A function is named after its namespace, class and name, joined with `_`:

| C++ | Julia |
|:----|:------|
| `shapes::midpoint(a, b)` | `shapes_midpoint(a, b)` |
| `shapes::Rect::area()` | `shapes_Rect_area(rect)` |
| `shapes::Rect::Rect(w, h)` | `shapes_Rect_Rect(storage, w, h)` |
| `shapes::Rect::~Rect()` | `shapes_Rect_destroy_Rect(rect)` |
| `hol::Box<double>::get()` | `hol_Box_double_get(box)` |

**A method takes the object first**, as a pointer. To choose your own names,
use [`[wrap.rename]`](toml.md#Names).

**An operator is named with a word**: `Vec3::operator<` is
`Vec3_operatorless(a, b)`, `operator[]` is `Vec3_operatorindex(v, i)` and
`operator()` is `Vec3_operatorcall(v, x)`. `names(M)` lists the ones your
wrapper has.

## Objects

Most C++ libraries hand you objects through functions that return a pointer.
Use that pointer as the first argument of the class's methods, and give it
back to the library's own function when you are done:

```julia
r = shapes_make_rect(2.0, 3.0)        # Shape* make_rect(double, double)

shapes_Rect_area(r)                   # 6.0
shapes_Shape_rename(r, "door")
shapes_Shape_name_abi_cxx11(r)        # "door"
shapes_Rect_scale(r, 0.5)

shapes_destroy_shape(r)               # the library's own release function
```

A method of a base class takes a pointer to a derived object as it is.

### Let Julia free the object

For a class the library can delete, the wrapper defines an owning type,
`Managed<Class>`. Wrap the pointer in it and Julia's garbage collector deletes
the object when nothing refers to it any more:

```julia
s = ManagedShape(shapes_make_rect(2.0, 3.0))
shapes_Rect_area(s)                   # pass it wherever the pointer goes
```

Wrap a pointer only if you own the object and it came from the library's
allocator. Do not wrap an object the library still owns (a body that belongs
to a physics world, for example), and do not also free it yourself.

For an `extern "C"` function, or a `noexcept` C++ function, that returns a
pointer to such a class, the wrapper also defines `<function>_safe`, which
returns the `Managed` object directly.

A class has a `Managed` type when the library has a release function for it (a
function taking a pointer to the class whose name contains delete, destroy or
free), or when the class has a virtual destructor.

### Construct an object yourself

A constructor wrapper builds the object in memory you provide, like placement
`new` in C++:

```julia
p = Ptr{Counter}(Libc.malloc(Shapes.struct_size(:Counter)))

shapes_Counter_Counter(p, 3)          # Counter::Counter(int)
shapes_Counter_next(p)                # 3
shapes_Counter_next(p)                # 4

shapes_Counter_destroy_Counter(p)     # Counter::~Counter(): runs the destructor, keeps the memory
Libc.free(p)                          # the memory was yours
```

Two destructor wrappers exist, and they do different things:

| Name | Does | Use it on |
|:-----|:-----|:----------|
| `Class_destroy_Class(p)` | Runs the destructor. Does not free the memory. | An object you constructed in your own memory. |
| `Class_delete_Class(p)` | Runs the destructor **and frees the object** (C++ `delete`). | An object that came from `new`. Never free it again afterwards. |

A class may have only one of the two: a class with a virtual destructor often
has only `delete`. `wrap` names those classes, and each function's docstring
says which kind it is.

!!! warning "`Rect()` is not a constructor"
    Every struct type has a zero-argument form that returns a value with all
    bytes zero. For a C++ class that is not a constructed object: no
    constructor has run, and calling a virtual method on it crashes. Construct
    objects with the constructor wrapper or a library function.

### Factory functions

When the library has a free function named `create_…`, `new_…`, `make_…`,
`alloc_…`, `init_…` or `build_…` that returns a pointer to a class, the
wrapper may turn that class into a Julia type you construct directly:
`M.Rect(2.0, 3.0)` calls the factory, the class's methods get short forms
(`M.area(r)`), and the object is deleted by the garbage collector.
`methods(M.Rect)` shows whether your wrapper has this.

When two factories of one class take the same argument types, name the one you
want: with `make_animal(int)` and `make_animal_doubled(int)`,
`M.Animal(Val(:make_animal), Cint(4))` and
`M.Animal(Val(:make_animal_doubled), Cint(4))` call each, and a bare
`M.Animal(Cint(4))` is a `MethodError` that lists both. The type's docstring
names them.

## What a pointer argument accepts

In a C++ wrapper, every pointer parameter, every reference and the object of a
method is checked before the C++ function runs. Each accepts:

| You pass | Accepted for |
|:---------|:-------------|
| A `Ptr` of any type, including `C_NULL` | Everything |
| `Ref(x)` or `Ref(array, i)`, where `x` is a number, a struct, a union or a bitfield record | Every pointer except a function pointer |
| A `Managed` object, an object made by a factory, a container such as `CppVector`, or your own type that defines `Base.unsafe_convert(::Type{Ptr{Cvoid}}, x)` | A pointer to an object, and `void*` |
| A union or bitfield record itself: `M.Flags_get_level(f)` | A pointer to an object, and `void*` |
| An `Array`, or a contiguous view, of `T` | `T*` where `T` is data (`const int*`), and `void*` |
| A string | `char*`, and byte pointers such as `unsigned char*` |
| A `@cfunction` pointer | A function pointer |

Anything else is a `MethodError`, before any C++ runs: a bare struct where the
object goes, a Julia function where a function pointer goes, a
`Vector{Float64}` for a `const int*`, a Julia `Vector` or a `Ref{String}` where
a `std::vector` or `std::string` goes.

A string passed for a `char*` must not contain a zero byte; one that does is an
`ArgumentError`. A `std::string` parameter has no such limit.

A `noexcept` function is called directly and is less strict about what it
converts: pass it a `Ptr`, a `Ref`, an array, or the class's own `Managed`
object.

A wrapper written by an older RepliBuild does not make these checks. Run
`wrap` again.

## Overloads and default arguments

C++ overloads of one name are one Julia function with several methods. Julia
picks the method from your argument types, so where two overloads differ you
must pass the right type:

- an overload taking `const char*` takes a `String`; one taking `const void*`
  at the same position takes a pointer;
- an overload taking an enum takes the enum value, not an integer;
- overloads that differ in integer width take that exact type (`Int32(1)` or
  `Int64(1)`).

An argument that fits no overload is a `MethodError`. It is never a call to
the wrong overload.

A function with default arguments has a shorter method for each argument you
may leave out, with the values C++ would use:

```julia
shapes_Rect_scale(r)          # void scale(double f = 2.0)
shapes_Rect_scale(r, 0.5)
```

This needs [`-fstandalone-debug`](toml.md#Types-declared-only-in-headers), and
covers constructors and member functions.

## std::string

- A function returning `std::string` returns a Julia `String`.
- A parameter declared `std::string` or `const std::string&` takes a Julia
  `String`.

No TOML entry is needed for either. Functions returning a string carry a
suffix on Linux: `shapes_Shape_name_abi_cxx11`. Rename them if you like.

Wide strings (`std::wstring` and the like) are not converted. A non-`const`
`std::string&`, which the function writes into, is not converted either: pass
a pointer to a string object the library gave you.

## Containers

List the containers you want to create in
[`[types] templates`](toml.md#C-templates-and-containers). Each gets a
constructor function, and behaves like a Julia collection:

```julia
v = create_std_vector_double()        # std::vector<double>
push!(v, 1.0)
push!(v, 2.0)
length(v)                             # 2
v[1]                                  # 1.0

shapes_total_area(v)                  # double total_area(const std::vector<double>&)
```

`std::vector`, `std::string`, `std::map` and `std::unordered_map` are
supported this way.

### Containers passed by value

A function that returns a `std::vector<int>` by value (or a map, a set, a
`shared_ptr`) returns the container object as raw bytes. Pass that value back
to a function that takes the container by value, or wrap it in a `Ref` for one
that takes a reference:

```julia
v = M.vec_iota(5)            # std::vector<int> vec_iota(int)
M.vec_sum_val(v)             # int vec_sum_val(std::vector<int>): 10
M.vec_sum(Ref(v))            # int vec_sum(const std::vector<int>&)
```

No copy is made when you pass it by value: the function works on your
container's storage. A function that changes its by-value argument (grows it,
or moves from it) can leave your value unusable.

## Standard types passed by value

When a function takes or returns one of these by value, the wrapper has a
struct for it:

| C++ | Julia |
|:----|:------|
| `std::pair<double, double>` | `pair_double_double`, with fields `first` and `second` |
| `std::array<double, 2>` | `array_double_2UL` |
| `std::tuple<…>`, `std::optional<…>` | A struct of their contents |
| `std::complex<double>` | `complex_double`, holding one `ComplexF64` |
| `std::string_view` | Its pointer and length |

```julia
r = M.minmax_pair(3.0, 1.5)           # std::pair<double, double>
(r.first, r.second)                   # (1.5, 3.0)
```

A library class returned by value comes back as a Julia struct holding its
bytes. Its destructor is not run. For a class that owns resources, prefer the
pointer-returning functions.

## Exceptions

A C++ exception becomes a Julia exception carrying the C++ message. That
includes one thrown by an `extern "C"` function of a C++ library:

```julia
try
    shapes_checked_sqrt(-1.0)
catch e
    e isa RepliBuild.JITManager.CxxException      # true
    showerror(stdout, e)                           # C++ exception: negative input
end
```

## Inheritance

Pass a pointer to a derived object wherever a base-class method wants its
object. That works as is for ordinary single inheritance.

A virtual method runs the override of the class the pointer really points at:
`Animal_legs(p)` on a `Dog` runs `Dog::legs`. That includes a pure virtual
method: `Animal_name(p)` works although `Animal` itself has no body for
`name()`. An abstract class cannot be constructed: `Animal_Animal(p, 4)`
raises an error, as C++ refuses `Animal a(4);`. Construct a class that defines
the method, or take the object from the library.

With multiple or virtual inheritance a base class may not start at the
beginning of the object. For those the wrapper defines
`<Derived>_as_<Base>(p)`. Call the base's methods on its result:

```julia
b = Derived_as_Base2(d)
Base2_method(b)
```

If the wrapper has such a function for a pair of classes, use it. If it has
none, pass the pointer directly.

## Small objects returned by value

Some libraries return small handle objects by value (pugixml's `xml_node`).
Methods want a pointer to them. A helper does the conversion:

```julia
withptr(f, h) = (r = Ref(h); GC.@preserve r f(Base.unsafe_convert(Ptr{typeof(h)}, r)))

name = withptr(node -> pugi_xml_node_name(node), root)
```

## Types declared inside a class

A type declared inside a C++ class has two names in the wrapper, its own and
one with the class in front: `ClosestRayResultCallback` and
`btCollisionWorld_ClosestRayResultCallback`. They are the same type.

---

# A CMake project's module

A [CMake project](graph.md) is one module with a submodule per library. Call
each library through its submodule. A type that several libraries share is
one type, so values pass between them:

```julia
include("julia/HelloWorld.jl")
H, W = HelloWorld.Hello, HelloWorld.World

W.Vec2 === H.Vec2                                        # true
W.hw_travel(H.hw_vec2(0f0, 0f0), H.hw_vec2(3f0, 4f0))    # 5.0f0
```

Load the top module, never one library's file by itself.
