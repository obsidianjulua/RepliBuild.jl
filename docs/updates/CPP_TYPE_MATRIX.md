# C++ standard-library types: where a wrapper stands

A measurement, not a design. One small library per type family is built,
wrapped, and called from Julia the way a wrapper's user would call it. For each
type: can Julia make one, pass one, read one back.

To measure again (about ten minutes, no Hub, no network):

```bash
julia --project=. test/tools/cpp_type_matrix.jl
```

The libraries and probes are `test/tools/cpp_type_matrix/<family>/`. A probe
uses what the manual gives a user (`docs/src/calling.md`) and nothing else.
Everything from "The measurement" on is the tool's output, unedited.

## What it shows

Measured 2026-11-27 with the C++ ownership change (RepliBuild 4.6.0 plus
unreleased changes), Julia 1.13, Linux x86-64, clang 23 with libstdc++ 16,
thunks compiled at load. Not measured: thunks compiled ahead of time, libc++,
AArch64, Windows. First measured 2026-11-23; what each measurement found and
has since been fixed is at the end of this section.

**No case is wrong and none kills the process.** 193 cases: 111 work, 32 are
handles, 20 take no plain Julia value, 5 are refused, 22 have nothing to
pass, 3 are errors (explained below). The measurement of 2026-11-24 had 10
wrong.

The coverage tool reads these wrappers as 97.01% bound: 162 of 167 functions,
5 refused. That counts a function whose thunk the wrapper invokes. The matrix
asks what a user can do with it.

### Works with a Julia value

Only `std::string`. A `String` goes in where C++ takes `std::string` or
`const std::string&`, and a returned `std::string` is a `String`.

### Works as a struct with fields

`std::pair`, `std::tuple`, `std::array`, `std::complex`, `std::chrono`
durations and time points, `std::bitset`, `std::string_view`, `std::span`,
`std::initializer_list`, `std::error_code`. Passed and read back; all can be
made from Julia but the last.

- **No Julia value converts.** A `ComplexF64`, a `Tuple`, a `Vector`, a
  `String` for a `string_view` is a `MethodError`: the struct has to be built.
- **A view is built by hand** from a pointer and a length (`string_view`,
  `span`, `initializer_list`), and the caller keeps the Julia object alive.
- **A tuple's fields have the C++ library's names**, last element first:
  `_M_head_impl`, `_M_head_impl_1`, `_M_head_impl_2`.

### Works through a container object

`std::vector<int>`, `<double>` and of a library struct, `std::map<int, int>`
and `std::unordered_map<int, int>`: `create_std_…()` gives an object that
behaves like a Julia collection, and the library takes it by value or by
reference.

- `std::vector<std::string>`, `std::vector<std::vector<int>>`, `std::set`,
  `std::deque`, `std::list` get no constructor.
- `std::map<std::string, int>` gets one whose keys are `Ptr{Cvoid}`: a `String`
  key is a `MethodError`.
- An empty `create_std_basic_string_char()` is zeroed memory, not a
  constructed string: `String(s)` on it is an error until the library has
  written into it.

### Held where C++ built it: a handle that is ended

Everything that owns memory and comes back by value: every container, the
smart pointers, `std::optional<std::string>`, `std::any`,
`std::filesystem::path`, wide strings, a `std::pair` holding a string. Since
2026-11-27 each is the C++ object itself, kept in the storage the call built
it in, and Julia runs its destructor when it collects the object or at
`RepliBuild.release!`.

- **It is valid where Julia holds it.** A map's last key, a walk to a list's
  end, a short string inside a pair or a path: all ten cases that were wrong
  read right (`intact after the return` in the table).
- **It is ended, once.** Of two `unique_ptr`s returned and dropped, both
  objects are deleted after a collection.
- **By value, C++ copies or moves.** A `shared_ptr` passed by value twice
  leaves the use count at 1 and the caller's pointer whole; a `unique_ptr`
  passed by value is deleted when the call returns and is spent, and passing
  it again is an `ArgumentError`.
- **Nothing reads what is inside** from Julia (a vector's elements, an
  optional's string): that is why these cells say OPAQUE and not WORKS. A
  record's fields do read, where it has them: a `std::weak_ptr`'s object
  pointer, a `std::pair`'s `second`.

**Where it does not reach.** The build asks its compiler about each type in
a unit of its own, and that unit has to be able to name the type. A class
defined in a source file cannot be included, so a `std::unique_ptr` or
`std::shared_ptr` to one keeps the old behaviour: bytes that nothing ends.
The two smart-pointer libraries here declare their classes in a header, as
a library does, for that reason.

### Bound, and nothing to pass

- **`std::variant` and `std::istream` by reference.** The parameter is a `Ref`
  to an empty struct and nothing makes one. Three functions the coverage tool
  counts as bound.
- **Nothing makes a `std::function`.** A Julia function and a `@cfunction` are
  not taken, by value or by reference. One the library owns passes both ways.
- **A `std::ostream&` does not take a `std::ostringstream*`** until the pointer
  is converted by hand.

### Refused

`std::function` returned by value, and `std::variant` and a fixed-extent
`std::span` by value. The refusal is correct. Its advice is not: it names
`-fstandalone-debug`, which these libraries are built with.

### The three errors

None is a wrong call: each is a `MethodError` or an `ArgumentError` raised in
Julia before or after C++ ran, and each is a gap listed above.

- **`std::map<std::string, int>`, made from Julia**: `m["one"] = 1` is a
  `MethodError`. The constructor function gives a map whose key type is
  `Ptr{Cvoid}` ("Works through a container object").
- **A `std::ostringstream*` at a `std::ostream&`** is a `MethodError`: the
  wrapper does not know the one derives from the other ("Bound, and nothing
  to pass").
- **An empty `create_std_basic_string_char()` read back** is an
  `ArgumentError`: it is zeroed memory, not a constructed string ("Works
  through a container object").

### Exceptions

A `std::runtime_error`, a `std::out_of_range`, a class derived from
`std::exception` and a thrown `int` each arrive as a Julia `CxxException`, the
first three with the C++ message, and the next call works.

### Fixed since the first measurement

Found on 2026-11-23. Everything here but the last item of the first list
reaches a library when it is rebuilt.

- **`std::vector<bool>` returned by value killed the process.** The wrapper
  gave it the 24 bytes of a `std::vector<T>`; it is 40. The size now comes
  from the DWARF.
- **`std::unique_ptr<T[]>` returned by value was called as a return in a
  register**, with no hidden return pointer: it killed the process or threw
  `std::bad_alloc`. It is now built in the caller's buffer.
- **`std::function` by value was refused.** It takes a pointer to one, and
  refuses anything else before any C++ runs.
- **A reference to a smart pointer refused the bytes one is returned as**
  (`sp_read(Ref(sp_make(5)))`, the manual's way for a returned container),
  and a `std::function*` the library returned came back untyped and was
  refused where a `const std::function&` is taken. Both pass since
  2026-11-24, with no rebuild.

Fixed on 2026-11-27, by the ownership change:

- **A value that points into itself was not valid in Julia's hands.** It was
  built in a buffer and its bytes were then copied into a Julia value. A
  `std::list`, `std::map`, `std::set` and `std::unordered_map` keep a pointer
  to a marker inside themselves, and a short `std::string` keeps its
  characters inside itself. Lookups happened to work; a map's last key read
  as 0 where it is 3; a walk of the list would not end.
- **Nothing ended it**, and a by-value pass lent the caller's bytes where C++
  copies or moves.

### Seen in passing

- A function returning a type that holds a `std::string` is named
  `<name>_abi_cxx11` (documented in the manual).
- The wrapper's namespace takes the C++ library's own enums: the six-function
  chrono library exports 100 names, 78 of them `std::errc`'s enumerators.
  The vector library exports ten `std::move_iterator<…>::base` functions.
- A `double` parameter is `::Cdouble`: `cx_make(3, 4)` is a `MethodError`
  where an integer parameter takes any `Integer`.

## The measurement

Measured 2026-11-27 with RepliBuild 4.6.0 on Julia 1.13.4 by `test/tools/cpp_type_matrix.jl`, on x86_64-linux-gnu. Probe libraries compiled with `-std=c++20 -O1 -fPIC -fstandalone-debug`; thunks compiled when the wrapper loads.

193 cases: 111 works, 32 opaque, 20 not taken, 5 refused, 22 no way, 3 error.

| Type | make | pass by value | pass by reference | return by value | return by reference | other |
|:--|:--|:--|:--|:--|:--|:--|
| `std::any` | WORKS | WORKS | WORKS | OPAQUE |  | make (holding a value): NO WAY |
| `std::array<int, 4>` | WORKS | WORKS | WORKS | WORKS |  | takes a Julia Tuple: NOT TAKEN |
| `std::array<double, 3>` | WORKS | WORKS |  | WORKS |  |  |
| `std::atomic<int>` | NO WAY |  | WORKS |  | OPAQUE |  |
| `std::mutex` | NO WAY |  | WORKS |  | OPAQUE |  |
| `std::chrono::milliseconds` | WORKS | WORKS | WORKS | WORKS |  |  |
| `std::chrono::duration<double>` | WORKS |  |  |  |  |  |
| `std::chrono::system_clock::time_point` | WORKS | WORKS |  | WORKS |  |  |
| `std::complex<double>` | WORKS | WORKS | WORKS | WORKS |  | takes a Julia Complex: NOT TAKEN; a double parameter takes an Integer: NOT TAKEN |
| `std::complex<float>` | WORKS |  |  | WORKS |  |  |
| `std::runtime_error` |  |  |  |  |  | thrown: WORKS; the next call after a throw: WORKS |
| `std::out_of_range` |  |  |  |  |  | thrown: WORKS |
| `a class derived from std::exception` |  |  |  |  |  | thrown: WORKS |
| `int (not a std::exception)` |  |  |  |  |  | thrown: WORKS |
| `std::error_code` | NO WAY | WORKS |  | WORKS |  |  |
| `std::function<int(int)>` | NO WAY | WORKS | WORKS | REFUSED | OPAQUE | takes a Julia function: NOT TAKEN; takes a @cfunction: NOT TAKEN; takes an untyped pointer: NOT TAKEN |
| `std::function<void(int)>` |  |  | NOT TAKEN |  |  |  |
| `std::function<double(double, double)>` |  |  | NOT TAKEN |  |  |  |
| `int (*)(int) (a plain function pointer)` | WORKS |  |  |  |  | takes a Julia function: NOT TAKEN |
| `std::initializer_list<int>` | WORKS | WORKS | WORKS |  |  | takes a Julia Vector: NOT TAKEN |
| `std::initializer_list<double>` |  | WORKS |  |  |  |  |
| `std::map<int, int>` | WORKS | WORKS | WORKS | OPAQUE |  | takes a Julia Dict: NOT TAKEN; intact after the return: WORKS |
| `std::map<std::string, int>` | ERROR |  |  | OPAQUE |  |  |
| `std::unordered_map<int, int>` | WORKS |  |  | OPAQUE |  | intact after the return: WORKS |
| `std::set<int>` | NO WAY |  |  | OPAQUE |  | intact after the return: WORKS |
| `std::optional<int>` | WORKS | WORKS | WORKS | OPAQUE |  | make (holding a value): NO WAY |
| `std::optional<double>` |  |  |  | OPAQUE |  |  |
| `std::optional<std::string>` |  |  |  | OPAQUE |  | intact after the return: WORKS |
| `std::pair<int, double>` | WORKS | WORKS | WORKS | WORKS |  | takes a Julia Pair or Tuple: NOT TAKEN |
| `std::pair<std::string, int>` |  |  |  | OPAQUE |  | intact after the return: WORKS |
| `std::tuple<int, double, char>` | WORKS | WORKS | WORKS | WORKS |  | takes a Julia Tuple: NOT TAKEN |
| `std::filesystem::path` | NO WAY | WORKS | WORKS | OPAQUE |  | takes a Julia String: NOT TAKEN; intact after the return: WORKS |
| `std::deque<int>` | NO WAY |  |  | OPAQUE |  |  |
| `std::list<int>` | NO WAY |  |  | OPAQUE |  | intact after the return: WORKS |
| `std::bitset<8>` | WORKS | WORKS |  | WORKS |  |  |
| `std::shared_ptr<SNode>` | NO WAY | WORKS | WORKS | OPAQUE |  | use count after a by-value pass: WORKS; copied and ended by Julia: WORKS |
| `std::weak_ptr<SNode>` |  |  | WORKS | WORKS |  |  |
| `std::span<const int>` | WORKS | WORKS | WORKS |  |  | takes a Julia Vector: NOT TAKEN |
| `std::span<int>` |  | WORKS |  | WORKS |  |  |
| `std::span<const int, 3> (fixed extent)` |  | REFUSED |  |  |  |  |
| `std::ostringstream` | NO WAY |  | WORKS |  | OPAQUE |  |
| `std::ostream` |  |  | ERROR, WORKS |  | OPAQUE |  |
| `std::istream` |  |  | NO WAY |  |  |  |
| `std::string` | WORKS | WORKS | WORKS | WORKS | OPAQUE |  |
| `std::string& (written by the function)` | WORKS | WORKS | WORKS |  |  | an empty CppString reads back as "": ERROR; takes a Julia String: NOT TAKEN |
| `std::string_view` | WORKS | WORKS | WORKS | WORKS |  | takes a Julia String: NOT TAKEN |
| `std::unique_ptr<UNode>` | NO WAY | WORKS | WORKS | OPAQUE |  |  |
| `std::unique_ptr<UCounted> (counts live objects)` |  |  |  |  |  | ended by Julia: WORKS; ended by release!: WORKS; ended by a by-value pass: WORKS |
| `std::unique_ptr<int[]>` |  |  |  | OPAQUE |  |  |
| `std::variant<int, double>` | NO WAY | REFUSED | NO WAY | REFUSED |  |  |
| `std::variant<int, std::string>` |  |  | NO WAY | REFUSED |  |  |
| `std::vector<int>` | WORKS | WORKS | WORKS | OPAQUE |  | takes a Julia Vector: NOT TAKEN |
| `std::vector<double>` | WORKS |  |  | OPAQUE |  |  |
| `std::vector<P2> (a library struct)` | WORKS |  |  | OPAQUE |  |  |
| `std::vector<std::string>` | NO WAY |  |  | OPAQUE |  |  |
| `std::vector<std::vector<int>>` | NO WAY |  |  | OPAQUE |  |  |
| `std::vector<bool>` | NO WAY |  |  | OPAQUE |  |  |
| `std::wstring` | NO WAY |  | WORKS | OPAQUE |  | takes a Julia String: NOT TAKEN; intact after the return: WORKS |
| `std::u16string` |  |  |  | OPAQUE |  | intact after the return: WORKS |
| `std::u8string` |  |  |  | NO WAY |  |  |

`make` is a value made from Julia with no function of the library making it. A cell with two words had two cases; every case is listed below with what was tried.

- **WORKS**: called from Julia with what a wrapper's user has, and the answer is the library's.
- **OPAQUE**: it crosses and the library reads it right, and the wrapper gives Julia no way to read or build what is inside: raw bytes or a pointer.
- **NOT TAKEN**: a plain Julia value of the natural type is a `MethodError` there; the wrapper's own type is needed.
- **REFUSED**: the wrapper refuses the call and says why.
- **NO WAY**: the function is bound, and nothing a wrapper gives can be passed to it, or what it returns cannot be used.
- **WRONG**: it ran and the answer is not the library's.
- **ERROR**: it threw something that is not a stated refusal.
- **ABSENT**: the wrapper has no such function.
- **CRASH**: the process died in the call, or never came back.

## Every case

### any

- `std::any`, make: **WORKS** — an empty one: `M.any()`, all zero
- `std::any`, make (holding a value): **NO WAY** — no create function in the wrapper; the struct is 16 bytes with no field for a value; the library's `any_int` is the way
- `std::any`, return by value: **OPAQUE** — the C++ object, a RepliBuild.CxxObjects.Owned{any}; what it holds is read only through the library
- `std::any`, pass by reference: **WORKS** — Ref of the returned struct
- `std::any`, pass by value: **WORKS** — passed twice

### array

- `std::array<int, 4>`, make: **WORKS** — the struct, from a tuple
- `std::array<int, 4>`, takes a Julia Tuple: **NOT TAKEN** — an NTuple{4,Cint}: no method arr_sum(::NTuple{4, Int32})
- `std::array<int, 4>`, pass by value: **WORKS**
- `std::array<int, 4>`, pass by reference: **WORKS** — const&, a Ref of the struct
- `std::array<int, 4>`, pass by reference: **WORKS** — the library writes through the Ref
- `std::array<int, 4>`, return by value: **WORKS** — read from the field `_M_elems`
- `std::array<double, 3>`, make: **WORKS**
- `std::array<double, 3>`, pass by value: **WORKS**
- `std::array<double, 3>`, return by value: **WORKS**

### atomic_mutex

- `std::atomic<int>`, make: **NO WAY** — no create function in the wrapper; `M.atomic_int()` is an empty struct
- `std::atomic<int>`, return by reference: **OPAQUE** — a Ptr{atomic_int}: the library counts through it, and Julia cannot read the value
- `std::atomic<int>`, pass by reference: **WORKS** — the pointer the library returned
- `std::mutex`, make: **NO WAY** — no create function in the wrapper; `M.mutex()` is an empty struct
- `std::mutex`, return by reference: **OPAQUE** — a Ptr{mutex}
- `std::mutex`, pass by reference: **WORKS** — locked and released twice

### chrono

- `std::chrono::milliseconds`, make: **WORKS** — the struct's one field, the count
- `std::chrono::milliseconds`, pass by value: **WORKS**
- `std::chrono::milliseconds`, pass by reference: **WORKS**
- `std::chrono::milliseconds`, return by value: **WORKS** — read from the field `_r`
- `std::chrono::duration<double>`, make: **WORKS**
- `std::chrono::system_clock::time_point`, return by value: **WORKS** — nanoseconds, two fields deep
- `std::chrono::system_clock::time_point`, pass by value: **WORKS**
- `std::chrono::system_clock::time_point`, make: **WORKS** — struct of a struct of the count

### complex

- `std::complex<double>`, make: **WORKS** — the struct, from a ComplexF64
- `std::complex<double>`, takes a Julia Complex: **NOT TAKEN** — a ComplexF64: no method cx_abs(::ComplexF64)
- `std::complex<double>`, pass by value: **WORKS**
- `std::complex<double>`, pass by reference: **WORKS**
- `std::complex<double>`, return by value: **WORKS** — a ComplexF64 in the field `_M_value`
- `std::complex<double>`, a double parameter takes an Integer: **NOT TAKEN** — an Int where C++ takes a double: no method cx_make(::Int64, ::Int64)
- `std::complex<float>`, make: **WORKS**
- `std::complex<float>`, return by value: **WORKS**

### exception

- `std::runtime_error`, thrown: **WORKS** — a Julia CxxException carrying the C++ message
- `std::out_of_range`, thrown: **WORKS** — a Julia CxxException carrying the C++ message
- `a class derived from std::exception`, thrown: **WORKS** — a Julia CxxException carrying the C++ message
- `int (not a std::exception)`, thrown: **WORKS** — a Julia CxxException: C++ exception: unknown C++ exception
- `std::runtime_error`, the next call after a throw: **WORKS**
- `std::error_code`, return by value: **WORKS** — the value is a field; the category is a pointer
- `std::error_code`, pass by value: **WORKS**
- `std::error_code`, make: **NO WAY** — the category is a pointer only the library can supply; the value of a code the library returned can be changed with setproperties (12 reads back)

### function

- `std::function<int(int)>`, make: **NO WAY** — no create function in the wrapper; `M.function_int_int()` is an empty struct, and nothing turns a Julia function or a @cfunction into a std::function
- `std::function<int(int)>`, takes a Julia function: **NOT TAKEN** — a Julia function, by reference: no method fn_apply_ref(::typeof(twice), ::Int64)
- `std::function<int(int)>`, takes a Julia function: **NOT TAKEN** — a Julia function, by value: no method pointer_arg(::DataType, ::typeof(twice))
- `std::function<int(int)>`, takes a @cfunction: **NOT TAKEN** — a @cfunction pointer: no method fn_apply_ref(::Ptr{Nothing}, ::Int64)
- `std::function<int(int)>`, return by reference: **OPAQUE** — one the library owns comes back as a Ptr{function_int_int}
- `std::function<int(int)>`, pass by value: **WORKS** — a pointer to one the library owns, passed twice; no copy is made
- `std::function<int(int)>`, pass by reference: **WORKS** — the pointer as the library returned it
- `std::function<int(int)>`, takes an untyped pointer: **NOT TAKEN** — a Ptr{Cvoid} to one the library owns: no method fn_apply_ref(::Ptr{Nothing}, ::Int64)
- `std::function<int(int)>`, return by value: **REFUSED** — cannot call 'fn_adder' (fn_adder(int)). It returns `function<int (int)>` by value, which C++ returns through a hidden pointer the thunk does not pass, and that record's l…
- `std::function<void(int)>`, pass by reference: **NOT TAKEN** — a Julia function: no method fn_each(::var"#51#52", ::Int64)
- `std::function<double(double, double)>`, pass by reference: **NOT TAKEN** — a Julia function: no method fn_fold(::typeof(+), ::Float64, ::Float64)
- `int (*)(int) (a plain function pointer)`, make: **WORKS** — @cfunction
- `int (*)(int) (a plain function pointer)`, takes a Julia function: **NOT TAKEN** — a Julia function: no method pointer_arg(::DataType, ::typeof(twice))

### initializer_list

- `std::initializer_list<int>`, make: **WORKS** — built by hand from pointer(v) and its length; the caller keeps the Vector alive
- `std::initializer_list<int>`, takes a Julia Vector: **NOT TAKEN** — a Vector{Cint}: no method il_sum(::Vector{Int32})
- `std::initializer_list<int>`, pass by value: **WORKS**
- `std::initializer_list<int>`, pass by reference: **WORKS**
- `std::initializer_list<double>`, pass by value: **WORKS**

### map

- `std::map<int, int>`, make: **WORKS** — create_std_map_int_int(), setindex!, getindex, haskey
- `std::map<int, int>`, takes a Julia Dict: **NOT TAKEN** — a Dict{Cint,Cint}: no method pointer_arg(::DataType, ::Dict{Int32, Int32})
- `std::map<int, int>`, pass by reference: **WORKS** — a CppMap
- `std::map<int, int>`, pass by value: **WORKS** — a CppMap, passed twice
- `std::map<int, int>`, return by value: **OPAQUE** — comes back as the C++ object, a RepliBuild.CxxObjects.Owned{NTuple{48, UInt8}}; lookups through Ref(r) are right
- `std::map<int, int>`, intact after the return: **WORKS**
- `std::map<std::string, int>`, make: **ERROR** — MethodError: no method convert(::DataType, ::String)
- `std::map<std::string, int>`, return by value: **OPAQUE** — the C++ object; the key is a Julia String
- `std::unordered_map<int, int>`, make: **WORKS** — create_std_unordered_map_int_int()
- `std::unordered_map<int, int>`, return by value: **OPAQUE** — the C++ object, a RepliBuild.CxxObjects.Owned{NTuple{56, UInt8}}; all 5 keys were found through Ref(r)
- `std::unordered_map<int, int>`, intact after the return: **WORKS**
- `std::set<int>`, make: **NO WAY** — no create function in the wrapper; the entry is in [types] templates and gets no constructor
- `std::set<int>`, return by value: **OPAQUE** — the C++ object
- `std::set<int>`, intact after the return: **WORKS**

### optional

- `std::optional<int>`, make: **WORKS** — an empty one: `M.optional_int()`, all zero
- `std::optional<int>`, make (holding a value): **NO WAY** — the struct's only field is _M_payload, a block of bytes: a value and its flag have to be written into the bytes by hand
- `std::optional<int>`, pass by value: **WORKS**
- `std::optional<int>`, pass by reference: **WORKS**
- `std::optional<int>`, return by value: **OPAQUE** — a struct of bytes: the value and the flag are read by hand at their offsets, no field names them
- `std::optional<double>`, return by value: **OPAQUE** — a struct of bytes
- `std::optional<std::string>`, return by value: **OPAQUE** — a struct of bytes; the string inside is not a Julia String
- `std::optional<std::string>`, intact after the return: **WORKS**

### pair_tuple

- `std::pair<int, double>`, make: **WORKS** — `pair_int_double()` zeroed, then setproperties; the positional constructor also wants the padding field
- `std::pair<int, double>`, takes a Julia Pair or Tuple: **NOT TAKEN** — a Tuple{Cint,Float64}: no method pair_sum(::Tuple{Int32, Float64})
- `std::pair<int, double>`, pass by value: **WORKS**
- `std::pair<int, double>`, pass by reference: **WORKS**
- `std::pair<int, double>`, return by value: **WORKS** — fields `first` and `second`
- `std::pair<std::string, int>`, return by value: **OPAQUE** — `second` is a field; `first` is the string's bytes, a NTuple{32, UInt8}, not a Julia String
- `std::pair<std::string, int>`, intact after the return: **WORKS**
- `std::tuple<int, double, char>`, return by value: **WORKS** — the elements are fields in the order the library stores them (last first): _M_head_impl, _M_head_impl_1, _M_head_impl_2
- `std::tuple<int, double, char>`, make: **WORKS** — zeroed, then setproperties on the library's internal field names
- `std::tuple<int, double, char>`, takes a Julia Tuple: **NOT TAKEN** — a Tuple{Cint,Float64,Cchar}: no method tup_sum(::Tuple{Int32, Float64, Int8})
- `std::tuple<int, double, char>`, pass by value: **WORKS**
- `std::tuple<int, double, char>`, pass by reference: **WORKS**

### path

- `std::filesystem::path`, make: **NO WAY** — no create function in the wrapper; `M.path()` is zeroed bytes, not a constructed path, and no field takes text
- `std::filesystem::path`, takes a Julia String: **NOT TAKEN** — a Julia String: no method path_len(::String)
- `std::filesystem::path`, return by value: **OPAQUE** — the C++ object, a RepliBuild.CxxObjects.Owned{path}; its text is read only through the library
- `std::filesystem::path`, pass by reference: **WORKS** — Ref of the value the library returned
- `std::filesystem::path`, pass by value: **WORKS** — and the std::string it returns is a Julia String
- `std::filesystem::path`, pass by value: **WORKS** — 68 characters
- `std::filesystem::path`, intact after the return: **WORKS**
- `std::filesystem::path`, return by value: **OPAQUE** — operator/ result

### sequence

- `std::deque<int>`, make: **NO WAY** — no create function in the wrapper; not in [types] templates, and a deque gets no constructor there either
- `std::deque<int>`, return by value: **OPAQUE** — comes back as the C++ object, a RepliBuild.CxxObjects.Owned{NTuple{80, UInt8}}
- `std::list<int>`, make: **NO WAY** — no create function in the wrapper; not in [types] templates, and a list gets no constructor there either
- `std::list<int>`, intact after the return: **WORKS**
- `std::list<int>`, return by value: **OPAQUE** — the C++ object, a RepliBuild.CxxObjects.Owned{NTuple{24, UInt8}}
- `std::bitset<8>`, make: **WORKS** — the struct's one field, a word
- `std::bitset<8>`, pass by value: **WORKS**
- `std::bitset<8>`, return by value: **WORKS** — read from the field `_M_w`

### shared_ptr

- `std::shared_ptr<SNode>`, make: **NO WAY** — no create function in the wrapper; the library's `sp_make` is the way
- `std::shared_ptr<SNode>`, return by value: **OPAQUE** — comes back as the C++ object, a RepliBuild.CxxObjects.Owned{NTuple{16, UInt8}}, held where C++ built it; its bytes are the object pointer, then the control block's
- `std::shared_ptr<SNode>`, pass by reference: **WORKS** — the object, and Ref(r) as the manual said while a container was its bytes
- `std::shared_ptr<SNode>`, pass by reference: **WORKS** — a hand-made Ptr{shared_ptr_SNode} to the object
- `std::shared_ptr<SNode>`, pass by value: **WORKS** — the object, passed twice: C++ copies it, and the caller keeps its own
- `std::shared_ptr<SNode>`, use count after a by-value pass: **WORKS**
- `std::shared_ptr<SNode>`, copied and ended by Julia: **WORKS** — copy(r) is a second owner; release!, or a collection, ends it
- `std::weak_ptr<SNode>`, return by value: **WORKS** — the C++ object, a RepliBuild.CxxObjects.Owned{weak_ptr_SNode}; the object pointer reads as a field
- `std::weak_ptr<SNode>`, pass by reference: **WORKS** — the returned object, and Ref of it

### span

- `std::span<const int>`, make: **WORKS** — built by hand: pointer(v), and the length inside a second struct; the caller keeps the Vector alive
- `std::span<const int>`, takes a Julia Vector: **NOT TAKEN** — a Vector{Cint}: no method span_sum(::Vector{Int32})
- `std::span<const int>`, pass by value: **WORKS**
- `std::span<const int>`, pass by reference: **WORKS**
- `std::span<int>`, pass by value: **WORKS** — the library writes the Julia Vector's memory
- `std::span<int>`, return by value: **WORKS** — pointer and length read from the struct's fields
- `std::span<const int, 3> (fixed extent)`, pass by value: **REFUSED** — cannot call 'span_fixed' (span_fixed(std::span<int const, 3ul>)). It takes `span<const int, 3UL>` by value, and that record's layout is not in this library's DWARF, so no…

### stream

- `std::ostringstream`, make: **NO WAY** — no create function in the wrapper; the library's `oss_new` is the way
- `std::ostringstream`, return by reference: **OPAQUE** — a pointer, a Ptr{basic_ostringstream_char_std_char_traits_char_std_allocator_char}; its text is read through the library, as a Julia String
- `std::ostringstream`, pass by reference: **WORKS** — const std::ostringstream&, the pointer
- `std::ostream`, pass by reference: **ERROR** — MethodError: no method os_write(::Ptr{basic_ostringstream_char_std_char_traits_char_std_allocator_char}, ::Int64)
- `std::ostream`, pass by reference: **WORKS** — the pointer converted by hand: Ptr{basic_ostream_char_std_char_traits_char}(o)
- `std::ostream`, return by reference: **OPAQUE** — std::cout comes back as a Ptr{basic_ostream_char_std_char_traits_char}
- `std::istream`, pass by reference: **NO WAY** — no create function in the wrapper; `is_read` is bound and takes a `Ref{basic_istream_char_std_char_traits_char}`; no function of the wrapper makes an input stream

### string

- `std::string`, make: **WORKS** — a Julia String is taken where a std::string is
- `std::string`, pass by value: **WORKS**
- `std::string`, pass by reference: **WORKS** — const std::string&
- `std::string`, return by value: **WORKS** — a Julia String
- `std::string`, return by value: **WORKS** — 100 characters
- `std::string`, return by reference: **OPAQUE** — const std::string& comes back as a Ptr{Nothing}: it passes back to the library, and nothing the manual names reads its text
- `std::string`, return by reference: **OPAQUE** — std::string* comes back as a Ptr{Nothing}
- `std::string& (written by the function)`, make: **WORKS** — create_std_basic_string_char(), a CppString; filled by the library, read back with String(s)
- `std::string& (written by the function)`, an empty CppString reads back as "": **ERROR** — ArgumentError: cannot convert NULL to string
- `std::string& (written by the function)`, pass by reference: **WORKS** — through a CppString
- `std::string& (written by the function)`, takes a Julia String: **NOT TAKEN** — a Julia String: no method pointer_arg(::DataType, ::String)
- `std::string& (written by the function)`, pass by value: **WORKS** — a CppString, passed twice

### string_view

- `std::string_view`, make: **WORKS** — built by hand from pointer(s) and its length; the caller keeps the String alive
- `std::string_view`, takes a Julia String: **NOT TAKEN** — a Julia String: no method sv_len(::String)
- `std::string_view`, pass by value: **WORKS**
- `std::string_view`, pass by reference: **WORKS**
- `std::string_view`, return by value: **WORKS** — a struct of length and pointer, read with unsafe_string
- `std::string_view`, return by value: **WORKS** — a view into the caller's String

### unique_ptr

- `std::unique_ptr<UNode>`, make: **NO WAY** — no create function in the wrapper; the library's `up_make` is the way
- `std::unique_ptr<UNode>`, return by value: **OPAQUE** — comes back as the C++ object, a RepliBuild.CxxObjects.Owned{NTuple{8, UInt8}}, held where C++ built it; its bytes are the pointer it owns
- `std::unique_ptr<UNode>`, pass by reference: **WORKS** — the object, and Ref(r) as the manual said while a container was its bytes
- `std::unique_ptr<UNode>`, pass by reference: **WORKS** — a hand-made Ptr{unique_ptr_UNode_std_default_delete_UNode} to the object
- `std::unique_ptr<UNode>`, pass by value: **WORKS** — moved, as C++ moves what it cannot copy: the object is spent, and passing it again is an ArgumentError
- `std::unique_ptr<UCounted> (counts live objects)`, ended by Julia: **WORKS** — both objects were deleted when Julia collected the two unique_ptrs
- `std::unique_ptr<UCounted> (counts live objects)`, ended by release!: **WORKS** — RepliBuild.release!(r) runs the destructor at once, and once
- `std::unique_ptr<UCounted> (counts live objects)`, ended by a by-value pass: **WORKS** — the object is deleted when the call returns, as in C++
- `std::unique_ptr<int[]>`, return by value: **OPAQUE** — the C++ object; its bytes are the pointer to the array

### variant

- `std::variant<int, double>`, make: **NO WAY** — no create function in the wrapper; `M.variant_int_double()` is an empty struct
- `std::variant<int, double>`, return by value: **REFUSED** — cannot call 'var_make' (var_make(int)). It returns `variant<int, double>` by value, and that record's layout is not in this library's DWARF, so no call could pass it corr…
- `std::variant<int, double>`, pass by value: **REFUSED** — cannot call 'var_index' (var_index(std::variant<int, double>)). It takes `variant<int, double>` by value, and that record's layout is not in this library's DWARF, so no c…
- `std::variant<int, double>`, pass by reference: **NO WAY** — `var_value` is bound and takes a `Ref{variant_int_double}`, an empty struct; nothing makes or returns one
- `std::variant<int, std::string>`, return by value: **REFUSED** — cannot call 'vars_make_abi_cxx11' (vars_make[abi:cxx11](int)). It returns `variant<int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >`…
- `std::variant<int, std::string>`, pass by reference: **NO WAY** — `vars_index` is bound and takes a Ref to an empty struct; nothing makes or returns one

### vector

- `std::vector<int>`, make: **WORKS** — create_std_vector_int(), push!, indexing
- `std::vector<int>`, takes a Julia Vector: **NOT TAKEN** — a Vector{Cint}: no method pointer_arg(::DataType, ::Vector{Int32})
- `std::vector<int>`, pass by reference: **WORKS** — const std::vector<int>&, a CppVector
- `std::vector<int>`, pass by reference: **WORKS** — std::vector<int>& the library grows
- `std::vector<int>`, pass by value: **WORKS** — a CppVector, passed twice
- `std::vector<int>`, return by value: **OPAQUE** — comes back as the C++ object, a RepliBuild.CxxObjects.Owned{NTuple{24, UInt8}}: it passes back by value (a copy) and by reference, and its elements cannot be read from Julia
- `std::vector<double>`, make: **WORKS**
- `std::vector<double>`, return by value: **OPAQUE** — the C++ object
- `std::vector<P2> (a library struct)`, make: **WORKS**
- `std::vector<P2> (a library struct)`, return by value: **OPAQUE** — the C++ object
- `std::vector<std::string>`, make: **NO WAY** — no create function in the wrapper; the entry is in [types] templates and gets no constructor
- `std::vector<std::string>`, return by value: **OPAQUE** — the C++ object; the strings cannot be read from Julia
- `std::vector<std::vector<int>>`, make: **NO WAY** — no create function in the wrapper; the entry is in [types] templates and gets no constructor
- `std::vector<std::vector<int>>`, return by value: **OPAQUE** — the C++ object
- `std::vector<bool>`, make: **NO WAY** — no create function in the wrapper; not in [types] templates
- `std::vector<bool>`, return by value: **OPAQUE** — the C++ object, a RepliBuild.CxxObjects.Owned{NTuple{40, UInt8}}

### wide_string

- `std::wstring`, make: **NO WAY** — no create function in the wrapper; no Julia value is converted either: wide strings are not decoded in either direction
- `std::wstring`, takes a Julia String: **NOT TAKEN** — a Julia String: no method pointer_arg(::DataType, ::String)
- `std::wstring`, return by value: **OPAQUE** — comes back as the C++ object, a RepliBuild.CxxObjects.Owned{NTuple{32, UInt8}}; the characters are not decoded
- `std::wstring`, return by value: **OPAQUE** — 40 characters
- `std::wstring`, pass by reference: **WORKS** — Ref of the object the library returned
- `std::wstring`, intact after the return: **WORKS**
- `std::u16string`, return by value: **OPAQUE** — the C++ object, a RepliBuild.CxxObjects.Owned{NTuple{32, UInt8}}
- `std::u16string`, intact after the return: **WORKS**
- `std::u8string`, return by value: **NO WAY** — comes back as a RepliBuild.CxxObjects.Owned{NTuple{32, UInt8}}, and no function of the library takes one to read it

## What the coverage tool says of the same wrappers

`cpp_thunk_coverage.jl` counts a function as bound when the wrapper's code invokes its thunk. Whether a user has anything to pass it is the table above.

```
| Library | Bound / callable | % | Tier 3 | Tier 2 | Traps | Dropped | Expected | Excluded / earlier | Types f/b/u/o/i | Generator |
|---|---|---|---|---|---|---|---|---|---|---|
| tm_any | 3/3 | 100.0 | 0 | 3 | 0 | 0 | 0 |  | 0/1/1/0/0 | 4.6.0 |
| tm_array | 6/6 | 100.0 | 0 | 6 | 0 | 0 | 0 |  | 2/0/0/0/0 | 4.6.0 |
| tm_atomic_mutex | 4/4 | 100.0 | 0 | 4 | 0 | 0 | 0 |  | 0/0/0/2/0 | 4.6.0 |
| tm_chrono | 6/6 | 100.0 | 0 | 6 | 0 | 0 | 0 |  | 4/0/0/0/0 | 4.6.0 |
| tm_complex | 5/5 | 100.0 | 0 | 5 | 0 | 0 | 0 |  | 2/0/0/0/0 | 4.6.0 |
| tm_exception | 8/8 | 100.0 | 0 | 8 | 0 | 0 | 0 |  | 1/1/0/1/0 | 4.6.0 |
| tm_function | 7/8 | 87.5 | 0 | 7 | 1 | 0 | 0 |  | 0/0/0/3/0 | 4.6.0 |
| tm_initializer_list | 3/3 | 100.0 | 0 | 3 | 0 | 0 | 0 |  | 2/0/0/0/0 | 4.6.0 |
| tm_map | 15/15 | 100.0 | 0 | 15 | 0 | 0 | 0 |  | 0/0/0/0/0 | 4.6.0 |
| tm_optional | 8/8 | 100.0 | 0 | 8 | 0 | 0 | 0 |  | 3/3/3/0/0 | 4.6.0 |
| tm_pair_tuple | 10/10 | 100.0 | 0 | 10 | 0 | 0 | 0 |  | 3/0/0/0/0 | 4.6.0 |
| tm_path | 6/6 | 100.0 | 0 | 6 | 0 | 0 | 0 |  | 2/0/0/1/0 | 4.6.0 |
| tm_sequence | 8/8 | 100.0 | 0 | 8 | 0 | 0 | 0 |  | 1/0/0/0/0 | 4.6.0 |
| tm_shared_ptr | 7/7 | 100.0 | 0 | 7 | 0 | 0 | 0 |  | 3/0/0/2/0 | 4.6.0 |
| tm_span | 4/5 | 80.0 | 0 | 4 | 1 | 0 | 0 |  | 3/0/0/1/0 | 4.6.0 |
| tm_stream | 6/6 | 100.0 | 0 | 6 | 0 | 0 | 0 |  | 0/0/0/3/0 | 4.6.0 |
| tm_string | 6/6 | 100.0 | 0 | 6 | 0 | 0 | 0 |  | 0/0/0/0/0 | 4.6.0 |
| tm_string_view | 5/5 | 100.0 | 0 | 5 | 0 | 0 | 0 |  | 1/0/0/0/0 | 4.6.0 |
| tm_unique_ptr | 10/10 | 100.0 | 0 | 10 | 0 | 0 | 2 |  | 2/0/0/3/0 | 4.6.0 |
| tm_variant | 2/5 | 40.0 | 0 | 2 | 3 | 0 | 0 |  | 0/0/0/2/0 | 4.6.0 |
| tm_vector | 24/24 | 100.0 | 0 | 24 | 0 | 0 | 0 |  | 1/0/0/0/0 | 4.6.0 |
| tm_wide_string | 9/9 | 100.0 | 0 | 9 | 0 | 0 | 0 |  | 0/0/0/0/0 | 4.6.0 |

libraries measured: 22 of 22 rows
surface (unique mangled, net of exclusions, not-exported and earlier-bound): 169
expected drops (several C++ entry points behind one Julia method, by design): 2
callable (surface less expected): 167
bound:   162/167 = 97.01%  (tier 3: 0, tier 2: 162)
traps:   5   dropped: 0   unexported: 0
pure virtuals (no symbol, called through the vtable): 0 bound of 0 in the surface
thunks in a manifest nothing invokes: 0   thunks invoked outside the surface (STL members): 0
types:   fields 30, blobs 5, byte records (unions/bitfields) 4, opaque 18, imported 0, owners (Managed/factory) 1

misses by cause (count, then the first symbols):
  5  trap: by-value record with no layout
        per library: tm_variant 3, tm_span 1, tm_function 1
        tm_function  fn_adder(int)
        tm_span  span_fixed(std::span<int const, 3ul>)
        tm_variant  var_make(int)
        tm_variant  var_index(std::variant<int, double>)
        tm_variant  vars_make[abi:cxx11](int)

expected drops (not misses):
  1  constructor variant (C1/C2: one Julia method per constructor)
        tm_unique_ptr  UCounted::UCounted(int)
  1  destructor variant (D0/D1/D2: one Julia method per class)
        tm_unique_ptr  UCounted::~UCounted()

trapped records (functions each blocks):
  2  variant<int, double>
  1  function<int (int)>
  1  span<const int, 3UL>
  1  variant<int, std::__cxx11::basic_string<char, std::char_traits<char>,…

cpp coverage: 162/167 (97.01%), traps 5, dropped 0, expected 2
```
