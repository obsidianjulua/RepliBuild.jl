# OwnershipUnit.jl — what the compiler says about the C++ types a library
# returns or takes by value and will not copy bytewise.
#
# A `std::vector`, a `std::map`, a `std::unique_ptr`, a class with a
# destructor: C++ builds one where it will live, ends it once, and copies it
# only through its copy constructor. Whether such a type must be ENDED, whether
# it can be COPIED or MOVED, and how big it is are facts only the compiler has:
# a record of one pointer owns memory when its author wrote a destructor, and a
# record with a copy constructor may own nothing. So the build asks. For each
# such type (`FunctionGen.owned_key`) one line of a generated unit,
# `replibuild_own.cpp`, names it and has the compiler state the facts and
# compile three functions into the library:
#
#     replibuild_own_end_<n>(p)          run the destructor on the object at p
#     replibuild_own_copy_<n>(dst, src)  copy-construct at dst from src
#     replibuild_own_move_<n>(dst, src)  move-construct at dst from src
#
# The same shape as the Rust front end's `__rb_drop_<n>` (`RustGlue`), and the
# same rule: never from the type's shape.
#
# A type is NAMED without spelling it wherever a function returns it: the line
# is `decltype` of that function called, with the overload picked by its
# parameter list (the demangled name spells it, qualified). A function's
# mangled name does not carry its return type, and DWARF names a type bare
# (`vector<int, std::allocator<int> >`), which is no C++ spelling. A type taken
# by value is spelled from the demangled parameter list. Up to three
# candidates per type are tried; the first that compiles is the type's row.
#
# The facts are read off the unit's own IR: each row explicitly instantiates
# `replibuild_force::own_facts<n, size, align, flags>()`, whose mangled name
# holds the four numbers. No variable is written, so nothing here reaches the
# metadata's globals, and the helper templates are in `replibuild_force`, the
# namespace the extraction already keeps out of the library's records and
# functions.
#
# Loaded after IRGen: which types these are is `FunctionGen`'s answer.
# `Compiler.compile_project` runs this as its step 4d.

module OwnershipUnit

import ..Compiler
import ..JLCSIRGenerator

const FG = JLCSIRGenerator.FunctionGen

"The prefix of the three functions a row compiles; `extract_symbols_from_binary` keeps it out of the API."
const HELPER_PREFIX = "replibuild_own_"
"How many ways of naming one type the unit tries."
const MAX_CANDIDATES = 3
# The rank of a type written by its own name (`own_name`): after every way
# that names it through a declaration.
const OWN_NAME_RANK = 9

# What a row's `flags` say (`replibuild_force::own<T>::flags` in the preamble).
const ENDS      = 0x01   # not trivially destructible: something must run its destructor
const COPYABLE  = 0x02   # copy-constructible from here
const MOVABLE   = 0x04   # move-constructible from here
const BITWISE   = 0x08   # trivially copyable: its bytes are the object
const REFERENCE = 0x10   # the line named a reference, not an object

helper(kind::AbstractString, id::Integer) = string(HELPER_PREFIX, kind, "_", id)

"""
    function_spelling(demangled) -> Union{Nothing,NamedTuple}

How the unit names a demangled function that has no class or namespace in
front of its name (`operator*(btMatrix3x3 const&, btMatrix3x3 const&)`,
`make_points(int)`): `(name, parameters)`. `nothing` for anything
`Compiler._method_spelling` reads (a name with `::`), a function template (its
demangling opens with a return type), an ABI tag, or a variadic list.
"""
function function_spelling(demangled::AbstractString)
    s = String(strip(demangled))
    endswith(s, ")") || return nothing
    depth = 0
    open = 0
    i = lastindex(s)
    while i >= firstindex(s)
        c = s[i]
        if c == ')'
            depth += 1
        elseif c == '('
            depth -= 1
            if depth == 0
                open = i
                break
            end
        end
        i = prevind(s, i)
    end
    open <= firstindex(s) && return nothing
    name = String(strip(s[1:prevind(s, open)]))
    inner = strip(s[nextind(s, open):prevind(s, lastindex(s))])
    (isempty(name) || occursin("::", name) || occursin("[abi:", name) || occursin('<', name)) && return nothing
    # `operator new`, `operator delete[]` and a conversion operator hold a
    # space; any other space before the name is a return type.
    (occursin(' ', name) && !startswith(name, "operator")) && return nothing
    params = isempty(inner) ? String[] : String[String(p) for p in Compiler._split_depth0_commas(inner)]
    any(p -> occursin("...", p) || occursin("(anonymous namespace)", p) || occursin("lambda", p) ||
             occursin("[abi:", p) || occursin("{", p) || isempty(p), params) && return nothing
    return (name = name, parameters = params)
end

# A demangled parameter the unit can write as a type: nothing local, unnamed
# or tagged.
_spellable(t::AbstractString) = !(isempty(t) || occursin("(anonymous namespace)", t) || occursin("lambda", t) ||
                                  occursin("[abi:", t) || occursin("{", t) || occursin("unnamed", t) ||
                                  occursin("...", t))

# The header the C++ standard puts each of its names in. A type written by
# its own name (a parameter's spelling, or `own_name`) needs the header that
# declares it, and a library whose functions no header declares gives none.
# A name not listed gets no header: its row then does not compile and is
# skipped with the compiler's message, as any other.
const STD_HEADERS = Dict{String,String}(
    "vector" => "<vector>", "deque" => "<deque>", "list" => "<list>", "forward_list" => "<forward_list>",
    "map" => "<map>", "multimap" => "<map>", "set" => "<set>", "multiset" => "<set>",
    "unordered_map" => "<unordered_map>", "unordered_multimap" => "<unordered_map>",
    "unordered_set" => "<unordered_set>", "unordered_multiset" => "<unordered_set>",
    "array" => "<array>", "queue" => "<queue>", "priority_queue" => "<queue>", "stack" => "<stack>",
    "basic_string" => "<string>", "char_traits" => "<string>", "string" => "<string>", "wstring" => "<string>",
    "basic_string_view" => "<string_view>", "string_view" => "<string_view>",
    "shared_ptr" => "<memory>", "unique_ptr" => "<memory>", "weak_ptr" => "<memory>",
    "allocator" => "<memory>", "default_delete" => "<memory>",
    "function" => "<functional>", "less" => "<functional>", "greater" => "<functional>",
    "equal_to" => "<functional>", "hash" => "<functional>", "reference_wrapper" => "<functional>",
    "pair" => "<utility>", "tuple" => "<tuple>", "optional" => "<optional>", "nullopt_t" => "<optional>",
    "variant" => "<variant>", "monostate" => "<variant>", "any" => "<any>",
    "complex" => "<complex>", "bitset" => "<bitset>", "span" => "<span>", "valarray" => "<valarray>",
    "initializer_list" => "<initializer_list>", "byte" => "<cstddef>",
    "filesystem" => "<filesystem>", "chrono" => "<chrono>", "ratio" => "<ratio>",
    "atomic" => "<atomic>", "mutex" => "<mutex>", "thread" => "<thread>",
    "future" => "<future>", "promise" => "<future>", "shared_future" => "<future>",
    "exception_ptr" => "<exception>", "error_code" => "<system_error>", "error_condition" => "<system_error>",
    "basic_regex" => "<regex>", "locale" => "<locale>", "type_index" => "<typeindex>",
    "basic_ostream" => "<ostream>", "basic_istream" => "<istream>", "basic_iostream" => "<istream>",
    "basic_stringstream" => "<sstream>", "basic_ostringstream" => "<sstream>",
    "basic_istringstream" => "<sstream>", "basic_stringbuf" => "<sstream>",
    "basic_ifstream" => "<fstream>", "basic_ofstream" => "<fstream>", "basic_fstream" => "<fstream>")

"""
    std_headers(expr) -> Vector{String}

The standard headers the names under `std::` in the C++ type `expr` are
declared in (`STD_HEADERS`), sorted: every segment of every `std::a::b` chain
is looked up, so `std::filesystem::path` and `std::__cxx11::basic_string`
both find theirs.
"""
function std_headers(expr::AbstractString)
    out = Set{String}()
    for m in eachmatch(r"\bstd::([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)", String(expr))
        for seg in split(m.captures[1], "::")
            h = get(STD_HEADERS, String(seg), nothing)
            h === nothing || push!(out, h)
        end
    end
    return sort!(collect(out))
end

"""
    own_name(key, c_type, structs, record_abi) -> nothing | (spelled, toolchain)

The type filed under `key`, written by its own name: the scope its
definition was read in, then the name (`std::vector<int, std::allocator<int>
>`). `nothing` where the build does not know the scope, where two scopes
declare the name, and for a name no unit can write (a lambda, an unnamed
type, a local one). `toolchain` says the definition is not the project's.

The compile still judges the line: a spelling that names nothing is a row
that does not compile, and one that names another type is a size other than
DWARF's (`checked!`).
"""
function own_name(key::AbstractString, c_type::AbstractString, structs, record_abi)
    def = get(structs, key, nothing)
    scope = nothing
    toolchain = true
    if def isa AbstractDict
        get(def, "conflict", false) == true && return nothing
        scope = get(def, "scope", nothing)
        toolchain = get(def, "toolchain", false) == true
    end
    if scope === nothing
        for cand in FG._scope_suffixes(FG._strip_top_cv(String(c_type)))
            facts = get(record_abi, cand, nothing)
            facts isa AbstractDict || continue
            (get(facts, "conflict", false) == true || get(facts, "scope_ambiguous", false) == true) && return nothing
            scope = get(facts, "scope", nothing)
            break
        end
    end
    scope isa AbstractString || return nothing
    spelled = isempty(scope) ? String(key) : String(scope) * "::" * String(key)
    _spellable(spelled) || return nothing
    return (spelled, toolchain)
end

"The bytes DWARF gives the record filed under `key`, or `nothing`: what the compiler's `sizeof` is checked against."
function _dwarf_bytes(key::AbstractString, c_type::AbstractString, structs, record_abi)
    def = get(structs, key, nothing)
    if def isa AbstractDict
        n = FG._parse_num(get(def, "byte_size", nothing))
        (n !== nothing && n > 0) && return Int(n)
    end
    for cand in FG._scope_suffixes(FG._strip_top_cv(String(c_type)))
        facts = get(record_abi, cand, nothing)
        facts isa AbstractDict || continue
        get(facts, "conflict", false) == true && return nothing
        n = FG._parse_num(get(facts, "byte_size", nothing))
        (n !== nothing && n > 0) && return Int(n)
    end
    return nothing
end

"""
    plan(functions, structs, record_abi; demangle) -> Dict

The types the unit asks the compiler about, from one extraction's metadata:
`candidates`, one row per way of naming a type (`id`, `key`, `expr`, the C++
type expression of its line, `headers`, `from`, and `bytes`, DWARF's size for
the type when it gives one), sorted by key and then by preference, at most
`MAX_CANDIDATES` per key; and `skipped`, a `key` nothing can name with its
`reason`.

A key (`FunctionGen.owned_key`) comes from a function's by-value return or
from a by-value parameter. In order of preference a type is named by:

  1. the return of a member function whose class a header declares
     (`decltype` of the member, the overload picked by its parameters);
  2. a by-value parameter's own spelling in a demangled name;
  3. the return of a function no class declares, which only the umbrella
     headers (`[wrap] shim_headers`) can have declared;
  4. its own name under the scope its definition was read in (`own_name`),
     which needs no declaration of any function: the one way for a library
     whose functions no header declares. Always among the candidates, last.

A line that writes a type by name (2 and 4) also gets the standard headers
its `std::` names are declared in (`std_headers`).

Passed over: a pure virtual row (no definition to name), a constructor or
destructor (no return), and for the return route a default-argument entry
point, whose name is the member's with fewer parameters than any overload
takes.
"""
function plan(functions, structs, record_abi; demangle = Compiler._demangle)
    # A declared member's symbol => the record that declares it.
    owner = Dict{String,String}()
    for name in sort!(collect(String, keys(record_abi)))
        entry = record_abi[name]
        entry isa AbstractDict || continue
        for row in get(entry, "methods", Any[])
            row isa AbstractDict || continue
            sym = String(get(row, "symbol", ""))
            isempty(sym) || get!(owner, sym, name)
        end
    end
    headers_of(rec) = begin
        def = rec === nothing ? nothing : get(structs, rec, nothing)
        def isa AbstractDict ? Compiler._inline_ctor_headers_of(get(def, "decl_files", String[])) : String[]
    end

    # What a line that writes a type by name has to include: the headers of
    # the project's records its identifiers name (a record's leading
    # identifier => its headers), and the standard ones (`std_headers`).
    lead_headers = Dict{String,Vector{String}}()
    for name in sort!(collect(String, keys(structs)))
        def = structs[name]
        (def isa AbstractDict && get(def, "toolchain", false) != true) || continue
        hs = headers_of(name)
        isempty(hs) && continue
        m = match(r"^[A-Za-z_][A-Za-z0-9_]*", name)
        m === nothing || append!(get!(() -> String[], lead_headers, String(m.match)), hs)
    end
    named_headers(spelled) = begin
        hs = String[]
        for m in eachmatch(r"[A-Za-z_][A-Za-z0-9_]*", String(spelled))
            append!(hs, get(lead_headers, String(m.match), String[]))
        end
        unique(vcat(hs, std_headers(spelled)))
    end

    found = Dict{String,Vector{Dict{String,Any}}}()
    bytes = Dict{String,Union{Nothing,Int}}()
    function add!(key, c_type, ret, rank, expr, headers, from)
        haskey(bytes, key) || (bytes[key] = _dwarf_bytes(key, c_type, structs, record_abi))
        push!(get!(found, key, Dict{String,Any}[]),
              Dict{String,Any}("key" => key, "rank" => rank, "expr" => expr,
                               "headers" => headers, "from" => from,
                               "c_type" => String(c_type), "ret" => ret))
        return
    end
    unnamed = Dict{String,String}()      # key => why no line could name it

    rows = Any[f for f in functions if f isa AbstractDict]
    sort!(rows; by = f -> String(get(f, "mangled", "")))
    for f in rows
        Compiler._is_vtable_only(f) && continue
        mangled = String(get(f, "mangled", ""))
        isempty(mangled) && continue
        dm = get(f, "demangled", nothing)
        dm === nothing && (dm = demangle(mangled))
        dm === nothing && continue
        dm = String(dm)
        isempty(dm) && continue
        # An ABI tag (`chain[abi:cxx11](int)`: the function returns a tagged
        # type) is part of the symbol, never of the name a unit writes.
        named = replace(dm, r"\[abi:[^\]]*\]" => "")
        sp = Compiler._method_spelling(named)
        fs = sp === nothing ? function_spelling(named) : nothing
        rec = get(owner, mangled, nothing)
        headers = headers_of(rec)
        dparams = sp !== nothing ? sp.parameters : fs !== nothing ? fs.parameters : nothing

        # The return.
        ret_c = String(get(get(f, "return_type", Dict()), "c_type", "void"))
        rkey = FG.owned_key(ret_c, structs, record_abi; ret = true)
        if rkey !== nothing
            shim = haskey(f, "default_arguments")
            if sp !== nothing && !sp.structor && !shim
                args = join(sp.parameters, ", ")
                expr = "decltype(replibuild_force::own_args<$args>::$(sp.isconst ? "mc" : "m")(&$(sp.type)::$(sp.name)))"
                add!(rkey, ret_c, true, isempty(headers) ? 2 : 0, expr, headers, "$mangled returns it")
            elseif fs !== nothing && !shim
                args = join(fs.parameters, ", ")
                add!(rkey, ret_c, true, 3, "decltype(replibuild_force::own_args<$args>::m(&::$(fs.name)))", String[],
                     "$mangled returns it")
            else
                get!(unnamed, rkey, "no function the unit can name returns it ($dm)")
            end
        end

        # The by-value parameters, matched to the demangled list by position.
        params = Any[p for p in get(f, "parameters", Any[]) if p isa AbstractDict]
        (!isempty(params) && String(get(params[1], "name", "")) == "this") && popfirst!(params)
        for (i, p) in enumerate(params)
            pc = String(get(p, "c_type", ""))
            pkey = FG.owned_key(pc, structs, record_abi)
            pkey === nothing && continue
            if dparams !== nothing && length(dparams) == length(params) && _spellable(dparams[i])
                add!(pkey, pc, false, isempty(headers) ? 2 : 1, String(dparams[i]),
                     unique(vcat(headers, named_headers(String(dparams[i])))), "parameter $i of $mangled")
            else
                get!(unnamed, pkey, "its spelling in $dm cannot be written in a unit")
            end
        end
    end

    # The type's own name, for every key: the headers are its own where the
    # project declares it in one, and what its spelling names.
    for key in sort!(unique(vcat(collect(String, keys(found)), collect(String, keys(unnamed)))))
        c_type = haskey(found, key) ? String(found[key][1]["c_type"]) : key
        named = own_name(key, c_type, structs, record_abi)
        named === nothing && continue
        (spelled, toolchain) = named
        hs = toolchain ? String[] : headers_of(key)
        add!(key, c_type, haskey(found, key) ? found[key][1]["ret"] : true, OWN_NAME_RANK, spelled,
             unique(vcat(hs, named_headers(spelled))), "its own name")
    end

    candidates = Dict{String,Any}[]
    for key in sort!(collect(keys(found)))
        ways = found[key]
        # One line per distinct expression, best first. The type's own name
        # is kept beside the `MAX_CANDIDATES` best of the others: it is the
        # one line that needs no function declared.
        sort!(ways; by = w -> (w["rank"], w["from"]))
        seen = Dict{String,Dict{String,Any}}()
        kept = 0
        for w in ways
            # The same expression reached another way is the same line: it
            # takes the headers of both.
            if haskey(seen, w["expr"])
                first_way = seen[w["expr"]]
                first_way["headers"] = unique(vcat(first_way["headers"], w["headers"]))
                continue
            end
            (kept >= MAX_CANDIDATES && w["rank"] != OWN_NAME_RANK) && continue
            seen[w["expr"]] = w
            w["rank"] == OWN_NAME_RANK || (kept += 1)
            w["id"] = length(candidates)
            w["bytes"] = bytes[key]
            push!(candidates, w)
        end
    end
    skipped = Dict{String,Any}[Dict{String,Any}("key" => k, "reason" => unnamed[k])
                               for k in sort!(collect(keys(unnamed))) if !haskey(found, k)]
    return Dict{String,Any}("candidates" => candidates, "skipped" => skipped)
end

"The C++ in front of the rows: the helper that names a function's return type, and what a row compiles."
const PREAMBLE = raw"""
// ── What the compiler says about a type (see OwnershipUnit.jl) ──────────────
// Never called from C++. `own_args<A...>::m(&T::f)` is the return type of the
// member, or function, taking exactly (A...), and `mc` of the `const` member:
// a mangled name does not carry it. `own<T>` ends, copies and moves a T, the
// last two only where T can be; `own_facts<n, size, align, flags>` is
// instantiated so that its NAME states the numbers. In `replibuild_force`,
// with the other generated units' helpers, which the extraction keeps out of
// the library's records and functions. C++11.
#include <new>
#include <type_traits>
#include <utility>

#if defined(_WIN32)
#  define RB_OWN_EXPORT __attribute__((used)) __declspec(dllexport)
#else
#  define RB_OWN_EXPORT __attribute__((used, visibility("default")))
#endif
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
#  define RB_OWN_TRY try
#  define RB_OWN_CATCH catch (...) { return 1; }
#else
#  define RB_OWN_TRY
#  define RB_OWN_CATCH
#endif

namespace replibuild_force {

template <class... A> struct own_args {
    template <class R, class C> static R m(R (C::*)(A...));
    template <class R>          static R m(R (*)(A...));
    template <class R, class C> static R mc(R (C::*)(A...) const);
};

template <class T> struct own {
    typedef typename std::remove_cv<T>::type V;
    static void end(void* p) { static_cast<V*>(p)->~V(); }
    static int copy_(std::true_type, void* d, const void* s) { ::new (d) V(*static_cast<const V*>(s)); return 0; }
    static int copy_(std::false_type, void*, const void*) { return 2; }
    static int copy(void* d, const void* s) { RB_OWN_TRY { return copy_(std::is_copy_constructible<V>(), d, s); } RB_OWN_CATCH }
    static int move_(std::true_type, void* d, void* s) { ::new (d) V(static_cast<V&&>(*static_cast<V*>(s))); return 0; }
    static int move_(std::false_type, void*, void*) { return 2; }
    static int move(void* d, void* s) { RB_OWN_TRY { return move_(std::is_move_constructible<V>(), d, s); } RB_OWN_CATCH }
    enum : unsigned long long {
        flags = (std::is_trivially_destructible<V>::value ? 0ull : 1ull) |
                (std::is_copy_constructible<V>::value ? 2ull : 0ull) |
                (std::is_move_constructible<V>::value ? 4ull : 0ull) |
                (std::is_trivially_copyable<V>::value ? 8ull : 0ull) |
                (std::is_reference<T>::value ? 16ull : 0ull)
    };
};

template <unsigned long long N, unsigned long long Size, unsigned long long Align, unsigned long long Flags>
void own_facts() {}

}  // namespace replibuild_force

#define RB_OWN_ROW(n, ...) \
    template void replibuild_force::own_facts<n, sizeof(replibuild_force::own<__VA_ARGS__>::V), \
                                              alignof(replibuild_force::own<__VA_ARGS__>::V), \
                                              replibuild_force::own<__VA_ARGS__>::flags>(); \
    extern "C" { \
    RB_OWN_EXPORT void replibuild_own_end_##n(void* p) { replibuild_force::own<__VA_ARGS__>::end(p); } \
    RB_OWN_EXPORT int replibuild_own_copy_##n(void* d, const void* s) { return replibuild_force::own<__VA_ARGS__>::copy(d, s); } \
    RB_OWN_EXPORT int replibuild_own_move_##n(void* d, void* s) { return replibuild_force::own<__VA_ARGS__>::move(d, s); } }
"""

"""
    unit_text(headers, rows) -> String

The unit: one `#include` per header, the preamble, and one `RB_OWN_ROW` line
per row. Every row is exactly one line, which is how a compiler diagnostic is
traced back to its row (`row_of_line`).
"""
function unit_text(headers::AbstractVector, rows::AbstractVector)::String
    io = IOBuffer()
    println(io, "// Auto-generated by RepliBuild: what the compiler says about each type this library")
    println(io, "// returns or takes by value and will not copy bytewise, and the functions that end,")
    println(io, "// copy and move one.")
    for h in headers
        hs = String(h)
        println(io, startswith(hs, "<") ? "#include $hs" : "#include \"$hs\"")
    end
    println(io)
    print(io, PREAMBLE)
    println(io)
    for r in rows
        from = replace(String(r["from"]), r"[\r\n\\]" => " ")
        println(io, "RB_OWN_ROW($(Int(r["id"])), $(r["expr"]))   // $from")
    end
    return String(take!(io))
end

"The row id each line of the unit states."
function row_of_line(text::AbstractString)::Dict{Int,Int}
    rows = Dict{Int,Int}()
    for (i, line) in enumerate(split(text, '\n'))
        m = match(r"^RB_OWN_ROW\((\d+),", line)
        m === nothing || (rows[i] = parse(Int, m.captures[1]))
    end
    return rows
end

"""
    facts(ir_path) -> Dict{Int,NamedTuple}

What the compiled unit states per row id: `(size, align, flags)`, read off the
mangled names of the `replibuild_force::own_facts<n, size, align, flags>`
instantiations it defines.
"""
function facts(ir_path::AbstractString)
    out = Dict{Int,NamedTuple{(:size, :align, :flags),Tuple{Int,Int,Int}}}()
    isfile(ir_path) || return out
    for line in eachline(ir_path)
        startswith(line, "define ") || continue
        m = match(r"@\"?_ZN16replibuild_force9own_factsIL[a-z](\d+)EL[a-z](\d+)EL[a-z](\d+)EL[a-z](\d+)EEEvv\"?\(", line)
        m === nothing && continue
        out[parse(Int, m.captures[1])] = (size = parse(Int, m.captures[2]), align = parse(Int, m.captures[3]),
                                         flags = parse(Int, m.captures[4]))
    end
    return out
end

"The unit's own strong definitions are its helpers; anything else a header defined."
_foreign_definitions(ir_path::AbstractString)::Vector{String} =
    filter!(s -> !startswith(s, HELPER_PREFIX), Compiler._strong_definitions(ir_path))

_first_error(out::AbstractString) = begin
    lines = split(out, '\n')
    k = findfirst(l -> occursin("error", l), lines)
    k === nothing ? String(strip(first(out, 200))) : String(strip(lines[k]))
end

"""
    generate(config, cpp_files, plan) -> (cpp_files, plan)

Write the unit for the plan's candidates under the build cache and append it
to `cpp_files`. The plan comes back with `rows` (one per type: the first
candidate that compiles) and `skipped` (a type none of whose candidates
compiled, with the compiler's reason).

The unit is compiled whole. A row a diagnostic names is dropped and the unit
compiled again; when no row is named, the include block is checked header by
header (one that does not compile alone leaves the unit) and then each row is
tried alone. A header that DEFINES functions with external linkage is an
implementation file the library's own unit already includes: it leaves the
unit too, since linking it twice is "multiple definition of"
(`Compiler._strong_definitions`). With no candidate, or none left, `cpp_files`
is returned as given and a stale unit removed.
"""
function generate(config, cpp_files::Vector{String}, plan::AbstractDict)
    rows = Dict{String,Any}[Dict{String,Any}(r) for r in get(plan, "candidates", Any[])]
    skipped = Dict{String,Any}[Dict{String,Any}(r) for r in get(plan, "skipped", Any[])]
    cache_dir = Compiler.get_cache_path(config)
    mkpath(cache_dir)
    path = Compiler._own_unit_path(config)
    done(files) = (files, Dict{String,Any}("rows" => winners(rows), "skipped" => skipped))
    if isempty(rows) || config.wrap.language != :cpp
        rm(path; force = true)
        empty!(rows)
        return done(cpp_files)
    end

    probe = joinpath(cache_dir, "replibuild_own_probe.cpp")
    probe_ir = joinpath(cache_dir, "replibuild_own_probe.ll")
    shim_headers = String[String(h) for h in config.wrap.shim_headers]
    row_headers() = sort!(unique(String[String(h) for r in rows for h in r["headers"]]))
    headers() = unique(vcat(shim_headers, row_headers()))
    why = Dict{String,String}()          # key => the last reason one of its rows was dropped
    function drop!(reason_of)
        keep = Dict{String,Any}[]
        for (k, r) in enumerate(rows)
            reason = reason_of(k, r)
            reason === nothing ? push!(keep, r) : (why[String(r["key"])] = reason)
        end
        rows = keep
        return
    end
    # Each header alone beside the umbrella; `judge` says what is wrong with
    # one, or `nothing`. A bad header leaves the umbrella and every row.
    function drop_headers!(judge, tag)
        bad = Dict{String,String}()
        kept = String[]
        for (i, h) in enumerate(copy(shim_headers))
            v = judge(vcat(kept, [h]), "$(tag)u$(i)")
            v === nothing ? push!(kept, h) : (bad[h] = v)
        end
        empty!(shim_headers)
        append!(shim_headers, kept)
        hs = row_headers()
        verdict = Vector{Union{Nothing,String}}(nothing, length(hs))
        Threads.@threads for i in eachindex(hs)
            verdict[i] = judge(vcat(shim_headers, [hs[i]]), "$(tag)h$(i)")
        end
        for i in eachindex(hs)
            verdict[i] === nothing || (bad[hs[i]] = verdict[i])
        end
        isempty(bad) && return false
        for r in rows
            r["headers"] = String[h for h in r["headers"] if !haskey(bad, h)]
        end
        return true
    end
    compiles(hs, tag) = begin
        p = joinpath(cache_dir, "replibuild_own_probe_$(tag).cpp")
        (ok, out) = Compiler._unit_probe(config, p, unit_text(hs, Any[]))
        rm(p; force = true)
        ok ? nothing : _first_error(out)
    end
    defines(hs, tag) = begin
        p = joinpath(cache_dir, "replibuild_own_probe_$(tag).cpp")
        ir = joinpath(cache_dir, "replibuild_own_probe_$(tag).ll")
        (ok, _) = Compiler._unit_probe(config, p, unit_text(hs, Any[]); ir = ir)
        found = ok && isfile(ir) ? _foreign_definitions(ir) : String[]
        rm(p; force = true)
        rm(ir; force = true)
        isempty(found) ? nothing : first(found)
    end

    # The include block by itself first: a header that does not compile alone
    # puts its errors on the lines of rows that are fine.
    headers_checked = false
    check_headers!() = (headers_checked || (headers_checked = true; drop_headers!(compiles, "c")); nothing)
    let (inc_ok, _) = Compiler._unit_probe(config, probe, unit_text(headers(), Any[]))
        inc_ok || check_headers!()
    end

    ok = false
    msg = ""
    tried_alone = false
    for pass in 1:2
        for _ in 1:12
            isempty(rows) && break
            text = unit_text(headers(), rows)
            (ok, out) = Compiler._unit_probe(config, probe, text; ir = probe_ir)
            ok && break
            bad = Compiler._unit_bad_lines(out, probe)
            ids = row_of_line(text)
            named = Dict{Int,String}(ids[line] => reason for (line, reason) in bad if haskey(ids, line))
            if !isempty(named)
                drop!((_, r) -> haskey(named, Int(r["id"])) ? "does not compile: $(named[Int(r["id"])])" : nothing)
            elseif !headers_checked
                check_headers!()
            elseif !tried_alone
                tried_alone = true
                hs = headers()
                all_rows = rows
                verdicts = Compiler._alone_verdicts(length(all_rows)) do g, round, k
                    p = joinpath(cache_dir, "replibuild_own_probe_r$(round)_$(k).cpp")
                    (rok, rout) = Compiler._unit_probe(config, p, unit_text(hs, all_rows[g]))
                    rm(p; force = true)
                    rok ? nothing : _first_error(rout)
                end
                drop!((k, _) -> verdicts[k] === nothing ? nothing : "does not compile: $(verdicts[k])")
            else
                msg = _first_error(out)
                break
            end
        end
        (ok && pass == 1 && isfile(probe_ir) && !isempty(_foreign_definitions(probe_ir))) || break
        drop_headers!(defines, "d") || break
        ok = false
    end
    if !ok && !isempty(rows)
        drop!((_, _) -> "the unit did not compile: $msg")
    end
    if ok && isfile(probe_ir)
        left = _foreign_definitions(probe_ir)
        isempty(left) ||
            drop!((_, _) -> "the unit would define a function the library's own unit defines ($(Compiler._demangle(first(left))))")
    end
    rm(probe; force = true)
    rm(probe_ir; force = true)

    # One row per type: its best candidate that compiled. A type with none
    # left is skipped with the reason its last candidate was dropped for.
    best = winners(rows)
    have = Set{String}(String(r["key"]) for r in best)
    for key in sort!(collect(keys(why)))
        key in have || push!(skipped, Dict{String,Any}("key" => key, "reason" => why[key]))
    end
    # The include block the rows were judged with, whole: a row may compile
    # only because another candidate's header declares what it names (a
    # function's return named beside the type's own header), and the block
    # of the winners alone would leave that header out.
    judged_with = headers()
    rows = best
    if isempty(rows)
        rm(path; force = true)
        return done(cpp_files)
    end
    Compiler._write_if_changed(path, unit_text(judged_with, rows))
    return done(vcat(cpp_files, [path]))
end

"The first row of each key, in the plan's order (best candidate first)."
function winners(rows::AbstractVector)
    seen = Set{String}()
    out = Dict{String,Any}[]
    for r in rows
        key = String(r["key"])
        key in seen && continue
        push!(seen, key)
        push!(out, r)
    end
    return out
end

function _refuse!(plan::AbstractDict, reason_of)
    kept = Dict{String,Any}[]
    for r in get(plan, "rows", Any[])
        reason = reason_of(r)
        reason === nothing ? push!(kept, r) :
            push!(plan["skipped"], Dict{String,Any}("key" => r["key"], "reason" => reason))
    end
    plan["rows"] = kept
    return plan
end

"""
    checked!(plan, ir_path) -> plan

Each row takes the facts the compiled unit at `ir_path` states for it (`size`,
`align`, `flags`). A row moves to `skipped` when the unit states none or does
not define its three functions, when the line named a reference, or when the
compiler's size is not the one DWARF gives the type the row is filed under:
the line then named some other type, and its helpers are not this one's.
"""
function checked!(plan::AbstractDict, ir_path::AbstractString)
    stated = facts(ir_path)
    (defined, _) = isfile(ir_path) ? Compiler._ir_defined_declared(ir_path) : (Set{String}(), Set{String}())
    return _refuse!(plan, r -> begin
        id = Int(r["id"])
        f = get(stated, id, nothing)
        f === nothing && return "the compiled unit states nothing for it"
        all(k -> helper(k, id) in defined, ("end", "copy", "move")) ||
            return "the compiled unit does not define its functions"
        (f.flags & REFERENCE) != 0 && return "its line named a reference, not an object"
        want = get(r, "bytes", nothing)
        (want !== nothing && Int(want) != f.size) &&
            return "the compiler's type is $(f.size) bytes where the library's is $(Int(want)) ($(r["from"]))"
        r["size"] = f.size
        r["align"] = f.align
        r["flags"] = f.flags
        nothing
    end)
end

"Every row refused with `why`: the library was built without the unit."
refused!(plan::AbstractDict, why::AbstractString) =
    _refuse!(plan, _ -> "the library did not build with its ownership unit: $why")

"""
    exported!(plan, runtime) -> plan

A row whose three functions are not among the library's runtime symbol names
(`runtime`; `nothing` is "could not look" and refuses none) moves to
`skipped`: nothing could call them.
"""
function exported!(plan::AbstractDict, runtime)
    runtime === nothing && return plan
    return _refuse!(plan, r -> all(k -> helper(k, Int(r["id"])) in runtime, ("end", "copy", "move")) ? nothing :
                               "the library does not export its functions")
end

"""
    metadata(plan, structs, record_abi) -> Dict

The metadata's `ownership`: `types`, the compiler's answer per key (`size`,
`align`, `ends`, `copy`, `move`, `bitwise`, and the three functions' names,
`end_symbol`, `copy_symbol`, `move_symbol`), and `skipped`, sorted by key.

A row is filed under the key `FunctionGen.owned_key` gives its C type in THIS
extraction (`structs`, `record_abi`), which is the one a generator will ask
with. The unit's own debug info can give a type a layout it had none of
before (it emits a copy constructor, and the class's definition with it), and
its key then changes from its spelling to its table entry.
"""
function metadata(plan::AbstractDict, structs, record_abi)
    types = Dict{String,Any}()
    for r in get(plan, "rows", Any[])
        haskey(r, "flags") || continue
        id = Int(r["id"])
        flags = Int(r["flags"])
        key = something(FG.owned_key(String(get(r, "c_type", "")), structs, record_abi;
                                     ret = get(r, "ret", false) == true),
                        String(r["key"]))
        haskey(types, key) && continue
        types[key] = Dict{String,Any}(
            "size" => Int(r["size"]), "align" => Int(r["align"]),
            "ends" => (flags & ENDS) != 0, "copy" => (flags & COPYABLE) != 0,
            "move" => (flags & MOVABLE) != 0, "bitwise" => (flags & BITWISE) != 0,
            "end_symbol" => helper("end", id), "copy_symbol" => helper("copy", id),
            "move_symbol" => helper("move", id), "from" => String(r["from"]))
    end
    skipped = sort!(Dict{String,Any}[Dict{String,Any}(s) for s in get(plan, "skipped", Any[])
                                     if !haskey(types, String(s["key"]))];
                    by = s -> String(s["key"]))
    return Dict{String,Any}("types" => types, "skipped" => skipped)
end

end # module OwnershipUnit
