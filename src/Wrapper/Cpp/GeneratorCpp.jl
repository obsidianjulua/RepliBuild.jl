"""
    _bare_type_name_cpp(cls) -> String

Innermost `::` component of a scope, template arguments removed:
`std::vector<int, std::allocator<int> >` → `vector`. Angle-bracket aware, so a
separator inside `<>` is not a cut point.
"""
function _bare_type_name_cpp(cls::AbstractString)::String
    depth = 0
    last_sep = 0
    i = firstindex(cls)
    n = lastindex(cls)
    while i <= n
        c = cls[i]
        if c == '<'
            depth += 1
        elseif c == '>'
            depth = max(0, depth - 1)
        elseif depth == 0 && c == ':' && i < n && cls[nextind(cls, i)] == ':'
            i = nextind(cls, i)
            last_sep = i
        end
        i = nextind(cls, i)
    end
    inner = last_sep == 0 ? cls : cls[nextind(cls, last_sep):end]
    j = findfirst('<', inner)
    return String(strip(j === nothing ? inner : inner[1:prevind(inner, j)]))
end

"""
    _is_ctor_or_dtor_cpp(class_name, func_name) -> Bool

True when the function is a constructor or destructor — which has a `this`
receiver by definition, whether or not its class reached `struct_types`.

**Mirror of `FunctionGen._is_ctor_or_dtor`; the two MUST agree**, because they
decide the same argument array from opposite ends of the pipeline. The
duplication is deliberate (generator isolation); the protection is
`test_symbol_hygiene.jl` §"Both receiver gates agree", which runs both over
every Hub package's metadata.

A destructor's name starts with `~`; a constructor's name equals its class's
bare name, and a member function may not share its class's name — so neither
test can fire on a free function, which is the direction that would shift
arguments.
"""
function _is_ctor_or_dtor_cpp(class_name::AbstractString, func_name::AbstractString)::Bool
    (isempty(class_name) || isempty(func_name)) && return false
    startswith(func_name, "~") && return true
    return func_name == _bare_type_name_cpp(class_name)
end

"Whether a `std::vector`'s element spelling (`bool`, `bool, std::allocator<bool> `) is `bool`: the bit-packed specialization."
_is_bool_vector_element(elem_cpp::AbstractString) = occursin(r"^\s*bool\s*(?:,|$)", elem_cpp)

"""
    _stl_vector_element_type(elem_cpp, defined_struct_names, infer) -> Union{String,Nothing}

The Julia type a `CppVector` holds for a `std::vector<elem_cpp>`: the struct
this module emitted for a record element (its innermost scope sanitized the
way records are named: `Clipper2Lib::Point<long>` → `Point_long`), else what
the registry infers for a scalar (`infer`), and `nothing` when neither gives a
type with a layout (`_UnsafeUnknown`, `Any`, a name nothing emitted).
"""
function _stl_vector_element_type(elem_cpp::AbstractString, defined_struct_names, infer)::Union{Nothing,String}
    bare = _cpp_innermost_scope(String(elem_cpp))
    for candidate in (_sanitize_cpp_type_name(bare), _sanitize_cpp_type_name(String(elem_cpp)))
        candidate in defined_struct_names && return candidate
    end
    jt = String(infer(String(elem_cpp)))
    (jt in ("Any", "_UnsafeUnknown", "") || startswith(jt, "Ptr{Cvoid}")) && return nothing
    occursin(r"[{}]", jt) && return jt                   # a composed type resolves through its parts
    (jt in defined_struct_names || isdefined(Base, Symbol(jt)) || isdefined(Core, Symbol(jt))) && return jt
    return nothing
end

"""
    _cpp_innermost_scope(cls) -> String

Last `::`-separated component of `cls`, at angle-bracket depth 0, **keeping**
template arguments: `ggml::cpu::repack::tensor_traits<block_q4_0, 8l>` →
`tensor_traits<block_q4_0, 8l>`.

Distinct from [`_bare_type_name_cpp`], which additionally strips the template
arguments. Both spellings are needed and they are not interchangeable: this one
is looked up in `struct_types` (which is keyed on raw DWARF names, template
arguments and all), while the stripped one is what a constructor is *named*.
Conflating them is exactly what made the first draft of the agreement test
report 26 phantom disagreements on llamacpp's `tensor_traits` instantiations.
"""
function _cpp_innermost_scope(cls::AbstractString)::String
    depth = 0
    last_sep = 0
    i = firstindex(cls)
    n = lastindex(cls)
    while i <= n
        c = cls[i]
        if c == '<'
            depth += 1
        elseif c == '>'
            depth = max(0, depth - 1)
        elseif depth == 0 && c == ':' && i < n && cls[nextind(cls, i)] == ':'
            i = nextind(cls, i)
            last_sep = i
        end
        i = nextind(cls, i)
    end
    return String(last_sep == 0 ? cls : cls[nextind(cls, last_sep):end])
end

"""
    _cpp_this_param(class_name, func_name, struct_types) -> Dict | Nothing

The C++ wrapper's receiver gate: the synthesized `this` parameter for a method
whose DWARF entry omits it, or `nothing` when no receiver should be added.

**This is the whole gate, in one place.** It used to be inlined in the emission
loop, which meant the only way to test it was to re-implement it — a third copy
of a decision that already exists twice, in a codebase whose worst outbreak
(Dear ImGui, 788 thunks) came from exactly two copies drifting. It is called by
the emission loop and by `test_symbol_hygiene.jl`, so the test exercises the
shipping code rather than a paraphrase of it.

The receiver is synthesized when EITHER:

  * the class is a known aggregate — checked under both the raw DWARF spelling
    (`Box<double>`, which is how `struct_types` is keyed) and the sanitized one
    (`Box_double`); without the raw check, template-class methods never get a
    receiver; or
  * the function is a constructor or destructor, which has one by definition.
    `struct_types` is NOT a complete list of the library's classes — a
    .cpp-local polymorphic class gets no type DIE — so box2d's seven internal
    contact subclasses were emitting zero-argument destructors (2026-08-13).

An unknown class has no emitted Julia type, so the receiver is typed
`Ptr{Cvoid}` rather than `Ptr{<undeclared name>}`: ABI-identical (both relax to
`::Any` in the signature, and `_assert_wrapper_loadable` would refuse the
undeclared spelling if it ever reached a ccall tuple).

Must agree with `FunctionGen._has_receiver` on every input.
"""
const _INTEGER_CTYPES = ("Cint", "Cuint", "Clong", "Culong", "Cshort", "Cushort",
                         "Clonglong", "Culonglong", "Csize_t", "Cssize_t", "Cchar")
const _NO_POSITIONS = Set{Int}()

"""
    _tier2_invoke_args(names, ctypes, needs_conversion, enum_base) -> Vector{String}

The argument list of a Tier 2 `invoke`. A thunk slot points at the argument's
storage and the thunk loads the C parameter's width from it, so an integer
narrower than its C type must be widened BEFORE it is boxed: `Ref(Int8(-1))`
holds one byte, and an `int` parameter reads three more past it (pugixml's
`as_int(attr, Int8(-1))` answered a different number on every call,
2026-11-03). `ccall` converts to the declared C type itself; a thunk cannot,
so the emitted body does — `Cint(def)` for a parameter in `_INTEGER_CTYPES`
(the ones declared `::Integer`), and the `@enum`'s base type for an emitted
enum, which the enum value and any Integer both convert to. `Bool`, floats
(declared at their exact type) and records by value cross as they are.

A pointer position (`pointer_types[i]`, from `_pointer_arg_type`: a pointer, a
reference, a `this`, a `char*`, a container the thunk takes by address) is
converted too, with `JITManager.pointer_arg(<declared type>, x)`: the thunk
dereferences what its slot holds, and `invoke` cannot tell a pointer position
from a value's, so the body says which it is and the conversion refuses what
is not a pointer to it.
"""
function _tier2_invoke_args(names, ctypes, needs_conversion, enum_base,
                            pointer_types = fill(nothing, length(names)))
    [needs_conversion[i] ? _integer_arg_conversion(ctypes[i], names[i]) :
     haskey(enum_base, ctypes[i]) ? "$(enum_base[ctypes[i]])($(names[i]))" :
     pointer_types[i] !== nothing ? "RepliBuild.JITManager.pointer_arg($(pointer_types[i]), $(names[i]))" :
     names[i]
     for i in eachindex(names)]
end

"""
    _by_address_ctype(ctype, c_type; object_address) -> String

The ccall type a parameter's SIGNATURE is spelled from. A container C++ takes
by value and the thunk takes by address (`FunctionGen._stl_by_value`) is
spelled as a pointer, so it relaxes to `::Any` like one, whatever its own name
is bound to: the value such a container is returned as, a `Ref` and a `Ptr`
all pass (`_pointer_arg_type`). So is any other record the thunk takes by the
address of the caller's object (`object_address`, from
`FunctionGen.takes_object_address`: a `std::function` by value): spelled by
its own name, the position took only a value of the empty placeholder struct
that name is bound to, which is nothing a pointer conversion accepts. Any
other parameter keeps `ctype`. The function loop and the overload prediction
both spell a position through this.
"""
_by_address_ctype(ctype::AbstractString, c_type::AbstractString; object_address::Bool = false) =
    (object_address || JLCSIRGenerator.FunctionGen._stl_by_value(c_type)) ? "Ptr{Cvoid}" : String(ctype)

"""
    _pointer_arg_type(ctype, c_type; resolves) -> Union{String,Nothing}

What a Tier 2 body tells `JITManager.pointer_arg` a parameter is, or `nothing`
for a parameter that takes a value. `ctype` is the parameter's ccall type
(`_ccall_param_type`), `c_type` its C spelling, and `resolves(name)` whether a
Julia type name is bound in the wrapper. One derivation for every pointer
position, `this` included:

  * a container C++ takes by value and the thunk by address
    (`FunctionGen._stl_by_value`): `ByValue{NTuple{N, UInt8}}`, the bytes the
    wrapper returns one as;
  * any other record C++ takes by value through the address of the caller's
    object, with no layout here (`object_address`, from
    `FunctionGen.takes_object_address`: a `std::function`): `Ptr{Opaque}`;
  * a function pointer: `Ptr{Code}`;
  * a pointer to plain `char`: `Cstring`, or `CharBuffer` when the `char` is
    not `const` (`_points_to_writable_char`: the function may write into it,
    so a Julia string is refused there). Any other byte pointer (`unsigned
    char*`, `int8_t*`) and a reference to a `char` are data: `Ptr{UInt8}`,
    `Ptr{Int8}`;
  * `void*`, and a pointer to a scalar Julia has no type for (`long double&`,
    which the metadata spells `Ptr{Cvoid}`: the address of a caller's
    buffer): `Ptr{Cvoid}`;
  * a pointer or reference to a type the wrapper binds: `Ptr{T}`;
  * a pointer or reference to anything else (an STL container, a class only
    declared): `Ptr{Opaque}`.
"""
function _pointer_arg_type(ctype::AbstractString, c_type::AbstractString; resolves = _ -> true,
                           record_abi = Dict{String,Any}(), object_address::Bool = false)
    FG = JLCSIRGenerator.FunctionGen
    jm = "RepliBuild.JITManager"
    c = String(strip(c_type))
    # The bytes are the ones a by-value return of the same container gives
    # (`stl_value_bytes`): a value passes where its own type is named.
    FG._stl_by_value(c) &&
        return "$jm.ByValue{NTuple{$(FG.stl_value_bytes(c, record_abi)), UInt8}}"
    # A record C++ takes by value through the address of the caller's object,
    # and that the wrapper has no value of (`FunctionGen.takes_object_address`).
    object_address && return "Ptr{$jm.Opaque}"
    # A `char*` whose `char` is not `const` may be written: a buffer, never a
    # Julia string (`JITManager.CharBuffer`).
    _points_to_writable_char(c) && (ctype == "Cstring" || occursin(r"^(?:Ptr|Ref)\{(?:UInt8|Int8|Cchar)\}$", ctype)) &&
        return "$jm.CharBuffer"
    ctype == "Cstring" && return "Cstring"
    m = match(r"^(?:Ptr|Ref)\{(.+)\}$", ctype)
    m === nothing && return nothing
    inner = String(m.captures[1])
    (startswith(c, "function_ptr") || occursin("(*", c)) && inner == "Cvoid" && return "Ptr{$jm.Code}"
    # What the C spelling points at: one declarator off, cv-qualifiers off.
    t = FG._strip_top_cv(c)
    is_pointer = endswith(t, "*")
    pointee = endswith(t, "&&") ? FG._strip_top_cv(chop(t; tail = 2)) :
              (is_pointer || endswith(t, "&")) ? FG._strip_top_cv(chop(t)) : ""
    inner in ("UInt8", "Int8", "Cchar", "Cuchar") &&
        return _points_to_plain_char(c) ? "Cstring" : "Ptr{$inner}"
    if inner == "Cvoid"
        pointee == "void" && return "Ptr{Cvoid}"
        mapped = JLCSIRGenerator.TypeUtils.map_cpp_type(pointee)
        scalar = !isempty(pointee) && mapped != "" && mapped != "!llvm.ptr" && !startswith(mapped, "!llvm.struct<\"")
        return scalar ? "Ptr{Cvoid}" : "Ptr{$jm.Opaque}"
    end
    names = (String(tok.match) for tok in eachmatch(r"[A-Za-z_][A-Za-z0-9_]*", inner))
    return all(resolves, names) ? "Ptr{$inner}" : "Ptr{$jm.Opaque}"
end

"""
    _type_name_bound(t, defined_names) -> Bool

Whether a parameter annotation `t` resolves when the wrapper is included: a
composed type (`Ref{ImVec2}`, `NTuple{4, UInt8}`, a `Union`) resolves through
its parts; a bare name must be a record this module emits or a name bound in
`Base`/`Core`. An emitted enum is NOT bound here on purpose: an enum parameter
relaxes to `::Any` like a pointer unless an overload set disagrees on it
(`_strict_param_spelling`), so a caller may keep passing an `Integer`.
"""
function _type_name_bound(t::AbstractString, defined_names)::Bool
    occursin(r"[{}<>,\s\.]", t) && return true
    t in defined_names && return true
    sym = Symbol(t)
    return isdefined(Base, sym) || isdefined(Core, sym)
end

"The `::Any`-relaxed kind a parameter's C type takes in a signature the loop builds."
_relaxed_param_kind(t::AbstractString, bound) =
    t in _INTEGER_CTYPES ? "Integer" :
    (startswith(t, "Ptr{") || !bound(t)) ? "Any" : String(t)

const _CHAR_POINTER_CTYPES = ("Ptr{UInt8}", "Ptr{Int8}", "Ptr{Cchar}", "Ptr{Cuchar}", "Cstring")
const _CHAR_POINTER_UNION = "Union{AbstractString, Cstring, Ptr{UInt8}, Ptr{Int8}, AbstractVector{UInt8}}"

"""
    _strict_param_spelling(t; bound, owners, is_enum) -> String

The dispatch-relevant Julia type a parameter of C type `t` takes at a position
an overload set disagrees on: what tells the overloads apart while admitting
what callers pass today through `::Any`.

  * an integer: its C type (`Cint`, `Clonglong`), as before;
  * `Ptr{Cvoid}`: `Any`, the catch-all the `void*` overload always was;
  * a `char*` of either sign, or `Cstring`: a `String`, a `Cstring`, a byte
    pointer or a byte vector (`_CHAR_POINTER_UNION`), which is what
    `JITManager.invoke` and `ccall` convert;
  * `Ptr{T}` for a `T` this module emits (or an enum): `Ptr{T}`, `Ref{T}`, an
    `AbstractVector{T}`, and the owning types that `unsafe_convert` to
    `Ptr{T}` (`Managed<T>`, a factory struct; `owners(T)`), so a `Managed`
    object still passes where it did;
  * `Ptr{T}` for a `T` nothing emits: `Any` (the name would not resolve);
  * an emitted enum: the enum type itself, never `Integer`: beside an integer
    overload, `Union{E, Integer}` would take the integers meant for it;
  * anything else: `t` as the loop already spells it.

The overload prediction (`_integer_width_overloads`) and the function loop
both spell a strict position through this, so they agree by construction.
"""
function _strict_param_spelling(t::AbstractString; bound = _ -> true, owners = _ -> String[],
                                is_enum = _ -> false)::String
    t = String(t)
    t in _INTEGER_CTYPES && return t
    t == "Ptr{Cvoid}" && return "Any"
    t in _CHAR_POINTER_CTYPES && return _CHAR_POINTER_UNION
    m = match(r"^Ptr\{(.+)\}$", t)
    if m !== nothing
        inner = String(m.captures[1])
        (bound(inner) || is_enum(inner)) || return "Any"
        parts = ["Ptr{$inner}", "Ref{$inner}", "AbstractVector{$inner}"]
        append!(parts, owners(inner))
        return "Union{" * join(unique(parts), ", ") * "}"
    end
    is_enum(t) && return t
    bound(t) || return "Any"
    return t
end

"""
    _deleting_destructor_doc(mangled, julia_name, only_one) -> String

The sentences a deleting destructor's (D0) docstring carries after "Wrapper
for": it frees, so the object must have come from the class's allocator and
must not be freed again; and whether the library has a non-deleting
destructor for the class (`<X>_destroy_<X>`), or this is the only one.
Empty for anything else.
"""
function _deleting_destructor_doc(mangled::AbstractString, julia_name::AbstractString, only_one::Bool)
    _is_deleting_destructor(mangled) || return ""
    destroy = replace(julia_name, "_delete_" => "_destroy_"; count = 1)
    tail = only_one ?
        "The library has no non-deleting destructor for this class (its destructor is implicit), " *
        "so there is no `$destroy`." :
        "To end the object without freeing it, call `$destroy`."
    return "\n\n**The deleting destructor.** Runs the destructor and then frees the object " *
           "through the class's `operator delete`, so `this` must have come from that " *
           "allocator (`new`, or the class's own) and must not be freed again. $tail"
end

"""
    _report_inline_constructors(plan)

The wrap report's lines for the inline constructors the build defined itself
(the metadata's `inline_constructors`, from
`Compiler.generate_inline_constructors`): the classes whose constructors the
forced unit emitted, the forced rows the unit's guard declined (an abstract
class, or a constructor not constructible from its declared parameters), and
the rows skipped before the unit with their reasons. Nothing for metadata
without the key, or with nothing to report. With `what = "member function"`
the same lines for the metadata's `inline_methods`.
"""
function _report_inline_constructors(plan; what::String = "constructor")
    plan isa AbstractDict || return
    forced = [f for f in get(plan, "forced", Any[]) if f isa AbstractDict]
    skipped = [s for s in get(plan, "skipped", Any[]) if s isa AbstractDict]
    # `emitted` is written by the build's second extraction; a row without it
    # was planned and never forced (a graph build runs no unit).
    emitted = [f for f in forced if get(f, "emitted", nothing) == true]
    declined = [f for f in forced if get(f, "emitted", nothing) == false]
    unforced = [f for f in forced if get(f, "emitted", nothing) === nothing]
    show(xs) = join(first(xs, 8), ", ") * (length(xs) > 8 ? ", …" : "")
    if !isempty(emitted)
        classes = sort!(unique(String[String(f["class"]) for f in emitted]))
        println("  wrap: $(length(emitted)) inline $(what)(s) of $(length(classes)) class(es) ",
                "declared in headers and defined by the build's forced unit: ", show(classes))
    end
    if !isempty(unforced)
        classes = sort!(unique(String[String(f["class"]) for f in unforced]))
        println("  wrap: $(length(unforced)) inline $(what)(s) of $(length(classes)) class(es) ",
                "declared in headers and defined by no translation unit; this build ran no forced unit: ",
                show(classes))
    end
    if !isempty(declined)
        println("  wrap: $(length(declined)) inline $(what)(s) the forced unit declined ",
                "(", what == "constructor" ? "an abstract class, or not constructible from the declared parameters" :
                                             "no definition came out of the unit", "): ",
                show(sort!(unique(String[String(f["class"]) for f in declined]))))
    end
    if !isempty(skipped)
        # A compiler's message is one row's own; the report groups those and the
        # metadata keeps each message.
        by_reason = Dict{String,Vector{String}}()
        for s in skipped
            reason = String(get(s, "reason", "?"))
            # A header's failure is its own group: the row is not at fault.
            startswith(reason, "its header does not compile on its own") ?
                (reason = "its header does not compile on its own (a fragment, or one that needs another included first; " *
                          "each compiler message is in the metadata's $(what == "constructor" ? "inline_constructors" : "inline_methods"))") :
            occursin("does not compile", reason) &&
                (reason = "does not compile in the unit (each compiler message is in the metadata's $(what == "constructor" ? "inline_constructors" : "inline_methods"))")
            push!(get!(() -> String[], by_reason, reason), String(get(s, "class", "?")))
        end
        for reason in sort!(collect(keys(by_reason)))
            classes = sort!(unique(by_reason[reason]))
            println("  wrap: $(length(classes)) class(es) with an inline $(what) left undefined, $reason: ",
                    show(classes))
        end
    end
    return
end

"""
    _integer_width_overloads(functions, struct_types, ccall_param_type; concrete, bound, owners, is_enum)
        -> Dict{String,Set{Int}}

For each mangled symbol whose wrapper would share its Julia name and dispatch
signature with another C++ entry point, the positions in `func["parameters"]`
the set disagrees on: where the strict spellings (`_strict_param_spelling`)
of the members differ. The signature is predicted the way the function loop
builds it: integer parameters `Integer`, pointers and unbound names (an enum
among them) `Any`, a receiver `Any` in front when the loop would inject one,
and the name from `_julia_function_name`. A function the loop skips (varargs)
is not grouped.

It began with integers alone (2026-10-31) and the name kept: a width is the
Julia type the C alias names on this host (`concrete`, by default
`_concrete_ctype`): `int` and `long` are one width on LLP64, `long` and `long
long` on LP64. A set that differs only in the spelling of one width is not a
disagreement; it stays `::Integer` and `_dedup_method_chunks` folds it as it
always did, instead of two `function`s with one dispatch signature, which
precompilation refuses. Tests pass another map to drive both hosts. Pointer
and enum positions joined it when 52 of the Hub's 89 dropped C++ functions
were overloads that differed in a pointer's target or in an enum against a
pointer (pugixml's `append_child(xml_node_type)` beside `(const char*)`,
imgui's `GetID(const char*)` beside `(const void*)`, 2026-11-13).
"""
function _integer_width_overloads(functions, struct_types, ccall_param_type;
                                  concrete = _concrete_ctype, bound = _ -> true,
                                  owners = _ -> String[], is_enum = _ -> false,
                                  string_param = _ -> false, object_address = _ -> false)
    groups = Dict{String,Vector{Tuple{String,Vector{String}}}}()
    for func in functions
        get(func, "is_vararg", false) == true && continue
        raw = get(func, "parameters", Any[])
        func_name = String(get(func, "name", ""))
        mangled = String(get(func, "mangled", func_name))
        class_name = String(get(func, "class", ""))
        is_method = get(func, "is_method", false) == true
        hop = get(func, "has_object_pointer", nothing)
        jname = is_method && !isempty(class_name) ? "$(class_name)_$(func_name)" : func_name
        jname = _julia_function_name(jname, mangled)
        has_this = !isempty(raw) && get(raw[1], "name", "") == "this"
        injects = is_method && !isempty(class_name) && hop !== false && !has_this &&
                  (_cpp_this_param(class_name, func_name, struct_types) !== nothing || hop === true)
        # A container taken by value crosses by address and is spelled as a
        # pointer, here as in the loop (`_by_address_ctype`).
        ctypes = String[_by_address_ctype(ccall_param_type(String(get(p, "julia_type", "Any")),
                                                           String(get(p, "c_type", ""))),
                                          String(get(p, "c_type", ""));
                                          object_address = object_address(String(get(p, "c_type", ""))))
                        for p in raw]
        # A `std::string` parameter the wrapper builds from a Julia string
        # (`_string_param_plan`) is spelled `_STD_STRING_PARAM` here as in the
        # loop, so a `const char*` sibling is another kind and its own method.
        kinds = [string_param(String(get(raw[i], "c_type", ""))) ? _STD_STRING_PARAM :
                 _relaxed_param_kind(ctypes[i], bound) for i in eachindex(ctypes)]
        injects && pushfirst!(kinds, "Any")
        # A receiver DWARF supplied is position 1 of `ctypes`; it is never a
        # strict position (below), so it is recorded as such.
        has_this && (ctypes[1] = "this")
        push!(get!(() -> Tuple{String,Vector{String}}[], groups, jname * "(" * join(kinds, ",") * ")"),
              (mangled, ctypes))
    end
    spell(t) = t in _INTEGER_CTYPES ? concrete(t) :
               _concrete_sig_type(_strict_param_spelling(t; bound, owners, is_enum))
    strict = Dict{String,Set{Int}}()
    for (_, members) in groups
        length(unique(first.(members))) >= 2 || continue
        n = length(members[1][2])
        all(m -> length(m[2]) == n, members) || continue
        for i in 1:n
            # The receiver stays `::Any`, whatever the metadata says it
            # points at: two entries of one destructor attributed to a base
            # and a derived class differ only there (tinyxml2's
            # `~XMLComment` entries, one carrying `XMLNode`'s D2 symbol),
            # and typing it would take the `::Any` receiver every caller
            # passes a `Ptr{Cvoid}` or a `Managed` object to.
            any(m -> m[2][i] == "this", members) && continue
            # Two spellings of one type (`int`/`long` on LLP64, `char*` of
            # either sign) are one type.
            length(Set(spell(m[2][i]) for m in members)) >= 2 || continue
            for (mangled, _) in members
                push!(get!(() -> Set{Int}(), strict, mangled), i)
            end
        end
    end
    return strict
end

"""
    _rvalue_shadowed(functions) -> Set{String}

The mangled symbols of overloads taking an rvalue reference (`T&&`) where
another overload of the same name takes the same parameters by lvalue
reference (`const T&`, `T&`). Julia has no rvalue: a caller passes a `Ref`
to either, and both are one Julia signature. The move overload steals from
its argument, so when `_dedup_method_chunks` kept it (msdfgen's
`Shape::addContour(Contour&&)`, pugixml's `xpath_node_set(xpath_node_set&&)`)
a caller copying an object emptied it. These are not emitted; the lvalue
overload is the one a Julia caller means. A `T&&` overload with no lvalue
twin is emitted as before.
"""
function _rvalue_shadowed(functions)::Set{String}
    norm(c) = replace(replace(String(c), r"\b(const|volatile)\b\s*" => ""), r"\s+" => "")
    key(f) = (String(get(f, "name", "")), String(get(f, "class", "")),
              [replace(norm(get(p, "c_type", "")), "&&" => "&") for p in get(f, "parameters", Any[])])
    lvalue = Set{Any}()
    for f in functions
        any(p -> endswith(norm(get(p, "c_type", "")), "&&"), get(f, "parameters", Any[])) && continue
        push!(lvalue, key(f))
    end
    out = Set{String}()
    for f in functions
        any(p -> endswith(norm(get(p, "c_type", "")), "&&"), get(f, "parameters", Any[])) || continue
        key(f) in lvalue && push!(out, String(get(f, "mangled", "")))
    end
    return out
end

import ..Compiler: _demangle, _param_list_paren, _matching_paren, _split_depth0_commas, _is_vtable_only

"""
    _structor_variant(func) -> Union{Nothing,Tuple{Symbol,Char}}

`(:ctor, v)` or `(:dtor, v)` for a constructor or destructor row of the
metadata, `v` the Itanium variant its symbol names: a constructor's `C1`
(complete object) or `C2` (base object), a destructor's `D0` (deleting), `D1`
(complete object) or `D2` (base object). `nothing` for any other function, for
a default-argument function standing under a constructor's name (its symbol is
`replibuild_default::`'s), and for a symbol the marker cannot be read from.

A destructor takes no parameters, so its marker closes the symbol (`…D1Ev`,
as `_is_deleting_destructor` reads it). A constructor's is the one `C<digit>`
before an `E`, `I` or `B`; where an identifier holds another (`C1Engine`), the
marker is the one whose digit can change without changing what the symbol
demangles to, which is what a variant is.
"""
function _structor_variant(func)
    name = String(get(func, "name", ""))
    _is_ctor_or_dtor_cpp(String(get(func, "class", "")), name) || return nothing
    haskey(func, "default_arguments") && return nothing
    sym = String(get(func, "mangled", ""))
    if startswith(name, "~")
        m = match(r"D([012])Ev$", sym)
        return m === nothing ? nothing : (:dtor, m.captures[1][1])
    end
    cands = collect(eachmatch(r"C([123])(?=[EIB])", sym))
    if length(cands) > 1
        spelled = _demangle(sym)
        cands = spelled === nothing ? empty(cands) : filter(cands) do c
            other = c.captures[1] == "1" ? "2" : "1"
            _demangle(sym[1:c.offset] * other * sym[c.offset + 2:end]) == spelled
        end
    end
    return length(cands) == 1 ? (:ctor, cands[1].captures[1][1]) : nothing
end

"""
    _has_virtual_base(class, dwarf_structs) -> Union{Bool,Nothing}

Whether `class` has a virtual base, its own or one a base of it has: the
classes whose base-object constructor and destructor take a VTT. `nothing`
when the metadata cannot say: no `struct_definitions` entry for the class, or
for a base on the way and no virtual base found elsewhere.
"""
function _has_virtual_base(class::AbstractString, dwarf_structs, seen = Set{String}())
    FG = JLCSIRGenerator.FunctionGen
    key = nothing
    for cand in FG._scope_suffixes(String(class))
        key = FG._fuzzy_struct_lookup(cand, dwarf_structs)
        key === nothing || break
    end
    key === nothing && return nothing
    key in seen && return false
    push!(seen, key)
    unknown = false
    for base in get(dwarf_structs[key], "base_classes", Any[])
        base isa AbstractDict || continue
        get(base, "virtual", false) == true && return true
        r = _has_virtual_base(replace(String(get(base, "type", "")), r"^(class|struct)\s+" => ""),
                              dwarf_structs, seen)
        r === true && return true
        r === nothing && (unknown = true)
    end
    return unknown ? nothing : false
end

"""
    _structor_parameters(func, kind) -> (declared, implicit)

A constructor's or destructor's metadata parameters, split: the ones the
declaration has (`this` and the written ones), and the implicit one a
base-object variant of a class with a virtual base takes after `this`, the
VTT. Metadata built since 2026-11-17 marks it `artificial`, and only a
base-object variant's row has one. Before that every variant of a member
shared the list of whichever definition the build read last, so in such
metadata the VTT is what a destructor's list holds after `this`, and in a
constructor's a `void**` named `vtt` there when the list is one longer than
the demangled name's.
"""
function _structor_parameters(func, kind::Symbol)
    params = collect(Any, get(func, "parameters", Any[]))
    marked = findall(p -> get(p, "artificial", false) == true, params)
    isempty(marked) || return (params[setdiff(eachindex(params), marked)], params[marked])
    # Unmarked: only a list that names its receiver and goes on past it.
    (length(params) >= 2 && get(params[1], "name", "") == "this") || return (params, Any[])
    kind === :dtor && return (params[1:1], params[2:end])
    (String(get(params[2], "name", "")) == "vtt" &&
     replace(String(get(params[2], "c_type", "")), r"\s+" => "") == "void**") || return (params, Any[])
    demangled = String(get(func, "demangled", ""))
    (open, _) = _param_list_paren(demangled)
    close = open == 0 ? 0 : _matching_paren(demangled, open)
    close == 0 && return (params, Any[])
    inner = strip(demangled[nextind(demangled, open):prevind(demangled, close)])
    written = isempty(inner) ? 0 : length(_split_depth0_commas(inner))
    length(params) - 1 == written + 1 || return (params, Any[])
    return (params[setdiff(eachindex(params), 2)], params[2:2])
end

"""
    _structor_bindings(functions, dwarf_structs) -> (skipped, refused)

Which symbol stands behind `<Class>_<Class>(this, …)` and
`<Class>_destroy_<Class>(this)`: the COMPLETE-object constructor (`C1`) and
destructor (`D1`), the ones `T x;` and `x.~T()` run. `skipped` is the set of
symbols not emitted because another variant of the same member is; `refused`
maps a symbol to why its wrapper is a refusal.

For a class with no virtual base, `C1` and `C2` (and `D1` and `D2`) are one
function, and clang often emits only the base-object one (an abstract class's
constructor, an inline destructor); either is bound, `C1`/`D1` when the
library has both. (An abstract class's constructor is refused afterwards, for
another reason: `_abstract_constructor_refusals!`.) For a class WITH a virtual base they are two: the
base-object variant takes a VTT after `this` and leaves the virtual base
alone, so a wrapper on it built half an object from a null VTT and crashed
(`VBottom_VBottom(p)`), and `<X>_destroy_<X>` was `(this, vtt)`. Until
2026-11-17 the two became one Julia method and `_dedup_method_chunks` kept
the last in the metadata's order, which was `C2` and `D2` for every class.
Such a class's base-object variant is never bound: with no complete-object
variant in the library the wrapper says so. Whether the class has a virtual
base is `_has_virtual_base`; where the metadata cannot say, a VTT in the
variant's own parameters does (`_structor_parameters`).

`D0` (deleting) is its own Julia function, `<X>_delete_<X>`, and never part
of the choice. A complete-object or deleting variant whose list still holds
a VTT is metadata from before each variant had its own list: refused, with
the rebuild that fixes it, since the thunk would be built from the same list.
"""
function _structor_bindings(functions, dwarf_structs)
    groups = Dict{Tuple{String,String,Symbol},Dict{Char,Any}}()
    for f in functions
        v = _structor_variant(f)
        v === nothing && continue
        g = get!(() -> Dict{Char,Any}(), groups,
                 (String(get(f, "class", "")), String(get(f, "demangled", "")), v[1]))
        get!(g, v[2], f)
    end
    skipped = Set{String}()
    refused = Dict{String,String}()
    virtual_base = Dict{String,Union{Bool,Nothing}}()      # asked once per class
    for ((cls, demangled, kind), g) in groups
        what = kind === :ctor ? "constructor" : "destructor"
        complete = get(g, '1', nothing)
        base = get(g, '2', nothing)
        vbase = get!(() -> _has_virtual_base(cls, dwarf_structs), virtual_base, cls)
        if complete !== nothing && base !== nothing
            push!(skipped, String(base["mangled"]))
        elseif base !== nothing &&
               (vbase === true || (vbase === nothing && !isempty(_structor_parameters(base, kind)[2])))
            refused[String(base["mangled"])] =
                "The library defines only the base-object $what of `$cls` (`$(base["mangled"])`), and `$cls` " *
                "has a virtual base: that variant takes a VTT and leaves the virtual base " *
                (kind === :ctor ? "unconstructed, so it cannot build" : "alive, so it cannot end") *
                " a whole object. The complete-object $what is not in the library (" *
                (kind === :ctor ? "an abstract class, or " : "") *
                "an inline $what nothing in the library runs on a whole object)."
        end
        vbase === false && continue
        for f in (complete, get(g, '0', nothing))
            (f === nothing || isempty(_structor_parameters(f, kind)[2])) && continue
            refused[String(f["mangled"])] =
                "The metadata lists a VTT among the parameters of `$demangled` (`$(f["mangled"])`), which " *
                "only the base-object $what takes: it predates per-variant parameter lists. Rebuild the " *
                "package (delete .replibuild_cache/project_hash) and wrap again."
        end
    end
    return (skipped, refused)
end

"""
    _abstract_constructor_refusals!(refused, functions, skipped, record_abi) -> Vector{String}

Refuse every constructor of an abstract class, adding to `refused` (symbol →
reason) what `_structor_bindings` left bindable; returns the classes, sorted.
A class is abstract here when the build recorded pure virtual member
functions an object of it would still have (`record_abi`'s `pure_virtuals`,
`Compiler._mark_abstract_records!`).

C++ builds no object of such a class except as the base of one that defines
those functions, and clang emits only its base-object constructor (`C2`).
Bound like any constructor, `Animal_Animal(p, 4)` left at `p` an object whose
vtable holds `__cxa_pure_virtual` in the pure slots: the first call of one,
through `Animal_name(p)` or from inside the library (`Animal::describe()`
calls `name()`), ends the process. Metadata without the key refuses nothing.
"""
function _abstract_constructor_refusals!(refused::AbstractDict, functions, skipped, record_abi)
    FG = JLCSIRGenerator.FunctionGen
    classes = Set{String}()
    for f in functions
        v = _structor_variant(f)
        (v === nothing || v[1] !== :ctor) && continue
        m = String(get(f, "mangled", ""))
        (m in skipped || haskey(refused, m)) && continue
        cls = String(get(f, "class", ""))
        suffixes = FG._scope_suffixes(cls)
        k = findfirst(c -> haskey(record_abi, c), suffixes)
        k === nothing && continue
        left = get(record_abi[suffixes[k]], "pure_virtuals", nothing)
        (left isa AbstractVector && !isempty(left)) || continue
        refused[m] = "`$cls` is an abstract class: " * join(("`$x`" for x in first(left, 3)), ", ") *
            (length(left) > 3 ? " and $(length(left) - 3) more" : "") *
            (length(left) == 1 ? " is" : " are") * " pure virtual in it, so C++ builds no object of `$cls` except " *
            "as the base of a class that defines " * (length(left) == 1 ? "it" : "them") * ". This constructor, run " *
            "on storage of its own, would leave an object whose vtable ends the process when a pure virtual " *
            "is called. Construct a derived class, or call a function of the library that returns one."
        push!(classes, cls)
    end
    return sort!(collect(classes))
end

"The sentences a pure virtual member function's docstring carries after \"Wrapper for\"."
const _PURE_VIRTUAL_DOC = "\n\n**Pure virtual.** The class declares this function and defines no body for it: " *
    "the call goes through the vtable of the object `this` points at, to the override of that object's own class."

"""
    _report_vtable_methods(plan)

The wrap report's lines for the pure virtual member functions (the
metadata's `vtable_methods`, `Compiler._vtable_only_plan`): how many are
bound through the vtable, of which classes, and the ones passed over, by
reason. Nothing for metadata without the key or with nothing to report.
"""
function _report_vtable_methods(plan)
    plan isa AbstractDict || return
    show(xs) = join(first(xs, 8), ", ") * (length(xs) > 8 ? ", …" : "")
    bound = [r for r in get(plan, "bound", Any[]) if r isa AbstractDict]
    if !isempty(bound)
        classes = sort!(unique(String[String(r["class"]) for r in bound]))
        println("  wrap: $(length(bound)) pure virtual member function(s) of $(length(classes)) class(es) have no ",
                "symbol and are called through the object's vtable: ", show(classes))
    end
    by_reason = Dict{String,Vector{String}}()
    for r in get(plan, "skipped", Any[])
        r isa AbstractDict || continue
        push!(get!(() -> String[], by_reason, String(get(r, "reason", "?"))), String(get(r, "class", "?")))
    end
    for reason in sort!(collect(keys(by_reason)))
        println("  wrap: $(length(by_reason[reason])) pure virtual member function(s) passed over, $reason: ",
                show(sort!(unique(by_reason[reason]))))
    end
    return
end

"""
    _string_return_plan(c_ret, record_abi) -> nothing | (reader, dtor) | String

What a wrapper does with a `std::basic_string` its function returns by value,
built in the caller's buffer (`FunctionGen.returns_in_caller_buffer`):
`nothing` when the return is no `basic_string`; `(reader, dtor)` when it is read
into a `String` (`StdStrings.take_string!`): the `StdStrings.Reader` from the
layout the build recorded (`record_abi`'s `string_layout`) and the
complete-object destructor's symbol (`D1`, from the declared `D4`); else the
reason it keeps the stand-in, which the wrapper writes beside the call, naming
the type (the coverage tool attributes stand-ins by that text).
"""
function _string_return_plan(c_ret::AbstractString, record_abi)
    t = JLCSIRGenerator.FunctionGen._strip_top_cv(String(c_ret))
    key = nothing
    for cand in JLCSIRGenerator.FunctionGen._scope_suffixes(t)
        if startswith(cand, "basic_string<") && haskey(record_abi, cand)
            key = cand
            break
        end
    end
    key === nothing && return nothing
    layout = get(record_abi[key], "string_layout", nothing)
    layout === nothing &&
        return "std::string `$key` returned as its raw bytes: the metadata predates `string_layout`; " *
               "rebuild the package (delete .replibuild_cache/project_hash) to record the string's layout"
    r = StdStrings.string_reader(layout)
    r isa String && return "std::string `$key` returned as its raw bytes: $r"
    d = get(layout, "destructor", nothing)
    d isa AbstractString ||
        return "std::string `$key` returned as its raw bytes: the DWARF declares no destructor for it"
    return (r, replace(String(d), r"D[24]Ev$" => "D1Ev"))
end

"""
The Julia type a `std::string` parameter takes: a string to build from, an
existing object's address, or the STL wrapper's own `CppString` (a
`[types] templates` entry's string, which `unsafe_convert`s to its object and
took the position while it was `::Any`).
"""
const _STD_STRING_PARAM = "Union{AbstractString, Ptr{Cvoid}, RepliBuild.STLWrappers.CppString}"

"""
    _string_param_plan(c_type, record_abi) -> nothing | (key, bytes, ctor, takes_allocator, dtor) | String

What a parameter of C type `c_type` does with a Julia string (2026-11-15).
`nothing` when it is not a `std::basic_string` taken by `const&` or by value:
a pointer, a non-`const` reference (an out-parameter the library writes to),
anything else. A `Tuple` when the wrapper builds the object
(`StdStrings.string_arg`): the record's key, its size, the complete-object
constructor (`C1`, from the `C4` the build recorded in `string_layout`, the
one building from a `const char*` and a length, `Compiler._string_constructor`),
whether it takes an allocator, and the complete-object destructor. A `String`
when it cannot, with the reason the wrapper writes beside the call: metadata
from before the constructor was recorded (rebuild), a wide string, no such
constructor or destructor declared. The signature then keeps today's spelling.
"""
function _string_param_plan(c_type::AbstractString, record_abi)
    FG = JLCSIRGenerator.FunctionGen
    t = strip(String(c_type))
    occursin(r"[*(\[]", FG._outside_template_args(t)) && return nothing
    byref = endswith(t, "&")
    byref && endswith(t, "&&") && return nothing
    core = byref ? strip(t[1:prevind(t, end)]) : t
    # Only `const T&` builds a temporary; `T&` is written to by the callee.
    is_const = occursin(r"^const\s", core) || occursin(r"\sconst$", core)
    byref && !is_const && return nothing
    core = FG._strip_top_cv(core)
    key = nothing
    for cand in FG._scope_suffixes(core)
        if startswith(cand, "basic_string<") && haskey(record_abi, cand)
            key = cand
            break
        end
    end
    key === nothing && return nothing
    facts = record_abi[key]
    layout = get(facts, "string_layout", nothing)
    layout === nothing &&
        return "std::string `$key` takes its byte stand-in: the metadata predates `string_layout`; " *
               "rebuild the package (delete .replibuild_cache/project_hash) to record the string's constructor"
    ct = get(layout, "char_type", nothing)
    ct == "char" || return "std::string `$key` takes its byte stand-in: its character type is " *
                          "`$(something(ct, "unknown"))`; only `char` strings are built from a Julia string"
    haskey(layout, "constructor") ||
        return "std::string `$key` takes its byte stand-in: the metadata predates the recorded constructor; " *
               "rebuild the package (delete .replibuild_cache/project_hash)"
    ctor = get(layout, "constructor", nothing)
    ctor isa AbstractString ||
        return "std::string `$key` takes its byte stand-in: the DWARF declares no constructor from a " *
               "`const char*` and a length for it"
    d = get(layout, "destructor", nothing)
    d isa AbstractString ||
        return "std::string `$key` takes its byte stand-in: the DWARF declares no destructor for it"
    bytes = try FG._parse_byte_size(string(get(facts, "byte_size", "0"))) catch; 0 end
    bytes > 0 || return "std::string `$key` takes its byte stand-in: the metadata records no size for it"
    return (key, bytes, _complete_object_ctor(String(ctor)), get(layout, "constructor_allocator", false) == true,
            replace(String(d), r"D[24]Ev$" => "D1Ev"))
end

# The complete-object constructor (`C1`) for a constructor the DWARF spelled
# unified (`C4`) or as the base-object variant (`C2`): the marker follows the
# class prefix, which `_dtor_variant_group`'s regex finds the same way.
function _complete_object_ctor(sym::String)
    m = match(r"^(.*?)C[24]E", sym)
    m === nothing && return sym
    return m.captures[1] * "C1E" * sym[m.offset + length(m.match):end]
end

_reader_literal(r) = "RepliBuild.StdStrings.Reader($(repr(r.kind)), $(r.ptr), $(r.len), $(r.flag_bit), " *
                     "$(r.short_data), $(r.short_len_bit), $(r.short_len_bits))"

"""
    _thunks_named_in_text(text, taken) -> Set{String}

The mangled symbols whose thunk the emitted `text` calls, in every spelling a
call site has: `"_mlir_ciface_<m>_thunk"` (JIT), an AOT slot `_FPTR_x[]`
resolved through `taken` (`_aot_fptr_const_name!`), and an AOT virtual
`ccall((:thunk_<m>, THUNKS_LIBRARY_PATH), …)`. The thunk manifest is the set
wrap derived intersected with this: a chunk `_dedup_method_chunks` dropped
must not leave a thunk the JIT compiles for no call site (87 across the Hub,
2026-11-13).
"""
function _thunks_named_in_text(text::AbstractString, taken::Dict{String,String})::Set{String}
    # `_thunk_inplace` too: a return built in the caller's buffer
    # (`FunctionGen.thunk_symbol`); `\b` alone stops at neither.
    out = Set{String}(String(m.captures[1])
                      for m in eachmatch(r"_mlir_ciface_(\S+?)_thunk(?:_inplace)?\b", text))
    for m in eachmatch(r"\b(_FPTR_\w+)\b", text)
        haskey(taken, m.captures[1]) &&
            push!(out, replace(taken[m.captures[1]], r"_thunk(?:_inplace)?$" => ""))
    end
    for m in eachmatch(r"ccall\(\(:thunk_(\S+?),\s*THUNKS_LIBRARY_PATH\)", text)
        push!(out, String(m.captures[1]))
    end
    return out
end

function _cpp_this_param(class_name::AbstractString, func_name::AbstractString,
                         struct_types)
    isempty(class_name) && return nothing

    bare_class = _cpp_innermost_scope(class_name)
    safe_class = _sanitize_cpp_type_name(bare_class)
    safe_class = replace(safe_class, r"[^A-Za-z0-9_]" => "")
    safe_class = replace(safe_class, r"_+"            => "_")
    safe_class = String(strip(safe_class, '_'))
    # Garbled (empty, or an operator spelling) — no usable Julia type name.
    if isempty(safe_class) || startswith(safe_class, "operator")
        safe_class = "Cvoid"
    end
    safe_class == "Cvoid" && return nothing

    known_class = bare_class in struct_types || safe_class in struct_types
    (known_class || _is_ctor_or_dtor_cpp(class_name, func_name)) || return nothing

    return Dict{String,Any}(
        "name" => "this",
        "c_type" => String(class_name) * "*",
        "julia_type" => known_class ? "Ptr{" * safe_class * "}" : "Ptr{Cvoid}",
        "position" => 0,
        "is_synthesized" => true
    )
end

"""
    _julia_field_layout(t, jl_layouts, jl_mutable) -> Union{Tuple{Int,Int},Nothing}

`(sizeof, alignment)` Julia gives a struct FIELD of emitted type `t`, or `nothing`
when it is not known here. Primitives come from GeneratorC's table
(`_C_PRIM_FIELD_LAYOUT`, same module), so both generators measure a field the
same way. A `mutable struct` field is a reference: pointer-sized, whatever the
type's own size.
"""
function _julia_field_layout(t::AbstractString, jl_layouts, jl_mutable)
    t = String(strip(t))
    haskey(_C_PRIM_FIELD_LAYOUT, t) && return _C_PRIM_FIELD_LAYOUT[t]
    (startswith(t, "Ptr{") || t == "Cstring") && return (8, 8)
    t in jl_mutable && return (8, 8)
    haskey(jl_layouts, t) && return jl_layouts[t]
    m = match(r"^NTuple\{(\d+),\s*(.+)\}$", t)
    if m !== nothing
        el = _julia_field_layout(String(m.captures[2]), jl_layouts, jl_mutable)
        el === nothing && return nothing
        return (parse(Int, m.captures[1]) * el[1], el[2])
    end
    return nothing
end

"""
    _prove_julia_layout(plan, byte_size, jl_layouts, jl_mutable)
        -> Union{Tuple{Int,Int},Symbol}

`plan` is the field list a struct would be emitted with, in order: `(julia type,
DWARF offset)` for a member, `(julia type, nothing)` for a pad. Returns the
struct's Julia `(sizeof, alignment)` when every member lands on its DWARF offset
and the struct on `byte_size`; `:mismatch` when Julia's layout provably differs;
`:unknown` when some field cannot be measured here, and the caller keeps its
previous behaviour.
"""
function _prove_julia_layout(plan, byte_size::Int, jl_layouts, jl_mutable)
    cur = 0
    maxal = 1
    for (t, off) in plan
        lay = _julia_field_layout(t, jl_layouts, jl_mutable)
        lay === nothing && return :unknown
        (sz, al) = lay
        cur = _align_up(cur, al)
        off !== nothing && cur != off && return :mismatch
        cur += sz
        maxal = max(maxal, al)
    end
    total = _align_up(cur, maxal)
    return total == byte_size ? (total, maxal) : :mismatch
end

const _SSE_JULIA_TYPES = ("Cfloat", "Cdouble", "Float16", "Float32", "Float64", "ComplexF32", "ComplexF64")

"""
    _type_has_sse(julia_type, dwarf_structs, record_key, is_enum; seen) -> Bool

Whether the C type a member's `julia_type` spells may hold a `float` or
`double`, which SysV puts in an SSE eightbyte: a floating scalar, an array of
one, or a record holding one at any depth, bases included. Integers, pointers
and enums do not. `record_key(name)` maps a record name to its `dwarf_structs`
key or `nothing`; a type that is neither resolved nor known counts as holding
one.
"""
function _type_has_sse(jt::AbstractString, dwarf_structs, record_key, is_enum;
                       seen::Set{String} = Set{String}())::Bool
    jt = String(strip(jt))
    jt in _SSE_JULIA_TYPES && return true
    (startswith(jt, "Ptr{") || haskey(_C_PRIM_FIELD_LAYOUT, jt)) && return false
    am = match(r"^NTuple\{\d+,\s*(.+)\}$", jt)
    am !== nothing && return _type_has_sse(am.captures[1], dwarf_structs, record_key, is_enum; seen)
    is_enum(jt) && return false
    key = record_key(jt)
    key === nothing && return true
    key in seen && return false
    push!(seen, key)
    info = dwarf_structs[key]
    for b in get(info, "base_classes", [])
        bt = replace(String(get(b, "type", "")), r"^(class|struct)\s+" => "")
        isempty(bt) && continue
        _type_has_sse(bt, dwarf_structs, record_key, is_enum; seen) && return true
    end
    for m in get(info, "members", [])
        # A static member is not part of the object.
        get(m, "offset", nothing) === nothing && !haskey(m, "bit_size") && continue
        _type_has_sse(String(get(m, "julia_type", "Any")), dwarf_structs, record_key, is_enum; seen) &&
            return true
    end
    return false
end

"""
    _crosses_small_blob(func, dwarf_structs, record_abi, blob_names) -> Bool

Whether `func` passes or returns by value, at 16 bytes or less, a record whose
Julia type puts bytes where C has a `float` or `double` (`blob_names`,
sanitized names): one the struct emitter wrote as a byte blob, or one with a
field it wrote as bytes because the member's type cannot be named there. A
ccall would carry those bytes, and `NTuple{N,UInt8}` claims INTEGER for every
SysV eightbyte where the record may have SSE ones, so such a function goes to
the thunk, which lays the record out from DWARF. Above 16 bytes a record
crosses in memory on both ABIs and the bytes are exact.
"""
function _crosses_small_blob(func, dwarf_structs, record_abi, blob_names)::Bool
    isempty(blob_names) && return false
    ts = Iterators.flatten(((get(func["return_type"], "c_type", ""),),
                            (get(p, "c_type", "") for p in func["parameters"])))
    for t in ts
        kind, key = JLCSIRGenerator.FunctionGen.by_value_crossing(String(t), dwarf_structs, record_abi)
        kind === :layout || continue
        jname = replace(replace(replace(_sanitize_cpp_type_name(key), "*" => "Ptr"), "&" => "Ref"),
                        r"[^a-zA-Z0-9_]" => "_")
        jname in blob_names && 1 <= _parse_dwarf_size(dwarf_structs[key]) <= 16 && return true
    end
    return false
end

"""
    _record_name_as_emitted(julia_type, records) -> String

The name a signature must use for a record or enum the module emits. A C++ type
named after a Base binding (`Pair`, `Vector`, `Type`, `String`, …) is emitted
as `c_<name>` by `_sanitize_cpp_type_name`, but parameter and return types kept
the raw name. `p::Pair` then meant `Base.Pair`: the emitted `c_Pair` could not
dispatch, and a `Base.Pair{Int,Int}` passed instead was marshalled as the
struct's bytes. Only a bare name, or one directly under `Ptr{…}`/`Ref{…}`, that
IS a record or enum in the metadata is rewritten; builtin spellings (`Cint`,
`Any`) never are.
"""
function _record_name_as_emitted(julia_type::AbstractString, records)::String
    m = match(r"^(Ref|Ptr)\{([A-Za-z_][A-Za-z0-9_]*)\}$", julia_type)
    if m !== nothing
        inner = String(m.captures[2])
        inner in records || return String(julia_type)
        return "$(m.captures[1]){$(_sanitize_cpp_type_name(inner))}"
    end
    return julia_type in records ? _sanitize_cpp_type_name(String(julia_type)) : String(julia_type)
end

"""
    _managed_safe_wrapper(julia_name, param_sig, param_names, julia_return_type, deleters, idiomatic_classes)
        -> Union{Nothing,Tuple{String,String}}

The `<f>_safe` wrapper of a function that returns a pointer to a class with a
`Managed<X>` type, as `(chunk, name)`: the same call, its result owned by a
`Managed<X>`. `nothing` when the return is no such pointer, or the class has an
idiomatic wrapper instead. It calls the function's own wrapper, so it is the
same text whichever tier that wrapper dispatches through.
"""
function _managed_safe_wrapper(julia_name::AbstractString, param_sig::AbstractString, param_names,
                               julia_return_type::AbstractString, deleters, idiomatic_classes)
    (startswith(julia_return_type, "Ptr{") && endswith(julia_return_type, "}")) || return nothing
    target_struct = julia_return_type[5:end-1]
    # Sanitize target struct name to match ManagedX keys
    safe_target = _sanitize_cpp_type_name(target_struct)
    ((haskey(deleters, safe_target) || haskey(deleters, target_struct)) &&
     !haskey(idiomatic_classes, safe_target) && !haskey(idiomatic_classes, target_struct)) || return nothing
    managed_name = "Managed$safe_target"
    safe_func_name = "$(julia_name)_safe"
    # Use original param_names (not ccall_args which may have converted names
    # like "a_c" that don't exist in the safe wrapper's scope)
    safe_call_args = join(param_names, ", ")
    return ("""
    \"""
        $safe_func_name($param_sig) -> $managed_name

    Safe wrapper for `$julia_name` that returns a managed object with automatic finalization.
    \"""
    function $safe_func_name($param_sig)::$managed_name
        ptr = $julia_name($safe_call_args)
        return $managed_name(ptr)
    end
    """, safe_func_name)
end

# The argument a factory class's one inner constructor takes before the handle,
# when its type is defined apart from its constructors (`_idiomatic_early_type`).
# `Base.`-qualified: the module's namespace is the library's, and a C++ class
# named `Val` would otherwise be the one called.
const _IDIOMATIC_ADOPT = "Base.Val(:_rb_adopt)"

"""
    _factory_ctor_plan(factories, julia_name) -> Vector{NamedTuple}

The constructors a factory class (the idiomatic wrapper) gets, one per
factory function, in the factories' order: `factory` (the function's Julia
name, `julia_name(name, mangled)`, the main loop's derivation), `sig_parts`
(`name::Type` per parameter), `call_args`, and `tagged`.

A constructor takes the factory's parameters, so two factories whose
parameters have one Julia dispatch signature gave the class one method
twice: `make_animal(int legs)` and `make_animal_doubled(int half)` both made
`Animal(::Cint)`. Included as a file the later definition replaced the
earlier with no message (`Animal(Cint(4))` made 8 legs), and in a
precompiled package Julia refused the overwrite, so the package did not load
(2026-11-18). The dedup keyed on the parameter TEXT, names included.

Signatures are compared as `_method_sig_keys` compares them, by the type each
C alias names on this host (`_concrete_sig_type`: `Cint` and `Int32` are one
type, `Clong` and `Clonglong` one on LP64). A signature one factory has stays
its plain constructor, so a class with no collision keeps its text. A
signature several factories share is none of theirs: each is `tagged`, and
its constructor takes `::Base.Val{:<factory>}` first, so the caller names the
factory and the bare call is a `MethodError` listing them. A factory listed
twice (one function, two DWARF entries) is one entry, the first.
"""
function _factory_ctor_plan(factories, julia_name::Function)
    entries = NamedTuple{(:factory, :sig_parts, :call_args, :key),
                         Tuple{String,Vector{String},Vector{String},String}}[]
    seen = Set{Tuple{String,String}}()
    for fac_func in factories
        fac_jl = julia_name(String(fac_func["name"]), String(get(fac_func, "mangled", "")))
        sig_parts  = String[]
        call_args  = String[]
        types      = String[]
        used_names = Set{String}()
        for (i, p) in enumerate(get(fac_func, "parameters", Any[]))
            p_nm = make_cpp_identifier(get(p, "name", "arg$(i)"))
            while p_nm in used_names; p_nm = "$(p_nm)_$(i)"; end
            push!(used_names, p_nm)
            p_jt = String(get(p, "julia_type", "Any"))
            push!(sig_parts, "$p_nm::$p_jt")
            push!(call_args, p_nm)
            push!(types, _concrete_sig_type(p_jt))
        end
        key = join(types, ",")
        (fac_jl, key) in seen && continue
        push!(seen, (fac_jl, key))
        push!(entries, (factory = fac_jl, sig_parts = sig_parts, call_args = call_args, key = key))
    end
    sharers = Dict{String,Int}()
    for e in entries
        sharers[e.key] = get(sharers, e.key, 0) + 1
    end
    return [(factory = e.factory, sig_parts = e.sig_parts, call_args = e.call_args,
             tagged = sharers[e.key] > 1) for e in entries]
end

"""
    _factory_ctor_doc(cls_nm, plan) -> String

The lines a factory class's docstring adds when factories share a signature
(`_factory_ctor_plan`): how to call each of them. Empty when none is tagged,
so a class with no collision keeps its docstring.
"""
function _factory_ctor_doc(cls_nm::AbstractString, plan)::String
    tagged = sort!([e for e in plan if e.tagged]; by = e -> e.factory)
    isempty(tagged) && return ""
    lines = String["",
                   "",
                   "Factories that take the same argument types are called by name, never",
                   "positionally (`$cls_nm(args...)` with those types is a `MethodError`):",
                   ""]
    for e in tagged
        args = join(["Val(:$(e.factory))"; e.call_args], ", ")
        push!(lines, "- `$cls_nm($args)` calls `$(e.factory)`")
    end
    return join(lines, "\n")
end

"""
    _idiomatic_early_type(cls_nm, deleter_doc; factory_doc = "") -> String

The type of a factory class (the idiomatic wrapper) that a record's field
names, without its constructors. `struct Cage { Animal *pet; }` emits
`pet::Ptr{Animal}`, which Julia resolves when `Cage` is defined, while the
factory class's type came after every record: the module failed to load
(`UndefVarError: Animal`). The type's one field is `handle::Ptr{Cvoid}`, so it
names nothing and can precede every record. Its constructors cannot: their
argument types are evaluated where they are defined, and a factory may take a
record by value or by pointer, the holder included, or another factory class.
So they are outer constructors in the idiomatic section, after every type, and
reach `new` through the one inner constructor here, which only stores the
handle; they attach the finalizer as the inner ones do. A class no field names
keeps its inner constructors, so its text does not change. `factory_doc` is
what `_factory_ctor_doc` adds to the docstring.
"""
function _idiomatic_early_type(cls_nm::AbstractString, deleter_doc::AbstractString;
                               factory_doc::AbstractString = "")::String
    return """
    \"\"\"
        $cls_nm

    Idiomatic Julia wrapper for C++ class `$cls_nm`.
    Memory is automatically managed: when Julia's GC collects this object,
    its finalizer $deleter_doc.$factory_doc
    \"\"\"
    mutable struct $cls_nm
        handle::Ptr{Cvoid}
        # Stores the handle and nothing else: the constructors, in the
        # idiomatic section, call the factory and attach the finalizer.
        $cls_nm(::Base.Val{:_rb_adopt}, handle::Ptr{Cvoid}) = new(handle)
    end

    """
end

function generate_introspective_module_cpp(config::RepliBuildConfig, lib_path::String,
                                      metadata, module_name::String,
                                      registry::TypeRegistry, generate_docs::Bool,
                                      thunks_lib_path::String="";
                                      aot_stage::Union{Nothing,Function}=nothing,
                                      emitted_facts::Union{Nothing,AbstractDict}=nothing)

    # Track exported symbols
    exports = String[]

    # Header with metadata
    header = """
    # Auto-generated Julia wrapper for $(config.project.name)
    # Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    # Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
    # Library: $(basename(lib_path))
    # Metadata: compilation_metadata.json

    module $module_name
    
    const Cintptr_t = Int
    const Cuintptr_t = UInt

    using Libdl
    import RepliBuild  # build-identity check: RepliBuild.SHA + pkgversion
    import Base: unsafe_convert

    # Resolve the library next to this file first: build artifacts travel as a
    # unit (wrapper + .so side by side, e.g. in ~/.replibuild/builds/<hash>/),
    # so the sibling copy is the one that belongs to this wrapper. The
    # generation-time absolute path is only a fallback — shared dirs like
    # ~/.replibuild/registry/julia/ get overwritten by later builds, stranding
    # any wrapper that baked them in.
    #
    # `repr` supplies the quotes AND the escaping. Hand-quoting as `\"\$(path)\"`
    # emits the path's own backslashes raw into generated Julia source, so on
    # Windows every wrapper baked `C:\\Users\\...` and failed to parse at `\\U`
    # ("invalid unicode escape sequence") — the generated module could not be
    # loaded at all. A path is data; it has to be escaped for the language it is
    # being written into, not concatenated into it.
    const LIBRARY_PATH = let baked = $(repr(abspath(lib_path)))
        sibling = Base.joinpath(@__DIR__, Base.basename(baked))
        Base.isfile(sibling) ? sibling : baked
    end
    const THUNKS_LIBRARY_PATH = let baked = $(repr(thunks_lib_path))
        sibling = Base.isempty(baked) ? \"\" : Base.joinpath(@__DIR__, Base.basename(baked))
        !Base.isempty(sibling) && Base.isfile(sibling) ? sibling : baked
    end

    # Verify library exists
    if !Base.isfile(LIBRARY_PATH)
        Base.error("Library not found: \$LIBRARY_PATH (no sibling copy in \$(@__DIR__) either)")
    end

    $(_libc_free_block(lib_path))
    # ── Build identity ────────────────────────────────────────────────────────
    # This wrapper is portable SOURCE, but its contents are a snapshot of ONE
    # compilation: struct field offsets, blob byte sizes, enum values and the
    # sliced IR all come from the library named below. Point it at a different
    # build — a JLL, a distro package, a rebuild with different flags — and
    # nothing fails loudly; it just reads the wrong offsets. So the identity of
    # the library it was generated against travels with it.
    #
    # Identity is the SHA-256 of the library file, not its GNU build ID. A build
    # ID only exists if the linker was asked for one, which RepliBuild never
    # does; a file hash needs no linker flag and is identical to the C bucket's.
    const BUILD_TARGET = \"$(Sys.MACHINE)\"
    const BUILD_GENERATOR = \"$(pkgversion(@__MODULE__) === nothing ? "" : pkgversion(@__MODULE__))\"
    const LIBRARY_SHA = \"$(_library_sha256(lib_path))\"

    \"\"\"
    Warn if this wrapper has drifted from what it was generated against.

    Two drifts, checked separately because they fail independently:

      * the LIBRARY changed under a wrapper that was not regenerated — struct
        offsets, enum values, blob sizes and any sliced IR go stale against the
        new binary;
      * the GENERATOR moved on under a library that did not — the wrapper is
        stale against RepliBuild's current codegen even though the .so matches.

    The file hash sees the first and is blind to the second; the version stamp
    is the reverse. Neither subsumes the other.

    Only warnings: a rebuild of the same source is usually fine, and refusing to
    load would break legitimate workflows. But layout drift is silent and
    unrecoverable at the call site, so the user gets told once.
    \"\"\"
    function _check_build_identity()
        if !Base.isempty(LIBRARY_SHA)
            actual = try
                Base.bytes2hex(Base.open(RepliBuild.SHA.sha256, LIBRARY_PATH))
            catch
                \"\"                                   # unreadable: nothing to compare
            end
            if !Base.isempty(actual) && actual != LIBRARY_SHA
                @warn \"\"\"
                $(module_name): the library loaded is NOT the build this wrapper was generated from.
                Struct layouts, enum values, blob sizes and any sliced IR in this wrapper are a
                snapshot of the original build; a mismatch can read wrong offsets with no error.
                Regenerate the wrapper against this library if the two are not known-compatible.\"\"\" *
                \"\\n  expected sha256: \$LIBRARY_SHA\\n  actual sha256:   \$actual\\n  library:         \$LIBRARY_PATH\" *
                \"\\n  generated for:   \$BUILD_TARGET by RepliBuild v\$BUILD_GENERATOR\"
            end
        end
        if !Base.isempty(BUILD_GENERATOR)
            running = try
                Base.string(Base.pkgversion(RepliBuild))
            catch
                \"\"                                   # RepliBuild absent: skip
            end
            if !Base.isempty(running) && running != BUILD_GENERATOR
                @warn \"\"\"
                $(module_name): this wrapper was generated by a different RepliBuild than the one loaded.
                The library itself may be fine — but codegen (struct layout, dispatch tiers, ABI
                handling) can have changed since, leaving this wrapper stale. Run wrap() again;
                a fix in what build() extracts also needs build(toml; clean=true) first.\"\"\" *
                \"\\n  wrapper generated by: RepliBuild v\$BUILD_GENERATOR\\n  currently loaded:     RepliBuild v\$running\"
            end
        end
        return nothing
    end

    # Flush C stdout so printf output appears immediately in the Julia REPL
    @inline _flush_cstdout() = ccall(:fflush, Cint, (Ptr{Cvoid},), C_NULL)

    """

    # Track if JIT is required (for virtual methods or complex ABI)
    requires_jit = false

    # Metadata section
    compiler_info = get(metadata, "compiler_info", Dict())
    lto_name = config.project.name
    lto_ir_block = config.link.enable_lto ? """
    # LTO: load LLVM bitcode at module parse time for Base.llvmcall zero-cost dispatch
    # Using bitcode (.bc) is significantly faster for Julia to parse than text IR (.ll)
    const LTO_IR_PATH = joinpath(@__DIR__, "$(lto_name)_lto.bc")
    const LTO_IR = isfile(LTO_IR_PATH) ? read(LTO_IR_PATH) : UInt8[]
    const THUNKS_LTO_IR_PATH = joinpath(@__DIR__, "$(lto_name)_thunks_lto.bc")
    const THUNKS_LTO_IR = isfile(THUNKS_LTO_IR_PATH) ? read(THUNKS_LTO_IR_PATH) : UInt8[]

    """ : """
    const LTO_IR = ""  # LTO disabled for this build
    const THUNKS_LTO_IR = ""

    """
    metadata_section = """
    # =============================================================================
    # Compilation Metadata
    # =============================================================================

    const METADATA = Dict(
        "llvm_version" => "$(get(compiler_info, "llvm_version", "unknown"))",
        "clang_version" => "$(get(compiler_info, "clang_version", "unknown"))",
        "optimization" => "$(get(compiler_info, "optimization_level", "unknown"))",
        "target_triple" => "$(get(compiler_info, "target_triple", "unknown"))",
        "function_count" => $(get(metadata, "function_count", 0)),
        "generated_at" => "$(get(metadata, "timestamp", "unknown"))"
    )

    """ * lto_ir_block

    # Extract metadata
    functions = metadata["functions"]
    dwarf_structs = get(metadata, "struct_definitions", Dict())
    # By-value record facts the thunk generator reads too (see the Tier-2
    # branch below). Absent from metadata built before it existed: every
    # record outside `struct_definitions` is then untypable, which traps.
    record_abi = get(metadata, "record_abi", Dict{String,Any}())
    # How each class's by-value temporary is made, the table the thunk
    # generator builds too: a class no temporary can be made for traps here
    # (`thunk_crossings`), where the thunk generator emits no thunk.
    class_raii = JLCSIRGenerator._collect_class_raii(metadata)

    # Struct definitions
    # Collect all struct names from DWARF (excluding enums which have __enum__ prefix)
    struct_types = Set{String}()
    for (name, info) in dwarf_structs
        if !startswith(name, "__enum__") && haskey(info, "members")
            push!(struct_types, name)
        end
    end

    # The STL screen, asked with the record's definition: a record whose
    # `scope` is outside the standard library is never STL-internal, whatever
    # its bare name (TypesCpp._is_stl_internal_record).
    _stl_internal(n) = _is_stl_internal_record(String(n), dwarf_structs)

    # Types an earlier library of the graph owns (TypeImports.jl): the pre-pass
    # gave their entries the owner's body for every reader of layouts, and
    # nothing in this wrapper defines them. Each emission loop below asks.
    _imported(k) = !_emits_type(k, get(dwarf_structs, k, nothing))
    imported_jl_names = Set{String}(_imported_type_names(dwarf_structs, _sanitize_cpp_type_name))
    # `__init__` checks the owners' manifests when there are any.
    _imports_init = isempty(imported_jl_names) ? "" : "\n            _check_type_imports()"

    # =============================================================================
    # AUTOMATIC FINALIZER GENERATION (Memory Safety)
    # =============================================================================

    # 1. Scan for Deleters: what a finalizer that owns an `X*` calls to free it.
    #
    #   * The library's deleter function for X: no receiver (a free function or
    #     a static member), one `X*` parameter, a name that says delete, destroy
    #     or free. The library allocated the object, so its deleter knows how.
    #   * Else X's deleting destructor (D0), which is C++ `delete` for an object
    #     whose dynamic type is X.
    #
    # A class with neither has no entry, so no `Managed<X>` and no `<f>_safe`.
    # Its other destructors (D1, D2) end the object's lifetime and leave its
    # storage allocated: a finalizer built on one leaked every object, and the
    # first destructor in metadata order was D2 for every class without a
    # virtual destructor (85 Managed types across the Hub, 2026-10-26). A
    # method with a receiver is never a deleter: imgui's `ManagedImDrawList`
    # called `ImDrawList::_ClearFreeMemory()`, whose one parameter is `this`,
    # and it frees the list's buffers, not the list. A static member is one:
    # tinyxml2's `XMLNode::DeleteNode(XMLNode*)` returns a node to its pool.
    # Several candidates for one class: the smallest mangled name.
    deleters = Dict{String, String}()          # struct_name -> what it calls, demangled
    deleters_mangled = Dict{String, String}()  # struct_name -> mangled symbol
    deleting_dtors = Set{String}()             # struct_names whose deleter is their D0

    deleter_fns = Dict{String, Any}()          # struct_name -> deleter function
    d0_funcs = Dict{String, Any}()             # struct_name -> deleting destructor
    _take_smallest!(d, k, func) =
        (haskey(d, k) && d[k]["mangled"] <= func["mangled"]) || (d[k] = func)

    for func in functions
        f_name = func["name"]
        demangled = func["demangled"]
        params = func["parameters"]

        # A DESTRUCTOR names its own class, and that is authoritative — it does
        # NOT depend on DWARF having supplied the `this` parameter. This scan
        # used to infer the class from `params[1]::Ptr{X}`, so when DWARF
        # stopped describing `this` for destructors the table silently
        # collapsed and every `Managed`/finalizer built on it vanished:
        # box2d 30 Managed types → 1, tinyxml2 11 → 2, llamacpp 127 → 18,
        # measured 2026-08-13 on the first rebuild since July. Same underlying
        # fact as the receiver gate (`_cpp_this_param`): a destructor has a
        # receiver by definition, so read the class, not the argument shape.
        #
        # The key must be a member of `struct_types` — that was the old
        # invariant (it came from a `Ptr{X}` whose `X` was checked) and the
        # rest of the emitter relies on it — so try each spelling and key on
        # whichever one actually matched.
        if contains(demangled, "~")
            endswith(func["mangled"], "D0Ev") || continue
            cls = String(get(func, "class", ""))
            if !isempty(cls)
                inner = _cpp_innermost_scope(cls)
                sname = inner in struct_types ? inner :
                        cls   in struct_types ? cls   :
                        let s = _sanitize_cpp_type_name(inner)
                            s in struct_types ? s : ""
                        end
                isempty(sname) || _imported(sname) || _take_smallest!(d0_funcs, sname, func)
            end
            continue
        end

        # `_has_receiver` is the receiver gate the thunks use: a method whose
        # declaration DIE left `this` out of `parameters` still has one.
        length(params) == 1 || continue
        JLCSIRGenerator.FunctionGen._has_receiver(func, dwarf_structs) && continue
        arg_type = params[1]["julia_type"]
        (startswith(arg_type, "Ptr{") && endswith(arg_type, "}")) || continue
        struct_name = arg_type[5:end-1]
        struct_name in struct_types || continue
        _imported(struct_name) && continue
        lower_name = lowercase(f_name)
        (contains(lower_name, "delete") || contains(lower_name, "destroy") ||
         contains(lower_name, "free")) || continue
        _take_smallest!(deleter_fns, struct_name, func)
    end
    for (s, func) in deleter_fns
        deleters[s] = func["demangled"]
        deleters_mangled[s] = func["mangled"]
    end
    for (s, func) in d0_funcs
        haskey(deleters, s) && continue
        deleters[s] = func["demangled"]
        deleters_mangled[s] = func["mangled"]
        push!(deleting_dtors, s)
    end

    # =============================================================================
    # IDIOMATIC CLASS WRAPPER PREPARATION
    # =============================================================================

    # The struct_types key a type spelling names, or `nothing`. Several keys can
    # sanitize to one name: the smallest wins, never the Dict order.
    sanitized_struct_key = Dict{String,String}()
    for k in sort!(collect(struct_types))
        get!(sanitized_struct_key, _sanitize_cpp_type_name(k), k)
    end
    _struct_key(t::AbstractString) = t in struct_types ? String(t) :
        get(sanitized_struct_key, _sanitize_cpp_type_name(String(t)), nothing)

    function _base_keys(key)
        found = String[]
        for base in get(get(dwarf_structs, key, Dict{String,Any}()), "base_classes", Any[])
            base isa AbstractDict || continue
            b = _struct_key(replace(String(get(base, "type", "")), r"^(class|struct)\s+" => ""))
            b === nothing || push!(found, b)
        end
        return found
    end
    _derives_from(d, b, seen = Set{String}()) =
        !(d in seen) && (push!(seen, d); any(x -> x == b || _derives_from(x, b, seen), _base_keys(d)))

    # Classes something in this library derives from. Their deleting
    # destructor, called directly, runs only their own destructor on an object
    # of a subclass.
    has_subclass = Set{String}()
    for key in keys(dwarf_structs)
        union!(has_subclass, _base_keys(key))
    end
    # ...and in every other library of the graph (`_graph_subclassed`): a
    # dependent deriving from a class of this library breaks the direct D0 call
    # the same way.
    for n in get(metadata, "graph_subclassed", String[])
        k = _struct_key(replace(String(n), r"^(class|struct)\s+" => ""))
        k === nothing || push!(has_subclass, k)
    end

    # The struct a function's pointer return names, or `nothing`.
    function _returned_key(func)
        ret = func["return_type"]
        ret_julia = String(get(ret, "julia_type", ""))
        if startswith(ret_julia, "Ptr{") && endswith(ret_julia, "}") && ret_julia != "Ptr{Cvoid}"
            k = _struct_key(ret_julia[5:end-1])
            k === nothing || return k
        end
        ret_c = String(get(ret, "c_type", ""))
        endswith(ret_c, "*") || return nothing
        base_c = replace(strip(ret_c[1:end-1]), r"^(const\s+|struct\s+|class\s+)+" => "")
        base_c = replace(base_c, r"\s+" => "")
        return isempty(base_c) || base_c == "void" ? nothing : _struct_key(base_c)
    end

    # Collect factory functions: free functions named create_/new_/make_/alloc_/init_/build_
    # that return a pointer to a known class. The class is the one that pointer
    # names, or the class the factory's name names when that class is the
    # returned one or derives from it (`create_circle()` returning `Shape*`). A
    # name never makes a class of its own: read first, `make_rect_nx` made a
    # struct `RectNx`, and a name that named an unrelated class gave the object
    # that class's finalizer.
    factory_prefixes = ("create_", "new_", "make_", "alloc_", "init_", "build_")
    factory_funcs = Dict{String, Vector{Any}}()  # struct_types key -> [func_info, ...]
    for func in functions
        get(func, "is_method", false) && continue
        f_name = func["name"]
        i = findfirst(p -> startswith(lowercase(f_name), p), factory_prefixes)
        i === nothing && continue
        returned = _returned_key(func)
        returned === nothing && continue
        # Snake case to CamelCase: create_dense_matrix -> DenseMatrix
        suffix = f_name[length(factory_prefixes[i])+1:end]
        named = isempty(suffix) ? nothing : _struct_key(join(uppercasefirst.(split(suffix, "_"))))
        cls = named !== nothing && (named == returned || _derives_from(named, returned)) ? named : returned
        push!(get!(factory_funcs, cls, Any[]), func)
    end

    # Collect C++ constructors: is_method=true AND func_name == class_name
    class_constructors = Dict{String, Vector{Any}}()  # class_name -> [ctor_func_info, ...]
    for func in functions
        is_m   = get(func, "is_method", false)
        cls_nm = get(func, "class", "")
        is_m && !isempty(cls_nm) && func["name"] == cls_nm || continue
        safe_cls = _sanitize_cpp_type_name(cls_nm)
        push!(get!(class_constructors, safe_cls, Any[]), func)
    end

    # Collect non-constructor, non-destructor instance methods
    class_methods = Dict{String, Vector{Any}}()  # class_name -> [method_func_info, ...]
    for func in functions
        is_m   = get(func, "is_method", false)
        cls_nm = get(func, "class", "")
        is_m && !isempty(cls_nm) || continue
        func["name"] == cls_nm   && continue  # skip constructors
        contains(func["demangled"], "~") && continue  # skip destructors
        safe_cls = _sanitize_cpp_type_name(cls_nm)
        push!(get!(class_methods, safe_cls, Any[]), func)
    end

    # Classes whose only destructor in the library is the deleting one (D0):
    # an implicit virtual destructor at -O2 leaves no D1/D2 (Bullet's
    # `btBoxShape`, 2026-11-14). Their `<X>_delete_<X>` is the only destructor
    # binding, and its docstring says so; the wrap report names them.
    d0_only_classes = Set{String}()
    let variants = Dict{String,Tuple{Set{Char},String}}()
        for func in functions
            m = String(get(func, "mangled", ""))
            g = _dtor_variant_group(m)
            g === nothing && continue
            entry = get!(variants, g, (Set{Char}(), String(get(func, "class", ""))))
            push!(entry[1], m[end - 2])              # the digit of `D0Ev`/`D1Ev`/`D2Ev`
        end
        for (_, (seen, cls)) in variants
            seen == Set(['0']) && !isempty(cls) && push!(d0_only_classes, cls)
        end
    end
    if !isempty(d0_only_classes)
        names = sort!(collect(d0_only_classes))
        println("  wrap: $(length(names)) class(es) whose only destructor is the deleting one (D0), ",
                "bound as <X>_delete_<X>, which frees the object: ", join(first(names, 8), ", "),
                length(names) > 8 ? ", …" : "")
    end
    _report_inline_constructors(get(metadata, "inline_constructors", nothing))
    _report_inline_constructors(get(metadata, "inline_methods", nothing); what = "member function")
    _report_vtable_methods(get(metadata, "vtable_methods", nothing))
    let bound = [r for r in get(get(metadata, "default_arguments", Dict{String,Any}()), "bound", Any[])
                 if r isa AbstractDict && get(r, "emitted", false) == true]
        isempty(bound) || println("  wrap: $(length(bound)) call(s) with default arguments left out, of ",
                                  "$(length(unique(String[String(r["symbol"]) for r in bound]))) function(s), bound under the functions' own names")
    end

    # Helper: sanitize a raw function name to the Julia identifier used in wrappers.
    # Proxies and deleters REFERENCE names the main loop defines, so this must be
    # the main loop's derivation, not a shorter copy of its list.
    # With the symbol, so a `[wrap.rename]` keyed on it reaches the reference.
    _sanitize_julia_fn(name::String, mangled::String = "")::String = _julia_function_name(name, mangled)

    # Determine which classes get full idiomatic wrappers: ≥1 factory function,
    # and a finalizer that frees what the factories return —
    #   1. the library's deleter function for the class, or for the type a
    #      factory returns (`create_circle()` returning `Shape*`, and
    #      `delete_shape(Shape*)`);
    #   2. else the class's deleting destructor, when nothing in the library
    #      derives from the class: it is called directly, so on an object of a
    #      subclass it would run only the class's own destructor.
    # The finalizer calls it by its mangled symbol. Called by Julia name it was
    # `<X>_destroy_<X>`, which D0, D1 and D2 share; dedup kept D2, and every
    # collected object leaked (2026-10-26). A class with neither keeps its raw
    # bindings, as do pure C++ constructor-only classes (no factory): they need
    # operator new and placement new.
    idiomatic_classes = Dict{String, Dict{String,Any}}()  # Julia struct name -> config dict

    for key in sort!(collect(keys(factory_funcs)))
        factories = factory_funcs[key]
        _imported(key) && continue          # the owner's wrapper defines the class
        cls_nm = _sanitize_cpp_type_name(key)
        haskey(idiomatic_classes, cls_nm) && continue
        owner = nothing
        for k in (key, (_returned_key(f) for f in factories)...)
            if haskey(deleters, k) && !(k in deleting_dtors)
                owner = k
                break
            end
        end
        if owner === nothing && key in deleting_dtors && !(key in has_subclass)
            owner = key
        end
        owner === nothing && continue

        idiomatic_classes[cls_nm] = Dict{String,Any}(
            "deleter"      => deleters_mangled[owner],
            "deleter_doc"  => owner in deleting_dtors ?
                "calls `$(deleters[owner])`, the deleting destructor (C++ `delete`)" :
                "calls `$(deleters[owner])`",
            "factories"    => factories,
            "ctors"        => get(class_constructors, cls_nm, Any[]),
        )
    end


    # =============================================================================
    # ENUM GENERATION (from DWARF)
    # =============================================================================

    # The enum section is held back as `_PendingEnum` records and rendered last
    # (`_render_enum_chunks`), once every other top-level name of the wrapper is
    # known: an enumerator that would rebind a record, a function, a global or
    # another enum's member is prefixed with its enum's name there. The text
    # still lands first in the file. Strings in this list are emitted verbatim.
    enum_chunks = _EnumChunk[]
    # Julia (sizeof, alignment) of each DWARF enum as emitted, by emitted name:
    # what a struct field of that enum type is. An alias has its canonical's.
    emitted_enum_layout = Dict{String,Tuple{Int,Int}}()
    # The `@enum X::T` base type `T` of each emitted enum, by emitted name: what
    # a Tier 2 body converts an enum argument to (`_tier2_invoke_args`).
    emitted_enum_base = Dict{String,String}()

    # Extract enums (stored with __enum__ prefix)
    enum_types = filter(k -> startswith(k, "__enum__"), keys(dwarf_structs))

    # Build set of enum names (without prefix) to exclude from struct generation
    enum_names = Set{String}()
    for enum_key in enum_types
        enum_name = replace(enum_key, "__enum__" => "")
        push!(enum_names, enum_name)
    end
    # Raw DWARF names of every record and enum, for `_record_name_as_emitted`.
    emitted_record_names = union(Set{String}(String(k) for k in keys(dwarf_structs)
                                             if !startswith(String(k), "__enum__")),
                                 enum_names)

    if !isempty(enum_types)
        push!(enum_chunks, """
        # =============================================================================
        # Enum Definitions (from DWARF debug info)
        # =============================================================================

        """)

        # Across-enum dedup. Within one @enum the loop below already drops
        # repeated names and values; nothing was watching for the SAME member
        # set arriving twice under two type names.
        #
        # DWARF records a std::map/unordered_map's member typedefs — `key_type`,
        # `value_type`, `mapped_type` — as separate enum types carrying the
        # underlying enum's enumerators verbatim. Emitting both means the later
        # `@enum` rebinds every member name, and Julia permits that silently:
        # after onednn's module loaded, `dnnl_f32` was a `key_type` rather than a
        # `dnnl_data_type_t`, and every ccall expecting the latter raised
        # `MethodError: Cannot convert an object of type key_type`. The module
        # parses, loads, and passes a smoke test that never passes a data type —
        # onednn's 15/15 did exactly that.
        #
        # Identical member sets mean it IS the same enum under another spelling,
        # so alias it. That keeps `key_type` resolvable for anything that
        # references the typedef, without touching the member bindings.
        emitted_enum_sigs = Dict{Vector{Pair{String,Any}},String}()

        for enum_key in sort(collect(enum_types))
            # Sanitize so the emitted `@enum` name matches every field/param
            # reference (those already route through _sanitize_cpp_type_name) and
            # never collides with a Base binding like `Type`.
            enum_name = _sanitize_cpp_type_name(replace(enum_key, "__enum__" => ""))
            enum_info = dwarf_structs[enum_key]

            # Get enum metadata
            underlying_type = get(enum_info, "underlying_type", "int")
            julia_underlying = get(enum_info, "julia_type", "Int32")

            # Fallback to Int32 if type is unknown or Any
            if julia_underlying == "Any" || julia_underlying == "unknown"
                julia_underlying = "Int32"
            end

            enumerators = get(enum_info, "enumerators", [])

            if _imported(enum_key)
                # The owner's module defines it; a field or Tier 2 argument of
                # this type still needs its width here.
                emitted_enum_base[enum_name] = julia_underlying
                haskey(_C_PRIM_FIELD_LAYOUT, julia_underlying) &&
                    (emitted_enum_layout[enum_name] = _C_PRIM_FIELD_LAYOUT[julia_underlying])
                continue
            end

            if !isempty(enumerators)
                # Sanitize first, then choose the shape. Mirrors GeneratorC.jl's
                # DWARF enum site exactly — a C++ `enum` used as a mask is the same
                # problem, and the two generators diverging on one decision is the
                # failure this codebase keeps re-learning. These are the BASE
                # names: `_render_enum_chunks` applies the collision prefix and
                # the keyword escape (`_enumerator_julia_name`).
                members = Pair{String,Any}[]
                seen_names = Set{String}()
                for enumerator in enumerators
                    name = _sanitize_cpp_type_name(get(enumerator, "name", "Unknown"))
                    name in seen_names && continue
                    push!(seen_names, name)
                    push!(members, name => _enumerator_value(get(enumerator, "value", 0), julia_underlying))
                end

                # Same member set already emitted under another name? Alias it
                # rather than re-binding every member. Sorting makes the
                # signature order-independent; `sort(collect(enum_types))` puts
                # `dnnl_data_type_t` ahead of `key_type` here, but relying on
                # alphabetical luck for which name wins is not a plan, so the
                # first emitted name is simply the canonical one.
                enum_sig = sort(members, by=first)
                if haskey(emitted_enum_sigs, enum_sig)
                    canonical = emitted_enum_sigs[enum_sig]
                    haskey(emitted_enum_layout, canonical) &&
                        (emitted_enum_layout[enum_name] = emitted_enum_layout[canonical])
                    haskey(emitted_enum_base, canonical) &&
                        (emitted_enum_base[enum_name] = emitted_enum_base[canonical])
                    if canonical != enum_name
                        push!(enum_chunks, _PendingEnumAlias(enum_name, canonical, """
                        # $enum_name is $canonical under a different DWARF spelling
                        # (a std::map member typedef carries its key/value enum's
                        # enumerators). Aliased rather than re-emitted: a second
                        # `@enum` would rebind every member name to this type and
                        # break ccalls expecting $canonical.
                        const $enum_name = $canonical

                        """))
                    end
                    continue
                end
                emitted_enum_sigs[enum_sig] = enum_name
                emitted_enum_base[enum_name] = julia_underlying
                # `@enum X::T` and the bitflag alias `const X = T` both have T's layout.
                haskey(_C_PRIM_FIELD_LAYOUT, julia_underlying) &&
                    (emitted_enum_layout[enum_name] = _C_PRIM_FIELD_LAYOUT[julia_underlying])

                push!(enum_chunks, _PendingEnum(enum_name,
                    "C++ enum: $enum_name (underlying type: $underlying_type)",
                    julia_underlying, members))
            else
                # Defined with no enumerators (`enum class byte : unsigned char
                # {};`, `std::byte`): an integer type of its own. Never an
                # alias of another empty enum, which the member-set dedup
                # above would make it.
                emitted_enum_base[enum_name] = julia_underlying
                haskey(_C_PRIM_FIELD_LAYOUT, julia_underlying) &&
                    (emitted_enum_layout[enum_name] = _C_PRIM_FIELD_LAYOUT[julia_underlying])
                push!(enum_chunks, _PendingEnum(enum_name,
                    "C++ enum: $enum_name (underlying type: $underlying_type, no enumerators)",
                    julia_underlying, Pair{String,Any}[]))
            end
        end

        push!(enum_chunks, "\n")
    end

    # =============================================================================
    # ENUM GENERATION (from Headers - supplementary)
    # =============================================================================
    # Add enums found in headers that weren't in DWARF (unused/optimized away)
    header_enums = get(metadata, "header_enums", Dict())
    if !isempty(header_enums)
        added_header_enums = 0
        # Sorted: emission order must not follow the Dict's (see the DWARF enums).
        for enum_name in sort!(collect(keys(header_enums)); by = string)
            members = header_enums[enum_name]
            # Skip if already defined from DWARF (match on the raw name)
            if enum_name in enum_names
                continue
            end
            # Sanitized form used for emission so it can't shadow a Base binding.
            enum_name_jl = _sanitize_cpp_type_name(string(enum_name))

            if !isempty(members)
                if added_header_enums == 0
                    push!(enum_chunks, """
                    # =============================================================================
                    # Enum Definitions (from Headers - supplementary)
                    # =============================================================================
                    # These enums were not in DWARF (unused code eliminated by compiler)

                    """)
                end

                # Choose underlying type based on value range
                min_val = minimum(v for (_, v) in members)
                max_val = maximum(v for (_, v) in members)
                underlying = if min_val >= 0 && max_val > typemax(Int32)
                    "UInt32"
                elseif min_val < typemin(Int32) || max_val > typemax(Int32)
                    "Int64"
                else
                    "Cint"
                end

                # Sanitize first, then choose the shape — see the DWARF enum site
                # above. `emitted`, not `members`: the loop variable is taken.
                # Base names; the renderer prefixes and escapes.
                emitted = Pair{String,Any}[]
                seen_names = Set{String}()
                for (member_name, value) in members
                    n = _sanitize_cpp_type_name(string(member_name))
                    n in seen_names && continue
                    push!(seen_names, n)
                    push!(emitted, n => value)
                end

                push!(enum_chunks, _PendingEnum(enum_name_jl,
                    "C++ enum: $enum_name (from header - not in DWARF)",
                    underlying, emitted))
                added_header_enums += 1
            end
        end

        if added_header_enums > 0
            push!(enum_chunks, "\n")
        end
    end

    # =============================================================================
    # STRUCT GENERATION (from DWARF)
    # =============================================================================

    # Create a mapping from sanitized Julia name back to raw C++ name for resolving dependencies
    julia_to_cpp_struct = Dict{String, String}()
    # Sorted, so of two names that sanitize alike the same one is kept every time.
    for struct_name in sort!(collect(struct_types))
        julia_to_cpp_struct[_sanitize_cpp_type_name(struct_name)] = struct_name
    end

    # Scan for ALL referenced struct types that need forward declarations.
    # This includes:
    # 1. Types referenced via Ptr{T} that have no DWARF definition (truly opaque)
    # 2. Types referenced via Ptr{T} that DO have definitions but may appear later
    #    in topological order (forward reference problem)
    # We use `mutable struct Foo end` for truly opaque types, and for defined types
    # we don't forward-declare (they'll be defined in topo order).
    opaque_structs = Set{String}()
    ptr_referenced_structs = Set{String}()  # Struct types referenced via Ptr{X}

    for (name, info) in dwarf_structs
        if !startswith(name, "__enum__") && haskey(info, "members")
            members = get(info, "members", [])
            for member in members
                julia_type = get(member, "julia_type", "Any")
                # Extract inner type from Ptr{T} (recursively for Ptr{Ptr{T}})
                inner = julia_type
                found_ptr = false
                while startswith(inner, "Ptr{") && endswith(inner, "}")
                    inner = inner[5:end-1]
                    found_ptr = true
                end

                # Scan for ALL struct type references in this julia_type
                # Handles: Ptr{Foo}, NTuple{N, Ptr{Foo}}, Ptr{Ptr{Foo}}, Ref{Foo}, etc.
                builtin_types = Set(["Cvoid", "Cint", "Cuint", "Cintptr_t", "Cuintptr_t", "Clong", "Culong", "Cshort", "Cushort",
                                     "Cchar", "Cuchar", "Cfloat", "Cdouble", "Bool", "UInt8", "Int8",
                                     "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64", "Csize_t",
                                     "Clonglong", "Culonglong", "Cptrdiff_t", "Cssize_t", "Cwchar_t",
                                     "Cstring", "Float32", "Float64", "ComplexF32", "ComplexF64",
                                     "Int128", "UInt128", "Any", "Nothing"])

                # Extract the base type by stripping known container prefixes
                base_ref = julia_type
                while true
                    if startswith(base_ref, "Ptr{") && endswith(base_ref, "}")
                        base_ref = base_ref[5:end-1]
                    elseif startswith(base_ref, "Ref{") && endswith(base_ref, "}")
                        base_ref = base_ref[5:end-1]
                    elseif startswith(base_ref, "NTuple{")
                        inner = peel_container_arg(base_ref)
                        if inner === nothing
                            break
                        end
                        base_ref = inner
                    else
                        break
                    end
                end
                
                base_ref = strip(base_ref)
                
                # If the inner type is not a builtin, register it
                if !(base_ref in builtin_types) && !isempty(base_ref)
                    # We compare the raw string with struct_types
                    if base_ref in struct_types
                        push!(ptr_referenced_structs, base_ref)
                    else
                        push!(opaque_structs, base_ref)
                    end
                end
            end
        end
    end

    # Also scan function parameter and return types for struct references.
    # Types like sqlite3_blob only appear in function signatures, not struct members.
    _builtin_types = Set(["Cvoid", "Cint", "Cuint", "Cintptr_t", "Cuintptr_t", "Clong", "Culong", "Cshort", "Cushort",
                          "Cchar", "Cuchar", "Cfloat", "Cdouble", "Bool", "UInt8", "Int8",
                          "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64", "Csize_t",
                          "Clonglong", "Culonglong", "Cptrdiff_t", "Cssize_t", "Cwchar_t",
                          "Cstring", "Float32", "Float64", "ComplexF32", "ComplexF64",
                                     "Int128", "UInt128", "Any", "Nothing"])
    for func in functions
        all_types = String[]
        for param in get(func, "parameters", [])
            push!(all_types, get(param, "julia_type", "Any"))
        end
        ret = get(func, "return_type", nothing)
        if !isnothing(ret)
            push!(all_types, get(ret, "julia_type", "Cvoid"))
        end
        for julia_type in all_types
            base_ref = julia_type
            while true
                if startswith(base_ref, "Ptr{") && endswith(base_ref, "}")
                    base_ref = base_ref[5:end-1]
                elseif startswith(base_ref, "Ref{") && endswith(base_ref, "}")
                    base_ref = base_ref[5:end-1]
                elseif startswith(base_ref, "NTuple{")
                    inner = peel_container_arg(base_ref)
                    if inner === nothing
                        break
                    end
                    base_ref = inner
                else
                    break
                end
            end
            base_ref = strip(base_ref)
            if !(base_ref in _builtin_types) && !isempty(base_ref)
                if base_ref in struct_types
                    push!(ptr_referenced_structs, base_ref)
                else
                    push!(opaque_structs, base_ref)
                end
            end
        end
    end

    struct_chunks = String[]
    union_accessor_defs = ""  # Deferred union accessors (emitted after all struct defs)
    blob_setters_emitted = false  # gates the one module-level `setproperties`
    field_setters_emitted = false  # gates the helper the field-struct setters call
    # Seeded inside the struct-emission block. Must exist even when DWARF
    # reports zero types (hello_world): function signatures still gate
    # parameter annotations on this set.
    defined_struct_names = Set{String}()

    # Emit forward declarations for:
    # 1. Truly opaque types (no DWARF definition) — as mutable struct
    # 2. Struct types referenced via Ptr{X} — as empty struct (will be redefined with fields later)
    # This ensures Ptr{X} fields can reference types defined later in the file.
    # BUT: skip forward declarations for types that will get a real DWARF definition,
    # since Julia doesn't allow redefining a struct with the same name.
    all_forward_decls = union(opaque_structs, ptr_referenced_structs)

    # Build set of sanitized names that will get real DWARF definitions
    dwarf_defined_names = Set{String}()
    for (name, info) in dwarf_structs
        _stl_internal(name) && continue
        s = _sanitize_cpp_type_name(name)
        s = replace(s, "*" => "Ptr")
        s = replace(s, "&" => "Ref")
        s = replace(s, r"[^a-zA-Z0-9_]" => "_")
        push!(dwarf_defined_names, s)
    end
    # Also exclude enum names — they're already defined via @enum, under the
    # name `_sanitize_cpp_type_name` gives them. The raw name missed every enum
    # it changes (`Type` is emitted as `c_Type`): box2d got `struct c_Type end`
    # after `@enum c_Type`, and Julia 1.12 lets that replace the enum, so every
    # field and signature naming it below meant an empty struct.
    union!(dwarf_defined_names, enum_names)
    union!(dwarf_defined_names, (_sanitize_cpp_type_name(e) for e in enum_names))

    if !isempty(all_forward_decls)
        push!(struct_chunks, """
        # =============================================================================
        # Forward Declarations (Opaque + Ptr-referenced types)
        # =============================================================================

        """)
        seen_forward_decls = Set{String}()
        for name in sort(collect(all_forward_decls))
            # Skip STL internal types
            _stl_internal(name) && continue
            # Skip blocklisted internal/system types
            name in _INTERNAL_TYPE_BLOCKLIST && continue

            # Sanitize
            s_name = _sanitize_cpp_type_name(name)
            s_name = replace(s_name, "*" => "Ptr")
            s_name = replace(s_name, "&" => "Ref")
            s_name = replace(s_name, r"[^a-zA-Z0-9_]" => "_")

            # Skip names that are Julia keywords or builtins
            if s_name in ("char", "int", "void", "bool", "float", "double", "long", "short")
                continue
            end

            # Skip duplicates (different C++ names can sanitize to the same Julia identifier)
            s_name in seen_forward_decls && continue
            push!(seen_forward_decls, s_name)

            # Skip forward declaration if this type will get a real DWARF definition later
            s_name in dwarf_defined_names && continue

            # Both opaque types and pointer-referenced types should be immutable structs
            # to preserve inline C++ ABI layout (0-byte empty structs) if they are used by value.
            push!(struct_chunks, "struct $s_name end\n")
        end
        push!(struct_chunks, "\n")
    end

    # A factory class's Julia type (the idiomatic section) is defined after
    # every record, but a record's field can name it: `struct Cage { Animal
    # *pet; }` gives `pet::Ptr{Animal}`, which Julia resolves when `Cage` is
    # defined. The types of the classes a field names go in this slot, ahead of
    # every record; their constructors stay in the idiomatic section
    # (`_idiomatic_early_type`). Filled there, from `idiomatic_named_early`,
    # which the field emitter fills with what it wrote. Empty otherwise, so a
    # wrapper no field of which names a factory class is unchanged.
    idiomatic_early_slot = 0
    idiomatic_named_early = Set{String}()

    if !isempty(struct_types)
        push!(struct_chunks, "")
        idiomatic_early_slot = length(struct_chunks)
        push!(struct_chunks, """
        # =============================================================================
        # Struct Definitions (from DWARF debug info)
        # =============================================================================

        """)

        # Topologically sort structs by dependencies
        # Build dependency graph: struct_name => [dependencies]
        struct_deps = Dict{String, Set{String}}()
        struct_soft_deps = Dict{String, Set{String}}()

        for struct_name in struct_types
            deps = Set{String}()
            soft_deps = Set{String}()

            if haskey(dwarf_structs, struct_name)
                struct_info = dwarf_structs[struct_name]
                members = get(struct_info, "members", [])

                for member in members
                    julia_type = get(member, "julia_type", "Any")

                    # Extract type name from various wrapper types:
                    #   Ptr{Foo} -> Foo (soft dep)
                    #   NTuple{N, Foo} -> Foo (hard dep, needs full definition first)
                    #   Ref{Foo} -> Foo (hard dep)
                    #   Foo -> Foo (hard dep, direct by-value embedding)
                    dep_type = nothing
                    is_soft = false

                    if startswith(julia_type, "Ptr{")
                        inner = peel_container_arg(julia_type)
                        if inner !== nothing
                            dep_type = unwrap_foreign_type(inner)
                            is_soft = true
                        end
                    elseif startswith(julia_type, "NTuple{")
                        inner = peel_container_arg(julia_type)
                        if inner !== nothing
                            # `NTuple{N, Ptr{Later}}` does not need Later's layout (a
                            # pointer is pointer-wide) but Julia still evaluates
                            # the name while defining the struct. Order Later
                            # first. A cycle is rewritten to `Ptr{Cvoid}` by
                            # `_resolve_forward_ptr`.
                            if startswith(inner, "Ptr{")
                                dep_type = unwrap_foreign_type(inner)
                            else
                                dep_type = unwrap_foreign_type(inner)
                            end
                        end
                    elseif startswith(julia_type, "Ref{")
                        inner = peel_container_arg(julia_type)
                        if inner !== nothing
                            dep_type = unwrap_foreign_type(inner)
                        end
                    else
                        dep_type = julia_type
                    end

                    if !isnothing(dep_type)
                        # Look up by raw name first, then by sanitized name
                        # (template types in Ref{...}/Ptr{...} use raw C++ names
                        #  but the dictionary is keyed by sanitized Julia names)
                        cpp_dep = if haskey(julia_to_cpp_struct, dep_type)
                            julia_to_cpp_struct[dep_type]
                        else
                            sanitized = _sanitize_cpp_type_name(dep_type)
                            haskey(julia_to_cpp_struct, sanitized) ? julia_to_cpp_struct[sanitized] : nothing
                        end
                        if !isnothing(cpp_dep) && cpp_dep != struct_name
                            if is_soft
                                push!(soft_deps, cpp_dep)
                            else
                                push!(deps, cpp_dep)
                            end
                        end
                    end
                end
                # A non-virtual base's members are this record's fields too
                # (`flatten_struct_members`), so the base is a hard dependency:
                # emitted first, its own dependencies before it. A record with
                # no member of its own had no dependency at all and sorted
                # first, so its inherited `btVector3` fields came out as bytes
                # and its `btConvexPolyhedron*` as `Ptr{Cvoid}` (2026-11-14).
                for base in get(struct_info, "base_classes", [])
                    base isa AbstractDict || continue
                    get(base, "virtual", false) == true && continue   # never flattened
                    bt = replace(String(get(base, "type", "")), r"^(class|struct)\s+" => "")
                    isempty(bt) && continue
                    cpp_dep = haskey(dwarf_structs, bt) ? bt :
                              get(julia_to_cpp_struct, _sanitize_cpp_type_name(bt), nothing)
                    (cpp_dep === nothing || cpp_dep == struct_name) || push!(deps, cpp_dep)
                end
            end

            struct_deps[struct_name] = deps
            struct_soft_deps[struct_name] = soft_deps
        end

        # Topological sort using Kahn's algorithm
        sorted_structs = String[]
        remaining_hard = Dict(k => copy(v) for (k, v) in struct_deps)
        remaining_soft = Dict(k => copy(v) for (k, v) in struct_soft_deps)

        while !isempty(remaining_hard)
            # Find structs with no hard AND no soft dependencies
            ready = [name for (name, deps) in remaining_hard if isempty(deps) && isempty(remaining_soft[name])]

            if isempty(ready)
                # Break soft dependency cycle: find structs with no hard dependencies
                ready = [name for (name, deps) in remaining_hard if isempty(deps)]
            end

            if isempty(ready)
                # Circular hard dependency - just take alphabetically first
                ready = [sort(collect(keys(remaining_hard)))[1]]
            end

            for name in sort(ready)
                push!(sorted_structs, name)
                delete!(remaining_hard, name)
                delete!(remaining_soft, name)

                # Remove this struct from all dependency lists
                for deps in values(remaining_hard)
                    delete!(deps, name)
                end
                for deps in values(remaining_soft)
                    delete!(deps, name)
                end
            end
        end

        # Helper to recursively flatten struct members from base classes.
        # `base_off` is the byte offset of the current subobject in the
        # OUTERMOST derived object: base members' DWARF offsets are relative
        # to their own class, so under multiple inheritance a secondary base's
        # members must be rebased by the base subobject offset (from
        # DW_AT_data_member_location on the inheritance edge) or they collide
        # with the primary base's layout. Single inheritance rebases by 0 —
        # byte-identical to the old behavior.
        function flatten_struct_members(s_name, visited=Set{String}(); base_off::Int=0)
            if s_name in visited
                return []
            end
            push!(visited, s_name)

            all_members = []
            seen_names = Set{String}()
            # Two bases may declare same-named members (or a derived member
            # may shadow a base's). Both exist in the layout, but a Julia
            # struct can't have duplicate field names — rename later
            # duplicates with the class they came from.
            function push_dedup!(m, origin_tag)
                nm = string(get(m, "name", "unknown"))
                if nm in seen_names
                    m = copy(m)
                    renamed = "$(origin_tag)_$(nm)"
                    while renamed in seen_names
                        renamed *= "_"
                    end
                    m["name"] = renamed
                    nm = renamed
                end
                push!(seen_names, nm)
                push!(all_members, m)
            end

            if haskey(dwarf_structs, s_name)
                info = dwarf_structs[s_name]

                # First, add members from base classes (layout order — the
                # extractor sorts bases by subobject offset)
                bases = get(info, "base_classes", [])
                for base in bases
                    base_type = get(base, "type", "")
                    # Strip "class " or "struct " prefix if present
                    base_type = replace(base_type, r"^(class|struct)\s+" => "")
                    isempty(base_type) && continue

                    # Virtual base: its offset is vtable-resident, not static —
                    # flattening it at a guessed offset would corrupt every
                    # access. By design its members are reached through the
                    # dynamic `<Derived>_as_<Base>` upcast helper instead.
                    if get(base, "virtual", false) == true
                        @debug "$(s_name): virtual base $(base_type) not flattened (offset is dynamic; use the $(s_name)_as_* upcast helper)"
                        continue
                    end

                    sub_off = get(base, "offset", 0)
                    sub_off = sub_off isa Integer ? Int(sub_off) : 0
                    base_tag = replace(base_type, r"[^A-Za-z0-9_]" => "_")
                    for m in flatten_struct_members(base_type, visited; base_off = base_off + sub_off)
                        push_dedup!(m, base_tag)
                    end
                end

                # Then add own members, rebased into the outermost object
                own_members = get(info, "members", [])
                own_tag = replace(s_name, r"[^A-Za-z0-9_]" => "_")
                for m in own_members
                    if base_off != 0
                        m = copy(m)
                        off_str = get(m, "offset", "0x0")
                        off_val = isnothing(off_str) ? 0 : parse(Int, off_str)
                        m["offset"] = "0x" * string(off_val + base_off, base=16)
                        # DWARF4+ bitfields carry an absolute bit offset from
                        # the struct start — rebase it too
                        if haskey(m, "data_bit_offset")
                            m["data_bit_offset"] = m["data_bit_offset"] + base_off * 8
                        end
                    end
                    push_dedup!(m, own_tag)
                end
            end
            return all_members
        end

        # Pre-merge DWARF keys that sanitize to the same Julia name.
        # Multiple C++ template nesting depths (e.g., "Matrix<double,-1,-1> >" vs
        # "Matrix<double,-1,-1> > >") sanitize to the same identifier.
        # Keep the one with the largest byte_size and most members.
        best_dwarf_key = Dict{String, String}()  # sanitized_name -> best struct_name
        for struct_name in sorted_structs
            struct_name in enum_names && continue
            _stl_internal(struct_name) && continue
            
            jname = _sanitize_cpp_type_name(struct_name)
            jname = replace(jname, "*" => "Ptr")
            jname = replace(jname, "&" => "Ref")
            jname = replace(jname, r"[^a-zA-Z0-9_]" => "_")
            
            if !haskey(best_dwarf_key, jname)
                best_dwarf_key[jname] = struct_name
            else
                old_key = best_dwarf_key[jname]
                if haskey(dwarf_structs, struct_name) && haskey(dwarf_structs, old_key)
                    old_info = dwarf_structs[old_key]
                    new_info = dwarf_structs[struct_name]
                    old_bs = try; s = get(old_info, "byte_size", "0"); startswith(string(s), "0x") ? parse(Int, string(s)[3:end], base=16) : parse(Int, string(s)); catch; 0; end
                    new_bs = try; s = get(new_info, "byte_size", "0"); startswith(string(s), "0x") ? parse(Int, string(s)[3:end], base=16) : parse(Int, string(s)); catch; 0; end
                    old_mc = length(get(old_info, "members", []))
                    new_mc = length(get(new_info, "members", []))
                    if new_bs > old_bs || (new_bs == old_bs && new_mc > old_mc)
                        best_dwarf_key[jname] = struct_name
                    end
                elseif haskey(dwarf_structs, struct_name) && !haskey(dwarf_structs, old_key)
                    best_dwarf_key[jname] = struct_name
                end
            end
        end

        seen_struct_defs = Set{String}()
        defined_struct_names = Set{String}()
        if @isdefined(seen_forward_decls)
            # Only seed with forward decls that were actually emitted (not DWARF-defined skips)
            for s in seen_forward_decls
                if !(s in dwarf_defined_names)
                    push!(defined_struct_names, s)
                end
            end
        end
        # A factory class is defined wherever a field names it (the slot above),
        # so a record sorted ahead of its DWARF entry (a pointer cycle broken in
        # the record's favour) keeps `Ptr{Class}` too, instead of `Ptr{Cvoid}`.
        union!(defined_struct_names, keys(idiomatic_classes))
        blob_struct_names = Set{String}()  # Track which structs became byte-blobs
        # Field-emitted structs with bytes over a float or double (see the
        # dispatch decision), and what `_type_has_sse` resolves names with.
        sse_bytes_struct_names = Set{String}()
        _record_key = function (name)
            haskey(dwarf_structs, name) && !startswith(name, "__enum__") && return name
            jn = replace(replace(replace(_sanitize_cpp_type_name(name), "*" => "Ptr"), "&" => "Ref"),
                         r"[^a-zA-Z0-9_]" => "_")
            k = get(best_dwarf_key, jn, nothing)
            return k !== nothing && haskey(dwarf_structs, k) ? k : nothing
        end
        _enum_jl_names = Set{String}(_sanitize_cpp_type_name(e) for e in enum_names)
        _is_enum_name = name -> name in enum_names || _sanitize_cpp_type_name(name) in _enum_jl_names
        # Julia (sizeof, alignment) of every type emitted so far, as a FIELD, for
        # `_prove_julia_layout`: proven-exact structs, blobs, anonymous unions and
        # enums. Named unions are `mutable` and embed as a pointer.
        # Enums as emitted: an alias has its canonical's layout, not its own
        # DWARF one.
        jl_layouts = Dict{String,Tuple{Int,Int}}(emitted_enum_layout)
        jl_mutable = Set{String}()
        # Every record emitted as a `mutable struct` of its C bytes (a named
        # union, a bitfield record), with its size. An accessor must build one
        # from bytes: `unsafe_load(Ptr{T})` reads C's bytes as an object
        # reference (abi_fuzz X5).
        jl_byte_records = Dict{String,Int}()
        for struct_name in sorted_structs
            # Skip if this is actually an enum (enums are generated separately)
            if struct_name in enum_names
                continue
            end

            # Skip STL internal implementation types (they leak from DWARF when
            # template instantiation is forced, but are not useful to Julia users)
            if _stl_internal(struct_name)
                continue
            end

            # Skip compiler/libc internal types
            if struct_name in _INTERNAL_TYPE_BLOCKLIST
                continue
            end

            # Sanitize struct name for Julia (replace < > , with _)
            julia_struct_name = _sanitize_cpp_type_name(struct_name)
            julia_struct_name = replace(julia_struct_name, "*" => "Ptr")
            julia_struct_name = replace(julia_struct_name, "&" => "Ref")
            julia_struct_name = replace(julia_struct_name, r"[^a-zA-Z0-9_]" => "_")

            # Skip duplicate sanitized names — only process the "best" DWARF key
            # (the one with largest byte_size / most members)
            if julia_struct_name in seen_struct_defs
                continue
            end
            if haskey(best_dwarf_key, julia_struct_name) && best_dwarf_key[julia_struct_name] != struct_name
                continue
            end
            push!(seen_struct_defs, julia_struct_name)
            push!(defined_struct_names, julia_struct_name)

            # Imported: one `const` naming the owner's type is its whole definition
            # here. What a FIELD of this type measures as comes from the owner's
            # manifest, so a record embedding it still proves its layout, and a
            # blob stays a blob for the dispatch decision.
            if _imported(struct_name)
                imp = _import_of(dwarf_structs[struct_name])
                lay = get(imp, "layout", nothing)
                kind = String(get(imp, "kind", "struct"))
                if kind in ("mutable", "idiomatic")
                    push!(jl_mutable, julia_struct_name)
                    kind == "mutable" && (jl_byte_records[julia_struct_name] = _parse_dwarf_size(dwarf_structs[struct_name]))
                elseif lay isa AbstractVector && length(lay) == 2
                    jl_layouts[julia_struct_name] = (Int(lay[1]), Int(lay[2]))
                end
                kind == "blob" && push!(blob_struct_names, julia_struct_name)
                get(imp, "class_risk", false) == true && push!(sse_bytes_struct_names, julia_struct_name)
                continue
            end

            # Skip DWARF struct generation if we are generating a high-level idiomatic wrapper
            if haskey(idiomatic_classes, julia_struct_name)
                continue
            end

            # Check if we have DWARF member information for this struct
            if haskey(dwarf_structs, struct_name)
                struct_info = dwarf_structs[struct_name]
                kind = get(struct_info, "kind", "struct")
                
                # SPECIAL HANDLING FOR UNIONS
                #
                # A `mutable struct` is a reference when it is a field, so a
                # union member embedded that way is a pointer: `{union; double}`
                # becomes 24 bytes instead of 16 and the double is read back as
                # 0. Anonymous unions exist only to be embedded. Emit them as an
                # immutable byte region of the DWARF size, which Julia inlines.
                # Named unions stay mutable: they are passed by value, and the
                # accessors need `pointer_from_objref`.
                if kind == "union"
                    byte_size_str = get(struct_info, "byte_size", "0x0")
                    byte_size = parse(Int, byte_size_str)
                    members = get(struct_info, "members", [])

                    if byte_size == 0
                        # Fallback if size missing
                        for m in members
                            m_size = get(m, "size", 0)
                            byte_size = max(byte_size, m_size)
                        end
                        if byte_size == 0; byte_size = 8; end # Panic fallback
                    end

                    # Anonymous → immutable, so a parent struct inlines the bytes.
                    # Named → mutable, so `pointer_from_objref` on a by-value
                    # union still works.
                    anon_union = get(struct_info, "anonymous", false) == true
                    union_head = anon_union ? "struct" : "mutable struct"
                    # As a FIELD: the anonymous one is its bytes, the named one a reference.
                    anon_union ? (jl_layouts[julia_struct_name] = (byte_size, 1)) :
                                 push!(jl_mutable, julia_struct_name)
                    anon_union || (jl_byte_records[julia_struct_name] = byte_size)

                    push!(struct_chunks, """
                    # C union: $struct_name (size $byte_size bytes)
                    $union_head $julia_struct_name
                        data::NTuple{$byte_size, UInt8}
                    end
                    $julia_struct_name() = $julia_struct_name(ntuple(i -> 0x00, $byte_size))

                    """)

                    # Every struct name PRESENT IN DWARF — this is the generator's
                    # intent, not its output. The two disagree constantly: names are
                    # dropped above as STL-internal (~894), blocklisted (~899), or
                    # as duplicate sanitized spellings (~911/914), and none of those
                    # are ever emitted.
                    #
                    # Screening the union accessors below against THIS set is what
                    # let llama.cpp emit `get__M_pos(...)::_normal_iterator_...` for
                    # an STL iterator that no chunk declares — an eager annotation,
                    # so the module died at load. Use `defined_struct_names` (filled
                    # as structs are actually emitted) for the emit/skip decision;
                    # this set is kept only for the reference lookups.
                    # Same lesson as the 2026-08-02 ccall fix: gate on what was
                    # emitted, never on what was intended.
                    known_struct_names = Set{String}()
                    for sn in keys(dwarf_structs)
                        push!(known_struct_names, _sanitize_cpp_type_name(sn))
                    end

                    # Julia primitive types that are always available
                    julia_builtins = Set(["Cvoid", "Cint", "Cuint", "Cchar", "Cuchar",
                        "Cshort", "Cushort", "Clong", "Culong", "Clonglong", "Culonglong",
                        "Cfloat", "Cdouble", "Cstring", "Csize_t", "Cssize_t", "Cptrdiff_t",
                        "Bool", "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32",
                        "Int64", "UInt64", "Float32", "Float64", "Nothing"])

                    # Generate typed accessor functions for each union member
                    for m in members
                        m_name = get(m, "name", "")
                        m_julia_type = get(m, "julia_type", "Any")
                        if isempty(m_name) || m_julia_type == "Any"
                            continue
                        end

                        # Sanitize member type name to match struct definitions
                        m_julia_type = _sanitize_cpp_type_name(m_julia_type)

                        # Sanitize member name for Julia identifier
                        safe_m_name = replace(m_name, r"[^A-Za-z0-9_]" => "_")

                        # Skip complex types (nested structs, arrays) — only primitive/pointer accessors
                        if startswith(m_julia_type, "NTuple{") || startswith(m_julia_type, "Vector{")
                            continue
                        end

                        # Check that Ptr{X} inner types are defined; fall back to Ptr{Cvoid} if not
                        ptr_match = match(r"^Ptr\{(.+)\}$", m_julia_type)
                        if !isnothing(ptr_match)
                            inner = ptr_match.captures[1]
                            # Recursively unwrap nested Ptr{}
                            while (pm = match(r"^Ptr\{(.+)\}$", inner)) !== nothing
                                inner = pm.captures[1]
                            end
                            if inner ∉ julia_builtins && inner ∉ defined_struct_names
                                m_julia_type = "Ptr{Cvoid}"
                            end
                        elseif m_julia_type ∉ julia_builtins && m_julia_type ∉ defined_struct_names
                            # Non-pointer, non-primitive, non-emitted struct — skip.
                            # Both arms test `defined_struct_names`, not
                            # `known_struct_names`: the annotations below are eager,
                            # so a name that DWARF mentions but no chunk declares
                            # takes the whole module down at load, not just this
                            # accessor.
                            continue
                        end

                        # Immutable unions have no object identity to store through.
                        # Read and write a `Ref` of the byte tuple instead.
                        _get_body = anon_union ?
                            "r = Ref(u.data)\n    return GC.@preserve r unsafe_load(Ptr{$m_julia_type}(pointer_from_objref(r)))" :
                            "return unsafe_load(Ptr{$m_julia_type}(pointer_from_objref(u)))"
                        _set_body = anon_union ?
                            "r = Ref(u.data)\n    GC.@preserve r unsafe_store!(Ptr{$m_julia_type}(pointer_from_objref(r)), v)\n    return $julia_struct_name(r[])" :
                            "unsafe_store!(Ptr{$m_julia_type}(pointer_from_objref(u)), v)"
                        union_accessor_defs *= """
                        \"\"\"Get union member `$m_name` as `$m_julia_type` from `$julia_struct_name`.\"\"\"
                        function get_$(safe_m_name)(u::$julia_struct_name)::$m_julia_type
                            $_get_body
                        end

                        \"\"\"Set union member `$m_name` as `$m_julia_type` in `$julia_struct_name`.\"\"\"
                        function set_$(safe_m_name)!(u::$julia_struct_name, v::$m_julia_type)
                            $_set_body
                        end

                        """
                        push!(exports, "get_$(safe_m_name)")
                        push!(exports, "set_$(safe_m_name)!")
                    end

                    continue
                end

                # BUG FIX: Recursively get all members including base classes
                members = flatten_struct_members(struct_name)

                if !isempty(members)
                    # Sort members by offset to ensure correct order
                    sort!(members, by = m -> begin
                        off = get(m, "offset", "0x0")
                        isnothing(off) ? 0 : parse(Int, off)
                    end)

                    # Check if this struct has bitfield members
                    has_bitfields = any(m -> haskey(m, "bit_size"), members)

                    if has_bitfields
                        byte_size_str = get(struct_info, "byte_size", "0x0")
                        byte_size = parse(Int, byte_size_str)
                        if byte_size == 0; byte_size = 8; end

                        push!(struct_chunks, """
                        # C struct with bitfields: $struct_name (size $byte_size bytes)
                        mutable struct $julia_struct_name
                            _data::NTuple{$byte_size, UInt8}
                        end
                        $julia_struct_name() = $julia_struct_name(ntuple(i -> 0x00, $byte_size))

                        """)
                        jl_byte_records[julia_struct_name] = byte_size

                        # Generate accessor functions for each member
                        for member in members
                            member_name = get(member, "name", "unknown")
                            julia_type = get(member, "julia_type", "Any")
                            safe_member = replace(member_name, r"[^A-Za-z0-9_]" => "_")

                            # Sanitize julia_type to avoid < > in function signatures
                            sanitized_type = julia_type
                            if occursin(r"[<>]", julia_type)
                                if occursin(r"Ptr\{[^}]+\}", julia_type)
                                    type_match = match(r"Ptr\{([^}]+)\}", julia_type)
                                    if !isnothing(type_match)
                                        inner_type = String(type_match.captures[1])
                                        if _stl_internal(inner_type)
                                            sanitized_type = "Ptr{Cvoid}"
                                        else
                                            sanitized_type = "Ptr{$(_sanitize_cpp_type_name(inner_type))}"
                                        end
                                    end
                                else
                                    if _stl_internal(julia_type)
                                        m_size = get(member, "size", 0)
                                        sanitized_type = m_size > 0 ? "NTuple{$m_size, UInt8}" : "Ptr{Cvoid}"
                                    else
                                        sanitized_type = _sanitize_cpp_type_name(julia_type)
                                    end
                                end
                            end
                            julia_type = sanitized_type

                            if haskey(member, "bit_size")
                                bit_size = member["bit_size"]

                                # Determine absolute bit offset
                                bit_offset = if haskey(member, "data_bit_offset")
                                    member["data_bit_offset"]
                                elseif haskey(member, "bit_offset_legacy")
                                    byte_off = parse(Int, get(member, "offset", "0x0"))
                                    byte_off * 8 + member["bit_offset_legacy"]
                                else
                                    0
                                end

                                byte_pos = bit_offset >> 3
                                bit_within_byte = bit_offset & 7
                                mask = (1 << bit_size) - 1
                                # A signed field reads sign-extended as its declared
                                # type and its setter takes that type's range
                                # (`_bitfield_signed_type`); an unsigned one is as it was.
                                signed_t = _bitfield_signed_type(member, dwarf_structs)
                                signed_store = signed_t === nothing ? "" :
                                    _bitfield_signed_store("set_$(safe_member)!", bit_size, signed_t)

                                # Getter with bit extraction
                                if bit_size <= 8 && bit_within_byte + bit_size <= 8
                                    # Single byte access
                                    getter_raw = "(s._data[$(byte_pos + 1)] >> $bit_within_byte) & $(mask)"
                                    push!(struct_chunks, """
                                    \"\"\"Get bitfield `$member_name` ($bit_size bits) from `$julia_struct_name`.\"\"\"
                                    function get_$(safe_member)(s::$julia_struct_name)::$(something(signed_t, "UInt32"))
                                        return $(signed_t === nothing ? "UInt32($getter_raw)" :
                                                 _bitfield_sign_extend(getter_raw, bit_size, signed_t))
                                    end

                                    \"\"\"Set bitfield `$member_name` ($bit_size bits) in `$julia_struct_name`.\"\"\"
                                    function set_$(safe_member)!(s::$julia_struct_name, v::Integer)
                                    $(signed_store)    data = collect(s._data)
                                        cleared = data[$(byte_pos + 1)] & ~UInt8($(mask) << $bit_within_byte)
                                        data[$(byte_pos + 1)] = cleared | UInt8((UInt32(v) & $(mask)) << $bit_within_byte)
                                        s._data = NTuple{$byte_size, UInt8}(data)
                                    end

                                    """)
                                else
                                    # Multi-byte bitfield. The bytes it spans are assembled
                                    # into an integer wide enough for them. A fixed UInt32
                                    # load threw on every field wider than 32 bits (its mask
                                    # does not fit) and could read past the struct; it had no
                                    # setter either (abi_fuzz X6).
                                    span = bit_within_byte + bit_size
                                    nbytes = cld(span, 8)
                                    raw_t = span <= 16 ? "UInt16" : span <= 32 ? "UInt32" :
                                            span <= 64 ? "UInt64" : "UInt128"
                                    ret_t = bit_size <= 32 ? "UInt32" : "UInt64"
                                    mask_lit = "$raw_t(0x" * string((UInt128(1) << bit_size) - 1, base = 16) * ")"
                                    gather(src) = join(["($raw_t($src[$(byte_pos + 1 + k)]) << $(8k))"
                                                        for k in 0:nbytes-1], " | ")
                                    scatter = join(["    data[$(byte_pos + 1 + k)] = (raw >> $(8k)) % UInt8"
                                                    for k in 0:nbytes-1], "\n")
                                    push!(struct_chunks, """
                                    \"\"\"Get bitfield `$member_name` ($bit_size bits) from `$julia_struct_name`.\"\"\"
                                    function get_$(safe_member)(s::$julia_struct_name)::$(something(signed_t, ret_t))
                                        raw = $(gather("s._data"))
                                        return $(signed_t === nothing ? "$ret_t((raw >> $bit_within_byte) & $mask_lit)" :
                                                 _bitfield_sign_extend("(raw >> $bit_within_byte) & $mask_lit", bit_size, signed_t))
                                    end

                                    \"\"\"Set bitfield `$member_name` ($bit_size bits) in `$julia_struct_name`.\"\"\"
                                    function set_$(safe_member)!(s::$julia_struct_name, v::Integer)
                                    $(signed_store)    data = collect(s._data)
                                        raw = $(gather("data"))
                                        m = $mask_lit << $bit_within_byte
                                        raw = (raw & ~m) | (((v % $raw_t) & $mask_lit) << $bit_within_byte)
                                    $scatter
                                        s._data = NTuple{$byte_size, UInt8}(data)
                                    end

                                    """)
                                end
                                push!(exports, "get_$(safe_member)")
                                push!(exports, "set_$(safe_member)!")
                            else
                                # Non-bitfield member in a bitfield struct — byte-offset accessor
                                byte_off = parse(Int, get(member, "offset", "0x0"))
                                if haskey(jl_byte_records, julia_type)
                                    # A named union or bitfield record is a `mutable struct`:
                                    # build it from its bytes (abi_fuzz X5).
                                    nb = jl_byte_records[julia_type]
                                    if byte_off + nb <= byte_size
                                        push!(struct_chunks, """
                                        \"\"\"Get non-bitfield member `$member_name` from `$julia_struct_name`.\"\"\"
                                        function get_$(safe_member)(s::$julia_struct_name)::$julia_type
                                            return $julia_type(ntuple(i -> s._data[$byte_off + i], $nb))
                                        end

                                        """)
                                        push!(exports, "get_$(safe_member)")
                                    end
                                elseif julia_type != "Any" && !startswith(julia_type, "NTuple{")
                                    push!(struct_chunks, """
                                    \"\"\"Get non-bitfield member `$member_name` from `$julia_struct_name`.\"\"\"
                                    function get_$(safe_member)(s::$julia_struct_name)::$julia_type
                                        p = pointer(collect(s._data)) + $byte_off
                                        return unsafe_load(Ptr{$julia_type}(p))
                                    end

                                    """)
                                    push!(exports, "get_$(safe_member)")
                                end
                            end
                        end

                        continue
                    end

                    # The members that are part of the object. A static data
                    # member, and clang's `__clang_vtable`, has no offset; sorted
                    # as offset 0 above, it would be a field at the front.
                    layout_members = filter(m -> get(m, "offset", nothing) !== nothing, members)

                    # Check if any member has unresolvable size (template types with no DWARF size info).
                    # When DWARF byte_size is known but member sizes aren't, use a byte blob to ensure
                    # the Julia struct has the correct total size for ABI safety.
                    _known_c_primitives = Set([
                        "int", "unsigned int", "int32_t", "uint32_t",
                        "long", "unsigned long", "long long", "unsigned long long", "int64_t", "uint64_t",
                        "short", "unsigned short", "int16_t", "uint16_t",
                        "char", "unsigned char", "signed char", "int8_t", "uint8_t",
                        "float", "double", "long double", "bool", "_Bool",
                        "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t",
                        "wchar_t",
                    ])
                    has_unresolvable = false
                    for m in layout_members
                        if get(m, "size", 0) == 0
                            c_type_raw = strip(replace(get(m, "c_type", ""), r"\bconst\b" => ""))
                            c_type_raw = strip(c_type_raw)
                            if endswith(c_type_raw, "*") || endswith(c_type_raw, "&")
                                continue
                            end
                            if c_type_raw in _known_c_primitives
                                continue
                            end
                            has_unresolvable = true
                            break
                        end
                    end

                    byte_size_str = get(struct_info, "byte_size", "0x0")
                    byte_size = isnothing(byte_size_str) ? 0 :
                        (startswith(string(byte_size_str), "0x") ?
                         parse(Int, string(byte_size_str)[3:end], base=16) :
                         parse(Int, string(byte_size_str)))

                    # The Julia type `julia_type` is written as in a field here,
                    # or `nothing` when this module cannot name it at this point.
                    _usable_field_type = function (julia_type::String)
                    # An array is usable when its element is. Peeled before any
                    # sanitizing, which folded `NTuple{2, ImVector<unsigned int>}`
                    # into one name nothing declares, and the element is checked
                    # like a bare field type: `NTuple{16, b2ContactRegister}` named
                    # a struct emitted later. Either one failed the module at load
                    # once DWARF gave such members a size.
                    am = match(r"^NTuple\{(\d+),\s*(.+)\}$", julia_type)
                    if am !== nothing
                        el = _usable_field_type(String(am.captures[2]))
                        el === nothing && return nothing
                        return el == am.captures[2] ? julia_type : "NTuple{$(am.captures[1]), $el}"
                    end
                    # Sanitize member types that reference other structs with template syntax
                    # Only sanitize custom struct names, not built-in Julia types like NTuple
                    sanitized_type = julia_type

                    # Don't sanitize built-in Julia types (NTuple, Ptr{Cint}, etc.)
                    builtin_types = ["NTuple", "Ptr", "Cint", "Cuint", "Cintptr_t", "Cuintptr_t", "Cdouble", "Cfloat", "Clong", "Culong", "Cshort", "Cushort", "Cchar", "Cuchar", "Culonglong", "Clonglong", "Cvoid", "Csize_t", "Cptrdiff_t", "Cssize_t", "Cwchar_t", "Cstring", "Bool", "UInt8", "Int8", "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64", "Int128", "UInt128", "Float32", "Float64", "ComplexF32", "ComplexF64"]
                    is_builtin = any(startswith(julia_type, bt) for bt in builtin_types)

                    if !is_builtin || occursin(r"[<>]", julia_type)
                        if occursin(r"Ptr\{[^}]+\}", julia_type)
                            # Extract type from Ptr{Type} for custom struct references
                            type_match = match(r"Ptr\{([^}]+)\}", julia_type)
                            if !isnothing(type_match)
                                inner_type = String(type_match.captures[1])
                                # Only sanitize if inner type is a custom struct (contains template chars)
                                if occursin(r"[<>]", inner_type)
                                    if _stl_internal(inner_type)
                                        # STL-internal inner type: use Ptr{Cvoid} — never defined
                                        sanitized_type = "Ptr{Cvoid}"
                                    else
                                        sanitized_inner = _sanitize_cpp_type_name(inner_type)
                                        sanitized_type = "Ptr{$sanitized_inner}"
                                    end
                                end
                            end
                        elseif occursin(r"[<>]", julia_type)
                            # Direct custom struct reference with template syntax
                            _stl_internal(julia_type) && return nothing
                            sanitized_type = _sanitize_cpp_type_name(julia_type)
                        end

                        # If the (now-sanitized) type refers to an STL-internal type that
                        # will be filtered out during struct generation, the member
                        # becomes bytes so the embedding struct remains valid Julia syntax.
                        sanitized_type == julia_type && _stl_internal(julia_type) && return nothing
                    end

                    # A pointer to an enum this module emitted keeps its type,
                    # under the enum's emitted name: enums precede structs in
                    # the file, so the name is defined.
                    let pm = match(r"^((?:Ptr\{)+)([A-Za-z_][A-Za-z0-9_]*)(\}+)$", sanitized_type)
                        if pm !== nothing && _is_enum_name(String(pm.captures[2]))
                            en = _sanitize_cpp_type_name(String(pm.captures[2]))
                            haskey(emitted_enum_base, en) && return pm.captures[1] * en * pm.captures[3]
                        end
                    end

                    # If the field is Ptr{X} and X hasn't been defined yet,
                    # substitute Ptr{Cvoid} to avoid UndefVarError (same ABI size).
                    sanitized_type = _resolve_forward_ptr(sanitized_type, defined_struct_names)

                    # Same hazard for a BARE field type, and worse. `Ptr{X}` at
                    # least has a fixed width; a bare `x::T` needs T to exist at
                    # struct-definition time, so one undeclared name raises
                    # UndefVarError and takes the whole module with it. Structs
                    # skipped above as STL-internal or blocklisted never enter
                    # `defined_struct_names`, yet a field can still reference
                    # them: llama.cpp's `_Sp_alloc_shared_tag` had
                    # `_M_a::Ref_allocator_void` (from `const allocator<void>&`,
                    # '&' → "Ref" at line ~906) — declared nowhere, used once,
                    # and all 98k lines failed to load. Emission is topological,
                    # so "not yet defined" here means "cannot be used", not
                    # "defined later".
                    #
                    # Defined is not enough: the layout proof must be able to
                    # measure the field. A bitfield record or an opaque
                    # placeholder is a `mutable struct`, a reference where C has
                    # the value inline, and a forward declaration is an empty
                    # `struct X end`; neither is in `jl_layouts`, the proof
                    # answered `:unknown`, and the fields went out unproven
                    # (llama.cpp's `llama_hparams`: 492 bytes for 29180). Named
                    # unions are in `jl_mutable`, so the proof measures them as
                    # the reference they are. Builtins match whole names: a
                    # prefix test took a struct named `BoolVector` for `Bool`.
                    (sanitized_type in builtin_types || startswith(sanitized_type, "Ptr{")) &&
                        return sanitized_type
                    sanitized_type in defined_struct_names &&
                        (haskey(jl_layouts, sanitized_type) || sanitized_type in jl_mutable) &&
                        return sanitized_type
                    # An enum the module emitted, under the name it has there
                    # (a raw `Type` is emitted as `c_Type`). Enums precede
                    # structs in the file.
                    if _is_enum_name(sanitized_type)
                        en = _sanitize_cpp_type_name(sanitized_type)
                        haskey(emitted_enum_layout, en) && return en
                    end
                    return nothing
                    end

                    # The Julia field type a member is emitted as, and whether it
                    # stands in for the member's type as bytes. One function for
                    # the layout proof below and for the emission loop, so the proof
                    # checks exactly the fields that get written. A type this module
                    # cannot name here degrades to the member's DWARF size as a byte
                    # region, so the containing struct keeps its exact layout;
                    # pointer width only when the size is unknown. So does a type of
                    # another size than the member's: fmt's `arg_id_kind&` is
                    # spelled as its enum but is a pointer. (A named union keeps its
                    # reference; the proof measures that.)
                    _member_field = function (member)
                        m_size = get(member, "size", 0)
                        t = _usable_field_type(String(get(member, "julia_type", "Any")))
                        if t !== nothing && m_size > 0 && !(t in jl_mutable)
                            lay = _julia_field_layout(t, jl_layouts, jl_mutable)
                            lay !== nothing && lay[1] != m_size && (t = nothing)
                        end
                        t === nothing || return (t, false)
                        return (m_size > 0 ? "NTuple{$m_size, UInt8}" : "Ptr{Cvoid}", true)
                    end
                    _member_field_type = member -> first(_member_field(member))

                    # Exact or opaque, never approximate — GeneratorC's
                    # `_resolve_exact_layout`, applied to the fields this struct
                    # would be emitted with. The field path puts `_pad_N` bytes in
                    # front of each member and trusts Julia to land it on its DWARF
                    # offset; natural alignment moves it anyway whenever C did not
                    # align it (`__attribute__((packed))`, `#pragma pack`). A packed
                    # `{char; int}` came out 8 bytes with the int at 4 instead of 5
                    # bytes with it at 1, so every read through a pointer, and every
                    # struct embedding it, was wrong. When the fields provably
                    # cannot reproduce the C layout, the byte blob below can.
                    field_plan = Tuple{String,Union{Int,Nothing}}[]
                    let cur = 0
                        for member in layout_members
                            ov = get(member, "offset", "0x0")      # parsed as the emission loop does
                            off = isnothing(ov) ? 0 : parse(Int, ov)
                            off > cur && push!(field_plan, ("NTuple{$(off - cur), UInt8}", nothing))
                            off > cur && (cur = off)
                            push!(field_plan, (String(_member_field_type(member)), off))
                            cur += get(member, "size", 0)
                        end
                        byte_size > cur && push!(field_plan, ("NTuple{$(byte_size - cur), UInt8}", nothing))
                    end
                    layout_proof = _prove_julia_layout(field_plan, byte_size, jl_layouts, jl_mutable)
                    if layout_proof === :mismatch && byte_size > 0 && !isempty(members)
                        has_unresolvable = true
                    end
                    # A named union is `mutable`: as a field it is a reference
                    # where C has the union's bytes inline, so a record holding
                    # one is a blob. The proof measures the reference as a
                    # pointer, which an 8-byte union at an aligned offset fits
                    # (abi_fuzz unit 104: `R5`'s `m6::T4` made `R5` non-isbits,
                    # and C's bytes in it crashed the GC), so check by type.
                    _in_mutable = function (t)
                        while (am = match(r"^NTuple\{\d+,\s*(.+)\}$", t)) !== nothing
                            t = String(am.captures[1])
                        end
                        return t in jl_mutable
                    end
                    if byte_size > 0 && any(m -> _in_mutable(first(_member_field(m))), layout_members)
                        has_unresolvable = true
                    end

                    if has_unresolvable && byte_size > 0
                        member_count = length(members)
                        push!(blob_struct_names, julia_struct_name)
                        jl_layouts[julia_struct_name] = (byte_size, 1)
                        push!(struct_chunks, """
                        # C++ struct: $struct_name ($member_count members, byte blob for ABI safety)
                        struct $julia_struct_name
                            _data::NTuple{$(byte_size), UInt8}
                        end

                        # Zero-initializer for $julia_struct_name
                        function $julia_struct_name()
                            return $julia_struct_name(ntuple(i -> 0x00, $byte_size))
                        end

                        """)

                        # Generate Base.getproperty accessor for named member access on byte-blob structs
                        _accessor_branches = String[]
                        # …and the matching writes. Built in the SAME loop off the
                        # same offset/type decision, so a member can never be
                        # readable but unsettable (see _blob_store_expr).
                        _setter_branches = String[]
                        _loadable_primitives = Dict(
                            "Cdouble" => ("Cdouble", 8), "Cfloat" => ("Cfloat", 4),
                            "Cint" => ("Cint", 4), "Cuint" => ("Cuint", 4),
                            "Clong" => ("Clong", sizeof(Clong)), "Culong" => ("Culong", sizeof(Culong)),
                            "Clonglong" => ("Clonglong", 8), "Culonglong" => ("Culonglong", 8),
                            "Cshort" => ("Cshort", 2), "Cushort" => ("Cushort", 2),
                            "Cchar" => ("Cchar", 1), "Cuchar" => ("Cuchar", 1),
                            "Csize_t" => ("Csize_t", 8), "Cptrdiff_t" => ("Cptrdiff_t", 8),
                            "Cssize_t" => ("Cssize_t", 8), "Bool" => ("Bool", 1),
                            "UInt8" => ("UInt8", 1), "Int8" => ("Int8", 1),
                            "UInt16" => ("UInt16", 2), "Int16" => ("Int16", 2),
                            "UInt32" => ("UInt32", 4), "Int32" => ("Int32", 4),
                            "UInt64" => ("UInt64", 8), "Int64" => ("Int64", 8),
                            "Float32" => ("Float32", 4), "Float64" => ("Float64", 8),
                        )
                        # Pre-compute in-context sizes for each member from offset gaps
                        _member_offsets = Int[]
                        for m in members
                            m_offset_str = get(m, "offset", nothing)
                            if isnothing(m_offset_str)
                                push!(_member_offsets, -1)
                            else
                                push!(_member_offsets, startswith(string(m_offset_str), "0x") ?
                                    parse(Int, string(m_offset_str)[3:end], base=16) :
                                    parse(Int, string(m_offset_str)))
                            end
                        end
                        for (mi, m) in enumerate(members)
                            m_name = get(m, "name", nothing)
                            isnothing(m_name) && continue
                            m_offset = _member_offsets[mi]
                            m_offset < 0 && continue
                            m_julia_type = get(m, "julia_type", "")
                            m_c_type = strip(get(m, "c_type", ""))

                            # Compute available space for this member from offset gaps
                            next_offset = byte_size
                            for j in (mi+1):length(_member_offsets)
                                if _member_offsets[j] >= 0
                                    next_offset = _member_offsets[j]
                                    break
                                end
                            end
                            available_size = next_offset - m_offset

                            # Pointer types
                            if endswith(m_c_type, "*")
                                push!(_accessor_branches, """
                                    if s === :$m_name
                                        return GC.@preserve x unsafe_load(Ptr{Ptr{Cvoid}}(pointer_from_objref(Ref(x._data)) + $m_offset))
                                    end""")
                                push!(_setter_branches, _blob_store_expr(m_name, m_offset, :pointer))
                            # Primitive types we can unsafe_load directly
                            elseif haskey(_loadable_primitives, m_julia_type)
                                jt, _ = _loadable_primitives[m_julia_type]
                                push!(_accessor_branches, """
                                    if s === :$m_name
                                        return GC.@preserve x unsafe_load(Ptr{$jt}(pointer_from_objref(Ref(x._data)) + $m_offset))
                                    end""")
                                push!(_setter_branches, _blob_store_expr(m_name, m_offset, :primitive; julia_type=jt))
                            # Nested struct types — extract sub-blob
                            else
                                if _stl_internal(String(m_c_type))
                                    continue
                                end
                                m_sanitized = _sanitize_cpp_type_name(m_c_type)
                                # Any record member the module emits. This used to
                                # require a spelling sanitization changes (a template
                                # or scoped name), which was all this blob path met
                                # while it only served members of unknown size. A
                                # layout Julia cannot reproduce (`_prove_julia_layout`)
                                # lands here too, and its `b2Vec2` or named-union
                                # members had no accessor at all.
                                if !isempty(m_sanitized) && m_sanitized in defined_struct_names
                                    # Find the nested struct's byte_size using best_dwarf_key map
                                    nested_info = nothing
                                    best_key = get(best_dwarf_key, m_sanitized, nothing)
                                    if !isnothing(best_key) && haskey(dwarf_structs, best_key)
                                        nested_info = dwarf_structs[best_key]
                                    else
                                        # Fallback: search all structs for sanitized name
                                        # match. Of several, the smallest key: the first
                                        # one met followed the Dict's order.
                                        hit = nothing
                                        for sk in keys(dwarf_structs)
                                            k = String(sk)
                                            _sanitize_cpp_type_name(k) == m_sanitized &&
                                                (hit === nothing || k < hit) && (hit = k)
                                        end
                                        hit === nothing || (nested_info = dwarf_structs[hit])
                                    end
                                    if !isnothing(nested_info)
                                        nested_bs_str = get(nested_info, "byte_size", "0x0")
                                        nested_bs = isnothing(nested_bs_str) ? 0 :
                                            (startswith(string(nested_bs_str), "0x") ?
                                             parse(Int, string(nested_bs_str)[3:end], base=16) :
                                             parse(Int, string(nested_bs_str)))
                                        if nested_bs > 0
                                            if m_sanitized in blob_struct_names
                                                # Target is a byte-blob struct — extract bytes
                                                actual_size = min(nested_bs, available_size)
                                                if actual_size == nested_bs
                                                    push!(_accessor_branches, """
                                    if s === :$m_name
                                        bytes = GC.@preserve x ntuple(i -> unsafe_load(Ptr{UInt8}(pointer_from_objref(Ref(x._data)) + $m_offset + i - 1)), $nested_bs)
                                        return $(m_sanitized)(bytes)
                                    end""")
                                                    push!(_setter_branches, _blob_store_expr(m_name, m_offset, :blob_struct_full;
                                                        struct_name=m_sanitized, nested_size=nested_bs))
                                                else
                                                    push!(_accessor_branches, """
                                    if s === :$m_name
                                        raw = GC.@preserve x ntuple(i -> unsafe_load(Ptr{UInt8}(pointer_from_objref(Ref(x._data)) + $m_offset + i - 1)), $actual_size)
                                        padded = ntuple(i -> i <= $actual_size ? raw[i] : 0x00, $nested_bs)
                                        return $(m_sanitized)(padded)
                                    end""")
                                                    push!(_setter_branches, _blob_store_expr(m_name, m_offset, :blob_struct_partial;
                                                        struct_name=m_sanitized, nested_size=nested_bs, actual_size=actual_size))
                                                end
                                            elseif haskey(jl_byte_records, m_sanitized)
                                                # A named union or bitfield record is a `mutable
                                                # struct`; `unsafe_load(Ptr{T})` would read C's bytes
                                                # as an object reference (abi_fuzz X5). Build it from
                                                # its bytes, and store its bytes back.
                                                if nested_bs <= available_size
                                                    push!(_accessor_branches, """
                                    if s === :$m_name
                                        bytes = GC.@preserve x ntuple(i -> unsafe_load(Ptr{UInt8}(pointer_from_objref(Ref(x._data)) + $m_offset + i - 1)), $nested_bs)
                                        return $(m_sanitized)(bytes)
                                    end""")
                                                    push!(_setter_branches, _blob_store_expr(m_name, m_offset, :mutable_record;
                                                        struct_name=m_sanitized, nested_size=nested_bs))
                                                end
                                            else
                                                # Target is a normal typed struct — unsafe_load directly
                                                push!(_accessor_branches, """
                                    if s === :$m_name
                                        return GC.@preserve x unsafe_load(Ptr{$m_sanitized}(pointer_from_objref(Ref(x._data)) + $m_offset))
                                    end""")
                                                push!(_setter_branches, _blob_store_expr(m_name, m_offset, :typed_struct;
                                                    struct_name=m_sanitized))
                                            end
                                        end
                                    end
                                end
                            end
                        end
                        if !isempty(_accessor_branches)
                            accessor_code = join(_accessor_branches, "\n")
                            push!(struct_chunks, """
                            function Base.getproperty(x::$julia_struct_name, s::Symbol)
                                s === :_data && return getfield(x, :_data)
                            $accessor_code
                                Base.error("type $julia_struct_name has no field \$s")
                            end

                            """)
                            filter!(!isempty, _setter_branches)
                            setter_code = _blob_setter_chunk(julia_struct_name, _setter_branches)
                            if !isempty(setter_code)
                                push!(struct_chunks, setter_code)
                                blob_setters_emitted = true
                            end
                        end

                        continue
                    end

                    # Bytes over C data that holds a float or double, in a field
                    # written as bytes or inside a field's own record: a ccall
                    # classifies them INTEGER (`_crosses_small_blob`). An enum
                    # written as bytes is INTEGER in C as well.
                    _sse_under_bytes = any(layout_members) do member
                        jt = String(get(member, "julia_type", "Any"))
                        (ft, as_bytes) = _member_field(member)
                        as_bytes && return _type_has_sse(jt, dwarf_structs, _record_key, _is_enum_name)
                        while (am = match(r"^NTuple\{\d+,\s*(.+)\}$", ft)) !== nothing
                            ft = String(am.captures[1])
                        end
                        ft in sse_bytes_struct_names && return true
                        ft in blob_struct_names || return false
                        key = get(best_dwarf_key, ft, nothing)
                        return key === nothing || _type_has_sse(key, dwarf_structs, _record_key, _is_enum_name)
                    end
                    _sse_under_bytes && push!(sse_bytes_struct_names, julia_struct_name)

                    member_count = length(layout_members)
                    push!(struct_chunks, """
                    # C++ struct: $struct_name ($member_count members)
                    struct $julia_struct_name
                    """)
                    
                    current_offset = 0
                    pad_idx = 0

                    for member in layout_members
                        member_name = get(member, "name", "unknown")
                        julia_type = get(member, "julia_type", "Any")
                        
                        offset_val = get(member, "offset", "0x0")
                        offset = isnothing(offset_val) ? 0 : parse(Int, offset_val)
                        
                        # Insert padding if needed
                        if offset > current_offset
                            pad_size = offset - current_offset
                            push!(struct_chunks, "    _pad_$(pad_idx)::NTuple{$(pad_size), UInt8}\n")
                            pad_idx += 1
                            current_offset = offset
                        end

                        # Sanitize member name for Julia (replace invalid characters and keywords)
                        sanitized_name = replace(member_name, '$' => '_')
                        sanitized_name = make_cpp_identifier(sanitized_name)

                        sanitized_type = _member_field_type(member)
                        # A factory class this field names must be defined
                        # before this record (`idiomatic_early_slot`).
                        for tm in eachmatch(r"[A-Za-z_][A-Za-z0-9_]*", sanitized_type)
                            haskey(idiomatic_classes, tm.match) && push!(idiomatic_named_early, tm.match)
                        end

                        push!(struct_chunks, "    $sanitized_name::$sanitized_type\n")
                        
                        # Update current offset
                        member_size = get(member, "size", 0)
                        # If size is 0 (e.g. unknown type), we can't reliably track offset
                        # But typically we rely on the next member's offset to insert padding
                        current_offset += member_size
                    end

                    # Add trailing padding if the struct size from DWARF is larger than the final offset
                    if byte_size > current_offset
                        pad_size = byte_size - current_offset
                        push!(struct_chunks, "    _pad_tail::NTuple{$(pad_size), UInt8}\n")
                    end
                    layout_proof isa Tuple && (jl_layouts[julia_struct_name] = layout_proof)

                    push!(struct_chunks, """
                    end

                    # Zero-initializer for $julia_struct_name
                    function $julia_struct_name()
                        ref = Ref{$julia_struct_name}()
                        GC.@preserve ref begin
                            ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), Base.unsafe_convert(Ptr{Cvoid}, ref), 0, sizeof($julia_struct_name))
                        end
                        return ref[]
                    end

                    """)
                    # The setters a byte blob gets, so a record keeps them when
                    # the struct emitter can lay it out instead.
                    push!(struct_chunks, _field_setter_chunk(julia_struct_name))
                    field_setters_emitted = true
                    blob_setters_emitted = true
                else
                    # A record with no member anywhere a field could come from:
                    # an empty class (a stateless functor, a tag), or one whose
                    # bases are all virtual. Its bytes, at its DWARF size, so
                    # `STRUCT_SIZES` and `sizeof` agree and a holder measures it
                    # right; the 8-byte placeholder stays only for a record whose
                    # size the metadata does not know.
                    byte_size = try
                        s = string(get(struct_info, "byte_size", "0"))
                        startswith(s, "0x") ? parse(Int, s[3:end], base = 16) : parse(Int, s)
                    catch
                        0
                    end
                    if byte_size > 0
                        push!(blob_struct_names, julia_struct_name)
                        jl_layouts[julia_struct_name] = (byte_size, 1)
                        push!(struct_chunks, """
                        # C++ struct: $struct_name (no data members, $byte_size byte(s))
                        struct $julia_struct_name
                            _data::NTuple{$(byte_size), UInt8}
                        end

                        # Zero-initializer for $julia_struct_name
                        function $julia_struct_name()
                            return $julia_struct_name(ntuple(i -> 0x00, $byte_size))
                        end

                        """)
                    else
                        push!(struct_chunks, """
                        # Opaque struct: $struct_name (no member info available)
                        mutable struct $julia_struct_name
                            data::NTuple{8, UInt8}  # Placeholder
                        end

                        """)
                    end
                end
            else
                # No DWARF info available - generate opaque struct
                push!(struct_chunks, """
                # Opaque struct: $struct_name (no DWARF info)
                mutable struct $julia_struct_name
                    data::NTuple{32, UInt8}  # Placeholder - compile with -g for member info
                end

                """)
            end
        end

        push!(struct_chunks, "\n")
    end

    # =============================================================================
    # GLOBAL VARIABLES GENERATION
    # =============================================================================
    
    global_vars = get(metadata, "globals", Dict())
    func_chunks = String[]
    
    if !isempty(global_vars)
        push!(func_chunks, """
        # =============================================================================
        # Global Variables
        # =============================================================================

        """)
        
        # The types a value accessor may load: what Julia binds itself, and the
        # records and enums this module emits other than a mutable one, which
        # `unsafe_load` would read as an object reference (abi_fuzz X5).
        holdable = _typed_return_names(join(struct_chunks))
        union!(holdable, _imported_value_type_names(dwarf_structs, _sanitize_cpp_type_name))
        union!(holdable, _pending_enum_type_names(enum_chunks))
        union!(holdable, _defined_type_names(join(c for c in enum_chunks if c isa String)))
        union!(holdable, setdiff(_BUILTIN_JULIA_SPELLINGS, ("Any", "Nothing", "Ref")), ("NTuple",))

        # Sorted: `globals` is a metadata Dict, iterated in hash order.
        for var_name in sort!(collect(keys(global_vars)); by = string)
            var_info = global_vars[var_name]
            julia_type = get(var_info, "julia_type", "Any")
            # The key is the symbol the loader binds (`_ZN3cfg5depthE`); the
            # accessor is named for the qualified name, scope first, the way a
            # method is (`cfg_depth`, `geo_Vec3_live_count`). Metadata older
            # than `symbol` and `scope` keys a global by its name.
            symbol = _global_symbol(var_name, var_info)
            scope = String(something(get(var_info, "scope", ""), ""))
            bare = String(get(var_info, "name", var_name))
            qualified = isempty(scope) ? bare : scope * "::" * bare
            # The value and the pointer accessor are screened apart, both named
            # from the library's spelling (GeneratorC's loop says why).
            (base_name, screen_src) = isempty(scope) ?
                (make_cpp_identifier(bare), var_name) :
                (make_cpp_identifier(replace(qualified, "::" => "_")), symbol)
            safe_name = _screened_function_name(base_name, screen_src)
            ptr_name = _screened_function_name(base_name * "_ptr", screen_src)
            sym_expr = Base.isidentifier(symbol) ? ":$symbol" : "Symbol($(repr(symbol)))"

            # ...and the TYPE too. Only the name was sanitized here, so a global
            # whose type is a C++ template kept its raw spelling and was emitted
            # into four identifier positions below (`::$julia_type` twice and two
            # cglobal args). llama.cpp's `unicode_ranges_nfd` is typed
            # `initializer_list<range_nfd>`, and the `<` made the whole module a
            # syntax error. `_assert_wrapper_loadable` deliberately does not look
            # at these positions — they are lazy, so an *undeclared* type there
            # costs one function — but a malformed one is a parse error and costs
            # everything, which is why the parse guard is the complement.
            # Residual, accepted: if the sanitized name is not a type this module
            # declares, these two accessors throw when called. One function, not
            # 98k lines.
            if !_is_julia_type_spelling(julia_type)
                julia_type = _sanitize_cpp_type_name(julia_type)
            end

            # A type the wrapper cannot hold by value (`std::string`, a type
            # this module does not emit, `Any`) gets the raw pointer only: a
            # value accessor typed with it threw on the call, or read the
            # bytes as a Julia object.
            # A thread-local global is read through the accessor the build
            # compiled for it, in the calling thread (`_thread_local_global_chunk`).
            if _is_thread_local_global(var_info)
                value = all(m -> m.match in holdable, eachmatch(r"[A-Za-z_][A-Za-z0-9_]*", julia_type))
                push!(func_chunks, _thread_local_global_chunk(value ? safe_name : nothing, ptr_name,
                                                              qualified, julia_type, var_info))
                value && push!(exports, safe_name)
                push!(exports, ptr_name)
                continue
            end

            if !all(m -> m.match in holdable, eachmatch(r"[A-Za-z_][A-Za-z0-9_]*", julia_type))
                push!(func_chunks, """
                \"""
                    $(ptr_name)()

                Get pointer to global variable `$qualified` (no Julia value type: raw pointer only).
                \"""
                function $(ptr_name)()::Ptr{Cvoid}
                    return cglobal(($sym_expr, LIBRARY_PATH))
                end

                """)
                push!(exports, ptr_name)
                continue
            end

            # 1. Accessor for the value (read-only)
            # Only for simple types or pointers, struct values might be tricky
            # We use unsafe_load on the pointer

            push!(func_chunks, """
            \"""
                $safe_name()

            Get value of global variable `$qualified`.
            \"""
            function $safe_name()::$julia_type
                ptr = cglobal(($sym_expr, LIBRARY_PATH), $julia_type)
                return unsafe_load(ptr)
            end

            \"""
                $(ptr_name)()

            Get pointer to global variable `$qualified`.
            \"""
            function $(ptr_name)()::Ptr{$julia_type}
                return cglobal(($sym_expr, LIBRARY_PATH), $julia_type)
            end

            """)

            push!(exports, safe_name)
            push!(exports, ptr_name)
        end
        
        push!(func_chunks, "\n")
    end

    # Function wrappers

    # 2. Generate Managed Types
    managed_chunks = String[]
    # =============================================================================
    # MI / VIRTUAL-INHERITANCE UPCAST HELPERS (base-subobject pointer adjustment)
    # =============================================================================
    # A method inherited from a non-primary base expects `this` to point at the
    # base SUBOBJECT, not the derived object's start. Emit <Derived>_as_<Base>
    # helpers for every class where an upcast is non-trivial:
    #   - static Itanium offsets for non-virtual bases (MI), and
    #   - DYNAMIC adjustment for virtual bases: the vbase offset is
    #     vtable-resident (a standalone Left and a Left-inside-Diamond place
    #     VBase at different offsets), so the helper reads the object's vptr
    #     and the vbase-offset entry below the vtable address point at
    #     runtime. The same helper is correct for every dynamic type.
    # Transitive virtual bases (Diamond -> Left -> virtual VBase) compose:
    # static-adjust to the intermediate base, dynamic from there.
    let upcast_seen = Set{String}(), upcast_chunks = String[]
        # Walk a class's base DAG collecting every reachable upcast target:
        # (base_type, :static, accumulated_offset, 0) for non-virtual chains,
        # (base_type, :virtual, static_prefix, vbase_vtable_offset) at virtual
        # edges (no recursion past those — positions beyond are dynamic too).
        function _collect_upcasts!(acc, s_name, static_off, visited)
            haskey(dwarf_structs, s_name) || return
            s_name in visited && return
            push!(visited, s_name)
            for base in get(dwarf_structs[s_name], "base_classes", [])
                base isa Dict || continue
                bt = replace(get(base, "type", ""), r"^(class|struct)\s+" => "")
                isempty(bt) && continue
                if get(base, "virtual", false) == true
                    vboff = get(base, "vbase_vtable_offset", nothing)
                    if vboff isa Integer && Int(vboff) != 0
                        push!(acc, (bt, :virtual, static_off, Int(vboff)))
                    else
                        # No usable vbase-offset, in either spelling the two
                        # producers use for "I could not parse it":
                        #   * Compiler.jl omits the key entirely -> nothing
                        #   * DWARFParser.jl leaves its pushed default -> 0
                        # 0 is independently impossible: the vbase-offset entry
                        # always sits BELOW the vtable address point, so any
                        # genuinely parsed value is negative.
                        #
                        # This used to be an @debug and fall through, which
                        # silently dropped the upcast helper — the wrapper then
                        # had no way to reach the virtual base at all, with no
                        # diagnostic at default log level. Fail loud instead.
                        # Both parsers handle DW_OP_constu and DW_OP_lit<N>;
                        # an unhandled form (DW_OP_const1u, DW_OP_consts, ...)
                        # lands here.
                        error("$(s_name): virtual base $(bt) has no usable " *
                              "vbase-offset (got $(repr(vboff))). The " *
                              "DW_AT_data_member_location expression on this " *
                              "inheritance edge was not parsed, so the upcast " *
                              "helper cannot be emitted. Check for an unhandled " *
                              "DW_OP form in Compiler.jl's readelf parser and " *
                              "DWARFParser.jl's llvm-dwarfdump parser — both " *
                              "must recognise the spelling.")
                    end
                else
                    off = get(base, "offset", 0)
                    off = off isa Integer ? Int(off) : 0
                    push!(acc, (bt, :static, static_off + off, 0))
                    _collect_upcasts!(acc, bt, static_off + off, visited)
                end
            end
        end

        for (s_name, s_info) in sort(collect(dwarf_structs), by = first)
            _imported(s_name) && continue
            bases = get(s_info, "base_classes", [])
            (bases isa Vector && !isempty(bases)) || continue

            acc = Tuple{String,Symbol,Int,Int}[]
            _collect_upcasts!(acc, s_name, 0, Set{String}())
            # Only emit when an upcast is non-trivial: a static base at a
            # non-zero offset (MI) or any virtual base. Plain offset-0 SI
            # chains need no helper.
            any(t -> (t[2] == :static && t[3] != 0) || t[2] == :virtual, acc) || continue

            d_name = replace(_sanitize_cpp_type_name(s_name), r"[^a-zA-Z0-9_]" => "_")
            for (base_type, kind, static_off, vboff) in acc
                b_name = replace(_sanitize_cpp_type_name(base_type), r"[^a-zA-Z0-9_]" => "_")
                fn = "$(d_name)_as_$(b_name)"
                fn in upcast_seen && continue
                push!(upcast_seen, fn)

                if kind == :static
                    push!(upcast_chunks, """
                    \"\"\"
                        $(fn)(obj) -> Ptr{Cvoid}

                    Upcast a `$(s_name)` to its `$(base_type)` base subobject
                    (+$(static_off) bytes, static Itanium layout). Accepts a raw `Ptr` or
                    any wrapper with a `.handle` field. Pass the result as `this`
                    when calling `$(base_type)` methods on a `$(s_name)` object.
                    \"\"\"
                    function $(fn)(obj)::Ptr{Cvoid}
                        p = obj isa Ptr ? Ptr{Cvoid}(obj) : Ptr{Cvoid}(obj.handle)
                        return p + $(static_off)
                    end
                    export $(fn)

                    """)
                else
                    prefix = static_off == 0 ? "" : " + $(static_off)"
                    push!(upcast_chunks, """
                    \"\"\"
                        $(fn)(obj) -> Ptr{Cvoid}

                    Upcast a `$(s_name)` to its VIRTUAL base `$(base_type)`. The
                    subobject's offset is dynamic (it depends on the most-derived
                    type), so this reads the object's vtable at runtime: the
                    vbase-offset entry lives $(vboff) bytes from the vtable
                    address point. Correct for every dynamic type — a standalone
                    `$(s_name)` and one embedded in a more-derived object resolve
                    different offsets through the same code. Accepts a raw `Ptr`
                    or any wrapper with a `.handle` field.
                    \"\"\"
                    function $(fn)(obj)::Ptr{Cvoid}
                        p = (obj isa Ptr ? Ptr{Cvoid}(obj) : Ptr{Cvoid}(obj.handle))$(prefix)
                        vptr = unsafe_load(Ptr{Ptr{Int64}}(p))
                        return p + unsafe_load(vptr + $(vboff))
                    end
                    export $(fn)

                    """)
                end
            end
        end
        if !isempty(upcast_chunks)
            push!(struct_chunks, """
            # =============================================================================
            # MI / Virtual-Inheritance Upcast Helpers (base-subobject adjustment)
            # =============================================================================

            """)
            append!(struct_chunks, upcast_chunks)
        end
    end

    if !isempty(deleters)
        push!(managed_chunks, """
        # =============================================================================
        # Managed Types (Auto-Finalizers)
        # =============================================================================

        """)
        
        # Sorted: `deleters` is a Dict, and its order is the hash order of the
        # class names, which changes with the Julia release.
        for s_name in sort!(collect(keys(deleters)))
            deleter = deleters[s_name]
            # Skip: a full idiomatic wrapper will be generated for this class
            (haskey(idiomatic_classes, s_name) ||
             haskey(idiomatic_classes, _sanitize_cpp_type_name(s_name))) && continue
            # Sanitize names
            safe_s_name = _sanitize_cpp_type_name(s_name)
            
            managed_name = "Managed$safe_s_name"
            
            # Use the mangled symbol for ccall (demangled names like ~Foo are invalid Julia symbols)
            deleter_sym = get(deleters_mangled, s_name, deleter)
            
            # Generate mutable struct with finalizer
            push!(managed_chunks, """
            mutable struct $managed_name
                handle::Ptr{$safe_s_name}
                
                function $managed_name(ptr::Ptr{$safe_s_name})
                    if ptr == C_NULL
                        Base.error("Cannot wrap NULL pointer in $managed_name")
                    end
                    obj = new(ptr)
                    finalizer(obj) do x
                        # Call deleter: $deleter_sym(x.handle)
                        ccall((:$deleter_sym, LIBRARY_PATH), Cvoid, (Ptr{$safe_s_name},), x.handle)
                    end
                    return obj
                end
            end

            # Allow passing Managed object to ccall expecting Ptr
            unsafe_convert(::Type{Ptr{$safe_s_name}}, obj::$managed_name) = obj.handle
            # ... and to a call through a thunk, which asks an owning object for an
            # untyped pointer (`JITManager.pointer_arg`): a base class's `this` takes it too
            unsafe_convert(::Type{Ptr{Cvoid}}, obj::$managed_name) = Ptr{Cvoid}(obj.handle)
            
            export $managed_name

            """)
        end
        
        append!(struct_chunks, managed_chunks)
    end

    # =============================================================================
    # IDIOMATIC CLASS WRAPPERS  (High-level Julian API — sits on top of raw FFI)
    # =============================================================================
    # For every C++ class that has:
    #   - ≥1 factory function (pointer-returning) identified in the scan above
    #   - A finalizer that frees what they return (a library deleter function, or
    #     the deleting destructor of a class nothing derives from)
    # we emit:
    #   1. A `mutable struct ClassName { handle::Ptr{Cvoid} }` with inner constructors
    #      that call the raw factory function and attach a GC finalizer.
    #   2. An `unsafe_convert` for transparent use in raw FFI calls.
    #   3. Proxy methods `method(obj::ClassName, ...) = GC.@preserve obj ClassName_method(obj.handle, ...)`.
    if !isempty(idiomatic_classes)
        push!(struct_chunks, """
        # =============================================================================
        # Idiomatic Class Wrappers
        # =============================================================================
        # High-level Julian structs wrapping C++ objects via opaque pointer handles.
        # Memory is automatically managed: when the last reference is dropped,
        # Julia's GC frees the C++ object (each struct's docstring says how).

        """)

        early_types = String[]
        for cls_nm in sort(collect(keys(idiomatic_classes)))
            cls_info    = idiomatic_classes[cls_nm]
            deleter_sym = cls_info["deleter"]
            deleter_doc = escape_string(cls_info["deleter_doc"], '$')
            factories   = cls_info["factories"]
            # A record's field names this class: its type goes ahead of every
            # record and its constructors become outer ones here
            # (`_idiomatic_early_type`). They cannot go with it: a constructor's
            # argument types are evaluated where it is defined, and a factory
            # may take a record, even the one holding the pointer.
            early = cls_nm in idiomatic_named_early

            # One constructor per factory function. A dispatch signature that
            # several factories share is no factory's positional constructor:
            # each of them is called by name (`_factory_ctor_plan`).
            ctor_blocks = String[]
            plan = _factory_ctor_plan(factories, _sanitize_julia_fn)
            let shared = sort!(unique!([e.factory for e in plan if e.tagged]))
                if !isempty(shared)
                    println("  wrap: class $cls_nm has factories sharing a constructor signature, ",
                            "each called by name as $cls_nm(Val(:<factory>), …): ", join(shared, ", "))
                end
            end
            factory_doc = _factory_ctor_doc(cls_nm, plan)

            for e in plan
                fac_jl   = e.factory
                # The early type's own constructor takes this tag (`_IDIOMATIC_ADOPT`).
                e.tagged && fac_jl == "_rb_adopt" &&
                    Base.error("factory `_rb_adopt` of $cls_nm: the name is the factory class's own constructor tag")
                call_sig = join(e.call_args, ", ")
                # A shared signature takes the factory's name first, as a `Val`.
                param_sig = e.tagged ?
                    join(["::Base.Val{:$(fac_jl)}"; e.sig_parts], ", ") :
                    join(e.sig_parts, ", ")

                push!(ctor_blocks, early ? """
                function $cls_nm($param_sig)
                    handle = $fac_jl($call_sig)
                    Ptr{Cvoid}(handle) == C_NULL && Base.error("$cls_nm: constructor returned NULL (allocation failed)")
                    obj = $cls_nm($_IDIOMATIC_ADOPT, Ptr{Cvoid}(handle))
                    finalizer(obj) do o
                        ccall((:$deleter_sym, LIBRARY_PATH), Cvoid, (Ptr{Cvoid},), o.handle)
                    end
                    return obj
                end""" : """
                    function $cls_nm($param_sig)
                        handle = $fac_jl($call_sig)
                        Ptr{Cvoid}(handle) == C_NULL && Base.error("$cls_nm: constructor returned NULL (allocation failed)")
                        obj = new(Ptr{Cvoid}(handle))
                        finalizer(obj) do o
                            ccall((:$deleter_sym, LIBRARY_PATH), Cvoid, (Ptr{Cvoid},), o.handle)
                        end
                        return obj
                    end""")
            end

            # No usable constructors: no type, unless a field names it.
            isempty(ctor_blocks) && !early && continue

            ctors_str = join(ctor_blocks, "\n")

            if early
                push!(early_types, _idiomatic_early_type(cls_nm, deleter_doc; factory_doc = factory_doc))
                push!(struct_chunks, """
                # Constructors of $cls_nm. Its type is defined ahead of the records,
                # since a record's field names it (Idiomatic Class Types).
                $ctors_str

                # Allow passing a $cls_nm directly to raw FFI functions expecting a pointer.
                Base.unsafe_convert(::Type{Ptr{Cvoid}}, obj::$cls_nm) = obj.handle
                Base.unsafe_convert(::Type{Ptr{$cls_nm}}, obj::$cls_nm) = Ptr{$cls_nm}(obj.handle)

                """)
                push!(exports, cls_nm)
                continue
            end

            push!(struct_chunks, """
            \"\"\"
                $cls_nm

            Idiomatic Julia wrapper for C++ class `$cls_nm`.
            Memory is automatically managed: when Julia's GC collects this object,
            its finalizer $deleter_doc.$factory_doc
            \"\"\"
            mutable struct $cls_nm
                handle::Ptr{Cvoid}
            $ctors_str
            end

            # Allow passing a $cls_nm directly to raw FFI functions expecting a pointer.
            Base.unsafe_convert(::Type{Ptr{Cvoid}}, obj::$cls_nm) = obj.handle
            Base.unsafe_convert(::Type{Ptr{$cls_nm}}, obj::$cls_nm) = Ptr{$cls_nm}(obj.handle)

            """)

            push!(exports, cls_nm)
        end
        if !isempty(early_types)
            idiomatic_early_slot > 0 ||
                Base.error("a record names a factory class, but the struct section has no slot for its type")
            struct_chunks[idiomatic_early_slot] = """
            # =============================================================================
            # Idiomatic Class Types (ahead of the records whose fields name them)
            # =============================================================================

            """ * join(early_types)
        end
    end

    # =============================================================================
    # METHOD PROXIES  (Julian dispatch on idiomatic structs)
    # =============================================================================
    # For each instance method of an idiomatic class, emit a Julian wrapper:
    #   method_name(obj::ClassName, other_args...) = GC.@preserve obj ClassName_method_name(obj.handle, ...)
    # This gives multi-dispatch semantics: the same method name works for multiple
    # classes (e.g. area(c::Circle) and area(r::Rectangle) dispatch independently).
    if !isempty(idiomatic_classes)
        for cls_nm in sort(collect(keys(idiomatic_classes)))
            methods = get(class_methods, cls_nm, Any[])
            isempty(methods) && continue

            push!(struct_chunks, """
            # Method proxies for $cls_nm
            """)

            # One proxy per dispatch signature. Keyed by the text, parameter
            # names included, two overloads one Julia signature stands for
            # (`f(long a)`, `f(long long b)` on LP64; `f(Foo*)`, `f(Bar*)`,
            # both `::Any` here) were one method defined twice. Proxies of one
            # signature calling one raw wrapper are one proxy (the raw wrapper
            # dispatches); calling different ones (a `[wrap.rename]` of one
            # overload) is no proxy at all, named, never a pick.
            proxies = Tuple{String,String,String,String}[]   # (key, raw_jl, text, proxy_nm)
            seen_proxy_sigs = Set{String}()

            for meth in methods
                meth_name = meth["name"]

                # The raw Julia wrapper for ClassName::method is named ClassName_method
                raw_jl = _sanitize_julia_fn("$(cls_nm)_$(meth_name)",
                                            String(get(meth, "mangled", "")))

                # Proxy method name: just the method name part (multi-dispatch via
                # Julia types). Not a binding of the symbol, so no rename applies.
                proxy_nm = _julia_function_name(meth_name; rename = false)

                # Build parameter list, skipping the implicit 'this' pointer
                meth_params = get(meth, "parameters", Any[])
                sig_parts   = String[]
                call_args   = String["obj.handle"]  # pass the raw C pointer, not the Julia wrapper
                used_names  = Set{String}()

                for (i, p) in enumerate(meth_params)
                    p_nm_raw = get(p, "name", "arg$(i)")
                    lowercase(p_nm_raw) == "this" && continue
                    p_nm = make_cpp_identifier(p_nm_raw)
                    while p_nm in used_names; p_nm = "$(p_nm)_$(i)"; end
                    push!(used_names, p_nm)
                    p_jt = get(p, "julia_type", "Any")
                    # Accept Any for pointer params so idiomatic wrappers pass transparently
                    dispatch_t = startswith(p_jt, "Ptr{") ? "Any" : p_jt
                    push!(sig_parts, "$p_nm::$dispatch_t")
                    push!(call_args, p_nm)
                end

                other_sig    = join(sig_parts, ", ")
                call_sig_str = join(call_args, ", ")

                # Deduplicate (same method may appear twice due to DWARF duplicate
                # entries), by dispatch signature (`_concrete_sig_type`).
                proxy_key = "$(proxy_nm)($(cls_nm)," *
                    join((_concrete_sig_type(String(last(split(s, "::"; limit = 2)))) for s in sig_parts), ",") * ")"

                full_sig = isempty(other_sig) ? "obj::$cls_nm" : "obj::$cls_nm, $other_sig"

                # `obj` is dead once its handle is read, and the raw wrapper
                # allocates before the call reaches C++ (a Tier 2 invoke packs
                # its slots), so a collection there ran `obj`'s finalizer, the
                # deleting destructor, on the object the call was about to use
                # (`area(Rect(2, 3))`: a segfault in the thunk, 2026-11-10).
                # The proxy keeps the owner alive across the call.
                push!(proxies, (proxy_key, raw_jl, """
                $proxy_nm($full_sig) = GC.@preserve obj $raw_jl($call_sig_str)
                """, proxy_nm))
            end

            targets = Dict{String,Set{String}}()
            for (k, raw, _, _) in proxies
                push!(get!(targets, k, Set{String}()), raw)
            end
            split_keys = sort!([k for (k, t) in targets if length(t) > 1])
            isempty(split_keys) ||
                println("  wrap: no method proxy of $cls_nm for $(length(split_keys)) signature(s) its overloads ",
                        "share under different Julia names (call those directly): ",
                        join(("$k → " * join(sort!(collect(targets[k])), ", ") for k in split_keys), "; "))
            for (k, _, text, proxy_nm) in proxies
                (k in seen_proxy_sigs || length(targets[k]) > 1) && continue
                push!(seen_proxy_sigs, k)
                push!(struct_chunks, text)
                push!(exports, proxy_nm)
            end

            push!(struct_chunks, "\n")
        end
    end



    # Classes of this library an earlier library of the graph reached by
    # declaration alone (`_earlier_placeholders`): its owning types convert to
    # that library's placeholder pointer type.
    let ph = get(metadata, "earlier_placeholders", Dict{String,Any}())
        if !isempty(ph)
            handle_types = Dict{String,Vector{String}}()
            for (k, _) in ph
                hts = String[]
                jn = _sanitize_cpp_type_name(String(k))
                haskey(idiomatic_classes, jn) && push!(hts, jn)
                (haskey(deleters, String(k)) && !haskey(idiomatic_classes, jn)) && push!(hts, "Managed" * jn)
                isempty(hts) || (handle_types[String(k)] = hts)
            end
            chunk = _placeholder_conversions_chunk(ph, handle_types)
            isempty(chunk) || push!(struct_chunks, chunk)
        end
    end

    # Track exported function names
    # exports already initialized at top

    # Dead-thunk elimination: collect mangled names of functions that actually
    # need MLIR thunks (Tier 2 dispatch). Written as thunk_manifest.json so
    # JITManager can skip generating thunks for Tier 3 (ccall) functions.
    needed_function_thunks = Set{String}()

    # Constant name => the mangled symbol that owns it, for the AOT thunk-address
    # slots. Keyed on `mangled` for the same reason `_slice_const_name!` is:
    # `julia_name` is not injective over it, and two functions sharing one slot
    # would each call whichever thunk `__init__` resolved last.
    aot_fptr_taken = Dict{String,String}()

    # Records whose Julia type puts bytes where C has a float or double: the
    # byte blobs, and field-emitted ones (see the dispatch decision).
    _blob_names = @isdefined(blob_struct_names) ? union(blob_struct_names, sse_bytes_struct_names) :
                  Set{String}()

    # A parameter's `ccall` type, from its metadata julia_type. The parameter
    # loop below spells each parameter so, and a callback's `@cfunction`
    # example spells the types of its signature so (`_callback_param_type`).
    function _ccall_param_type(julia_type::AbstractString, c_type_name::AbstractString)::String
        # A reference or pointer to a type Julia cannot hold (`long double&`,
        # `unsigned __int128&`) arrives as `Ref{Any}`/`Ptr{Any}` and is the
        # address of a caller's buffer (`_pointee_safe_type`; Boost.Charconv's
        # `long double` from_chars, 33 signatures in that wrapper, 2026-11-01).
        julia_type = _pointee_safe_type(julia_type)
        # A `char` by value is the host's `char` (`_char_param_ctype`).
        julia_type = _char_param_ctype(julia_type, c_type_name)
        # Replace blocklisted internal types (e.g. _IO_FILE) with Cvoid
        let bm = match(r"^(Ref|Ptr)\{(.*)\}$", julia_type)
            if bm !== nothing && String(bm.captures[2]) in _INTERNAL_TYPE_BLOCKLIST
                julia_type = "Ptr{Cvoid}"
            end
        end

        # Sanitize C++ template types in parameter julia_type (e.g. "Ref{allocator<int> >}")
        if occursin(r"[<>]", julia_type)
            m = match(r"^(Ref|Ptr)\{(.*)\}$", julia_type)
            if m !== nothing
                wrapper_kw, inner = m.captures
                # If inner type is an STL-internal (allocator<>, char_traits<>, etc.),
                # fall back to Ptr{Cvoid} — the type is opaque and not user-accessible.
                if _stl_internal(String(inner))
                    julia_type = "Ptr{Cvoid}"
                else
                    inner_safe = _sanitize_cpp_type_name(inner)
                    julia_type = "$wrapper_kw{$inner_safe}"
                end
            else
                julia_type = _sanitize_cpp_type_name(julia_type)
            end
        end
        # Also sanitize namespace separators (::) and other non-identifier chars
        # that can appear in Ptr{namespace::type} style references from DWARF
        # Allow spaces (e.g. "NTuple{4, UInt8}") to pass through unsanitized
        if occursin(r"::|[^A-Za-z0-9_{},\[\] ]", julia_type) && julia_type != "Any"
            m = match(r"^(Ref|Ptr)\{(.*)\}$", julia_type)
            if m !== nothing
                wrapper_kw, inner = m.captures
                inner_safe = _sanitize_cpp_type_name(inner)
                julia_type = isempty(inner_safe) ? "Ptr{Cvoid}" : "$wrapper_kw{$inner_safe}"
            else
                julia_type = _sanitize_cpp_type_name(julia_type)
            end
        end
        julia_type = _record_name_as_emitted(julia_type, emitted_record_names)

        # Determine the actual C type for ccall
        # If julia_type is "Any" but c_type has a name, it's likely a struct
        return if julia_type == "Any" && !isempty(c_type_name) && c_type_name in struct_types
            String(c_type_name)  # Use the struct name directly
        else
            julia_type
        end
    end
    _callback_type = _callback_param_type(_ccall_param_type, dwarf_structs)

    # Overloads that differ only in an integer parameter's width. An integer
    # parameter is annotated `::Integer` so a caller can pass any Julia
    # integer, which makes `f(int)` and `f(long long)` one Julia signature; the
    # dedup then keeps the last definition and the other C++ entry point is
    # unreachable (Boost.Charconv's integer `to_chars` kept only the `unsigned
    # long long` one, 2026-10-31). Where two or more functions would share a
    # name and a signature, the parameters whose C integer type differs
    # between them keep that type (`value::Cint`, `value::Clonglong`), so
    # dispatch tells them apart; the parameters the group agrees on stay
    # `::Integer`. Positions index `func["parameters"]`, before any `this`.
    # The same for a pointer's target and for an enum beside a pointer: those
    # positions take `_strict_param_spelling`, with the names this module
    # emits, the owning types that convert to each record's pointer, and the
    # enums it emitted, which the loop below spells the same way.
    _sig_type_bound(t) = _type_name_bound(t, defined_struct_names)
    _sig_is_enum(t) = haskey(emitted_enum_base, String(t))
    _sig_owners(t) = begin
        jn = String(t)
        out = String[]
        haskey(idiomatic_classes, jn) && push!(out, jn)
        any(k -> _sanitize_cpp_type_name(k) == jn && !haskey(idiomatic_classes, jn), keys(deleters)) &&
            push!(out, "Managed" * jn)
        out
    end
    # An rvalue-reference overload beside its lvalue twin is not emitted at
    # all (`_rvalue_shadowed`), so it takes no part in the prediction either.
    rvalue_shadowed = _rvalue_shadowed(functions)
    # The member functions with a vtable slot, by symbol: the `virtual` rows
    # the build writes for every member a library record declares
    # (`record_abi`'s `methods`). Metadata without the rows has none, and a
    # `noexcept` virtual keeps its `ccall` until a rebuild.
    virtual_symbols = Set{String}(String(get(row, "symbol", ""))
                                  for facts in values(record_abi) if facts isa AbstractDict
                                  for row in get(facts, "methods", Any[])
                                  if row isa AbstractDict && get(row, "virtual", false) == true)
    # One constructor and one destructor binding per class, the complete-object
    # variant (`_structor_bindings`): the other variant of a member is not
    # emitted, and one the library cannot back is a refusal with its declared
    # parameters, which is how the prediction sees it too.
    (structor_skipped, structor_refused) = _structor_bindings(functions, dwarf_structs)
    _declared_structor(f) = !haskey(structor_refused, String(get(f, "mangled", ""))) ? f :
        merge(f, Dict{String,Any}("parameters" => _structor_parameters(
            f, startswith(String(get(f, "name", "")), "~") ? :dtor : :ctor)[1]))
    let classes = sort!(unique(String[String(get(f, "class", "")) for f in functions
                                      if haskey(structor_refused, String(get(f, "mangled", "")))]))
        isempty(classes) || println("  wrap: $(length(structor_refused)) constructor/destructor wrapper(s) of ",
                                    "$(length(classes)) class(es) are refusals (a class with a virtual base ",
                                    "whose complete-object variant the library lacks, or metadata to rebuild): ",
                                    join(first(classes, 8), ", "), length(classes) > 8 ? ", …" : "")
    end
    # ...and no constructor of an abstract class is bound at all
    # (`_abstract_constructor_refusals!`).
    let classes = _abstract_constructor_refusals!(structor_refused, functions, structor_skipped, record_abi)
        isempty(classes) || println("  wrap: the constructor wrapper(s) of $(length(classes)) abstract class(es) are ",
                                    "refusals (C++ builds no object of a class with a pure virtual left): ",
                                    join(first(classes, 8), ", "), length(classes) > 8 ? ", …" : "")
    end
    # The pure virtual member functions refused at emission, for the report.
    vtable_refusals = String[]
    strict_int_positions = _integer_width_overloads(
        Any[_declared_structor(f) for f in functions
            if !(String(get(f, "mangled", "")) in rvalue_shadowed) &&
               !(String(get(f, "mangled", "")) in structor_skipped)],
        struct_types, _ccall_param_type;
        bound = _sig_type_bound, owners = _sig_owners, is_enum = _sig_is_enum,
        string_param = c -> _string_param_plan(c, record_abi) isa Tuple,
        object_address = c -> JLCSIRGenerator.FunctionGen.takes_object_address(c, dwarf_structs, record_abi))

    # What a pointer or reference return may be typed with (`_typed_pointer_return`).
    typed_return_names = _typed_return_names(join(struct_chunks))
    union!(typed_return_names, _imported_value_type_names(dwarf_structs, _sanitize_cpp_type_name))
    # ...and with an enum this module emits: `Color*` and `Color&` are `Ptr{Color}`.
    union!(typed_return_names, _pending_enum_type_names(enum_chunks))

    for func in functions
        func_name = func["name"]
        mangled = func["mangled"]
        demangled = func["demangled"]

        # A move overload beside its copy overload: one Julia signature, and
        # the lvalue one is what a Julia caller means (`_rvalue_shadowed`).
        mangled in rvalue_shadowed && continue
        # The base-object constructor or destructor beside the complete-object
        # one: the same Julia name, and never the one a caller means.
        mangled in structor_skipped && continue

        # =========================================================
        # TIERED DISPATCH DECISION
        # =========================================================

        # Determine if we should use MLIR or ccall.
        # Tier 2 is decided by is_ccall_safe, and by what ccall would carry:
        # the Julia type a record crosses by value must be its C layout. A
        # record `_prove_julia_layout` could not reproduce is a byte blob, and
        # at 16 bytes or less a blob claims INTEGER for every SysV eightbyte
        # where the record may have SSE ones. So does a field written as bytes
        # over a float or double. The thunk lays the record out from DWARF.
        # (Above 16 bytes the record crosses in memory on both ABIs and the
        # bytes are exact. Unions and bitfield records, the `mutable` ones,
        # never reach here: `_record_natural_layout` rejects them.)
        # A `std::string` the wrapper builds for a parameter (`_string_param_plan`)
        # is passed by address and ended after the call in the Tier 2 body; a
        # `noexcept` function taking one goes there too, so the body is one.
        use_mlir_dispatch = !is_ccall_safe(func, dwarf_structs; record_abi=record_abi) ||
                            _crosses_small_blob(func, dwarf_structs, record_abi, _blob_names) ||
                            any(p -> _string_param_plan(String(get(p, "c_type", "")), record_abi) isa Tuple,
                                func["parameters"])
        # A virtual member function is reached through the object's vtable,
        # and only a thunk does that (`jlcs.vcall`). A `ccall` names the
        # symbol, which is `p->Class::f()`: a `noexcept` virtual was routed
        # there, and `Base_g(derived)` ran `Base::g`.
        use_mlir_dispatch = use_mlir_dispatch || mangled in virtual_symbols
        # A pure virtual member function has no symbol at all
        # (`Compiler._is_vtable_only`): its thunk is a `jlcs.vcall` and there
        # is nothing a `ccall` could name.
        vtable_only = _is_vtable_only(func)
        use_mlir_dispatch = use_mlir_dispatch || vtable_only

        # BUG FIX: Make copies to allow modification (injecting 'this', refining types) without affecting metadata
        params = copy(func["parameters"])
        return_type = copy(func["return_type"])
        # A refused constructor or destructor keeps its name and its declared
        # parameters (no VTT), and raises the reason (`_structor_bindings`).
        structor_refusal = get(structor_refused, mangled, nothing)
        structor_refusal === nothing || (params = _declared_structor(func)["parameters"])

        is_method = get(func, "is_method", false)
        is_vararg = get(func, "is_vararg", false)
        class_name = get(func, "class", "")

        # No DIE describes this symbol, and it is no alias of a function that
        # has one: the signature in hand is a guess from the name, so the
        # function is refused where it is defined (`_lacks_debug_info`), under
        # the name the loop below would have given it.
        if _lacks_debug_info(func)
            stub_name = _julia_function_name(
                is_method && !isempty(class_name) ? "$(class_name)_$(func_name)" : func_name, mangled)
            push!(func_chunks, _no_debug_info_stub(stub_name, mangled, demangled, metadata, generate_docs))
            push!(exports, stub_name)
            continue
        end

        # Constructors are emitted. The old `func_name == class_name` skip
        # only fired at global scope (`T::T`, where `class` is the bare name)
        # and silently dropped those; `ns::T::T` was emitted because `class`
        # is the qualified prefix. Both go through the same naming
        # (`$(class)_$(name)`), so a global constructor is `T_T`, not a
        # redeclaration of the struct `T`.

        # Inject missing 'this' pointer for methods.
        # A definition DIE that recorded `has_object_pointer == false` is a
        # static member: the class is a real aggregate, so the heuristic below
        # would invent a `this` and the first real argument would be read as
        # the receiver. Declaration DIEs omit the attribute, so a missing key
        # must not take this branch.
        hop = get(func, "has_object_pointer", nothing)
        this_injected = false
        if is_method && !isempty(class_name) && hop !== false
            # Check if 'this' is already present (it should be first)
            has_this = !isempty(params) && (params[1]["name"] == "this")
            
            if !has_this
                this_param = _cpp_this_param(class_name, func_name, struct_types)
                if this_param === nothing && hop === true
                    # The definition DIE says there is a receiver, and the
                    # class has no emitted Julia type. `Ptr{Cvoid}` is the
                    # same width; leaving it off would disagree with the thunk.
                    this_param = Dict{String,Any}(
                        "name" => "this",
                        "c_type" => String(class_name) * "*",
                        "julia_type" => "Ptr{Cvoid}",
                        "position" => 0,
                        "is_synthesized" => true,
                    )
                end
                this_param === nothing || (pushfirst!(params, this_param); this_injected = true)
            end
        end

        # Replace blocklisted internal types in return type (e.g. _IO_FILE -> Cvoid)
        let bm = match(r"^(Ref|Ptr)\{(.*)\}$", get(return_type, "julia_type", ""))
            if bm !== nothing && String(bm.captures[2]) in _INTERNAL_TYPE_BLOCKLIST
                return_type["julia_type"] = "Ptr{Cvoid}"
            end
        end

        # BUG FIX: Refine return types (Ptr{Cvoid} -> Ptr{Struct})
        # If c_type indicates a pointer to a known struct, but julia_type is generic Ptr{Cvoid}, fix it.
        c_ret = get(return_type, "c_type", "")
        j_ret = get(return_type, "julia_type", "")
        
        if (j_ret == "Ptr{Cvoid}" || j_ret == "Any") && endswith(c_ret, "*")
             base_c = strip(c_ret[1:end-1])
             # Remove const/struct/class prefixes
             base_c = replace(base_c, r"^(const\s+|struct\s+|class\s+)+" => "")
             base_c = strip(base_c)
             
             if base_c in struct_types && !(base_c in _INTERNAL_TYPE_BLOCKLIST)
                 # Sanitize
                 safe_base = _sanitize_cpp_type_name(base_c)
                 return_type["julia_type"] = "Ptr{$safe_base}"
             end
        end

        # A pointer or reference return the step above left untyped takes the
        # type its declaration names (`_typed_pointer_return`): `ImGui::GetIO`
        # (`ImGuiIO&`) is `Ptr{ImGuiIO}`. Only the type: such a return never
        # becomes a `Managed` owner (`newly_typed_return` below). A reference
        # never transfers ownership, and nothing here says a pointer does.
        newly_typed_return = false
        let jt = String(get(return_type, "julia_type", "Cvoid"))
            if occursin(r"^(Ptr\{)*(Cvoid|Any)\}*$", jt)
                typed = _typed_pointer_return(String(get(return_type, "c_type", "")),
                                              typed_return_names, _sanitize_cpp_type_name)
                if typed !== nothing
                    return_type["julia_type"] = typed
                    newly_typed_return = true
                end
            end
        end

        # BUG FIX: Sanitize return types that are template instantiations (e.g. Box<double> -> Box_double)
        # This handles by-value returns of template types
        ret_type_str = get(return_type, "julia_type", "Cvoid")
        if occursin(r"[<>]", ret_type_str) && !startswith(ret_type_str, "Ptr{") && !startswith(ret_type_str, "Ref{")
            # If the raw return type is an STL container, use NTuple with the
            # size the DWARF gave this instantiation (`stl_value_bytes`).
            stl_sz = JLCSIRGenerator.FunctionGen.stl_value_bytes(ret_type_str, record_abi)
            if stl_sz > 0
                return_type["julia_type"] = "NTuple{$stl_sz, UInt8}"
            elseif _stl_internal(ret_type_str)
                return_type["julia_type"] = "Ptr{Cvoid}"
            else
                safe_ret = _sanitize_cpp_type_name(ret_type_str)
                return_type["julia_type"] = safe_ret
            end
        end
        return_type["julia_type"] = _record_name_as_emitted(
            String(get(return_type, "julia_type", "Cvoid")), emitted_record_names)

        # =========================================================
        # VARARGS INTERCEPTION
        # =========================================================
        # Never for a pure virtual: the interception is a `@ccall` by name,
        # and the build makes no variadic row for one. BRANCH 1 refuses it.
        if is_vararg && !vtable_only
            # Sanitize function name for Julia (same logic as below)
            va_julia_name = func_name
            if is_method && !isempty(class_name)
                va_julia_name = "$(class_name)_$(func_name)"
            end
            va_julia_name = _julia_function_name(va_julia_name, mangled)

            overloads = get(config.wrap.varargs_overloads, func_name, Vector{Vector{String}}())
            if isempty(overloads)
                @warn "Varargs function '$func_name' has no overloads configured in [wrap.varargs]. Generating base wrapper only."
            end

            # A `long double` crossing is the same trap the plain path emits
            # below; this branch used to `continue` past it.
            va_x87 = _x87_crossings(func)
            # ...and so is a fixed parameter the type map does not know.
            va_fixed = [p for p in params if get(p, "name", "") != "varargs..."]
            va_unmapped = _unmapped_crossings(va_fixed, String[String(get(p, "julia_type", "Any")) for p in va_fixed])
            va_code, va_exports = generate_vararg_wrappers(
                func_name, mangled, va_julia_name,
                params, return_type, overloads,
                generate_docs, demangled, :cpp,
                cstring_free=get(config.wrap.cstring_owned, func_name, ""),
                conflicts=JLCSIRGenerator.FunctionGen.conflict_crossings(func, dwarf_structs),
                trap=!isempty(va_x87) ? (n -> _x87_trap_body(n, va_x87)) :
                     !isempty(va_unmapped) ? (n -> _unmapped_trap_body(n, va_unmapped)) : nothing
            )
            push!(func_chunks, va_code)
            append!(exports, va_exports)
            continue  # Skip normal wrapper generation
        end

        # Build parameter list with ergonomic types
        param_names = String[]
        param_types = String[]  # C types for ccall
        julia_param_types = String[]  # Julia types for function signature (may differ)
        needs_conversion = Bool[]
        string_plans = Any[]          # per parameter: nothing, a plan Tuple, or a reason String
        pointer_types = Union{Nothing,String}[]   # per parameter: what `pointer_arg` is told (`_pointer_arg_type`)
        cstring_args = Union{Nothing,String}[]    # per parameter: the `ccall` path's string conversion (`_cstring_param_arg`)

        for (i, param) in enumerate(params)
            # Sanitize parameter name (e.g., avoid 'end', 'function' keywords)
            safe_name = make_cpp_identifier(param["name"])
            # Ensure unique parameter names (duplicate names cause Julia syntax errors)
            if safe_name in param_names
                safe_name = "$(safe_name)_$(i)"
            end
            push!(param_names, safe_name)
            actual_c_type = _ccall_param_type(param["julia_type"], get(param, "c_type", ""))
            push!(param_types, actual_c_type)

            # The ergonomic Julia type: any Integer for a C integer, `Any` for
            # a pointer (so a `Managed` object, a `Ref`, a `String` pass through
            # `Base.unsafe_convert`) and for a name this module never emitted
            # (an enum, an STL internal; an unbound name in a signature kills
            # the module at include, see `_type_name_bound`). At a position an
            # overload set disagrees on (`strict_int_positions`) the type is
            # the one that tells the overloads apart (`_strict_param_spelling`):
            # the C width, the pointer's target, the enum.
            raw_i = i - (this_injected ? 1 : 0)
            strict = raw_i in get(strict_int_positions, mangled, _NO_POSITIONS)
            # A `std::string` taken by `const&` or by value: a Julia string, or
            # the address of an existing object (`_STD_STRING_PARAM`), the
            # spelling the prediction above used for this position.
            splan = _string_param_plan(String(get(param, "c_type", "")), record_abi)
            push!(string_plans, splan)
            # A container taken by value crosses by address: spelled as a
            # pointer (`_by_address_ctype`), as the prediction spelled it.
            # So does any other record the thunk takes by the caller's
            # object's address (`takes_object_address`).
            object_address = JLCSIRGenerator.FunctionGen.takes_object_address(
                String(get(param, "c_type", "")), dwarf_structs, record_abi)
            sig_c_type = _by_address_ctype(actual_c_type, String(get(param, "c_type", "")); object_address)
            push!(julia_param_types,
                  splan isa Tuple ? _STD_STRING_PARAM :
                  strict ? _strict_param_spelling(sig_c_type; bound = _sig_type_bound,
                                                  owners = _sig_owners, is_enum = _sig_is_enum) :
                           _relaxed_param_kind(sig_c_type, _sig_type_bound))
            push!(needs_conversion, actual_c_type in _INTEGER_CTYPES)
            push!(cstring_args, _cstring_param_arg(safe_name, String(get(param, "c_type", "")), actual_c_type))
            # What the Tier 2 body converts a pointer argument to. A `std::string`
            # the wrapper builds is passed by the address `string_arg` gives.
            push!(pointer_types, splan isa Tuple ? nothing :
                  _pointer_arg_type(actual_c_type, String(get(param, "c_type", ""));
                                    resolves = t -> _sig_type_bound(t) || _sig_is_enum(t), record_abi,
                                    object_address))
        end

        # Julia function name (avoid conflicts and sanitize)
        julia_name = func_name
        if is_method && !isempty(class_name)
            julia_name = "$(class_name)_$(func_name)"
        end

        # One derivation for every function name in both generators — see
        # `_julia_function_name` (Wrapper/Utils.jl).
        julia_name = _julia_function_name(julia_name, mangled)

        # Build function signature using ergonomic Julia types.
        #
        # A parameter annotation naming a type this module never emitted is fatal:
        # argument types are resolved EAGERLY at method definition, so an unbound
        # name raises UndefVarError at include and kills the entire module, not just
        # that one call. `julia_param_types` comes from DWARF, which describes plenty
        # of types the generator deliberately declines to emit (STL internals,
        # blocklisted system types), so intent and output disagree here exactly as
        # they do for the union accessors above — gate on the EMITTED set.
        #
        # Latent until DW_AT_specification merging recovered real parameter lists
        # (2026-08-09): `pugi::xml_attribute::set_name(std::basic_string_view<char,
        # std::char_traits<char>>)` had its only parameter silently dropped as an
        # unnamed declaration-DIE child, so the undeclared type never reached a
        # signature. With the merge it does — 44 uses, wrapper dead at load.
        #
        # `Any` is ABI-neutral for these: Tier 2 marshals through
        # JITManager.invoke, which packs by runtime value, and `this::Any` is
        # already the norm. Only bare identifiers are rewritten — a composed type
        # (`Ref{ImVec2}`, `Ptr{Cvoid}`) resolves through its parts and is left
        # alone. `julia_param_types` is already spelled that way
        # (`_relaxed_param_kind`, `_strict_param_spelling`); the check here is
        # the guard that keeps it so.
        param_sig_parts = String[]
        for (name, typ) in zip(param_names, julia_param_types)
            if name == "varargs..."
                push!(param_sig_parts, name)
            else
                safe_typ = (_sig_type_bound(String(typ)) || _sig_is_enum(String(typ))) ? typ : "Any"
                push!(param_sig_parts, "$name::$safe_typ")
            end
        end
        param_sig = join(param_sig_parts, ", ")

        # Documentation
        doc_comment = ""
        if generate_docs
            # Build detailed argument documentation, detecting function pointers
            arg_docs = String[]
            callback_docs = String[]

            for param in params
                _param_docs!(arg_docs, callback_docs, param, metadata, registry, _callback_type)
            end

            # Advertise what the function actually returns. Every `char*` return
            # is emitted through _cstring_wrapper_pair on BOTH dispatch paths, so
            # the policy type is unconditional here — the raw pointer is reachable
            # as `<name>_ptr`, which carries its own docstring. (The C generator
            # has always done this; the C++ side did not, so all 33 of llamacpp's
            # Union-returning functions advertised `Cstring`.)
            doc_ret = String(return_type["julia_type"])
            doc_ret == "Cstring" && (doc_ret = "Union{String,Nothing}")

            # Build the docstring
            doc_parts = """
            \"\"\"
                $julia_name($param_sig) -> $doc_ret

            Wrapper for `$demangled`$(_deleting_destructor_doc(mangled, julia_name, class_name in d0_only_classes))$(vtable_only ? _PURE_VIRTUAL_DOC : "")

            # Arguments
            $(join(arg_docs, "\n"))

            # Returns
            - `$doc_ret`"""

            # Add callback documentation if any
            if !isempty(callback_docs)
                doc_parts *= "\n\n            # Callback Signatures\n"
                doc_parts *= join(callback_docs, "\n\n")
            end

            doc_parts *= """


            # Metadata
            - Mangled symbol: `$mangled`
            \"\"\"
            """

            # Chomp the heredoc's trailing newline — see the identical note in
            # GeneratorC.jl. A blank line between the docstring and `function`
            # detaches it, silently, and the function ships undocumented.
            doc_comment = chomp(doc_parts)
        end

        if structor_refusal !== nothing
            push!(func_chunks, """
            $doc_comment
            function $julia_name($param_sig)
                Base.error($(repr("ABI Safety Trap: cannot call '$julia_name' ($demangled). " * structor_refusal)))
            end
            """)
            push!(exports, julia_name)
            continue
        end

        # =========================================================
        # BRANCH 1: MLIR DISPATCH (Robust Path)
        # =========================================================
        if use_mlir_dispatch
            # One verdict for both ends: the thunk generator asks the same
            # function about the same metadata, and emits no thunk when this
            # says trap. A by-value record neither side can type (a declared-only
            # stub, `std::string_view`, a derived class with no members of its
            # own) used to become a pointer in the thunk — plausible wrong
            # answers, NaN, or a segfault — so the call is refused where it is
            # made, with the reason, and the rest of the wrapper still loads.
            # A record whose name several types share (`conflict_trap`) is
            # refused the same way: the thunk would lay out whichever one the
            # metadata kept.
            crossings = JLCSIRGenerator.FunctionGen.thunk_crossings(func, dwarf_structs, record_abi; class_raii)
            if crossings.trap !== nothing
                trap_msg = "ABI Safety Trap: cannot call '$julia_name' ($demangled)."
                if crossings.layout_trap !== nothing
                    trap_msg *= " It $(crossings.layout_trap), and that record's layout is not in this " *
                        "library's DWARF, so no call could pass it correctly. A type " *
                        "only declared here (header-only, constructor homing) is " *
                        "recovered by building with `-Xclang -fno-use-ctor-homing` or " *
                        "`-fstandalone-debug` in [compile] flags; otherwise keep the " *
                        "function out with [wrap] exclude_symbols."
                end
                # The convention's own refusal: the record's layout is known,
                # and SysV still has no register the thunk could read it from,
                # or its class cannot be told from DWARF.
                if crossings.abi_trap !== nothing
                    trap_msg *= " It $(crossings.abi_trap), so no thunk could call it. Keep the " *
                        "function out with [wrap] exclude_symbols, or wrap a shim that takes " *
                        "the record by pointer or returns a `double`."
                end
                if crossings.conflict_trap !== nothing
                    trap_msg *= " It $(crossings.conflict_trap). " * _CONFLICT_TRAP_NOTE
                end
                push!(func_chunks, """
                $doc_comment
                function $julia_name($param_sig)
                    Base.error($(repr(trap_msg)))
                end
                """)
                push!(exports, julia_name)
                continue
            end
            # A pure virtual member function is bound only where the thunk
            # generator will write its vtable call: with no thunk there is no
            # symbol to fall back on, so the generator is asked
            # (`vtable_only_refusal`) and its reason is the refusal.
            if vtable_only
                why = JLCSIRGenerator.vtable_only_refusal(func, dwarf_structs, record_abi; class_raii)
                if why !== nothing
                    push!(vtable_refusals, String(demangled))
                    push!(func_chunks, """
                    $doc_comment
                    function $julia_name($param_sig)
                        Base.error($(repr("ABI Safety Trap: cannot call '$julia_name' ($demangled). " *
                                          uppercasefirst(why) * ". Call the override through its own class's " *
                                          "wrapper, or keep the function out with [wrap] exclude_symbols.")))
                    end
                    """)
                    push!(exports, julia_name)
                    continue
                end
            end
            # An empty record returned by value. The thunk is void on both
            # ABIs (SysV's callee is void; Win64's returns iN and the thunk
            # discards it), and the value is the type's only instance.
            empty_ret = crossings.ret[1] === :ignore || crossings.ret[1] === :regpad

            # Build argument list for invoke: every scalar at its C width.
            # A `std::string` built for a parameter: `_rb_sN` holds the object
            # (`StdStrings.string_arg`, before the call), its address is the
            # argument, and `release!` ends it in a `finally` around the call,
            # whether the call returns or throws. A reason (old metadata, a
            # wide string) is written beside the call and the stand-in stays.
            string_setup = ""
            string_syms = String[]
            invoke_names = copy(param_names)
            for (si, splan) in enumerate(string_plans)
                if splan isa String
                    string_setup *= "# $(splan)\n    "
                elseif splan isa Tuple
                    sym = "_rb_s$(si)"
                    push!(string_syms, sym)
                    string_setup *= "$sym = RepliBuild.StdStrings.string_arg($(param_names[si]), $(splan[2]), LIBRARY_PATH, " *
                                    "$(repr(splan[3])), $(splan[4]), $(repr(splan[5])), $(repr(julia_name)))\n    "
                    invoke_names[si] = "RepliBuild.StdStrings.address($sym)"
                end
            end
            _with_strings(call) = isempty(string_syms) ? call :
                "(try GC.@preserve $(join(string_syms, " ")) $call finally " *
                join(("RepliBuild.StdStrings.release!($s)" for s in string_syms), "; ") * " end)"
            invoke_args = join(_tier2_invoke_args(invoke_names, param_types, needs_conversion,
                                                  emitted_enum_base, pointer_types), ", ")
            jit_ret_type = return_type["julia_type"]
            jit_c_ret = get(return_type, "c_type", "void")

            # Resolve "Any" return type to the Julia struct if we have a DWARF definition
            # This is critical: invoke(name, Any, ...) creates Ref{Any}() which corrupts
            # memory when the JIT writes raw struct bytes into it.
            if jit_ret_type == "Any" && jit_c_ret != "void"
                # Every lookup reads the spelling with its top-level cv removed,
                # as `by_value_crossing` does. DWARF gives
                # `key_value_pair::key()` the return `const basic_string_view<char>`
                # and keys the record without the `const`; looked up as spelled,
                # the return stayed `Any` and the JIT refused the call, while the
                # same record crossed as a parameter (Boost.JSON, 2026-11-01).
                _bare_ret = JLCSIRGenerator.FunctionGen._strip_top_cv(jit_c_ret)
                # Enum returns first: ABI integers keyed __enum__<name> in DWARF,
                # so every struct branch below misses them and the Any sentinel
                # survived into invoke() — whose Ref{Any}() sret path throws
                # UndefRefError (found live: tinyxml2 XMLDocument::Parse → XMLError).
                # The generated @enum type is a Julia primitive type, so
                # _invoke_call takes the scalar ccall path with it.
                stl_size = _stl_internal(_bare_ret) ? JLCSIRGenerator.FunctionGen.stl_value_bytes(_bare_ret, record_abi) : 0
                if haskey(dwarf_structs, "__enum__$(_bare_ret)")
                    jit_ret_type = _sanitize_cpp_type_name(_bare_ret)
                elseif stl_size > 0
                    jit_ret_type = "NTuple{$stl_size, UInt8}"
                elseif _stl_internal(_bare_ret)
                    info = get(dwarf_structs, _bare_ret, nothing)
                    bs = isnothing(info) ? 0 : (let s = string(get(info, "byte_size", "0x0")); startswith(s, "0x") ? parse(Int, s[3:end], base=16) : parse(Int, s) end)
                    jit_ret_type = bs > 0 ? "NTuple{$bs, UInt8}" : "Ptr{Cvoid}"
                elseif haskey(dwarf_structs, _bare_ret)
                    jit_ret_type = _sanitize_cpp_type_name(_bare_ret)
                else
                    # Fuzzy match: DWARF keys may have trailing " >" nesting artifacts
                    matched_key = _fuzzy_dwarf_lookup(_bare_ret, dwarf_structs)
                    if matched_key !== nothing
                        stl_size = _stl_internal(matched_key) ? JLCSIRGenerator.FunctionGen.stl_value_bytes(matched_key, record_abi) : 0
                        if stl_size > 0
                            jit_ret_type = "NTuple{$stl_size, UInt8}"
                        elseif _stl_internal(matched_key)
                            info = dwarf_structs[matched_key]
                            bs = let s = string(get(info, "byte_size", "0x0")); startswith(s, "0x") ? parse(Int, s[3:end], base=16) : parse(Int, s) end
                            jit_ret_type = bs > 0 ? "NTuple{$bs, UInt8}" : "Ptr{Cvoid}"
                        else
                            jit_ret_type = _sanitize_cpp_type_name(matched_key)
                        end
                    end
                end
            end

            # A return still `Any` is one no Julia type was resolved for.
            # `invoke` refuses it at the call ("unresolved `Any` sentinel"), so
            # the function was uncallable anyway; refuse it where it is
            # defined, as the other ABI traps do, and compile no thunk for it.
            if jit_ret_type == "Any" && jit_c_ret != "void" && !empty_ret
                push!(func_chunks, """
                $doc_comment
                function $julia_name($param_sig)
                    Base.error($(repr(_unresolved_return_trap_msg(julia_name, demangled, jit_c_ret))))
                end
                """)
                push!(exports, julia_name)
                continue
            end

            requires_jit = true
            push!(needed_function_thunks, mangled)

            # Determine if we need the return type overload of invoke
            # Void returns: invoke(name, args...)
            # Struct returns: invoke(name, RetType, args...)
            is_void_ret = jit_ret_type == "Cvoid" || jit_c_ret == "void" || empty_ret
            # What the Julia function returns after a void call for an empty
            # record: its singleton when the type is emitted, else `nothing`.
            empty_ret_val = !empty_ret ? "" :
                (jit_ret_type in defined_struct_names ? "$(jit_ret_type)()" : "nothing")
            ret_stmt(call) = string_setup * (empty_ret ? "$call\n    return $empty_ret_val" : "return $call")

            # A `char*` return gets the same policy here as on the ccall path.
            # The tier decides how the call is MADE; it does not get to decide
            # how the result is presented (see _cstring_wrapper_pair).
            is_cstring_ret = !is_void_ret && jit_ret_type == "Cstring"
            cstring_free_sym = is_cstring_ret ?
                get(config.wrap.cstring_owned, func_name, "") : ""

            # A record C++ returns through a hidden pointer it constructs
            # into (`DW_CC_pass_by_reference`): the thunk builds it in the
            # caller's buffer, `<m>_thunk_inplace`. The same predicate the
            # thunk generator asks, so the call and the thunk always agree.
            # The wrapper allocates the buffer (`return_slot`), keeps it
            # where it is across the call, and returns its value: the Julia
            # return type is what it was.
            in_place = !is_void_ret && !is_cstring_ret &&
                       JLCSIRGenerator.FunctionGen.returns_in_caller_buffer(func, dwarf_structs, record_abi)
            # A `std::string` built there is read into a `String` and ended
            # (`_string_return_plan`, `StdStrings.take_string!`), or keeps
            # its bytes with the reason written beside the call.
            string_plan = in_place ? _string_return_plan(jit_c_ret, record_abi) : nothing
            in_place_body(call) =
                (string_plan isa String ? "    # $(string_plan)\n" : "") *
                "    " * string_setup * "_rb_ret = RepliBuild.JITManager.return_slot($jit_ret_type)\n" *
                "    GC.@preserve _rb_ret $call\n" *
                (string_plan isa Tuple ?
                    "    return RepliBuild.StdStrings.take_string!(_rb_ret, $(_reader_literal(string_plan[1])), " *
                    "LIBRARY_PATH, $(repr(string_plan[2])), $(repr(julia_name)))" :
                    "    return RepliBuild.JITManager.slot_value(_rb_ret)")
            in_place_args = isempty(invoke_args) ? "RepliBuild.JITManager.slot_address(_rb_ret)" :
                            "RepliBuild.JITManager.slot_address(_rb_ret), $invoke_args"

            if config.compile.aot_thunks
                # AOT dispatch is the JIT dispatch with a different way of
                # finding the function: `invoke_aot` takes the thunks handle and
                # dlsyms it, `invoke` asks the JIT engine. Everything after that
                # — argument marshalling, scalar-vs-sret return convention,
                # pending-exception check — is shared code in JITManager.
                #
                # This branch used to emit all of that inline, and both copies
                # drifted from the originals. Marshalling forgot AbstractString
                # and boxed Ref, so string arguments raised
                # `unsafe_convert(Ptr{UInt8}, ::Cstring)`. Worse, the return
                # convention was chosen by testing the type's NAME against a
                # hardcoded list of scalar spellings — but RepliBuild emits
                # enums as real primitive types, so `isprimitivetype(XMLError)`
                # is true while the string "XMLError" matches no list entry and
                # never could. An `i32 (void**)` thunk was then called as
                # `void (i32*, void**)`, which reads args_ptr out of the
                # register holding the return buffer: a segfault inside the
                # thunk for every enum-returning function in every C++ package.
                #
                # Both were the same mistake — approximating, in generated text,
                # a question the type system answers exactly at the call site.
                # Neither is fixable here; only delegation fixes them.
                # The thunks .so is fully linked the moment `__init__` dlopens
                # it, and this generator knows every thunk name right here — so
                # the address is a load-time fact, not a per-call question.
                # Passing `(handle, name)` made every call re-answer it through a
                # lock and a Dict that re-hashed the ~40-char symbol: measured on
                # hello_world, 37.3 ns of a 53.6 ns call, against 3.5 ns for the
                # thunk itself. The slot is filled once in `__init__`; see
                # `_aot_thunk_slot_chunk`, which derives the slots from the FINAL
                # wrapper text so a call site cannot reference one that no
                # `__init__` fills.
                _aot_slot = _aot_fptr_const_name!(aot_fptr_taken, mangled;
                    thunk = JLCSIRGenerator.FunctionGen.thunk_symbol(mangled, in_place)) * "[]"
                if in_place
                    invoke_call = "RepliBuild.JITManager.invoke_aot_ptr($_aot_slot, $in_place_args)"
                elseif is_void_ret
                    invoke_call = if isempty(invoke_args)
                        "RepliBuild.JITManager.invoke_aot_ptr($_aot_slot)"
                    else
                        "RepliBuild.JITManager.invoke_aot_ptr($_aot_slot, $invoke_args)"
                    end
                else
                    invoke_call = if isempty(invoke_args)
                        "RepliBuild.JITManager.invoke_aot_ptr($_aot_slot, $jit_ret_type)"
                    else
                        "RepliBuild.JITManager.invoke_aot_ptr($_aot_slot, $jit_ret_type, $invoke_args)"
                    end
                end

                invoke_call = _with_strings(invoke_call)
                if is_cstring_ret
                    _aot_head = "    # [Tier 2] Dispatch to MLIR AOT Thunk (Complex ABI / Packed / Union)\n    " * string_setup
                    func_def = _cstring_wrapper_pair(julia_name, param_sig,
                        _aot_head * "ptr = " * invoke_call,
                        _aot_head * "return " * invoke_call,
                        cstring_free_sym; doc_comment=doc_comment)
                elseif in_place
                    func_def = """
                    $doc_comment
                    function $julia_name($param_sig)
                        # [Tier 2] Dispatch to MLIR AOT Thunk, built in the caller's buffer
                    $(in_place_body(invoke_call))
                    end
                    """
                else
                    func_def = """
                    $doc_comment
                    function $julia_name($param_sig)
                        # [Tier 2] Dispatch to MLIR AOT Thunk (Complex ABI / Packed / Union)
                        $(ret_stmt(invoke_call))
                    end
                    """
                end
            else
                if in_place
                    invoke_call = "RepliBuild.JITManager.invoke(\"_mlir_ciface_$(JLCSIRGenerator.FunctionGen.thunk_symbol(mangled, true))\", $in_place_args)"
                elseif is_void_ret
                    invoke_call = if isempty(invoke_args)
                        "RepliBuild.JITManager.invoke(\"_mlir_ciface_$(mangled)_thunk\")"
                    else
                        "RepliBuild.JITManager.invoke(\"_mlir_ciface_$(mangled)_thunk\", $invoke_args)"
                    end
                else
                    invoke_call = if isempty(invoke_args)
                        "RepliBuild.JITManager.invoke(\"_mlir_ciface_$(mangled)_thunk\", $jit_ret_type)"
                    else
                        "RepliBuild.JITManager.invoke(\"_mlir_ciface_$(mangled)_thunk\", $jit_ret_type, $invoke_args)"
                    end
                end

                invoke_call = _with_strings(invoke_call)
                if is_cstring_ret
                    _jit_head = "    # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)\n    " * string_setup
                    func_def = _cstring_wrapper_pair(julia_name, param_sig,
                        _jit_head * "ptr = " * invoke_call,
                        _jit_head * "return " * invoke_call,
                        cstring_free_sym; doc_comment=doc_comment)
                elseif in_place
                    func_def = """
                    $doc_comment
                    function $julia_name($param_sig)
                        # [Tier 2] Dispatch to MLIR JIT, built in the caller's buffer
                    $(in_place_body(invoke_call))
                    end
                    """
                else
                    func_def = """
                    $doc_comment
                    function $julia_name($param_sig)
                        # [Tier 2] Dispatch to MLIR JIT (Complex ABI / Packed / Union)
                        $(ret_stmt(invoke_call))
                    end
                    """
                end
            end

            push!(func_chunks, func_def)
            push!(exports, julia_name)
            is_cstring_ret && push!(exports, "$(julia_name)_ptr")
            # A function with C linkage keeps its `<f>_safe` on this path. It
            # had one while every unmangled name took ccall, and an `extern "C"`
            # factory calls `new`, so it is never proven `nounwind`
            # (`_is_nounwind`): which tier makes the call must not decide what
            # the function offers. A mangled function has one on the ccall path
            # alone, as before.
            if !newly_typed_return && !_is_cpp_function(func)
                safe = _managed_safe_wrapper(julia_name, param_sig, param_names,
                                             String(return_type["julia_type"]), deleters, idiomatic_classes)
                safe === nothing || (push!(func_chunks, safe[1]); push!(exports, safe[2]))
            end
            continue # Skip the rest of the loop (ccall generation)
        end

        # =========================================================
        # BRANCH 2: CCALL (Fast Path) - Existing Logic
        # =========================================================
        # Every path of BRANCH 1 ends the iteration, and a pure virtual
        # always takes it. Held here because a `ccall` of its name would bind
        # nothing, or another library's function of that name.
        vtable_only && Base.error("internal error: the pure virtual `$demangled` reached the ccall path")

        # Build conversion logic for parameters
        conversion_code = ""
        ccall_param_names = String[]

        for (i, (name, c_type, needs_conv)) in enumerate(zip(param_names, param_types, needs_conversion))
            if needs_conv
                converted_name = "$(name)_c"
                push!(ccall_param_names, converted_name)

                # Generate range-checked conversion
                conversion_code *= "    $converted_name = $(_integer_arg_conversion(c_type, name))\n"
            elseif cstring_args[i] !== nothing
                # A string with `Cstring` semantics (`_cstring_param_arg`).
                converted_name = "$(name)_c"
                push!(ccall_param_names, converted_name)
                conversion_code *= "    $converted_name = $(cstring_args[i])\n"
            else
                push!(ccall_param_names, name)
            end
        end

        ccall_args = join(ccall_param_names, ", ")

        # ccall needs tuple expression (Type1, Type2) not Tuple{Type1, Type2}
        ccall_types = if isempty(param_types)
            "()"
        else
            "($(join(param_types, ", ")),)"  # Note: trailing comma for single-element tuples
        end

        # llvmcall needs Tuple{Type1, Type2} — built from raw param_types (no trailing comma)
        # CRITICAL: Ref{T} in Julia maps to ptr addrspace(10) in LLVM, but C++ IR uses
        # plain ptr (addrspace 0). For llvmcall we must use Ptr{T} and convert explicitly.
        llvmcall_param_types = String[]
        llvmcall_arg_names = String[]
        llvmcall_ref_args = String[]  # args that need GC.@preserve
        llvmcall_conversion_lines = String[]
        for (i, pt) in enumerate(param_types)
            arg_name = ccall_param_names[i]
            m_ref = match(r"^Ref\{(.+)\}$", pt)
            m_ptr = match(r"^Ptr\{(.+)\}$", pt)
            if m_ref !== nothing
                # Ref{T} → Ptr{T}: llvmcall sees Ref as ptr addrspace(10),
                # but C++ IR uses plain ptr (addrspace 0)
                inner_type = m_ref.captures[1]
                push!(llvmcall_param_types, "Ptr{$inner_type}")
                ptr_name = "__ptr_$(arg_name)"
                push!(llvmcall_arg_names, ptr_name)
                push!(llvmcall_ref_args, arg_name)
                push!(llvmcall_conversion_lines, "        $ptr_name = Base.unsafe_convert(Ptr{$inner_type}, $arg_name)")
            elseif m_ptr !== nothing
                # Ptr{T} params: the Julia signature is relaxed to ::Any,
                # so callers may pass Ref{T} or Vector{T}. ccall auto-converts
                # via cconvert/unsafe_convert, but llvmcall doesn't.
                push!(llvmcall_param_types, pt)
                ptr_name = "__ptr_$(arg_name)"
                cc_name = "__cc_$(arg_name)"
                push!(llvmcall_arg_names, ptr_name)
                push!(llvmcall_ref_args, cc_name)  # preserve cconvert result (holds the GC root)
                push!(llvmcall_conversion_lines, "        $cc_name = Base.cconvert($pt, $arg_name)")
                push!(llvmcall_conversion_lines, "        $ptr_name = Base.unsafe_convert($pt, $cc_name)")
            else
                push!(llvmcall_param_types, pt)
                push!(llvmcall_arg_names, arg_name)
            end
        end
        llvmcall_types = isempty(llvmcall_param_types) ? "" : join(llvmcall_param_types, ", ")
        llvmcall_args = join(llvmcall_arg_names, ", ")
        has_ref_params = !isempty(llvmcall_ref_args)

        # Generate function body based on return type and conversions
        julia_return_type = return_type["julia_type"]
        c_return_type = return_type["c_type"]

        # Check if return type is a struct (not primitive, not pointer)
        is_struct_return = julia_return_type == "Any" && !contains(c_return_type, "*") && !contains(c_return_type, "void") && c_return_type != "unknown"
        
        # Also detect resolved struct returns: julia_return_type was mapped to a
        # known struct name (e.g. Matrix_double_minus_1...) but is_struct_return
        # missed it because julia_return_type != "Any".
        returns_known_struct = is_struct_return ||
            haskey(julia_to_cpp_struct, julia_return_type) ||
            c_return_type in struct_types

        # Check if this is a virtual method that requires dynamic dispatch
        is_virtual = get(func, "is_virtual", false)

        # Build the llvmcall expression with proper Ref{T} → Ptr{T} handling.
        # ccall handles Ref{T} → C pointer automatically, but llvmcall sees Ref{T}
        # as ptr addrspace(10) while C++ IR uses plain ptr (addrspace 0).
        function _build_llvmcall_expr(ret_type_str, indent="        ")
            call_expr = "Base.llvmcall((LTO_IR, \"$mangled\"), $ret_type_str, Tuple{$llvmcall_types}, $llvmcall_args)"
            if !has_ref_params
                return "$(indent)return $call_expr"
            end
            lines = String[]
            for l in llvmcall_conversion_lines
                push!(lines, "$indent$l")
            end
            preserve_list = join(llvmcall_ref_args, " ")
            push!(lines, "$(indent)GC.@preserve $preserve_list begin")
            push!(lines, "$(indent)    return $call_expr")
            push!(lines, "$(indent)end")
            return join(lines, "\n")
        end

        # LTO eligibility: safe for primitive/pointer/small-POD structs (filtered earlier by is_ccall_safe),
        # non-virtual, no Cstring anywhere (llvmcall won't auto-convert String→Cstring like ccall does),
        # and NOT returning a struct by value (Base.llvmcall doesn't handle the sret ABI).
        lto_eligible = config.link.enable_lto &&
            !is_virtual &&
            !returns_known_struct &&
            julia_return_type != "Cstring" &&
            !any(t -> t == "Cstring", param_types)

        # Check for _UnsafeUnknown trap to prevent segfaults
        has_unknown_param = any(t -> t == "_UnsafeUnknown", param_types)
        # An `Any` return whose C type is an aggregate is a by-value struct
        # return the mapper could not NAME. `is_struct_return` above detects the
        # shape but assumes a nameable type; when there is none the emission
        # fell through to a plain ccall declaring `::Any`, and `Any` in a foreign
        # return position tells Julia the callee returned a `jl_value_t*`, so the
        # result is dereferenced as a Julia object — a segfault inside dispatch,
        # far from the call site.
        #
        # It cannot degrade to `Cvoid` either, which is what the guard's message
        # suggests for the enum case: a MEMORY-class aggregate is returned
        # through a hidden sret pointer, so dropping the value leaves the ABI
        # wrong rather than merely lossy.
        #
        # llamacpp is the live case — `format[abi:cxx11]` returns
        # `std::basic_string`, which cf09702 correctly drops from
        # `struct_definitions` as an STL type, so nothing can name it. The
        # wrapper shipped that `::Any` ccall for as long as the function has
        # existed; `_assert_no_any_ccall_return` is what finally refused it.
        #
        # Refusing at the CALL SITE is what `_UnsafeUnknown` already does for
        # unmappable parameters: the module still loads and every other function
        # works, and this one explains itself if anybody calls it.
        is_unnameable_agg_return = is_struct_return &&
            !(c_return_type in struct_types) &&
            !haskey(julia_to_cpp_struct, julia_return_type)

        is_unknown_return = julia_return_type == "_UnsafeUnknown" ||
                            is_unnameable_agg_return

        x87 = _x87_crossings(func)
        # A parameter the type map has no Julia type for is still `Any` here,
        # and a ccall declaring it would cost the whole wrapper
        # (`_unmapped_crossings`, as in the C generator).
        unmapped = _unmapped_crossings(params, param_types)
        if !isempty(x87)
            func_def = """
            $doc_comment
            function $julia_name($param_sig)
            $(_x87_trap_body(julia_name, x87))end

            """
        elseif !isempty(unmapped)
            func_def = """
            $doc_comment
            function $julia_name($param_sig)
            $(_unmapped_trap_body(julia_name, unmapped))end

            """
        elseif has_unknown_param || is_unknown_return
            func_def = """
            $doc_comment
            function $julia_name($param_sig)
                Base.error(\"\"\"
                FFI Safety Trap: Cannot call function '$julia_name'.
                One or more types could not be mapped to Julia safely:
                Return type: $julia_return_type (C++: $c_return_type)
                Parameter types: $(join(param_types, ", "))

                To fix this, map the type under [types.custom] in replibuild.toml
                (for example MyHandle = "Ptr{Cvoid}"), then build and wrap again.
                Calling this function would have caused a segmentation fault.
                \"\"\")
            end

            """
        elseif is_virtual
            requires_jit = true
            
            # Sanitize return type if needed
            safe_c_ret = c_return_type
            if occursin(r"[<>]", safe_c_ret)
                 safe_c_ret = _sanitize_cpp_type_name(safe_c_ret)
            end
            ret_type_sig = (julia_return_type == "Any" || julia_return_type == "Cstring") ? safe_c_ret : julia_return_type
            
            if config.compile.aot_thunks
                # AOT-based virtual dispatch wrapper
                # We call the statically generated MLIR thunk which natively handles the vtable math
                thunk_sym = ":thunk_$(mangled)"
                
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$ret_type_sig
                $conversion_code    return ccall(($thunk_sym, THUNKS_LIBRARY_PATH), $ret_type_sig, $ccall_types, $ccall_args)
                end

                """
            else
                # JIT-based virtual dispatch wrapper
                cls_name = get(func, "class", "")
                
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$ret_type_sig
                    # Get thunk for virtual dispatch (JIT compiled on-demand)
                    thunk_ptr = RepliBuild.JITManager.get_jit_thunk("$cls_name", "$func_name")
                    
                    # Call thunk
                $conversion_code    return ccall(thunk_ptr, $ret_type_sig, $ccall_types, $ccall_args)
                end

                """
            end
        elseif is_struct_return
            # Struct-valued return - Julia uses the struct type directly
            # ccall will handle struct returns automatically if the Julia type matches
            
            # Sanitize C return type if it contains template chars (Box<int> -> Box_int)
            safe_c_ret = c_return_type
            if occursin(r"[<>]", safe_c_ret)
                 safe_c_ret = _sanitize_cpp_type_name(safe_c_ret)
            end

            if lto_eligible
                llvmcall_body = _build_llvmcall_expr(safe_c_ret)
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$safe_c_ret
                $conversion_code    if !isempty(LTO_IR)
                $llvmcall_body
                    else
                        return ccall((:$mangled, LIBRARY_PATH), $safe_c_ret, $ccall_types, $ccall_args)
                    end
                end

                """
            else
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$safe_c_ret
                $conversion_code    return ccall((:$mangled, LIBRARY_PATH), $safe_c_ret, $ccall_types, $ccall_args)
                end

                """
            end
        elseif julia_return_type == "Cstring"
            # Cstring return: NULL → nothing, else copy to String; owned buffers
            # ([wrap.cstring_owned]) freed through the library's deallocator.
            # Raw `_ptr` variant emitted alongside. Shared with the MLIR-dispatch
            # branch above and with the C generator — see _cstring_wrapper_pair.
            cstring_free_sym = get(config.wrap.cstring_owned, func_name, "")
            _cs_call = "ccall((:$mangled, LIBRARY_PATH), Cstring, $ccall_types, $ccall_args)"
            func_def = _cstring_wrapper_pair(julia_name, param_sig,
                "$conversion_code    ptr = $_cs_call",
                "$conversion_code    return $_cs_call",
                cstring_free_sym; doc_comment=doc_comment)
            push!(exports, "$(julia_name)_ptr")
        elseif !isempty(conversion_code)
            # Has parameter conversions
            if lto_eligible
                llvmcall_body = _build_llvmcall_expr(julia_return_type)
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                $conversion_code    if !isempty(LTO_IR)
                $llvmcall_body
                    else
                        return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
                    end
                end

                """
            else
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                $conversion_code    return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
                end

                """
            end
        else
            # Standard wrapper - no conversions needed
            if lto_eligible
                llvmcall_body = _build_llvmcall_expr(julia_return_type)
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                    if !isempty(LTO_IR)
                $llvmcall_body
                    else
                        return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
                    end
                end

                """
            else
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                    ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
                end

                """
            end
        end

        push!(func_chunks, func_def)
        push!(exports, julia_name)

        # Generate safe wrapper if return type is a managed struct. Not for a
        # return typed from its declaration alone: `GetIO()` hands back memory
        # the context owns, and a finalizer on it frees what imgui still uses.
        if !newly_typed_return
            safe = _managed_safe_wrapper(julia_name, param_sig, param_names, julia_return_type,
                                         deleters, idiomatic_classes)
            safe === nothing || (push!(func_chunks, safe[1]); push!(exports, safe[2]))
        end

        # Generate convenience wrappers for ergonomic APIs
        # One type: const primitive array pointers -> accept Vector directly with GC.@preserve
        #
        # Deliberately NOT emitted: struct-by-value overloads for Ptr{Struct} params
        # (f(x::MyStruct) passing Ref(local copy) to the ccall). A function taking
        # T* may free, mutate-and-retain, or store that pointer; handing it a pointer
        # into a temporary Julia-owned copy is undefined behavior for every such
        # function (crash-proven on the C side: cJSON_Delete(::cJSON) → glibc
        # double-free abort). Ownership is not recoverable from DWARF, so no name
        # heuristic can gate this safely. The base wrapper's ::Any params already
        # accept Ref(x)/pointers.

        has_array_ptr_params = false
        array_ptr_indices = Int[]
        convenience_param_types = String[]
        convenience_param_names = String[]

        for (i, (ptype, pname)) in enumerate(zip(param_types, param_names))
            # Non-C-prefixed pointer types (struct pointers, Ptr{Int32}-style aliases)
            # pass through unchanged (no by-value overload — see note above); this arm
            # also keeps them out of the array-pointer heuristic below.
            if startswith(ptype, "Ptr{") && !startswith(ptype, "Ptr{C") && ptype != "Ptr{Cvoid}"
                push!(convenience_param_types, ptype)
                push!(convenience_param_names, pname)
            # Check if this is Ptr{Cdouble}, Ptr{Cfloat}, Ptr{Cint}, etc. (array pointers)
            # These benefit from Vector{T} wrapper with automatic GC.@preserve
            elseif ptype in ["Ptr{Cdouble}", "Ptr{Cfloat}", "Ptr{Cint}", "Ptr{Cuint}",
                             "Ptr{Int32}", "Ptr{Int64}", "Ptr{UInt32}", "Ptr{UInt64}",
                             "Ptr{Float32}", "Ptr{Float64}"]
                # Check if parameter name suggests it's an input array (not output)
                # Common patterns: x, y, data, array, vector, signal, values, input, src
                pname_lower = lowercase(pname)
                is_likely_input = any(pattern -> contains(pname_lower, pattern),
                    ["x", "y", "data", "array", "vec", "signal", "value", "input", "src", "coef"])

                # Also check if it's NOT likely an output
                is_likely_output = any(pattern -> contains(pname_lower, pattern),
                    ["out", "result", "dest", "dst", "buffer"])

                if is_likely_input && !is_likely_output
                    has_array_ptr_params = true
                    push!(array_ptr_indices, i)
                    # Convert Ptr{Cdouble} -> Vector{Float64}, etc.
                    elem_type = replace(ptype, "Ptr{" => "", "}" => "")
                    julia_vec_type = if elem_type == "Cdouble"
                        "Vector{Float64}"
                    elseif elem_type == "Cfloat"
                        "Vector{Float32}"
                    elseif elem_type in ["Cint", "Int32"]
                        "Vector{Int32}"
                    elseif elem_type in ["Cuint", "UInt32"]
                        "Vector{UInt32}"
                    elseif elem_type == "Int64"
                        "Vector{Int64}"
                    elseif elem_type == "UInt64"
                        "Vector{UInt64}"
                    elseif elem_type == "Float64"
                        "Vector{Float64}"
                    elseif elem_type == "Float32"
                        "Vector{Float32}"
                    else
                        ptype  # Fallback to pointer type
                    end
                    push!(convenience_param_types, julia_vec_type)
                    push!(convenience_param_names, pname)
                else
                    push!(convenience_param_types, ptype)
                    push!(convenience_param_names, pname)
                end
            else
                push!(convenience_param_types, ptype)
                push!(convenience_param_names, pname)
            end
        end

        # Generate convenience wrapper if we have array pointer parameters
        if has_array_ptr_params
            convenience_sig = join(["$pname::$ptype" for (pname, ptype) in zip(convenience_param_names, convenience_param_types)], ", ")

            # Resolve the return type the same way the base wrapper does. For a
            # struct-valued return, julia_return_type is the "Any" sentinel — using
            # it as the ccall return type makes Julia treat the returned struct as a
            # boxed object pointer and segfault. Mirror the base wrapper's safe_c_ret.
            convenience_ret_type = julia_return_type
            if is_struct_return
                convenience_ret_type = occursin(r"[<>]", c_return_type) ?
                    _sanitize_cpp_type_name(c_return_type) : c_return_type
            end

            # Build the ccall arguments
            convenience_ccall_args = String[]
            for (i, pname) in enumerate(param_names)
                if i in array_ptr_indices
                    # Array parameter: use pointer()
                    push!(convenience_ccall_args, "pointer($pname)")
                else
                    # Regular parameter: pass as-is
                    push!(convenience_ccall_args, pname)
                end
            end
            convenience_ccall = join(convenience_ccall_args, ", ")

            # The vectors backing pointer() must stay rooted across the call
            preserve_vars = join([param_names[i] for i in array_ptr_indices], " ")

            if julia_return_type == "Cstring"
                # Match the base wrapper's Cstring policy (NULL → nothing,
                # String copy, [wrap.cstring_owned] free)
                convenience_func = """
                # Convenience wrapper - accepts arrays directly with automatic GC preservation
                function $julia_name($convenience_sig)::Union{String,Nothing}
                    ptr = GC.@preserve $preserve_vars begin
                        ccall((:$mangled, LIBRARY_PATH), Cstring, $ccall_types, $convenience_ccall)
                    end
                $(_cstring_policy_lines(get(config.wrap.cstring_owned, func_name, "")))
                end

                """
            else
                convenience_func = """
                # Convenience wrapper - accepts arrays directly with automatic GC preservation
                function $julia_name($convenience_sig)::$convenience_ret_type
                    return GC.@preserve $preserve_vars begin
                        ccall((:$mangled, LIBRARY_PATH), $convenience_ret_type, $ccall_types, $convenience_ccall)
                    end
                end

                """
            end

            push!(func_chunks, convenience_func)
        end
    end
    isempty(vtable_refusals) ||
        println("  wrap: $(length(vtable_refusals)) pure virtual member function(s) are refusals (a record by value: ",
                "a call through the vtable carries scalars and pointers only): ",
                join(first(sort!(vtable_refusals), 6), ", "), length(vtable_refusals) > 6 ? ", …" : "")

    # =================================================================
    # STL CONTAINER FACTORY FUNCTIONS
    # =================================================================
    stl_methods = get(metadata, "stl_methods", Dict())
    if !isempty(stl_methods)
        requires_jit = true
        push!(func_chunks, """

        # =============================================================================
        # STL Container Factories
        # =============================================================================

        """)

        # Sorted: `stl_methods` comes back from compilation_metadata.json, and a
        # Dict's iteration order follows the key order of the file it was read
        # from, which differs between JSON majors. Same for every metadata-derived
        # collection below that orders emitted code or the export list.
        for container_type in sort!(collect(keys(stl_methods)))
            methods = stl_methods[container_type]
            # Build thunks dict for this container
            thunks_entries = String[]
            for m in methods
                method_name = get(m, "method", "")
                mangled_name = get(m, "mangled", "")
                if !isempty(method_name) && !isempty(mangled_name)
                    push!(thunks_entries, "\"$method_name\" => \"$mangled_name\"")
                end
            end
            thunks_dict_str = "Dict{String,String}(" * join(thunks_entries, ", ") * ")"

            # Get container byte_size from DWARF struct_definitions
            byte_size = get_stl_container_size(container_type)
            if byte_size == 0
                byte_size = 24  # fallback default
            end
            # Try to get exact size from DWARF if available
            # DWARF keys may lack std:: prefix and include allocator args
            container_short = replace(container_type, "std::" => "")
            for sname in sort!(collect(keys(dwarf_structs)))
                sinfo = dwarf_structs[sname]
                if contains(sname, container_type) || contains(sname, container_short) || _normalize_stl_for_dwarf(sname) == container_type
                    bs_str = get(sinfo, "byte_size", "")
                    if !isempty(bs_str)
                        byte_size = bs_str isa Integer ? bs_str : (startswith(bs_str, "0x") ? parse(Int, bs_str[3:end], base=16) : parse(Int, bs_str))
                    end
                    break
                end
            end

            # Generate sanitized factory name
            safe_name = replace(replace(replace(container_type, "::" => "_"), "<" => "_"), ">" => "")
            safe_name = replace(replace(safe_name, " " => ""), "," => "_")

            # Determine element type for CppVector. The element is read from
            # the compiler's spelling of the container when the table recorded
            # one (`container`: `std::vector<Clipper2Lib::Point<long>>` for an
            # entry written with the typedef `Point64`), and a record element
            # is the struct this module emitted for it (`Point_long`), which
            # `CppVector{T}` reads with `unsafe_load(Ptr{T})`. An element with
            # no Julia layout gets no factory: `CppVector{_UnsafeUnknown}` would
            # read garbage, so the entry is named instead.
            if startswith(container_type, "std::vector<")
                canonical = String(get(first(methods), "container", container_type))
                elem_match = match(r"^std::vector<(.+)>$", canonical)
                elem_match === nothing && (elem_match = match(r"^std::vector<(.+)>$", container_type))
                if !isnothing(elem_match)
                    elem_cpp = String(strip(elem_match.captures[1]))
                    # `std::vector<bool>` is the bit-packed specialization: its
                    # `operator[]` returns a proxy object and `push_back` takes
                    # a `bool` by value, where a `CppVector` reads and writes
                    # elements through a pointer. Made as one, `push!` stored
                    # wrong bits and indexing read the proxy as an address.
                    if _is_bool_vector_element(elem_cpp)
                        @warn "STL: no CppVector factory for $(repr(container_type)): `std::vector<bool>` is " *
                              "bit-packed, and its members take and return bits where a CppVector reads " *
                              "and writes elements through a pointer"
                        continue
                    end
                    elem_julia = _stl_vector_element_type(elem_cpp, defined_struct_names,
                                                          t -> _normalize_stl_elem_type(infer_julia_type(registry, t; context="STL factory element type")))
                    if elem_julia === nothing
                        @warn "STL: no CppVector factory for $(repr(container_type)): its element type " *
                              "$(repr(elem_cpp)) has no Julia layout in this wrapper"
                        continue
                    end

                    factory_name = "create_$(safe_name)"
                    push!(exports, factory_name)

                    push!(func_chunks, """
                    const _THUNKS_$(uppercase(safe_name)) = $thunks_dict_str

                    \"\"\"
                        $factory_name() -> CppVector{$elem_julia}

                    Create a new `$container_type` and return a Julia `CppVector{$elem_julia}` wrapper.
                    \"\"\"
                    function $factory_name()
                        buf = Libc.malloc($byte_size)
                        # Zero-initialize
                        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), buf, 0, $byte_size)
                        # Call C++ constructor
                        ctor = get(_THUNKS_$(uppercase(safe_name)), "constructor", "")
                        if !isempty(ctor)
                            RepliBuild.JITManager.invoke("_mlir_ciface_\$(ctor)_thunk", buf)
                        end
                        return RepliBuild.STLWrappers.CppVector{$elem_julia}(buf, true, _THUNKS_$(uppercase(safe_name)); byte_size=$byte_size)
                    end

                    """)
                end
            elseif startswith(container_type, "std::basic_string") || container_type == "std::string"
                factory_name = "create_$(safe_name)"
                push!(exports, factory_name)

                push!(func_chunks, """
                    const _THUNKS_$(uppercase(safe_name)) = $thunks_dict_str

                    \"\"\"
                        $factory_name() -> CppString

                    Create a new `$container_type` and return a Julia `CppString` wrapper.
                    \"\"\"
                    function $factory_name()
                        buf = Libc.malloc($byte_size)
                        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), buf, 0, $byte_size)
                        ctor = get(_THUNKS_$(uppercase(safe_name)), "constructor", "")
                        if !isempty(ctor)
                            RepliBuild.JITManager.invoke("_mlir_ciface_\$(ctor)_thunk", buf)
                        end
                        return RepliBuild.STLWrappers.CppString(buf, true, _THUNKS_$(uppercase(safe_name)); byte_size=$byte_size)
                    end

                """)

            elseif startswith(container_type, "std::map<") || startswith(container_type, "std::unordered_map<")
                # Extract K, V from "std::map<K, V>" or "std::unordered_map<K, V>"
                tmpl_start = findfirst('<', container_type)
                tmpl_end = findlast('>', container_type)
                if !isnothing(tmpl_start) && !isnothing(tmpl_end)
                    inner = container_type[tmpl_start+1:tmpl_end-1]
                    parts = _split_template_args(inner)
                    if length(parts) >= 2
                        key_cpp = String(strip(parts[1]))
                        val_cpp = String(strip(parts[2]))
                        key_julia = infer_julia_type(registry, key_cpp; context="STL map key type")
                        val_julia = infer_julia_type(registry, val_cpp; context="STL map value type")

                        # Normalize to ccall-compatible types
                        key_julia = _normalize_stl_elem_type(key_julia)
                        val_julia = _normalize_stl_elem_type(val_julia)

                        factory_name = "create_$(safe_name)"
                        push!(exports, factory_name)

                        push!(func_chunks, """
                    const _THUNKS_$(uppercase(safe_name)) = $thunks_dict_str

                    \"\"\"
                        $factory_name() -> CppMap{$key_julia,$val_julia}

                    Create a new `$container_type` and return a Julia `CppMap{$key_julia,$val_julia}` wrapper.
                    \"\"\"
                    function $factory_name()
                        buf = Libc.malloc($byte_size)
                        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), buf, 0, $byte_size)
                        ctor = get(_THUNKS_$(uppercase(safe_name)), "constructor", "")
                        if !isempty(ctor)
                            RepliBuild.JITManager.invoke("_mlir_ciface_\$(ctor)_thunk", buf)
                        end
                        return RepliBuild.STLWrappers.CppMap{$key_julia,$val_julia}(buf, true, _THUNKS_$(uppercase(safe_name)); byte_size=$byte_size)
                    end

                        """)
                    end
                end
            end
        end
    end

    # Export statement (unique to handle overloaded functions)
    # Include enum types, enum values, struct types, and functions
    all_exports = copy(exports)

    # Enum types and enumerators go here, in `sort!(collect(enum_types))`
    # order, but are spliced in below once `_render_enum_chunks` has settled
    # each enumerator's final name: the export list must carry the spelling
    # the `@enum` binds (`_enumerator_julia_name`), and that is decided against
    # the rest of the module. Pushing the raw DWARF name here exported
    # `bool_` for an enumerator bound as `bool`.
    enum_exports_at = length(all_exports) + 1

    # Add struct types (filter internal/compiler types)
    for struct_name in sort!(collect(struct_types))
        if !(struct_name in enum_names)
            if _stl_internal(struct_name)
                continue
            end
            # Sanitize struct name for export
            julia_struct_name = _sanitize_cpp_type_name(struct_name)
            julia_struct_name = replace(julia_struct_name, "*" => "Ptr")
            julia_struct_name = replace(julia_struct_name, "&" => "Ref")
            julia_struct_name = replace(julia_struct_name, r"[^a-zA-Z0-9_]" => "_")
            # Skip internal/compiler types
            if julia_struct_name in _INTERNAL_TYPE_BLOCKLIST || struct_name in _INTERNAL_TYPE_BLOCKLIST
                continue
            end
            push!(all_exports, julia_struct_name)
        end
    end

    # The bulk setter, once, only if some struct actually got a `setproperty`.
    # Skipped outright if the library already owns the name: our definition
    # takes no type-annotated argument, so it would not merely add a method —
    # it would shadow the library's own `setproperties` for every call. The
    # per-struct `setproperty` methods are safe either way (they dispatch on a
    # concrete blob type, and `_dedup_method_chunks` settles exact collisions).
    field_setters_emitted && push!(struct_chunks, _field_setter_helpers_chunk())
    if blob_setters_emitted
        if !("setproperties" in all_exports)
            push!(struct_chunks, _blob_setproperties_chunk())
            push!(all_exports, "setproperties")
        else
            @warn "wrap: the library exports `setproperties`; skipping the generated " *
                  "bulk field setter. Use `setproperty(x, :field, v)` per field."
        end
        "setproperty" in all_exports || push!(all_exports, "setproperty")
    end

    # NOTE: `export_statement` is built AFTER `_dedup_method_chunks` below, not
    # here — it is filtered against the definitions that actually ship, and
    # dedup runs between this point and the final join.

    # Footer
    footer = """

    end # module $module_name
    """

    # Unbuffer C stdout so printf output is visible in the REPL. Must run in
    # __init__, not the module body: top-level side effects execute during
    # PRECOMPILATION and never re-run at load time in a precompiled package.
    # Pre-load [ingest] extra_link_libs. Must precede every dlopen below,
    # so it is spliced at the TOP of each __init__ variant, not appended
    # with the other snippets.
    _preload_snippet = _extra_link_libs_snippet(config) * _link_dir_preload_snippet(config)

    # `stdout` is an exported data symbol in glibc, and is NOT a symbol at all on
    # Windows: the UCRT declares it as a macro over `__acrt_iob_func(1)`. So
    # `cglobal(:stdout, …)` throws "could not load symbol" there — and because
    # this runs inside `__init__`, it took the entire generated module down with
    # it, turning a cosmetic buffering tweak into "the wrapper cannot be loaded".
    #
    # The branch is emitted as `@static if` rather than resolved here, so the
    # decision belongs to the machine that LOADS the wrapper, not the one that
    # generated it. And the whole thing is best-effort: a module that could not
    # unbuffer someone else's stdout is still a working module.
    # (Kept byte-identical to the C generator's copy in GeneratorC.jl — if you
    # change one, change both.)
    _setvbuf_snippet = """
            # Unbuffer C stdout so printf output appears immediately in the REPL
            try
                c_stdout = @static if Sys.iswindows()
                    ccall(:__acrt_iob_func, Ptr{Cvoid}, (Cuint,), 1)
                else
                    unsafe_load(cglobal(:stdout, Ptr{Cvoid}))
                end
                if c_stdout != C_NULL
                    ccall(:setvbuf, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Csize_t), c_stdout, C_NULL, 2, 0)
                end
            catch e
                @debug "could not unbuffer the C runtime's stdout" exception=e
            end"""

    # Generate initialization block
    init_block = if config.compile.aot_thunks
        """
        # Library handles for manual management if needed
        const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
        const THUNKS_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

        function __init__()
$_preload_snippet            # Load main library explicitly to ensure symbols are available
            LIB_HANDLE[] = Libdl.dlopen(LIBRARY_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)

            # Load AOT thunks library if it was successfully generated
            if !isempty(THUNKS_LIBRARY_PATH) && isfile(THUNKS_LIBRARY_PATH)
                # Not a bare `dlopen`: the thunks import `jlcs_catch_current_exception`
                # from libJLCS, which has to be the same copy `MLIRNative` ccalls and
                # is on no loader search path on PE. See `open_thunks_library`.
                THUNKS_HANDLE[] = RepliBuild.JITManager.open_thunks_library(THUNKS_LIBRARY_PATH)
            elseif $requires_jit
                @warn "AOT Thunks library not found, but advanced FFI features are required. These features will fail at runtime."
            end

            # Resolve every thunk address once, here, so no call site pays for a
            # lookup. A miss is collected rather than raised: a stale thunks .so
            # is usually missing a set of symbols, and the whole set named at
            # load beats the first one raised at some later call. Functions whose
            # thunk did resolve keep working; calling one whose thunk did not
            # raises with a pointer back to this warning.
            let _missing = String[]
                for (_slot, _sym) in _THUNK_SLOTS
                    RepliBuild.JITManager.resolve_thunk!(_slot, THUNKS_HANDLE[], _sym, _missing)
                end
                RepliBuild.JITManager.warn_missing_thunks(_missing, THUNKS_LIBRARY_PATH)
            end

            _check_build_identity()$(_imports_init)
        $_setvbuf_snippet
        end
        """
    else
        if requires_jit
            """
            function __init__()
$_preload_snippet                # Initialize this library's JIT engine (one engine per binary)
                RepliBuild.JITManager.initialize_global_jit(LIBRARY_PATH)
                _check_build_identity()$(_imports_init)
            $_setvbuf_snippet
            end
            """
        else
            """
            # Library handle for manual management if needed
            const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

            function __init__()
$_preload_snippet                # Load library explicitly to ensure symbols are available
                LIB_HANDLE[] = Libdl.dlopen(LIBRARY_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
                _check_build_identity()$(_imports_init)
            $_setvbuf_snippet
            end
            """
        end
    end

    # Colliding definitions (dtor D1/D2 pairs, overloads collapsing to the
    # same ::Any signature) must not survive into the module — method
    # overwriting is a hard error under package precompilation. Keep the
    # last definition per dispatch signature (include()-identical semantics).
    # Before it, `operator()`/`[]`/`,` get their old shared name back beside
    # their own, deduplicated as it always was (`_operator_legacy_aliases`).
    (func_chunks, _legacy_operator_names) = _operator_legacy_aliases(func_chunks)
    append!(all_exports, _legacy_operator_names)
    func_chunks = _dedup_method_chunks(func_chunks)

    # Each section is joined ONCE and reused for both the export filter and the
    # file, so the list of names exported and the text defining them cannot be
    # built from different inputs — which is exactly how 102 undefined exports
    # reached the Hub (see `_export_statement`). Must stay below the dedup.
    _structs = join(struct_chunks)
    _funcs   = join(func_chunks)

    # The thunk set is what the FINAL text calls. The loop recorded a thunk
    # for every Tier 2 chunk it emitted, and the dedup above dropped chunks;
    # a thunk no call site names would be compiled by the JIT for nothing,
    # and listed as bound by anything that reads the manifest
    # (`_thunks_named_in_text`).
    let named = _thunks_named_in_text(_structs * _funcs, aot_fptr_taken)
        filter!(m -> m in named, needed_function_thunks)
    end

    # Derived from the FINAL chunks: what dispatches how (C++ is Tier 2/3, so
    # this is a lookup table with no kernels), and the layout facts consumers
    # were re-parsing out of compilation_metadata.json. The enum section is
    # not text yet; its type names are known.
    (_tiers, _kernels) = _dispatch_facts(func_chunks)
    _layout_names = setdiff(union(_defined_type_names(_structs), _pending_enum_type_names(enum_chunks)),
                            imported_jl_names)
    # A class-nested type also answers to the name its functions carry
    # (`_nested_type_aliases`), unless the module binds that name already.
    _nested_aliases = _nested_type_aliases(
        dwarf_structs, _layout_names, _sanitize_cpp_type_name,
        union(Set{String}(keys(_toplevel_names(_structs * _funcs))), _pending_enum_type_names(enum_chunks),
              imported_jl_names))
    introspection = _dispatch_tier_chunk(_tiers, _kernels) *
                    _layout_chunk(dwarf_structs, _layout_names, _sanitize_cpp_type_name;
                                  aliases = _nested_aliases) *
                    _nested_type_alias_chunk(_nested_aliases)

    # Also derived from the FINAL text: the thunk-address slots the emitted call
    # sites read. Must stay below the dedup for the same reason the tier table
    # does — a chunk dropped for a colliding dispatch signature must not leave a
    # slot behind, and a slot a surviving chunk reads must exist.
    thunk_slots = config.compile.aot_thunks ?
        _aot_thunk_slot_chunk(_funcs, aot_fptr_taken) : ""

    # The manifest is written on every wrap, JIT mode included: JITManager
    # compiles only the thunks it lists, and without one it compiles a thunk for
    # every function in metadata (`test_thunk_manifest.jl`). It lands before the
    # AOT stage, so a refused wrap still records the set it derived.
    _write_thunk_manifest(get_output_path(config), needed_function_thunks)

    # AOT: wrap's last stage compiles exactly the thunks this text binds, then
    # the library is asked what it defines before call sites that assume it ship
    # — a slot resolving to C_NULL is a hard error on first call, found by a
    # user rather than by the wrap. When the stage writes no library,
    # `wrap_library` generates the wrapper again with JIT dispatch.
    #
    # The set comes from here and nowhere else. AOT used to run at build, before
    # this generator had decided it, and read the manifest an earlier wrap left:
    # a new package had none, its first wrap refused or fell back to JIT, and only
    # a second build and wrap converged. Without a stage (a direct call), the
    # caller supplies `thunks_lib_path` already built.
    if config.compile.aot_thunks
        if aot_stage !== nothing
            aot_stage(needed_function_thunks) || throw(AOTStageFailed(thunks_lib_path))
        end
        _assert_aot_thunks_present(_funcs, aot_fptr_taken, thunks_lib_path)
    end

    # The enum section last, against every other top-level name the module
    # binds, read off the text it will be written with (minus the enums and
    # the export line). An enumerator that would rebind one of them is
    # prefixed with its enum's name (`_render_enum_chunks`).
    # Imported types and their enumerators first, against the names the rest
    # of the module binds; then the own enums against everything, imports included.
    own_text = header * thunk_slots * init_block * metadata_section *
               _structs * union_accessor_defs * introspection * _funcs * footer
    (imports_section, imported_enum_names) =
        _type_imports_chunk(dwarf_structs, _sanitize_cpp_type_name, metadata, get_output_path(config),
                            _toplevel_names(own_text))
    taken = _toplevel_names(own_text * imports_section)
    (rendered_enums, enum_member_names) = _render_enum_chunks(enum_chunks, taken)
    _enums = join(rendered_enums)

    # The enum segment of the export list, at the position recorded above, in
    # the names the `@enum`s bind. An alias carries its canonical's members; an
    # imported enum's are the aliases the import section bound.
    enum_exports = String[]
    for enum_key in sort!(collect(enum_types))
        enum_name = _sanitize_cpp_type_name(replace(enum_key, "__enum__" => ""))
        push!(enum_exports, enum_name)
        append!(enum_exports, get(enum_member_names, enum_name, get(imported_enum_names, enum_name, String[])))
    end
    splice!(all_exports, enum_exports_at:enum_exports_at-1, enum_exports)

    export_statement = _export_statement(all_exports,
                                         imports_section * _enums * _structs * union_accessor_defs *
                                         introspection * _funcs)

    wrapper_content = join([header, thunk_slots, init_block, metadata_section, imports_section,
                            _enums, _structs,
                            union_accessor_defs, introspection, export_statement,
                            _funcs, footer])
    if emitted_facts !== nothing
        # What a wrap leaves for the libraries after it (`_type_manifest`).
        emitted_facts["layouts"] = @isdefined(jl_layouts) ?
            Dict{String,Any}(k => Any[v[1], v[2]] for (k, v) in jl_layouts) : Dict{String,Any}()
        emitted_facts["class_risk"] = _blob_names
        emitted_facts["enumerators"] = _enumerator_facts(enum_chunks, enum_member_names)
    end
    return (wrapper_content, needed_function_thunks)
end

