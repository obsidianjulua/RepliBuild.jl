module FunctionGen

using ..TypeUtils
using ..StructGen # for get_struct_type_string

export generate_function_thunks

"""
    _fuzzy_struct_lookup(c_type, structs) -> Union{String, Nothing}

Fuzzy-match a C type name against struct definition keys.
DWARF keys often have trailing " >" nesting artifacts from template types.
"""
function _fuzzy_struct_lookup(c_type::String, structs::Any)
    haskey(structs, c_type) && return c_type
    key = c_type * " >"
    haskey(structs, key) && return key
    c_norm = rstrip(c_type, [' ', '>'])
    for k in keys(structs)
        rstrip(String(k), [' ', '>']) == c_norm && return String(k)
    end
    return nothing
end

"""
    by_value_crossing(c_type, structs, record_abi) -> (kind, key)

How a parameter or return of C type `c_type` crosses a Tier-2 thunk. `kind`:

  * `:direct`   — no record decision to make: a scalar, pointer, reference,
                  function pointer, or STL container (which has its own blob
                  path). `key` is `nothing`.
  * `:enum`     — an enum; `key` is its `__enum__` entry in `structs`, whose
                  `julia_type` is the integer it travels as.
  * `:layout`   — a record with a DWARF member layout; `key` is its entry in
                  `structs`. A template or qualified name (`Box<double>`) is
                  looked up too, not only a bare identifier.
  * `:ignore`   — an empty record passed by value. SysV gives it no class, so
                  it takes no register and no stack slot: clang drops the
                  argument, and an empty return is `void`.
  * `:indirect` — a record C++ passes by invisible reference (non-trivial for
                  the purposes of calls) and that has no layout here. As a
                  parameter the callee takes a pointer, which the thunk passes.
  * `:opaque`   — a record the thunk cannot type: only declared here
                  (constructor homing), a toolchain type left out of
                  `structs` (`std::string_view`), a derived class with no
                  members of its own, or a spelling the scalar map does not
                  know (`long double`).

Every `:opaque` crossing used to become `!llvm.ptr`, from `map_cpp_type`'s
fallback or `generate_jlcs_ir`'s bodyless-struct rewrite. A 16-byte
`string_view` then went in as one pointer, an empty `Tag` shifted every later
argument by one register, and a `Box<double>` return was read from RAX while
the callee wrote XMM0. None of these crashed reliably; most returned a
plausible wrong value. `GeneratorCpp` asks this same function, so a function
the thunk cannot call is emitted as a trap on the Julia side, not as a call.
"""
function by_value_crossing(c_type::AbstractString, structs, record_abi)
    t = strip(String(c_type))
    t = replace(t, r"^(?:(?:const|volatile)\s+)+" => "")
    t = String(replace(t, r"(?:\s+(?:const|volatile))+$" => ""))
    (isempty(t) || t in ("void", "unknown")) && return (:direct, nothing)
    (occursin(r"[*&(\[]", t) || startswith(t, "function_ptr")) && return (:direct, nothing)

    mapped = map_cpp_type(t)
    named = startswith(mapped, "!llvm.struct<\"") && endswith(mapped, "\">")
    # A scalar the map knows: `i32`, `f64`, `i1`, …
    (!named && mapped != "!llvm.ptr" && mapped != "") && return (:direct, nothing)
    # STL containers keep their own paths (pointer in, byte blob out).
    get_stl_container_size(t) > 0 && return (:direct, nothing)

    # The tables are keyed on the bare DWARF name (`JoinType`, `Box<double>`);
    # a signature inferred from the demangler spells it qualified
    # (`Clipper2Lib::JoinType`). Longest suffix first, as `_has_receiver` does.
    suffixes = _scope_suffixes(t)
    for cand in suffixes
        haskey(structs, "__enum__" * cand) && return (:enum, "__enum__" * cand)
        key = _fuzzy_struct_lookup(cand, structs)
        key !== nothing && return (:layout, key)
    end

    for cand in suffixes
        facts = get(record_abi, cand, nothing)
        (facts === nothing || get(facts, "conflict", false) == true) && continue
        pass = get(facts, "pass", nothing)
        pass == "reference" && return (:indirect, nothing)
        if get(facts, "empty", false) == true
            bs = try _parse_byte_size(string(get(facts, "byte_size", "1"))) catch; 1 end
            # C++ says pass-by-value; a C record carries no convention, and
            # there only the GNU size-0 empty struct is empty.
            (pass == "value" || (pass === nothing && bs == 0)) && return (:ignore, nothing)
        end
        break
    end
    return (:opaque, nothing)
end

"""
    record_passes_by_reference(c_type, key, record_abi) -> Bool

Whether DWARF says the record `c_type` names (table key `key`, when it has a
layout) is non-trivial for the purposes of calls: `DW_CC_pass_by_reference`. The
Itanium ABI then passes it as a pointer to a caller-owned temporary and returns
it through a hidden sret pointer, whatever its size — a 4-byte class with a
destructor is not returned in EAX. `false` when the facts are absent (metadata
from before `record_abi`, or a C record), which keeps the size-based path.
"""
function record_passes_by_reference(c_type::AbstractString, key, record_abi)::Bool
    t = strip(String(c_type))
    t = replace(t, r"^(?:(?:const|volatile)\s+)+" => "")
    t = String(replace(t, r"(?:\s+(?:const|volatile))+$" => ""))
    cands = key === nothing ? _scope_suffixes(t) : vcat(String(key), _scope_suffixes(t))
    for cand in cands
        facts = get(record_abi, cand, nothing)
        facts === nothing && continue
        get(facts, "conflict", false) == true && return false
        return get(facts, "pass", nothing) == "reference"
    end
    return false
end

"""
    thunk_crossings(func, structs, record_abi) -> NamedTuple

The per-function answer both generators act on: `params` (one
`by_value_crossing` per DWARF parameter, `this` excluded), `ret`, and `trap` —
a human-readable reason when some crossing is `:opaque`, or when the return is
`:indirect` (an sret the thunk does not model), else `nothing`.
"""
function thunk_crossings(func, structs, record_abi)
    # Typed so the thunk generator can prepend `this` as (:direct, nothing).
    params = Tuple{Symbol,Union{Nothing,String}}[
        by_value_crossing(String(get(p, "c_type", "")), structs, record_abi)
        for p in get(func, "parameters", [])]
    ret_c = String(get(get(func, "return_type", Dict()), "c_type", "void"))
    ret = by_value_crossing(ret_c, structs, record_abi)
    reasons = String[]
    for (p, (k, _)) in zip(get(func, "parameters", []), params)
        k === :opaque && push!(reasons, "takes `$(get(p, "c_type", "?"))` by value")
    end
    ret[1] === :opaque && push!(reasons, "returns `$ret_c` by value")
    ret[1] === :indirect && push!(reasons,
        "returns `$ret_c` by value, which C++ returns through a hidden pointer the thunk does not pass")
    return (params = params, ret = ret,
            trap = isempty(reasons) ? nothing : join(reasons, "; "))
end

"""
    _has_receiver(func, structs) -> Bool

True only when `func`'s enclosing scope is a real class/struct, i.e. when the
Itanium ABI actually passes a `this` pointer.

`is_method` is a **string heuristic** upstream — `contains(_name_prefix(demangled),
"::")` ([Compiler.jl](../../Builder/Compiler.jl)) — and Itanium mangles a
namespace-scoped free function and a static member identically
(`_ZN5ImGui10GetVersionEv` vs `_ZN3Obj3getEv`), so it is true for both. Only one
of them has a receiver. DWARF does distinguish them (`DW_TAG_namespace` vs
`DW_TAG_class_type`), but the extractor records neither that nor
`DW_AT_object_pointer`, so the aggregate table is the witness available here — a
namespace is never a key in it. This mirrors the `struct_types` gate the Julia
generator already applies (`GeneratorCpp.jl`, "If it's a namespace name (e.g.
\"pugi\") rather than a class, skip").

Note the aggregate table is keyed on the **bare** type name (`xml_document`),
while `class` carries the full scope (`pugi::xml_document`), so every `::`
suffix is tried. Splitting is angle-bracket-depth aware — a naive `split("::")`
cuts `std::vector<std::string>` at the inner separator. Trying every suffix is
deliberately a superset of the Julia generator's `bare_class`/`safe_class`
check: this gate can only over-admit relative to it, never wrongly deny a real
method its receiver (the failure that direction is silent argument shifting).
"""
function _has_receiver(func, structs)::Bool
    # A constructor or destructor ALWAYS has a receiver — that is a fact about
    # C++, not an inference from the aggregate table, so it is checked FIRST and
    # independently. It matters because the table is not a complete list of the
    # library's classes: a .cpp-local polymorphic class gets no type DIE, so
    # box2d's internal contact subclasses (`b2CircleContact` and six siblings)
    # are absent from it and their destructors were emitted as ZERO-ARGUMENT
    # wrappers — the same shape as the adjustor-thunk bug, found the same day
    # (2026-08-13) by rebuilding box2d for the first time since Jul 17.
    _is_ctor_or_dtor(func) && return true
    # Definition DIE fact. `false` is a static member: the aggregate table
    # still contains the class, and the heuristic below would synthesize a
    # phantom `this` that shifts every argument. `true` is an instance method.
    # Absent (no definition DIE, or a C function) keeps the heuristic.
    hop = get(func, "has_object_pointer", nothing)
    hop isa Bool && return hop
    cls = String(get(func, "class", ""))
    isempty(cls) && return false
    for cand in _scope_suffixes(cls)
        _fuzzy_struct_lookup(cand, structs) !== nothing && return true
    end
    return false
end

"""
    _is_ctor_or_dtor(func) -> Bool

True when `func` is a constructor or destructor, from the demangled `class` and
`name` metadata alone.

**This predicate is duplicated in `GeneratorCpp.jl` and the two copies MUST
agree** — they decide the same argument array from opposite ends of the
pipeline, which is the `ffe_call`/`try_call` hazard shape. The duplication is
deliberate (the generators are isolated on purpose); the protection is
`test_symbol_hygiene.jl` §"Both receiver gates agree", which drives BOTH gates
over every Hub package's metadata and fails on any disagreement.

Tests, and why each is safe in the dangerous direction (a false positive
synthesizes `this` for a free function and shifts every argument):

  * **Destructor** — the name begins with `~`. Nothing else in C++ does. This is
    the same test `GeneratorCpp.jl`'s deleter scan and `_collect_class_raii`
    already use, so all readers of this metadata agree.
  * **Constructor** — the name equals the class's own bare name. A member
    function may not share its class's name; that IS the constructor, so there
    is no false positive to have.

Template arguments are stripped from the class before comparing, because the
demangler prints `Box<int>::Box()` — name `Box`, class `Box<int>`.
"""
function _is_ctor_or_dtor(func)::Bool
    cls   = String(get(func, "class", ""))
    fname = String(get(func, "name", ""))
    (isempty(cls) || isempty(fname)) && return false
    startswith(fname, "~") && return true
    return fname == _bare_type_name(cls)
end

"""
    _bare_type_name(cls) -> String

The innermost `::` component of a scope with template arguments removed:
`std::vector<int, std::allocator<int> >` → `vector`, `pugi::xml_document` →
`xml_document`. Splitting is angle-bracket aware (via [`_scope_suffixes`]), so
a separator inside `<>` is not a cut point.
"""
function _bare_type_name(cls::AbstractString)::String
    inner = last(_scope_suffixes(cls))
    i = findfirst('<', inner)
    return String(strip(i === nothing ? inner : inner[1:prevind(inner, i)]))
end

"""
    _scope_suffixes(name) -> Vector{String}

`name` plus each suffix obtained by dropping leading `::`-separated scopes,
longest first. Separators inside `<>` are ignored, so
`A::B<C::D>` yields `["A::B<C::D>", "B<C::D>"]` — not a cut at `C::D`.
"""
function _scope_suffixes(name::AbstractString)::Vector{String}
    out = String[String(name)]
    depth = 0
    i = 1
    n = lastindex(name)
    while i < n
        c = name[i]
        if c == '<'
            depth += 1
        elseif c == '>'
            depth = max(0, depth - 1)
        elseif depth == 0 && c == ':' && name[i+1] == ':'
            push!(out, String(name[i+2:end]))
            i += 1
        end
        i += 1
    end
    return out
end

"""
    _dwarf_member_offsets(info) -> Vector{Int}

Each member's DWARF byte offset, falling back to `StructGen.get_julia_offsets`
only when a member has none recorded.
"""
function _dwarf_member_offsets(info)
    members = get(info, "members", [])
    offs = Int[]
    for m in members
        o = get(m, "offset", nothing)
        o === nothing && return StructGen.get_julia_offsets(info)
        push!(offs, o isa Integer ? Int(o) : _parse_byte_size(string(o)))
    end
    return offs
end

"""
    _byte_blob_type(byte_size) -> String

Generate a packed MLIR struct type of exactly `byte_size` bytes.
Uses i64 chunks with i8 remainder for compactness.
"""
function _byte_blob_type(byte_size::Int)
    byte_size <= 0 && return "!llvm.ptr"
    n_i64 = div(byte_size, 8)
    n_rem = byte_size % 8
    parts = fill("i64", n_i64)
    for _ in 1:n_rem
        push!(parts, "i8")
    end
    return "!llvm.struct<packed ($(join(parts, ", ")))>"
end

function _parse_byte_size(s::AbstractString)
    startswith(s, "0x") ? parse(Int, s[3:end], base=16) : parse(Int, s)
end

function generate_function_thunks(functions::Vector, structs::Any=Dict(); may_throw::Bool=false,
                                  class_raii::Dict{String,Dict{Symbol,String}}=Dict{String,Dict{Symbol,String}}(),
                                  vcall_info::AbstractDict=Dict{String,Any}(),
                                  record_abi::AbstractDict=Dict{String,Any}())
    io = IOBuffer()
    println(io, "// Function Thunks")

    generated = Set{String}()

    for func in functions
        name = get(func, "name", "")
        mangled = get(func, "mangled", name)

        if mangled in generated
            continue
        end

        # Skip varargs functions — they use va_start/va_end which MLIR cannot handle.
        # These are already routed through direct ccall via varargs interception.
        if get(func, "is_vararg", false)
            continue
        end

        push!(generated, mangled)

        # A function with a by-value record the thunk cannot type gets no
        # thunk: every spelling of one would be a wrong call. GeneratorCpp
        # reaches the same verdict and emits a trap. The line stays in the
        # .mlir, which is the file a debugger shows.
        crossings = thunk_crossings(func, structs, record_abi)
        if crossings.trap !== nothing
            println(io, "// no thunk for $(mangled): $(replace(crossings.trap, '\n' => ' '))")
            println(io, "")
            continue
        end

        params = copy(get(func, "parameters", []))
        # One crossing per entry of `params`, kept aligned when `this` is added.
        param_kinds = copy(crossings.params)
        ret_info = get(func, "return_type", Dict())
        is_method = get(func, "is_method", false)

        # The receiver gate must match the Julia wrapper's, or the two sides
        # disagree about the arg array. Synthesizing `this` for a namespace-scoped
        # free function emits a thunk that loads args_ptr[0] and DEREFERENCES it,
        # while the Julia wrapper — which DOES check (`struct_types`,
        # GeneratorCpp.jl) — correctly emits zero arguments and passes an EMPTY
        # `Ptr{Cvoid}[]`. Slot 0 is then read past the end of that array and the
        # second load segfaults on the first call. Found on Dear ImGui 2026-08-08:
        # 788 thunked `ImGui::` functions, 174 of them zero-arg (immediate SIGSEGV
        # on any call), the remaining 614 with every argument shifted one slot.
        # Two generators, one guard — the same shape as `ffe_call`/`try_call`
        # carrying independent copies of the SysV coercion.
        if is_method && _has_receiver(func, structs)
            if isempty(params) || get(params[1], "name", "") != "this"
                pushfirst!(params, Dict("c_type" => "void*", "name" => "this"))
                pushfirst!(param_kinds, (:direct, nothing))
            end
        end

        # Per-param scope-RAII decision: a by-value class param whose class has
        # an emitted destructor is non-trivial for the purposes of calls under
        # the Itanium ABI — the callee expects a POINTER to a caller-owned
        # temporary, not raw bits in registers. Passing the raw struct (the old
        # behavior) miscompiled those calls. The producer builds the temporary
        # (copy-ctor when resolvable, bit-copy when the copy is trivial), passes
        # its address, and destructs it after the call via jlcs.scope.
        # nothing = not an RAII param; NamedTuple = transform it.
        raii_specs = Vector{Any}(nothing, length(params))
        # What the slot is loaded as, per param, decided once here and read by
        # the call builder below: `nothing` for an ignored param, else
        # (type, packed, struct_info). The call builder used to recompute it,
        # which is how the two ends of one argument drift apart.
        param_load = Vector{Any}(nothing, length(params))

        arg_types = String[]
        for (pi, p) in enumerate(params)
            kind, rec_key = param_kinds[pi]
            # An empty record by value: SysV gives it no class, clang drops the
            # argument, and every later argument keeps its register. The Julia
            # side still fills the slot; nothing reads it.
            kind === :ignore && continue

            t = get(p, "c_type", "void*")
            mlir_t = map_cpp_type(t)
            s_name = nothing
            lookup_key = nothing
            is_packed_param = false

            # Resolve full struct type if available
            if startswith(mlir_t, "!llvm.struct<\"") && endswith(mlir_t, "\">")
                s_name = mlir_t[15:end-2]
                lookup_key = haskey(structs, s_name) ? s_name : (haskey(structs, "__enum__$(s_name)") ? "__enum__$(s_name)" : nothing)
            end
            if lookup_key === nothing && kind === :layout
                # A template or qualified spelling (`Box<double>`,
                # `pugi::xml_node_iterator`) never maps to a named struct, and
                # fell to `!llvm.ptr`: the record's bytes loaded as a pointer.
                s_name = lookup_key = rec_key
            elseif lookup_key === nothing && kind === :enum
                ju = String(get(structs[rec_key], "julia_type", "Cint"))
                mlir_t = map_cpp_type(ju == "Any" ? "int" : ju)
                mlir_t == "" && (mlir_t = "i32")
            elseif kind === :indirect
                mlir_t = "!llvm.ptr"   # Itanium: the callee takes the object's address
            end
            # Laid out, non-trivial for calls, and no destructor to run: the
            # callee still takes a POINTER. `invoke` already put a private copy
            # of the bytes behind the slot (`Ref(arg)`), so that address is the
            # temporary. Passed in registers, the callee read the object's first
            # eightbyte as its address.
            by_address = kind === :layout && lookup_key !== nothing &&
                         !haskey(class_raii, strip(replace(t, r"\bconst\b" => ""))) &&
                         record_passes_by_reference(t, rec_key, record_abi)
            if by_address
                param_load[pi] = (type = "!llvm.ptr", packed = false, info = nothing, address = true)
                push!(arg_types, "!llvm.ptr")
                continue
            end
            if lookup_key !== nothing
                # Scope-RAII takes precedence over both by-value marshalling
                # paths: a class with an emitted destructor is non-trivial
                # for the purposes of calls (Itanium), so the callee expects
                # a POINTER to a caller-owned temporary regardless of byte
                # layout. (is_struct_packed classifies any padding-free
                # struct as "packed", so it must not gate this decision.)
                cls_key = strip(replace(t, r"\bconst\b" => ""))
                bsz = try _parse_byte_size(get(structs[lookup_key], "byte_size", "0")) catch; 0 end
                if haskey(class_raii, cls_key) && bsz > 0
                    raii_specs[pi] = (cls = cls_key, size = bsz,
                                      dtor = class_raii[cls_key][:dtor],
                                      copy_ctor = get(class_raii[cls_key], :copy_ctor, ""))
                    mlir_t = "!llvm.ptr"   # Itanium: pass address of the temporary
                elseif StructGen.is_struct_packed(structs[lookup_key])
                    # For packed structs, use LLVM packed type to avoid type mismatch with thunk marshalling
                    mlir_t = StructGen.get_llvm_equivalent_type_string(s_name, structs[lookup_key], structs)
                    is_packed_param = true
                else
                    # Use FULL definition string to avoid alias parser issues in function signatures
                    mlir_t = StructGen.get_struct_definition_string(s_name, structs[lookup_key], structs)
                end
            end

            param_load[pi] = (type = mlir_t, packed = is_packed_param,
                              info = lookup_key === nothing ? nothing : structs[lookup_key],
                              address = false)
            push!(arg_types, mlir_t)
        end
        has_raii = any(!isnothing, raii_specs)

        ret_c_type = get(ret_info, "c_type", "void")
        ret_type = map_cpp_type(ret_c_type)

        # Track packed struct return for thunk conversion
        is_packed_ret = false
        ret_packed_type = ""
        ret_aligned_type = ""
        ret_struct_info = nothing
        ret_num_members = 0

        ret_kind, ret_key = crossings.ret
        _ret_named = startswith(ret_type, "!llvm.struct<\"") && endswith(ret_type, "\">")
        if ret_kind === :ignore
            # An empty record return: clang makes the callee `void`, and reading
            # RAX for it hands Julia whatever the callee left there.
            ret_type = ""
        elseif ret_kind === :enum && !_ret_named
            ju = String(get(structs[ret_key], "julia_type", "Cint"))
            ret_type = map_cpp_type(ju == "Any" ? "int" : ju)
            ret_type == "" && (ret_type = "i32")
        elseif ret_kind === :layout && !(_ret_named && haskey(structs, ret_type[15:end-2]))
            # A template or qualified spelling takes the named-struct path below
            # under its table key. It used to take the fuzzy branch, which
            # always answered with an integer byte blob: `Box<double>` came back
            # from RAX while the callee returned it in XMM0.
            ret_type = "!llvm.struct<\"$(ret_key)\">"
        end

        # Enum return → bare underlying integer, NOT a single-member struct.
        # A struct result (even `struct<(i32)>`) makes MLIR's emit_c_interface
        # use the sret convention (void ciface(T* sret, void** args)), but the
        # Julia side calls the @enum back as a scalar (T ciface(void** args)) —
        # the args pointer lands in the sret slot and the call derefs garbage.
        # Found live: tinyxml2 XMLDocument::Parse → XMLError. Bare int returns
        # by value, matching the scalar ccall; @enum is ABI-identical to its base.
        if startswith(ret_type, "!llvm.struct<\"") && endswith(ret_type, "\">")
            _ename = ret_type[15:end-2]
            if haskey(structs, "__enum__$(_ename)")
                ju = String(get(structs["__enum__$(_ename)"], "julia_type", "Cint"))
                ret_type = map_cpp_type(ju == "Any" ? "int" : ju)
                ret_type == "" && (ret_type = "i32")
            end
        end

        # Resolve full struct type for return
        if startswith(ret_type, "!llvm.struct<\"") && endswith(ret_type, "\">")
            s_name = ret_type[15:end-2]
            lookup_key = haskey(structs, s_name) ? s_name : (haskey(structs, "__enum__$(s_name)") ? "__enum__$(s_name)" : nothing)
            if lookup_key !== nothing
                # For packed structs, use LLVM packed type for external decl/ffe_call
                if StructGen.is_struct_packed(structs[lookup_key])
                    ret_type = StructGen.get_llvm_equivalent_type_string(s_name, structs[lookup_key], structs)
                    ret_struct_info = structs[lookup_key]
                    aligned = StructGen.get_llvm_aligned_type_string(s_name, ret_struct_info, structs)
                    lay = StructGen._mlir_layout(aligned, structs)
                    c_size = try _parse_byte_size(string(get(ret_struct_info, "byte_size", "0"))) catch; 0 end
                    if lay !== nothing && lay[1] == c_size
                        # Padding-free and naturally aligned: the two spellings
                        # are one layout, and the aligned one is what this
                        # thunk has always returned.
                        is_packed_ret = true
                        ret_packed_type = ret_type
                        ret_aligned_type = aligned
                        ret_num_members = length(get(ret_struct_info, "members", []))
                    end
                    # Otherwise the natural layout is NOT the C layout (a
                    # misaligned member: `__attribute__((packed))`). The wrapper
                    # holds the C layout, so the packed value goes back as it
                    # is; re-laying it out wrote an 8-byte `{char; int}` into
                    # a 5-byte Julia value.
                else
                    # Check if the struct has members that can't be sized correctly in MLIR.
                    # When members are template types with size=0, get_struct_definition_string
                    # would produce a struct with wrong total size. Use a byte blob instead.
                    info = structs[lookup_key]
                    byte_size = try _parse_byte_size(get(info, "byte_size", "0")) catch; 0 end
                    members = get(info, "members", [])
                    has_unsizable = any(m -> begin
                        ms = try; s = get(m, "size", 0); s isa String ? parse(Int, s) : s; catch; 0; end
                        ct = strip(replace(get(m, "c_type", ""), r"\bconst\b" => ""))
                        ms == 0 && !endswith(ct, "*") && !endswith(ct, "&") && !occursin(r"^[A-Za-z0-9_]+$", ct)
                    end, members)
                    if has_unsizable && byte_size > 0
                        ret_type = _byte_blob_type(byte_size)
                    else
                        ret_type = StructGen.get_struct_definition_string(s_name, structs[lookup_key], structs)
                    end
                end
            end
        elseif ret_type == "!llvm.ptr" && !contains(ret_c_type, "*") && ret_c_type != "void" && ret_c_type != "unknown"
            # Template types (e.g. "Matrix<double, -1, -1>") fall through map_cpp_type
            # to !llvm.ptr because they contain non-alphanumeric chars.
            # Fuzzy-match against DWARF struct definitions to recover the correct size.
            matched_key = _fuzzy_struct_lookup(ret_c_type, structs)
            if matched_key !== nothing
                info = structs[matched_key]
                byte_size = try _parse_byte_size(get(info, "byte_size", "0")) catch; 0 end
                if byte_size > 0
                    ret_type = _byte_blob_type(byte_size)
                end
            else
                # Fallback for known STL containers that might be missing from DWARF struct_defs
                stl_size = get_stl_container_size(ret_c_type)
                if stl_size > 0
                    ret_type = _byte_blob_type(stl_size)
                end
            end
        end

        # A record that is non-trivial for the purposes of calls comes back
        # through a hidden pointer — Itanium puts it FIRST, before `this` — and
        # the callee returns void, at every size. The size-based classifier
        # only agrees above 16 bytes (MEMORY class); below that it read EAX
        # while the callee stored the object through whatever the first
        # argument register held. The thunk owns the slot and returns the
        # bytes, exactly as for a MEMORY-class return.
        ret_sret = ret_kind === :layout && ret_type != "" &&
                   record_passes_by_reference(ret_c_type, ret_key, record_abi)
        sret_slot_type = is_packed_ret ? ret_packed_type : ret_type

        # External declaration uses the actual C type (packed for packed structs)
        ext_ret_type = ret_sret ? "" : ret_type
        ext_mlir_ret = ext_ret_type == "" || ext_ret_type == "!llvm.void" ? "" : "-> $ext_ret_type"

        # Thunk returns aligned type so Julia can read it correctly
        thunk_ret_type = is_packed_ret ? ret_aligned_type : ret_type
        thunk_mlir_ret = thunk_ret_type == "" || thunk_ret_type == "!llvm.void" ? "" : "-> $thunk_ret_type"
        func_ret = ret_sret || ret_type == "" || ret_type == "!llvm.void" ? "" : ret_type
        ret_sret && pushfirst!(arg_types, "!llvm.ptr")

        # The epilogue after a call that returns nothing: plain `return`, or
        # the sret slot read back (and re-laid-out when the C layout is packed).
        emit_void_epilogue = function ()
            if !ret_sret
                println(io, "  return")
            elseif is_packed_ret
                println(io, "  %sret_val = llvm.load %sret_slot : !llvm.ptr -> $(sret_slot_type)")
                println(io, "  %ret_aligned = jlcs.marshal_ret %sret_val { numMembers = $(ret_num_members) : i64 } : ($(ret_packed_type)) -> $(ret_aligned_type)")
                println(io, "  return %ret_aligned : $(ret_aligned_type)")
            else
                println(io, "  %sret_val = llvm.load %sret_slot : !llvm.ptr -> $(sret_slot_type)")
                println(io, "  return %sret_val : $(sret_slot_type)")
            end
        end

        # 1. External Declaration (Real C++ Symbol)
        # Use real mangled name so JIT can link to it
        println(io, "func.func private @$(mangled)($(join(arg_types, ", "))) $ext_mlir_ret")

        # 2. Thunk (Exposed to JIT)
        # Append _thunk suffix to avoid collision
        # Add llvm.emit_c_interface to generate _mlir_ciface_ wrapper for invokePacked
        println(io, "func.func @$(mangled)_thunk(%args_ptr: !llvm.ptr) $(thunk_mlir_ret) attributes { llvm.emit_c_interface } {")
        if ret_sret
            println(io, "  %sret_one = llvm.mlir.constant(1 : i64) : i64")
            println(io, "  %sret_slot = llvm.alloca %sret_one x $(sret_slot_type) : (i64) -> !llvm.ptr")
        end

        # Scope-RAII entry allocas: one caller-owned temporary per non-trivial
        # by-value param, plus (for non-void calls) a slot the call result
        # escapes through — jlcs.scope has no results, so values leave via memory.
        raii_ret_slot_type = ""
        if has_raii
            println(io, "  %raii_one = llvm.mlir.constant(1 : i64) : i64")
            for (pi, spec) in enumerate(raii_specs)
                isnothing(spec) && continue
                println(io, "  %raii_tmp_$(pi) = llvm.alloca %raii_one x !llvm.array<$(spec.size) x i8> : (i64) -> !llvm.ptr")
            end
            raii_ret_slot_type = ret_sret ? "" : (is_packed_ret ? ret_packed_type : func_ret)
            if raii_ret_slot_type != ""
                println(io, "  %raii_retslot = llvm.alloca %raii_one x $(raii_ret_slot_type) : (i64) -> !llvm.ptr")
            end
        end

        call_args = String[]

        for (i, p) in enumerate(params)
            # The type decided with arg_types above; `nothing` is an ignored
            # empty record, whose slot is never read. The slot OFFSET still
            # counts it, because the Julia side passes every argument.
            load = param_load[i]
            load === nothing && continue
            mlir_t = load.type
            is_packed_struct = load.packed
            struct_info = load.info

            # Argument slot read. The ciface convention hands the thunk a
            # `void**`, so slot i-1 is a field at byte offset 8*(i-1) — which
            # is what jlcs.get_field says, in one op, instead of a constant
            # plus a pointer-scaled GEP plus a load. Same address arithmetic:
            # `getelementptr ptr, ptr %args, i64 N` and
            # `getelementptr i8, ptr %args, i64 8N` compute the same address,
            # and the emitted machine code is instruction-identical (proven
            # per-shape in test_jlcs_producers.jl §E).
            #
            # This is the op stating the convention that CLAUDE.md records as
            # having cost a debugging session: the result is the address of
            # the argument's STORAGE, so scalars and pointers both need the
            # second load below. The .mlir is the debugger's source file, so
            # what is written here is what gdb shows at the breakpoint.
            slot_off = 8 * (i - 1)
            println(io, "  %val_ptr_$(i) = \"jlcs.get_field\"(%args_ptr) {fieldOffset = $(slot_off) : i64} : (!llvm.ptr) -> !llvm.ptr")

            if load.address
                push!(call_args, "%val_ptr_$(i)")   # the slot IS the temporary's address
                continue
            end

            if !isnothing(raii_specs[i])
                # Non-trivial by-value param: the temporary is copy-constructed
                # inside the jlcs.scope (emitted at the call site below); here we
                # only record the source pointer. The callee receives %raii_tmp_i.
                push!(call_args, "%raii_tmp_$(i)")
                continue
            end

            if is_packed_struct
                # Emit jlcs.marshal_arg op — the MLIR lowering pass handles the field-by-field reconstruction.
                llvm_t = mlir_t  # Already the LLVM packed type

                # Where each member sits in the JULIA value behind the slot. The
                # wrapper emits every struct at its C layout — fields proven to
                # land on the DWARF offsets, or a byte blob when they cannot
                # (GeneratorCpp `_prove_julia_layout`) — so those are the DWARF
                # offsets. `get_julia_offsets` guessed them from `min(size, 8)`
                # alignment, which put a packed `{char; int}`'s int at 4 and a
                # `char data[3]` member at 3.
                offsets = _dwarf_member_offsets(struct_info)
                members = get(struct_info, "members", [])
                member_types_strs = [map_cpp_type(get(m, "c_type", "void*")) for m in members]

                member_types_attr = "[" * join(member_types_strs, ", ") * "]"
                offsets_attr = "[" * join(["$(o) : i64" for o in offsets], ", ") * "]"

                println(io, "  %packed_$(i) = jlcs.marshal_arg %val_ptr_$(i) { memberTypes = $(member_types_attr), juliaOffsets = $(offsets_attr) } : (!llvm.ptr) -> $(llvm_t)")
                arg_value_name = "%packed_$(i)"
            else
                println(io, "  %val_$(i) = llvm.load %val_ptr_$(i) : !llvm.ptr -> $(mlir_t)")
                arg_value_name = "%val_$(i)"
            end

            push!(call_args, arg_value_name)
        end

        # The hidden return pointer is the first argument (declared above).
        ret_sret && pushfirst!(call_args, "%sret_slot")

        # Determine whether to use jlcs.try_call (exception-safe) or jlcs.ffe_call
        # Per-function noexcept: if the function is marked noexcept, use ffe_call even when may_throw is set
        func_is_noexcept = get(func, "is_noexcept", false)
        use_try_call = may_throw && !func_is_noexcept
        call_op = use_try_call ? "jlcs.try_call" : "jlcs.ffe_call"

        # Virtual dispatch: methods with vtable coordinates get a jlcs.vcall
        # (read vptr → index slot → indirect call) so OVERRIDES are honored —
        # a direct symbol call is `p->Class::method()` static semantics. The
        # vcall lowering does no sret/packed ABI coercion, so only
        # scalar/pointer signatures are eligible; struct-shaped ones keep the
        # direct-call path (rare for virtual methods). may_throw on the op
        # picks the invoke+landing-pad lowering (same sentinel-continue EH
        # model as try_call).
        vc = get(vcall_info, mangled, nothing)
        use_vcall = vc !== nothing && is_method && !has_raii && !is_packed_ret && !ret_sret &&
                    !isempty(call_args) &&
                    !occursin("struct", func_ret) &&
                    !any(t -> occursin("struct", t) || occursin("array", t), arg_types)
        if use_vcall
            vcall_attrs = "class_name = @$(vc.class), vtable_offset = $(vc.vtable_offset) : i64, " *
                          "slot = $(vc.slot) : i64, this_offset = 0 : i64"
            use_try_call && (vcall_attrs *= ", may_throw")
        end

        # Destructor thunks: jlcs.dtor_call encodes exactly this shape — one
        # object pointer in, void out — so the ABI coercion ffe_call/try_call
        # carry has nothing to do, and the op states in the IR what the symbol
        # name only implies. Destructor-ness is tested the same way both other
        # readers of this metadata test it (`~` in the demangled name,
        # GeneratorCpp.jl:125, JLCSIRGenerator._collect_class_raii).
        #
        # The ARITY GATE IS LOAD-BEARING, not defensive: under Itanium a
        # base-object destructor (D2) of a class with virtual bases takes a
        # second VTT argument — vi_test has three (`_ZN4LeftD2Ev(this, vtt)`)
        # — and dtor_call has no operand to carry it. Those keep the
        # ffe_call/try_call path; emitting dtor_call for them would drop the
        # VTT silently. Same reason the gate is on the FINAL call_args, after
        # `this` synthesis, rather than on the DWARF parameter list.
        #
        # may_throw comes from the same `may_throw && !is_noexcept` rule that
        # picks try_call over ffe_call, so a thunk moving onto this op keeps
        # the landing pad it had. (DWARF marks none of these noexcept, so in
        # practice C++ destructor thunks land on the EH path.)
        is_destructor = occursin("~", String(get(func, "demangled", "")))
        use_dtor_call = is_destructor && !has_raii && !use_vcall && !ret_sret &&
                        func_ret == "" && length(call_args) == 1 &&
                        length(arg_types) == 1 && arg_types[1] == "!llvm.ptr"
        dtor_attrs = use_try_call ? " { may_throw }" : ""

        # Call using jlcs.ffe_call or jlcs.try_call (via Dialect)
        if has_raii
            # Scope-RAII: copy-construct temporaries, call inside the scope,
            # destructors fire in reverse order at scope exit. try_call converts
            # C++ exceptions to sentinel-return-and-continue, so the normal path
            # (where the scope emits dtors) is the only path — coverage is total.
            raii_idx = [i for i in 1:length(params) if !isnothing(raii_specs[i])]
            tmp_list = join(["%raii_tmp_$(i)" for i in raii_idx], ", ")
            tmp_types = join(fill("!llvm.ptr", length(raii_idx)), ", ")
            # managed_ptrs and dtors are co-generated from the same index list —
            # the arity invariant the (still unverified) op relies on
            dtor_list = join(["@$(raii_specs[i].dtor)" for i in raii_idx], ", ")

            println(io, "  jlcs.scope($(tmp_list) : $(tmp_types)) dtors([$(dtor_list)]) {")
            for i in raii_idx
                spec = raii_specs[i]
                if !isempty(spec.copy_ctor)
                    println(io, "    jlcs.ctor_call @$(spec.copy_ctor)(%raii_tmp_$(i), %val_ptr_$(i)) : (!llvm.ptr, !llvm.ptr) -> ()")
                else
                    # No copy-ctor symbol → copy is trivial; bit-copy the bytes
                    println(io, "    %raii_blob_$(i) = llvm.load %val_ptr_$(i) : !llvm.ptr -> !llvm.array<$(spec.size) x i8>")
                    println(io, "    llvm.store %raii_blob_$(i), %raii_tmp_$(i) : !llvm.array<$(spec.size) x i8>, !llvm.ptr")
                end
            end
            if func_ret == ""
                println(io, "    $(call_op) $(join(call_args, ", ")) { callee = @$(mangled) } : ($(join(arg_types, ", "))) -> ()")
            else
                inner_ret = is_packed_ret ? ret_packed_type : func_ret
                println(io, "    %raii_ret = $(call_op) $(join(call_args, ", ")) { callee = @$(mangled) } : ($(join(arg_types, ", "))) -> $(inner_ret)")
                println(io, "    llvm.store %raii_ret, %raii_retslot : $(inner_ret), !llvm.ptr")
            end
            println(io, "    jlcs.yield")
            println(io, "  }")
            if func_ret == ""
                emit_void_epilogue()
            elseif is_packed_ret
                println(io, "  %ret_packed = llvm.load %raii_retslot : !llvm.ptr -> $(ret_packed_type)")
                println(io, "  %ret_aligned = jlcs.marshal_ret %ret_packed { numMembers = $(ret_num_members) : i64 } : ($(ret_packed_type)) -> $(ret_aligned_type)")
                println(io, "  return %ret_aligned : $(ret_aligned_type)")
            else
                println(io, "  %ret_val = llvm.load %raii_retslot : !llvm.ptr -> $(func_ret)")
                println(io, "  return %ret_val : $(func_ret)")
            end
        elseif func_ret == ""
             # Void return
             if use_vcall
                 println(io, "  \"jlcs.vcall\"($(join(call_args, ", "))) { $(vcall_attrs) } : ($(join(arg_types, ", "))) -> ()")
             elseif use_dtor_call
                 println(io, "  jlcs.dtor_call @$(mangled)($(call_args[1]))$(dtor_attrs) : (!llvm.ptr) -> ()")
             else
                 println(io, "  $(call_op) $(join(call_args, ", ")) { callee = @$(mangled) } : ($(join(arg_types, ", "))) -> ()")
             end
             emit_void_epilogue()
        elseif is_packed_ret
             # Packed struct return: call returns packed type, marshal to aligned for Julia
             println(io, "  %ret_packed = $(call_op) $(join(call_args, ", ")) { callee = @$(mangled) } : ($(join(arg_types, ", "))) -> $(ret_packed_type)")
             println(io, "  %ret_aligned = jlcs.marshal_ret %ret_packed { numMembers = $(ret_num_members) : i64 } : ($(ret_packed_type)) -> $(ret_aligned_type)")
             println(io, "  return %ret_aligned : $(ret_aligned_type)")
        else
             # Value return
             if use_vcall
                 println(io, "  %ret_val = \"jlcs.vcall\"($(join(call_args, ", "))) { $(vcall_attrs) } : ($(join(arg_types, ", "))) -> $(func_ret)")
             else
                 println(io, "  %ret_val = $(call_op) $(join(call_args, ", ")) { callee = @$(mangled) } : ($(join(arg_types, ", "))) -> $(func_ret)")
             end
             println(io, "  return %ret_val : $(func_ret)")
        end

        println(io, "}")
        println(io, "")
    end

    return String(take!(io))
end

end
