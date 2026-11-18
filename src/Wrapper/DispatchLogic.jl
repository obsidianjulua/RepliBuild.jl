# =============================================================================
# Dispatch logic — Per-function tier routing decisions
# =============================================================================
# Separated from Utils.jl because is_ccall_safe() depends on
# is_stl_container_type() (defined in Cpp/TypesCpp.jl), which is included
# later in the load order.  is_c_lto_safe() has no such dependency but lives
# here to keep all dispatch decisions in one file.

"""
    _is_enum_type(type_name, dwarf_structs) -> Bool

Check if a type name corresponds to an enum in DWARF metadata.
Enums are stored with `__enum__` prefix (e.g. `__enum__MathOp`).
Enums are integers at the ABI level — always safe for ccall.
"""
function _is_enum_type(type_name::AbstractString, dwarf_structs)::Bool
    return haskey(dwarf_structs, "__enum__$(type_name)")
end

# =============================================================================
# is_c_lto_safe() — C-specific dispatch gate
# =============================================================================

"""
    is_c_lto_safe(func_info, dwarf_structs)::Bool

Determine if a C function can be dispatched via direct ccall/llvmcall.
Returns false only when the **return type** is:

- A record whose layout is not natural (`_record_natural_layout`: packed,
  `#pragma pack`, an explicit alignment, a bitfield or union inside, or a
  member type it cannot measure)
- A union by value

Such a return is a candidate for GeneratorC's explicit-sret branch, which
decides per target where C returns it (`_c_explicit_sret`), and is never a
Tier 1 slice candidate.

C has no templates, vtables, STL, or inheritance — the only ABI hazards
are packed struct returns and union returns where ccall's return convention
doesn't match the actual layout.

By-value parameters are not decided here: GeneratorC's blob guards check them.
"""
function is_c_lto_safe(func_info, dwarf_structs)::Bool
    ret_type = String(get(func_info["return_type"], "c_type", ""))

    # Void, pointer, primitive returns — always safe
    if ret_type == "void" || contains(ret_type, "*") || _is_primitive_type(ret_type)
        return true
    end

    # Enum returns are integers at the ABI level — safe
    cleaned_ret = strip(replace(ret_type, r"\bconst\b" => ""))
    if _is_enum_type(cleaned_ret, dwarf_structs)
        return true
    end

    # Check struct return safety
    if haskey(dwarf_structs, cleaned_ret)
        s_info = dwarf_structs[cleaned_ret]

        # Union return — ccall can't handle by-value union returns
        if get(s_info, "kind", "struct") == "union"
            return false
        end

        # A record whose layout is not natural (the same question C++'s
        # `_is_struct_unsafe` asks)
        _record_natural_layout(s_info, dwarf_structs) === true || return false
    end

    return true
end

"""
    _is_noexcept(func_info) -> Bool

Check if a function is marked `noexcept` in metadata.
Functions marked noexcept are guaranteed not to throw C++ exceptions,
so they can safely use ccall/llvmcall without exception handling.
"""
function _is_noexcept(func_info)::Bool
    return get(func_info, "is_noexcept", false)
end

"""
    _is_cpp_function(func_info) -> Bool

Check if a function has C++ linkage (Itanium ABI mangled name starting with `_Z`).
C++ functions may throw exceptions unless marked `noexcept`.
An unmangled name says nothing either way: see `_is_nounwind`.
"""
function _is_cpp_function(func_info)::Bool
    mangled = get(func_info, "mangled", "")
    return startswith(mangled, "_Z")
end

"""
    _is_nounwind(func_info) -> Bool

Whether the function's own definition is `nounwind` in the IR the library was
linked from (metadata `nounwind`, recorded by `Compiler._mark_noexcept!` from
`_nounwind_definitions`), so no exception can leave it. `is_noexcept` is that
proof plus a `noexcept` in the sources, and implies it.

This is the question for a function with C linkage in a C++ library.
`extern "C"` changes the name, not the language of the body: `extern "C" int
f(int x) { if (x < 0) throw std::runtime_error("…"); return x; }` throws, and
through a plain `ccall` there is no landing pad, so the process ends in
`std::terminate` (2026-11-17). Metadata without the key (built before it was
recorded, or with no IR to read) proves nothing, and the function takes the
thunk, like any C++ function that may throw.
"""
function _is_nounwind(func_info)::Bool
    return get(func_info, "nounwind", false) === true || _is_noexcept(func_info)
end

"""
    _lacks_debug_info(func_info) -> Bool

Whether no DIE describes this symbol: its return type was not read from DWARF
(`return_type_source == "inferred"`, set by `extract_compilation_metadata` when
neither the symbol nor an exported function sharing its address has one). What the
metadata holds for it then is a guess from its name, which for a C symbol is no
parameters and `Cint`. Both generators emit `_no_debug_info_stub` for it
instead of a call. A function with a DIE and no prototype (K&R `int f()`) has
its return type from DWARF and is not one of these.
"""
_lacks_debug_info(func_info)::Bool = get(func_info, "return_type_source", "") == "inferred"

"""
    _no_debug_info_stub(julia_name, mangled, demangled, metadata, generate_docs) -> String

The chunk both generators emit for a function `_lacks_debug_info` names: a
method taking anything that raises, saying the library has no debug
information for the symbol. Its parameters are unknown, so it has no typed
signature a call could match instead.
"""
function _no_debug_info_stub(julia_name::AbstractString, mangled::AbstractString,
                             demangled::AbstractString, metadata, generate_docs::Bool)::String
    msg = "ABI Safety Trap: cannot call '$julia_name'. The library exports `$mangled` and has no " *
          "debug information for it: no DWARF entry describes that symbol, or an exported function " *
          "at its address, so its parameters and return type are unknown and no call could pass them " *
          "correctly. Compile the file that defines it with `-g`, or keep it out with " *
          "[wrap] exclude_symbols."
    get(metadata, "alias_signatures", false) === true ||
        (msg *= " This metadata was written before an alias took the signature of the function " *
                "it shares an address with: if `$mangled` is one, rebuild the package (delete " *
                ".replibuild_cache/project_hash) and wrap again.")
    doc = generate_docs ? """
    \"\"\"
        $julia_name(args...)

    `$demangled` has no debug information in this library, so its signature is
    unknown. Calling it raises an error.

    # Metadata
    - Mangled symbol: `$mangled`
    \"\"\"
    """ : ""
    return doc * """
    function $julia_name(args...)
        Base.error($(repr(msg)))
    end

    """
end

"""
    is_ccall_safe(func_info, dwarf_structs)::Bool

Determine if a function is safe for standard `ccall`.
Returns false (→ route to MLIR Tier 2) if any of:

**Exception safety:**
- C++ function not marked `noexcept` (may throw → needs MLIR try_call)
- C-linkage function whose definition is not proven `nounwind` (`_is_nounwind`)

**Return type:**
- STL container by value
- Template type (contains `<`)
- Not a natural layout (`_record_natural_layout`: packed, a bitfield or union inside, unmeasurable)
- Over-aligned (`_c_record_align` > 8: the record's own, a member's, a nested record's)
- Polymorphic (has vtable)
- Inherits from base class
- Non-POD class with members
- Contains unsafe nested struct member
- Unknown struct not in DWARF (can't verify safety)

**Parameters (by value, not pointer/ref):**
- STL container
- Template type (contains `<`)
- Union
- Not a natural layout
- Overaligned
- Polymorphic / inherits
- Contains unsafe nested member
- Unknown struct not in DWARF
"""
function is_ccall_safe(func_info, dwarf_structs; record_abi=Dict{String,Any}())
    # ── -1. Exception safety: C++ functions that may throw ───────────────
    # Conservative: all C++ functions (mangled _Z* names) may throw unless
    # noexcept. Route to MLIR Tier 2 for exception-safe thunks.
    # A function with C linkage is C++ too here (only GeneratorCpp asks), and
    # reaches ccall only when its definition is proven `nounwind`
    # (`_is_nounwind`).
    if _is_cpp_function(func_info) ? !_is_noexcept(func_info) : !_is_nounwind(func_info)
        return false
    end

    # ── -0.5. Non-trivial for calls (DWARF DW_CC_pass_by_reference) ───────
    # Such a record travels as a pointer to a temporary and comes back
    # through a hidden sret pointer, at any size. `ccall` lays a by-value
    # struct out by the C rules instead, so a `noexcept` function taking or
    # returning one goes to the thunk, which does both.
    let FG = JLCSIRGenerator.FunctionGen
        for t in Iterators.flatten(((get(func_info["return_type"], "c_type", ""),),
                                    (get(p, "c_type", "") for p in func_info["parameters"])))
            s = String(t)
            (isempty(s) || occursin(r"[*&(\[]", FG._outside_template_args(s))) && continue
            FG.record_passes_by_reference(s, nothing, record_abi) && return false
            # A name several types share (`FG.conflicted_within`): the Julia
            # struct is one of them, maybe not this function's. The thunk path
            # is where the trap is emitted. An empty record (`:ignore`,
            # `:regpad`) is one the convention drops or pads, which a ccall of
            # its one-byte Julia type does not: the thunk does (2026-11-14,
            # when every defined record gained an entry).
            FG.by_value_crossing(s, dwarf_structs, record_abi)[1] in (:conflict, :ignore, :regpad) && return false
        end
    end

    # ── -0.25. SIMD vectors by value ──────────────────────────────────────
    # C passes a vector by its size (one SSE register on SysV, by reference on
    # Win64) and `ccall` passes the tuple of its lanes, and the two agree only
    # by luck. A bare vector (`crosses_as_vector`) and a record holding one at
    # any depth (`holds_vector`) take the thunk, which states a record's class
    # from DWARF and refuses a bare vector on every host. A plain ccall of the
    # same shape read the wrong registers on SysV and faulted on Win64; the C
    # generator had refused these since 2026-10-31 and this gate had not
    # (2026-11-01).
    let FG = JLCSIRGenerator.FunctionGen
        FG.crosses_as_vector(func_info["return_type"]) && return false
        any(p -> FG.crosses_as_vector(p), func_info["parameters"]) && return false
        for t in Iterators.flatten(((get(func_info["return_type"], "c_type", ""),),
                                    (get(p, "c_type", "") for p in func_info["parameters"])))
            s = String(strip(replace(String(t), r"\bconst\b" => "")))
            s = String(replace(s, r"^(?:struct|union|class|enum)\s+" => ""))
            (isempty(s) || occursin(r"[*&(\[]", FG._outside_template_args(s))) && continue
            FG.holds_vector(s, dwarf_structs) && return false
        end
    end

    # ── -0.2. 128-bit integers by value ───────────────────────────────────
    # `ccall` passes an `Int128` as two separate halves, which is where C
    # puts one only while both halves get registers (`_c_int128_misplaced`);
    # a thunk passes `i128`. `__int128` always took the thunk, as an
    # identifier the primitive set below does not hold; `unsigned __int128`,
    # which is no identifier, did not.
    any(e -> String(get(e, "julia_type", "")) in _C_INT128_TYPES,
        Iterators.flatten(((func_info["return_type"],), func_info["parameters"]))) && return false

    # ── 0. STL containers (never ccall-safe by value) ───────────────────
    ret_type_str = String(get(func_info["return_type"], "c_type", ""))
    if is_stl_container_type(ret_type_str)
        return false
    end
    for param in func_info["parameters"]
        c_type = get(param, "c_type", "")
        if contains(c_type, "&") || contains(c_type, "*")
            continue
        end
        if is_stl_container_type(c_type)
            return false
        end
    end

    # ── 1. Return type analysis ─────────────────────────────────────────
    ret_type = ret_type_str

    # Quick exit: pointer, void, or primitive returns are safe
    is_template_ret = occursin('<', ret_type)
    is_primitive_ret = !is_template_ret && _is_primitive_type(ret_type)
    is_pointer_ret = contains(ret_type, "*")
    is_void_ret = ret_type == "void" || strip(replace(ret_type, r"\bconst\b" => "")) == "void"

    if !is_pointer_ret && !is_void_ret && !is_primitive_ret
        # Template return types — always route to MLIR.
        # DWARF may not have the exact specialization.
        if is_template_ret
            return false
        end

        # Enums are integers at the ABI level — always safe
        cleaned_ret = strip(replace(ret_type, r"\bconst\b" => ""))
        if _is_enum_type(cleaned_ret, dwarf_structs)
            # safe, fall through
        elseif haskey(dwarf_structs, cleaned_ret)
            s_info = dwarf_structs[cleaned_ret]

            # Full struct safety check (packed, overaligned, polymorphic, etc.)
            if _is_struct_unsafe(s_info, dwarf_structs)
                return false
            end
        else
            # Return type is a struct name but NOT in DWARF — can't verify safety.
            # If it looks like an identifier (not a known primitive), route to MLIR.
            if occursin(r"^[A-Za-z_][A-Za-z0-9_:]*$", cleaned_ret) && !_is_primitive_type(cleaned_ret)
                return false
            end
        end
    end

    # ── 2. Parameter analysis ───────────────────────────────────────────
    for param in func_info["parameters"]
        c_type = get(param, "c_type", "")

        # Pointers and references are always safe at ABI level
        if contains(c_type, "*") || contains(c_type, "&")
            continue
        end

        # Template parameter by value — always route to MLIR
        if occursin('<', c_type)
            return false
        end

        # Clean const prefix for base type lookup
        base_type = String(strip(replace(c_type, r"\bconst\b" => "")))

        # Enums are integers at the ABI level — always safe
        if _is_enum_type(base_type, dwarf_structs)
            continue
        end

        if haskey(dwarf_structs, base_type)
            s_info = dwarf_structs[base_type]

            # Full struct safety check
            if _is_struct_unsafe(s_info, dwarf_structs)
                return false
            end
        else
            # Unknown struct parameter not in DWARF — can't verify safety
            cleaned = strip(base_type)
            if occursin(r"^[A-Za-z_][A-Za-z0-9_:]*$", cleaned) && !_is_primitive_type(cleaned)
                return false
            end
        end
    end

    return true
end
