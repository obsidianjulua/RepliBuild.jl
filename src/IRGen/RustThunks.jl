# RustThunks.jl — thunks that call a Rust function the way rustc compiled it.
#
# A Rust function's call is whatever its `define` line says (`RustABI`). Where
# every type on that line is a scalar or a pointer, `ccall` makes the same
# call. Where one is not (a pair returned in two registers, a three-byte
# integer, `i128`, a hidden return pointer on AArch64), nothing in Julia can
# spell the call, and a thunk makes it: a small function, generated here as
# MLIR, whose body is one `jlcs.native_call` carrying the define line's own
# types and attributes. The dialect lowers that op to exactly that call.
#
# The thunk's own signature is one that `ccall` does spell, on every host:
#
#     void thunk(void **args, void *ret)
#
#   args[k]   points at the storage of the k-th IR parameter's value, in the
#             order of the define line, the hidden return pointer left out.
#             A parameter that is itself a pointer (`ptr`) is stored as a
#             pointer: one slot per IR parameter, no exceptions.
#   ret       where the result goes. For a function that returns through a
#             hidden pointer it IS that pointer: Rust builds the value in the
#             caller's buffer, which must be as large and as aligned as the
#             type. Otherwise the returned scalar is stored there, and the
#             two scalars of a pair each at the offset DWARF gives it.
#
# A `bool` (`i1`) is a byte in storage, on both sides.
#
# The thunk makes a PLAIN call. A Rust function that panics unwinds through
# it, and only Rust may end a Rust panic: the catching frame is a Rust
# function outside the thunk, which is not built yet. Until it is, a panic in
# a call made here ends the process.
#
# Design: docs/updates/RUST_FRONT_END.md §13 and Part III.

module RustThunks

using Libdl

import ..MLIRNative
import ..RustABI

export thunk_name, thunk_module, load, call_raw

"The thunk's name for the function `symbol`."
thunk_name(symbol::AbstractString) = String(symbol) * "_rbthunk"

# An MLIR symbol reference: bare where the name allows it, quoted otherwise.
_sym(name::AbstractString) = occursin(r"^[A-Za-z_$.][A-Za-z0-9_$.-]*$", name) ? "@" * name : "@" * repr(String(name))

"""
    mlir_type(ir_type) -> Union{String,Nothing}

The MLIR spelling of an LLVM type on a `define` line: integers of any width,
the floating types, `ptr`, and a literal struct of those. `nothing` for a
type a thunk does not carry (a vector, an array, a named struct).
"""
function mlir_type(t::AbstractString)::Union{String,Nothing}
    t = strip(t)
    t == "ptr" && return "!llvm.ptr"
    t == "double" && return "f64"
    t == "float" && return "f32"
    t == "half" && return "f16"
    t == "fp128" && return "f128"
    occursin(r"^i\d+$", t) && return String(t)
    if startswith(t, "{") && endswith(t, "}")
        elems = RustABI._brace_elements(t)
        isempty(elems) && return nothing
        parts = String[]
        for e in elems
            m = mlir_type(e)
            m === nothing && return nothing
            push!(parts, m)
        end
        return "!llvm.struct<(" * join(parts, ", ") * ")>"
    end
    return nothing
end

"""
    _arg_attrs(attrs) -> Union{String,Nothing}

The `arg_attrs` dictionary for one IR parameter: the attributes that change
how the argument is passed, under the LLVM dialect's names. Everything else
on a `define` line (`noundef`, `noalias`, `nonnull`, `dereferenceable`,
`captures`, `range`, `readonly`, …) says something about the value, not about
the call, and is left out. `nothing` for an attribute that does change the
call and that this does not carry.
"""
function _arg_attrs(attrs)::Union{String,Nothing}
    out = String[]
    memory = false
    align = nothing
    for a in attrs
        if a in ("zeroext", "signext", "inreg")
            push!(out, "llvm." * a)
        elseif startswith(a, "sret(") || startswith(a, "byval(")
            m = match(r"^(sret|byval)\(\[(\d+) x i8\]\)$", a)
            m === nothing && return nothing
            push!(out, "llvm.$(m.captures[1]) = !llvm.array<$(m.captures[2]) x i8>")
            memory = true
        elseif startswith(a, "align ")
            align = tryparse(Int, a[7:end])
        elseif startswith(a, "inalloca") || startswith(a, "preallocated") || a == "nest" ||
               startswith(a, "byref(") || startswith(a, "swift")
            return nothing
        end
    end
    # An alignment is part of the call only for memory the call itself
    # describes; on a plain pointer it is the callee's assumption about the
    # caller's value.
    (memory && align !== nothing) && push!(out, "llvm.align = $align : i64")
    return "{" * join(out, ", ") * "}"
end

"""
    thunk(row, types) -> (text, callee_decl) or a refusal String

One function's thunk and its callee's declaration, from its metadata row
(`RustABI.join_function`) and the type table. A `String` says why there is
no thunk: the row is refused, or its signature has a type or an attribute a
thunk does not carry.
"""
function thunk(row::AbstractDict, types::AbstractDict)
    row["refusal"] === nothing || return "the function is refused: " * String(row["refusal"])
    sym = String(row["symbol"])
    ir = row["ir"]
    params = ir["parameters"]
    ret = row["return"]
    ret_ir = String(ir["return"]["type"])
    sret = get(ret, "pass", "") == "indirect"

    arg_types = String[]
    arg_attrs = String[]
    for p in params
        t = mlir_type(p["type"])
        t === nothing && return "a parameter of LLVM type `$(p["type"])` is not one a thunk carries"
        a = _arg_attrs(p["attrs"])
        a === nothing && return "a parameter carries an attribute a thunk does not: $(join(p["attrs"], " "))"
        push!(arg_types, t)
        push!(arg_attrs, a)
    end
    ret_type = ""
    if ret_ir != "void"
        t = mlir_type(ret_ir)
        t === nothing && return "the return's LLVM type `$ret_ir` is not one a thunk carries"
        ret_type = t
    end

    io = IOBuffer()
    name = thunk_name(sym)
    println(io, "  func.func $(_sym(name))(%args: !llvm.ptr, %ret: !llvm.ptr) {")
    operands = String[]
    slot = 0
    for (i, p) in enumerate(params)
        if sret && i == 1
            push!(operands, "%ret")
            continue
        end
        t = arg_types[i]
        println(io, "    %s$i = llvm.getelementptr %args[$slot] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr")
        println(io, "    %p$i = llvm.load %s$i : !llvm.ptr -> !llvm.ptr")
        if t == "i1"
            # A bool is a byte in storage and one bit in the call.
            println(io, "    %b$i = llvm.load %p$i {alignment = 1 : i64} : !llvm.ptr -> i8")
            println(io, "    %v$i = llvm.trunc %b$i : i8 to i1")
        else
            println(io, "    %v$i = llvm.load %p$i {alignment = 1 : i64} : !llvm.ptr -> $t")
        end
        push!(operands, "%v$i")
        slot += 1
    end
    attrs = any(a -> a != "{}", arg_attrs) ? ", arg_attrs = [" * join(arg_attrs, ", ") * "]" : ""
    call = "jlcs.native_call " * join(operands, ", ") * " { callee = $(_sym(sym))$attrs } : (" *
           join(arg_types, ", ") * ") -> " * (isempty(ret_type) ? "()" : ret_type)
    function store(value, type, offset)
        v = value
        if type == "i1"
            println(io, "    $(value)z = llvm.zext $value : i1 to i8")
            v = value * "z"
            type = "i8"
        end
        at = "%ret"
        if offset != 0
            at = "$(value)q"
            println(io, "    $at = llvm.getelementptr %ret[$offset] : (!llvm.ptr) -> !llvm.ptr, i8")
        end
        println(io, "    llvm.store $v, $at {alignment = 1 : i64} : $type, !llvm.ptr")
    end
    pass = get(ret, "pass", "")
    if isempty(ret_type)
        println(io, "    ", call)
    elseif pass == "pair"
        leaves = RustABI.shape(types, ret["type"]).leaves
        elems = RustABI._brace_elements(ret_ir)
        (leaves isa Vector{RustABI.Leaf} && length(leaves) == 2 && length(elems) == 2) ||
            return "the returned pair's two places in the return type are not known"
        println(io, "    %r = ", call)
        for (k, leaf) in enumerate(leaves)
            println(io, "    %r$k = llvm.extractvalue %r[$(k - 1)] : $ret_type")
            store("%r$k", mlir_type(elems[k]), leaf.offset)
        end
    elseif pass in ("direct", "cast")
        println(io, "    %r = ", call)
        store("%r", ret_type, 0)
    else
        return "the return is passed as `$pass`, which a thunk does not carry"
    end
    println(io, "    return")
    println(io, "  }")
    decl = "  func.func private $(_sym(sym))(" * join(arg_types, ", ") * ")" *
           (isempty(ret_type) ? "" : " -> " * ret_type)
    return (String(take!(io)), decl)
end

"""
    thunk_module(functions, types; only = nothing) -> (ir, made, skipped)

The MLIR module holding a thunk for each function row that can have one, in
the rows' order. `made` lists their symbols; `skipped` maps each other symbol
to its reason. `only`, when given, is the set of symbols to consider.

An alias (`alias_of`) is called by its own name: it is its own symbol in the
library, and its call is its target's.
"""
function thunk_module(functions, types::AbstractDict; only = nothing)
    decls = String[]
    bodies = String[]
    made = String[]
    skipped = Dict{String,String}()
    for row in functions
        sym = String(row["symbol"])
        (only === nothing || sym in only) || continue
        t = thunk(row, types)
        if t isa String
            skipped[sym] = t
            continue
        end
        push!(bodies, t[1])
        push!(decls, t[2])
        push!(made, sym)
    end
    ir = "module {\n" * join(decls, "\n") * (isempty(decls) ? "" : "\n") * join(bodies, "") * "}\n"
    return (ir, made, skipped)
end

"""
A library's thunks, JIT'd. `thunks` maps a function's symbol to its thunk's
address; `skipped` maps every other function of the metadata to the reason it
has none. The context and the engine live as long as the session.
"""
struct Session
    library::String
    ctx::Ptr{Cvoid}
    jit::Ptr{Cvoid}
    thunks::Dict{String,Ptr{Cvoid}}
    skipped::Dict{String,String}
    functions::Dict{String,Any}
    types::Dict{String,Any}
end

"""
    load(library, metadata; opt_level = 2) -> Session

Generate, lower and JIT the thunks for a Rust library from its
`compilation_metadata.json` (parsed).

Every callee is looked up through the library's own handle first, and a
function the handle does not resolve gets no thunk: the JIT does not raise
for a symbol it cannot find, it waits forever.
"""
function load(library::AbstractString, metadata::AbstractDict; opt_level::Int = 2)::Session
    get(metadata, "language", "") == "rust" ||
        error("RustThunks.load: $(library)'s metadata is not a Rust library's")
    library = abspath(String(library))
    functions = metadata["functions"]
    types = metadata["types"]
    handle = Libdl.dlopen(library, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
    bound = Set{String}()
    unbound = Dict{String,String}()
    for row in functions
        sym = String(row["symbol"])
        if Libdl.dlsym_e(handle, sym) == C_NULL
            unbound[sym] = "the library does not export `$sym`"
        else
            push!(bound, sym)
        end
    end
    (ir, made, skipped) = thunk_module(functions, types; only = bound)
    merge!(skipped, unbound)
    by_symbol = Dict{String,Any}(String(r["symbol"]) => r for r in functions)
    ctx = MLIRNative.create_context()
    thunks = Dict{String,Ptr{Cvoid}}()
    isempty(made) && return Session(library, ctx, C_NULL, thunks, skipped, by_symbol, types)
    mod = MLIRNative.parse_module(ctx, ir; debug_base = MLIRNative.debug_dir_for(library))
    MLIRNative.lower_to_llvm(mod) ||
        error(MLIRNative._with_diagnostics("RustThunks.load: the thunk module for $library did not lower."))
    jit = MLIRNative.create_jit(mod; opt_level, shared_libs = [library, MLIRNative.libJLCS_path()])
    for sym in made
        addr = MLIRNative.lookup(jit, thunk_name(sym))
        if addr == C_NULL
            skipped[sym] = "its thunk was not found in the JIT"
        else
            thunks[sym] = addr
        end
    end
    return Session(library, ctx, jit, thunks, skipped, by_symbol, types)
end

"""
    call_raw(session, symbol, ret, values...)

Call `symbol` through its thunk. `values` are the IR-level arguments, one per
parameter of the `define` line in its order, the hidden return pointer left
out: each an `isbits` value of exactly the parameter's size (`UInt8` for an
`i8` or a `bool`, `Ptr` for a `ptr`, `UInt128` for an `i128`, an
`NTuple{3,UInt8}` for an `i24`). `ret` is the buffer the result is written
to: at least as large as what the function returns, and for a function that
returns through a hidden pointer, as aligned as its type.

This is the thunk's own convention with the bookkeeping done, for tests and
for trying a library by hand; a generated wrapper will write these slots
itself. A count or a size that does not match the metadata is an error
before the call, never a wrong call.
"""
function call_raw(s::Session, symbol::AbstractString, ret::Union{AbstractVector{UInt8},Base.RefValue,Ptr}, values...)
    sym = String(symbol)
    fp = get(s.thunks, sym, C_NULL)
    fp == C_NULL && error("RustThunks.call_raw: no thunk for `$sym`" *
                          (haskey(s.skipped, sym) ? ": " * s.skipped[sym] : " (not a function of this library)"))
    row = s.functions[sym]
    params = row["ir"]["parameters"]
    sret = get(row["return"], "pass", "") == "indirect"
    expected = [p for (i, p) in enumerate(params) if !(sret && i == 1)]
    length(values) == length(expected) ||
        error("RustThunks.call_raw: `$sym` takes $(length(expected)) IR arguments " *
              "($(join((p["type"] for p in expected), ", "))), and $(length(values)) were given")
    for (v, p) in zip(values, expected)
        isbits(v) || error("RustThunks.call_raw: an argument for `$(p["type"])` is not an isbits value: $(typeof(v))")
        bytes = RustABI._ir_scalar_bytes(p["type"])
        bytes === nothing && error("RustThunks.call_raw: `$(p["type"])` is not a scalar this can size")
        sizeof(v) == bytes ||
            error("RustThunks.call_raw: `$sym` takes `$(p["type"])` ($bytes bytes) and was given a " *
                  "$(typeof(v)) ($(sizeof(v)) bytes)")
    end
    refs = Any[Ref(v) for v in values]
    GC.@preserve refs ret begin
        slots = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, r) for r in refs]
        rp = ret isa Ptr ? Ptr{Cvoid}(ret) : Ptr{Cvoid}(Base.unsafe_convert(Ptr{eltype(ret)}, ret))
        GC.@preserve slots ccall(fp, Cvoid, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}), slots, rp)
    end
    return ret
end

end # module RustThunks
