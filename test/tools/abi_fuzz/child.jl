#!/usr/bin/env julia
# test/tools/abi_fuzz/child.jl — the calling half of test/tools/abi_fuzz.jl.
#
#     julia --project=<env> child.jl <plan.jls>
#
# Runs in a fresh process (the parent starts it through `run_isolated`), loads
# one generated wrapper and checks it against the C table the unit exports.
# Protocol, one line each, flushed:
#
#     RUNNING <tag>/<check>          before a check, so a crash can be blamed
#     PROBE <tag>/<check>: <verdict> PASS | FAIL … | GAP … | XFAIL …
#     GAP <tag>/<kind>: <detail>     a coverage gap (not a check)
#     PROBE_DONE
#
# Verdicts: PASS, FAIL (a wrong value, size, offset, or an exception from a
# function that should have worked), XFAIL (an expected refusal: `long double`),
# GAP (the wrapper refused or cannot reach something it should have).

include(joinpath(@__DIR__, "model.jl"))
using .AbiFuzzModel
const AF = AbiFuzzModel
using Serialization
using Libdl

const PLAN = deserialize(ARGS[1])::AF.Plan
const U = PLAN.unit
const TAG = PLAN.tag
const LIB = PLAN.libpath

if !isempty(PLAN.neuter)
    import RepliBuild
    include(joinpath(@__DIR__, "neuter.jl"))
    apply_neuter(RepliBuild, PLAN.neuter; side = :child)
end

out(s) = (println(s); flush(stdout))
running(check) = out("RUNNING $TAG/$check")
probe(check, verdict) = out("PROBE $TAG/$check: " * replace(verdict, '\n' => " | "))
const GAPS_SEEN = Set{String}()
function gap(kind, detail)
    line = "GAP $TAG/$kind: " * replace(detail, '\n' => " | ")
    line in GAPS_SEEN && return
    push!(GAPS_SEEN, line)
    out(line)
end

clip(s, n = 400) = (s = String(s); length(s) <= n ? s : first(s, n) * "…")
errstr(e) = clip(sprint(showerror, e), 600)

wanted(check) = !(check in PLAN.skip) && (isempty(PLAN.only) || check in PLAN.only)

# ── The C table, read through cglobal (no call into the library) ────────────

# A library that does not load (on Windows, Smart App Control can refuse a
# freshly built DLL) is reported as a load failure with the loader's own
# message, which the parent classifies; it must not look like a crash.
const LIBH = try
    Libdl.dlopen(LIB)
catch e
    running("load")
    probe("load", "FAIL library did not load: " * errstr(e))
    out("PROBE_DONE")
    exit(0)
end
const TABLE = Dict{Symbol,Ptr{UInt64}}(
    :af_meta => cglobal((:af_meta, LIB), UInt64), :af_prims => cglobal((:af_prims, LIB), UInt64),
    :af_esizes => cglobal((:af_esizes, LIB), UInt64), :af_sizes => cglobal((:af_sizes, LIB), UInt64),
    :af_offs => cglobal((:af_offs, LIB), UInt64), :af_ready => cglobal((:af_ready, LIB), UInt64))
_tab(sym) = TABLE[sym]
_u64(sym, i) = unsafe_load(_tab(sym), i)

struct Table
    ok::String                         # "" or why the table cannot be trusted
    prims::Dict{Symbol,Int}
    esize::Vector{Int}                 # by index into the main unit's enums
    size::Dict{Int,Int}                # record => sizeof
    off::Dict{Tuple{Int,String},Int}   # (record, cpath) => offsetof
    bfbits::Dict{Tuple{Int,String},Vector{Int}}   # (record, cpath) => image bit indices
    char_signed::Bool
end

function read_table()
    meta = [_u64(:af_meta, i) for i in 1:8]
    recs = AF.named_records(U)
    probes = AF.bitfield_probes(U)
    npath = sum((length(AF.paths(U, r)) for r in recs); init = 0)
    nenum = count(e -> e.twin != 2, U.enums)
    problems = String[]
    meta[1] == 0xAFAB1F00 || push!(problems, "bad magic $(repr(meta[1]))")
    meta[2] == length(recs) || push!(problems, "records $(meta[2]) != $(length(recs))")
    meta[3] == npath || push!(problems, "paths $(meta[3]) != $npath")
    meta[4] == length(probes) || push!(problems, "probes $(meta[4]) != $(length(probes))")
    meta[5] == length(AF.PRIM_ORDER) || push!(problems, "prims $(meta[5])")
    meta[7] == nenum || push!(problems, "enums $(meta[7]) != $nenum")
    prims = Dict(k => Int(_u64(:af_prims, i)) for (i, k) in enumerate(AF.PRIM_ORDER))
    esize = [Int(_u64(:af_esizes, i)) for i in 1:nenum]
    size = Dict(r => Int(_u64(:af_sizes, i)) for (i, r) in enumerate(recs))
    off = Dict{Tuple{Int,String},Int}()
    k = 0
    for r in recs, p in AF.paths(U, r)
        k += 1
        off[(r, p.cpath)] = Int(_u64(:af_offs, k))
    end
    stride = Int(meta[6])
    ready = unsafe_load(_tab(:af_ready)) == 0xAF5EED
    ready || isempty(probes) || push!(problems, "af_ready not set: the load-time initializer did not run")
    bfp = cglobal((:af_bfp, LIB), UInt8)
    bfbits = Dict{Tuple{Int,String},Vector{Int}}()
    for (i, (r, l)) in enumerate(probes)
        bits = Int[]
        for byte in 0:size[r]-1, b in 0:7
            (unsafe_load(bfp, (i - 1) * stride + byte + 1) >> b) & 1 == 1 && push!(bits, 8byte + b)
        end
        length(bits) == l.bits || push!(problems, "probe $(l.cpath) has $(length(bits)) bits, not $(l.bits)")
        bfbits[(r, l.cpath)] = bits
    end
    return Table(join(problems, "; "), prims, esize, size, off, bfbits, meta[8] != 0)
end

const P_REC = cglobal((:af_rec, LIB), UInt64)
const P_RECN = cglobal((:af_rec_n, LIB), Cint)
const P_RECFN = cglobal((:af_rec_fn, LIB), Cint)

# ── Widths, images, recorded values ──────────────────────────────────────────

function enum_slot(e::Int)
    # af_esizes lists the main unit's enums in order; the companion's TwinB has
    # TwinA's size
    U.enums[e].twin == 2 && (e = findfirst(x -> x.twin == 1, U.enums))
    return count(i -> U.enums[i].twin != 2, 1:e)
end

width(T::Table, t::AF.CT) = t isa AF.EnumT ? T.esize[enum_slot(t.e)] :
                            (t::AF.Prim).k == :ldouble ? 8 : T.prims[t.k]

isfloat(t) = t isa AF.Prim && t.k in (:float, :double, :ldouble)
leafsigned(T::Table, ty, signed) = ty isa AF.Prim && ty.k == :char ? T.char_signed : signed

int_bytes(v, w) = UInt8[UInt8((mod(Int128(v), Int128(1) << (8w)) >> (8i)) & 0xff) for i in 0:w-1]

function leaf_bytes(T::Table, ty, v)
    w = width(T, ty)
    ty isa AF.EnumT && return int_bytes(v, w)
    k = ty.k
    k == :float && return collect(reinterpret(UInt8, [Float32(v)]))
    k in (:double, :ldouble) && return collect(reinterpret(UInt8, [Float64(v)]))
    return int_bytes(v, w)
end

"Write leaf `l` of record `r` holding `v` into the C image `img`."
function put_leaf!(T::Table, img::Vector{UInt8}, r::Int, l::AF.Leaf, v)
    if l.bits > 0
        x = mod(Int128(v), Int128(1) << l.bits)
        for (i, b) in enumerate(T.bfbits[(r, l.cpath)])
            mask = UInt8(1) << (b % 8)
            img[b ÷ 8 + 1] = ((x >> (i - 1)) & 1 == 1) ? (img[b ÷ 8 + 1] | mask) : (img[b ÷ 8 + 1] & ~mask)
        end
    else
        o = T.off[(r, l.cpath)]
        bs = leaf_bytes(T, l.ty, v)
        img[o+1:o+length(bs)] = bs
    end
    return img
end

"The C-width bit pattern of leaf `l` as it sits in `img`."
function image_bits(T::Table, img::Vector{UInt8}, r::Int, l::AF.Leaf)
    if l.bits > 0
        x = UInt64(0)
        for (i, b) in enumerate(T.bfbits[(r, l.cpath)])
            b ÷ 8 + 1 <= length(img) || return nothing
            (img[b ÷ 8 + 1] >> (b % 8)) & 1 == 1 && (x |= UInt64(1) << (i - 1))
        end
        return x
    end
    o = T.off[(r, l.cpath)]
    w = width(T, l.ty)
    o + w <= length(img) || return nothing
    x = UInt64(0)
    for i in 0:w-1
        x |= UInt64(img[o+i+1]) << (8i)
    end
    return x
end

"The C-width bit pattern of a model value."
function value_bits(T::Table, ty, bits, v)
    if isfloat(ty)
        return ty.k == :float ? UInt64(reinterpret(UInt32, Float32(v))) : reinterpret(UInt64, Float64(v))
    end
    nb = bits > 0 ? bits : 8 * width(T, ty)
    return UInt64(mod(Int128(v), Int128(1) << nb))
end

"The bit pattern of a value read back through the wrapper, or a reason it has none."
function julia_bits(T::Table, ty, bits, val)
    if isfloat(ty)
        ty.k == :float && val isa Float32 && return reinterpret(UInt32, val) % UInt64
        ty.k != :float && val isa Float64 && return reinterpret(UInt64, val)
        return "a $(typeof(val)) for a C $(AF.PRIM_C[ty.k])"
    end
    x = val isa Enum ? Int128(Integer(val)) :
        val isa Bool ? Int128(val) :
        val isa Integer ? Int128(val) :
        val isa Ptr ? Int128(UInt(val)) : nothing
    x === nothing && return "a $(typeof(val))"
    val isa AbstractFloat && return "a $(typeof(val))"
    nb = bits > 0 ? bits : 8 * width(T, ty)
    return UInt64(mod(x, Int128(1) << nb))
end

"What `AF_R*` stores for value `v` of a leaf or scalar parameter."
function recorded(T::Table, ty, bits, signed, v)
    ty isa AF.EnumT && return AF.recorded_bits(v, 64, true)
    k = ty.k
    k == :float && return UInt64(reinterpret(UInt32, Float32(v)))
    k in (:double, :ldouble) && return reinterpret(UInt64, Float64(v))
    k in (:ptr, :bool) && return UInt64(v)
    nb = bits > 0 ? bits : 8 * width(T, ty)
    return AF.recorded_bits(v, nb, leafsigned(T, ty, signed))
end

# ── The wrapper ──────────────────────────────────────────────────────────────

LOAD_ERROR = ""
LOAD_TIME = @elapsed try
    include(PLAN.wrapper)
catch e
    global LOAD_ERROR = errstr(e)
end

# Function inventory, read off the wrapper's source: for each linker symbol,
# the Julia function that wraps it and whether that function makes a foreign
# call or is a refusal stub.
struct Wrapped
    name::Symbol
    kind::Symbol      # :callable, :trap, :other
    msg::String
end

function _defname(ex)
    ex isa Expr || return nothing
    if ex.head === :function || (ex.head === :(=) && ex.args[1] isa Expr && ex.args[1].head in (:call, :where, :(::)))
        sig = ex.args[1]
        while sig isa Expr && sig.head in (:where, :(::))
            sig = sig.args[1]
        end
        sig isa Expr && sig.head === :call || return nothing
        f = sig.args[1]
        f isa Symbol && return f
        f isa Expr && f.head === :. && return nothing
        return nothing
    end
    return nothing
end

function _scan(body, acc)
    if body isa Expr
        if body.head === :call && body.args[1] === :ccall
            t = body.args[2]
            if t isa Expr && t.head === :tuple && t.args[1] isa QuoteNode
                push!(acc[:syms], String(t.args[1].value))
            end
            acc[:foreign] = true
        elseif body.head === :call && body.args[1] isa Expr && body.args[1].head === :. &&
               string(body.args[1]) in ("RepliBuild.JITManager.invoke", "RepliBuild.JITManager.invoke_aot",
                                         "RepliBuild.JITManager.invoke_aot_ptr")
            acc[:foreign] = true
            for a in body.args[2:end]
                a isa String && push!(acc[:syms], a)
            end
        elseif body.head === :call && body.args[1] isa Expr && string(body.args[1]) == "Base.error"
            for a in body.args[2:end]
                a isa String && push!(acc[:errors], a)
                a isa Expr && a.head === :string && push!(acc[:errors], join(filter(x -> x isa String, a.args)))
            end
        elseif body.head === :macrocall && body.args[1] === Symbol("@ccall")
            acc[:foreign] = true
        end
        for a in body.args
            _scan(a, acc)
        end
    elseif body isa String && occursin("ABI Safety Trap", body)
        push!(acc[:errors], body)
    end
end

function inventory(path)
    ast = Meta.parseall(read(path, String))
    out = Dict{String,Wrapped}()
    function visit(ex, doc)
        ex isa Expr || return
        if ex.head === :macrocall && length(ex.args) >= 4 && (ex.args[1] isa GlobalRef ?
                ex.args[1].name === Symbol("@doc") : ex.args[1] === Symbol("@doc"))
            d = ex.args[3]
            visit(ex.args[4], d isa String ? d : "")
            return
        end
        if ex.head in (:module, :toplevel, :block)
            for a in ex.args
                visit(a, "")
            end
            return
        end
        name = _defname(ex)
        name === nothing && return
        acc = Dict{Symbol,Any}(:syms => String[], :errors => String[], :foreign => false)
        _scan(ex.args[2], acc)
        m = match(r"Mangled symbol: `([^`]+)`", doc)
        keys_ = m !== nothing ? [String(m.captures[1])] : acc[:syms]
        kind = acc[:foreign] ? :callable :
               any(e -> occursin("ABI Safety Trap", e) || occursin("cannot call", e), acc[:errors]) ? :trap : :other
        msg = isempty(acc[:errors]) ? "" : first(acc[:errors])
        for k in keys_
            # a convenience overload must not shadow the primary definition
            haskey(out, k) && out[k].kind === :callable && continue
            out[k] = Wrapped(name, kind, msg)
        end
    end
    visit(ast, "")
    return out
end

# ── Reaching the wrapper's types and members ─────────────────────────────────

name_candidates(s::String) = unique(filter(!isempty, [s, all(==('_'), s) ? "c__" : "",
                                                      s * "_", "c_" * s, "_" * s]))

ownbinding(MOD, sym) = isdefined(MOD, sym) && Base.which(MOD, sym) === MOD

function julia_type(MOD, r::Int)
    rec = U.records[r]
    for base in unique(filter(!isempty, [rec.name, rec.tag])), c in name_candidates(base)
        s = Symbol(c)
        ownbinding(MOD, s) || continue
        T = getfield(MOD, s)
        T isa DataType && return T
    end
    return nothing
end

function julia_enum(MOD, e::Int)
    en = U.enums[e]
    for base in unique(filter(!isempty, [en.name, en.tag])), c in name_candidates(base)
        s = Symbol(c)
        ownbinding(MOD, s) || continue
        T = getfield(MOD, s)
        T isa DataType && T <: Enum && return T
    end
    return nothing
end

zero_of(JT) = try
    x = Base.invokelatest(JT)
    x isa JT ? x : nothing
catch
    isbitstype(JT) ? (b = zeros(UInt8, sizeof(JT)); GC.@preserve b unsafe_load(Ptr{JT}(pointer(b)))) : nothing
end

inline_layout(T) = isbitstype(T) || (ismutabletype(T) && isconcretetype(T) &&
                   all(i -> isbitstype(fieldtype(T, i)), 1:fieldcount(T)))

function value_bytes(x)
    T = typeof(x)
    if isbitstype(T)
        r = Ref(x)
        return GC.@preserve r copy(unsafe_wrap(Array, Ptr{UInt8}(pointer_from_objref(r)), sizeof(T)))
    elseif inline_layout(T)
        return GC.@preserve x copy(unsafe_wrap(Array, Ptr{UInt8}(pointer_from_objref(x)), sizeof(T)))
    end
    return nothing
end

function from_bytes(JT, img::Vector{UInt8})
    sizeof(JT) == length(img) || return nothing
    if isbitstype(JT)
        return GC.@preserve img unsafe_load(Ptr{JT}(pointer(img)))
    elseif inline_layout(JT)
        x = zero_of(JT)
        x === nothing && return nothing
        GC.@preserve x img unsafe_copyto!(Ptr{UInt8}(pointer_from_objref(x)), pointer(img), length(img))
        return x
    end
    return nothing
end

"The wrapper offers no way to reach a member: a coverage gap."
struct NoAccess
    why::String
end
"The wrapper's accessor exists and threw: a bug, reported as a FAIL detail."
struct AccessError
    why::String
end
const Unreached = Union{NoAccess,AccessError}

const BLOB_FIELDS = (:_data, :data)

# "No such field" from a plain struct or a generated getproperty's fallthrough
# is the absence of an accessor; any other exception is the accessor failing.
nofield(e) = (isdefined(Base, :FieldError) && e isa Base.FieldError) ||
             (e isa ErrorException && (occursin("has no field", e.msg) || occursin("has no settable field", e.msg)))

# Read member `nm` of `x` the ways a generated wrapper offers: a real field,
# a `getproperty` overload, or a `get_<member>` / `get_<Type>_<member>`
# function.
function G(MOD, x, nm::String)
    T = typeof(x)
    for c in name_candidates(nm)
        s = Symbol(c)
        if s in fieldnames(T) && !(s in BLOB_FIELDS && fieldcount(T) == 1)
            return getfield(x, s)
        end
        v = try
            Base.invokelatest(getproperty, x, s)
        catch e
            nofield(e) ? NoAccess("") :
                return AccessError("getproperty(::$(nameof(T)), :$c) threw " * errstr(e))
        end
        # a C member spelled like the blob's own storage field reads the storage
        v isa NoAccess || (s in BLOB_FIELDS && fieldcount(T) == 1 && v === getfield(x, 1)) || return v
        for f in (Symbol("get_", c), Symbol("get_", nameof(T), "_", c))
            isdefined(MOD, f) || continue
            fn = getfield(MOD, f)
            Base.invokelatest(applicable, fn, x) || continue
            return try
                Base.invokelatest(fn, x)
            catch e
                AccessError("$f(::$(nameof(T))) threw " * errstr(e))
            end
        end
    end
    return NoAccess("no getter for $(nameof(T)).$nm")
end

function to_julia(TT, v)
    v isa TT && return v
    TT <: Enum && return TT(Int64(v isa Enum ? Integer(v) : v))
    TT === Bool && return v isa Bool ? v : Bool(Int128(v isa Enum ? Integer(v) : v) != 0)
    TT <: Integer && return Int128(v isa Enum ? Integer(v) : v isa Ptr ? UInt(v) : v) % TT
    TT <: AbstractFloat && return TT(v)
    TT <: Ptr && return TT(UInt(v isa Integer ? v : UInt(v)))
    return convert(TT, v)
end

# Set member `nm` of `x` to `v`; returns the new value of `x` (setters on
# immutable types return a copy, on mutable ones they mutate) or `NoAccess`.
function S(MOD, x, nm::String, v)
    T = typeof(x)
    for c in name_candidates(nm)
        s = Symbol(c)
        if s in fieldnames(T) && !(s in BLOB_FIELDS && fieldcount(T) == 1)
            vv = try
                to_julia(fieldtype(T, s), v)
            catch e
                return AccessError("cannot convert to $(fieldtype(T, s)) for $(nameof(T)).$nm: " * errstr(e))
            end
            if ismutabletype(T)
                setfield!(x, s, vv)
                return x
            end
            if !isdefined(MOD, :with)
                # what a Julia user does without `with`: the default constructor
                gap("no-with", "$(nameof(T)): the wrapper defines no `with`")
                return try
                    T((f === s ? vv : getfield(x, f) for f in fieldnames(T))...)
                catch e
                    AccessError("$(nameof(T))(…fields…) threw " * errstr(e))
                end
            end
            return try
                Base.invokelatest(getfield(MOD, :with), x; (s => vv,)...)
            catch e
                AccessError("with(::$(nameof(T)); $s = …) threw " * errstr(e))
            end
        end
        cur = G(MOD, x, c)
        cur isa AccessError && return cur
        cur isa NoAccess && continue
        TT = typeof(cur)
        vv = try
            to_julia(TT, v)
        catch e
            return AccessError("cannot convert to $(TT) for $(nameof(T)).$nm: " * errstr(e))
        end
        if isdefined(MOD, :setproperty)
            sp = getfield(MOD, :setproperty)
            if Base.invokelatest(applicable, sp, x, s, vv)
                r = try
                    Base.invokelatest(sp, x, s, vv)
                catch e
                    nofield(e) ? NoAccess("") : AccessError("setproperty(::$(nameof(T)), :$c, …) threw " * errstr(e))
                end
                r isa NoAccess || return r
            end
        end
        for f in (Symbol("set_", c), Symbol("set_", c, "!"), Symbol("set_", nameof(T), "_", c, "!"),
                  Symbol("set_", nameof(T), "_", c))
            isdefined(MOD, f) || continue
            fn = getfield(MOD, f)
            Base.invokelatest(applicable, fn, x, vv) || continue
            r = try
                Base.invokelatest(fn, x, vv)
            catch e
                return AccessError("$f(::$(nameof(T)), …) threw " * errstr(e))
            end
            return r isa T ? r : x
        end
    end
    return NoAccess("no setter for $(nameof(T)).$nm")
end

anon_ordinal(r::Int, i::Int) = count(j -> isempty(U.records[r].members[j].name) &&
                                          U.records[r].members[j].bits < 0, 1:i)

memname(r, i) = (m = U.records[r].members[i]; isempty(m.name) ? "_anon$(anon_ordinal(r, i))" : m.name)

"Read the scalar at `steps` through the wrapper's accessors."
function getpath(MOD, x, steps, k = 1)
    k > length(steps) && return x
    (kind, a, b) = steps[k]
    if kind === :i
        x isa Tuple || return NoAccess("array member is a $(typeof(x)), not a tuple")
        return getpath(MOD, x[a+1], steps, k + 1)
    end
    m = U.records[a].members[b]
    if isempty(m.name) && k < length(steps) && steps[k+1][1] === :m
        # C11 injection first (`x.ax`), then the invented `_anonK` field
        (_, a2, b2) = steps[k+1]
        inner = U.records[a2].members[b2]
        if !isempty(inner.name)
            v = G(MOD, x, inner.name)
            v isa Unreached || return getpath(MOD, v, steps, k + 2)
        end
    end
    v = G(MOD, x, memname(a, b))
    v isa Unreached && return v
    return getpath(MOD, v, steps, k + 1)
end

"Set the scalar at `steps` to `v` through the wrapper's accessors."
function setpath(MOD, x, steps, v, k = 1)
    (kind, a, b) = steps[k]
    if kind === :i
        x isa Tuple || return NoAccess("array member is a $(typeof(x)), not a tuple")
        e = if k == length(steps)
            try
                to_julia(typeof(x[a+1]), v)
            catch err
                AccessError("cannot convert to $(typeof(x[a+1])): " * errstr(err))
            end
        else
            setpath(MOD, x[a+1], steps, v, k + 1)
        end
        e isa Unreached && return e
        return Base.setindex(x, e, a + 1)
    end
    nm = memname(a, b)
    k == length(steps) && return S(MOD, x, nm, v)
    child = G(MOD, x, nm)
    child isa Unreached && return child
    nc = setpath(MOD, child, steps, v, k + 1)
    nc isa Unreached && return nc
    return S(MOD, x, nm, nc)
end

# ── Building an argument ─────────────────────────────────────────────────────

"""
Build record `r` with `vals[i]` in leaf `i`: the C image the callee should see,
and the Julia value made through the wrapper's own setters. A leaf the wrapper
cannot set (no setter: a gap; a setter that threw: an error, returned in
`notes` prefixed `setter threw`) is written as raw bytes at its C location
instead, so the crossing is still tested.
"""
function build_record(MOD, T::Table, r::Int, JT, vals)
    ls = AF.leaves(U, r)
    img = zeros(UInt8, T.size[r])
    for (l, v) in zip(ls, vals)
        put_leaf!(T, img, r, l, v)
    end
    x = zero_of(JT)
    x === nothing && return (nothing, img, String["no zero value for $(nameof(JT))"])
    notes = String[]
    raw = String[]
    for (l, v) in zip(ls, vals)
        y = setpath(MOD, x, l.steps, v)
        if y isa NoAccess
            push!(raw, l.cpath)
            gap("no-setter", string(nameof(JT), l.cpath, ": ", y.why))
        elseif y isa AccessError
            push!(raw, l.cpath)
            push!(notes, string("setter threw for ", l.cpath, ": ", y.why))
        else
            x = y
        end
    end
    if !isempty(raw)
        # raw writes go last: the value bytes ARE the C layout when the layout
        # check passes, and a raw write before a setter could be undone by it
        b = value_bytes(x)
        if b === nothing || length(b) != length(img)
            push!(notes, "cannot write $(join(raw, ", ")) raw: $(nameof(JT)) is $(b === nothing ? "not inline" : string(length(b), " bytes"))")
        else
            for l in ls
                l.cpath in raw || continue
                v = vals[findfirst(==(l), ls)]
                put_leaf!(T, b, r, l, v)
            end
            y = from_bytes(JT, b)
            y === nothing ? push!(notes, "raw rebuild failed") : (x = y)
        end
    end
    return (x, img, notes)
end

"Leaves whose bytes in the Julia value differ from the C image (setter bugs)."
function setter_diffs(T::Table, r::Int, x, img)
    b = value_bytes(x)
    (b === nothing || length(b) != length(img)) && return String[]
    d = String[]
    for l in AF.leaves(U, r)
        got = image_bits(T, b, r, l); want = image_bits(T, img, r, l)
        got == want || push!(d, string(l.cpath, " holds 0x", string(got; base = 16), " want 0x", string(want; base = 16)))
    end
    return d
end

natural_julia(k::Symbol) = Dict(:char => Cchar, :schar => Int8, :uchar => UInt8, :short => Cshort,
    :ushort => Cushort, :int => Cint, :uint => Cuint, :long => Clong, :ulong => Culong,
    :llong => Clonglong, :ullong => Culonglong, :float => Cfloat, :double => Cdouble,
    :bool => Bool, :ptr => Ptr{Cvoid}, :ldouble => Cdouble)[k]

function scalar_arg(MOD, T::Table, ty, v)
    if ty isa AF.EnumT
        JE = julia_enum(MOD, ty.e)
        return JE === nothing ? Int64(v) % Cint : JE(Int64(v))
    end
    return to_julia(natural_julia(ty.k), v)
end

# ── Checks ───────────────────────────────────────────────────────────────────

function check_layout(MOD, T::Table, r::Int)
    rec = U.records[r]
    JT = julia_type(MOD, r)
    JT === nothing && return "GAP no Julia type for $(rec.name)"
    bad = String[]
    csz = T.size[r]
    jsz = try sizeof(JT) catch; -1 end
    jsz == csz || push!(bad, "sizeof($(nameof(JT))) = $jsz, C says $csz")
    if isdefined(MOD, :struct_size)
        s = try Base.invokelatest(getfield(MOD, :struct_size), nameof(JT)) catch e; e end
        s isa Integer ? (s == csz || push!(bad, "struct_size(:$(nameof(JT))) = $s, C says $csz")) :
                        gap("no-size", "struct_size(:$(nameof(JT))): " * errstr(s))
    else
        gap("no-size", "no struct_size in the module")
    end
    mo = isdefined(MOD, :member_offset) ? getfield(MOD, :member_offset) : nothing
    for p in AF.paths(U, r)
        p.direct || continue
        want = T.off[(r, p.cpath)]
        nm = p.cpath[2:end]
        got = nothing
        if mo !== nothing
            for c in name_candidates(nm)
                got = try Base.invokelatest(mo, nameof(JT), Symbol(c)) catch; nothing end
                got === nothing || break
            end
            if got === nothing
                # an injected name: `_anonK` plus the member within it
                got = anon_offset(MOD, mo, r, JT, nm)
            end
        end
        if got === nothing
            gap("no-offset", "member_offset(:$(nameof(JT)), :$nm)")
        elseif got != want
            push!(bad, "member_offset(:$(nameof(JT)), :$nm) = $got, C says $want")
        end
        # a named Julia field must sit at the C offset
        if isbitstype(JT)
            for c in name_candidates(nm)
                i = findfirst(==(Symbol(c)), fieldnames(JT))
                i === nothing && continue
                fo = Int(fieldoffset(JT, i))
                fo == want || push!(bad, "fieldoffset($(nameof(JT)), :$c) = $fo, C says $want")
                break
            end
        end
    end
    return isempty(bad) ? "PASS" : "FAIL " * join(bad, "; ")
end

function anon_offset(MOD, mo, r, JT, nm)
    rec = U.records[r]
    for (i, m) in enumerate(rec.members)
        (isempty(m.name) && m.bits < 0) || continue
        inner = (m.ty::AF.RecT).r
        any(mm -> mm.name == nm, U.records[inner].members) || continue
        base = try Base.invokelatest(mo, nameof(JT), Symbol("_anon", anon_ordinal(r, i))) catch; nothing end
        base === nothing && return nothing
        x = zero_of(JT)
        x === nothing && return nothing
        av = G(MOD, x, "_anon$(anon_ordinal(r, i))")
        av isa Unreached && return nothing
        sub = try Base.invokelatest(mo, nameof(typeof(av)), Symbol(nm)) catch; nothing end
        sub === nothing && return nothing
        return base + sub
    end
    return nothing
end

function check_access(MOD, T::Table, r::Int)
    JT = julia_type(MOD, r)
    JT === nothing && return "GAP no Julia type for $(U.records[r].name)"
    ls = AF.leaves(U, r)
    vals = [AF.access_value(U, r, i - 1, l.ty, l.bits, l.signed, width(T, l.ty)) for (i, l) in enumerate(ls)]
    img = zeros(UInt8, T.size[r])
    for (l, v) in zip(ls, vals)
        put_leaf!(T, img, r, l, v)
    end
    bad = String[]
    # getters: a value made from the C image reads back each leaf
    x = from_bytes(JT, img)
    if x === nothing
        push!(bad, "cannot view a $(T.size[r])-byte C image as $(nameof(JT)) ($(sizeof(JT)) bytes)")
    else
        for (l, v) in zip(ls, vals)
            got = getpath(MOD, x, l.steps)
            if got isa NoAccess
                gap("no-getter", string(nameof(JT), l.cpath, ": ", got.why))
                continue
            elseif got isa AccessError
                push!(bad, string("get ", l.cpath, ": ", got.why))
                continue
            end
            gb = julia_bits(T, l.ty, l.bits, got)
            wb = value_bits(T, l.ty, l.bits, v)
            gb == wb || push!(bad, string("get ", l.cpath, " = ", gb isa String ? gb : "0x" * string(gb; base = 16),
                                          " want 0x", string(wb; base = 16)))
        end
    end
    # setters: setting every leaf on a zero value reproduces the C image
    (y, _, notes) = build_record(MOD, T, r, JT, vals)
    append!(bad, notes)
    if y !== nothing
        for d in setter_diffs(T, r, y, img)
            push!(bad, "set " * d)
        end
    end
    return isempty(bad) ? "PASS" : "FAIL " * join(bad, "; ")
end

isldouble(t) = t isa AF.Prim && t.k == :ldouble

function refusal_verdict(f, msg)
    hasld = any(p -> isldouble(p.ty), f.params) || isldouble(f.ret)
    first_line = clip(first(split(msg, '\n')), 200)
    hasld && (occursin("long double", msg) || occursin("x87", msg)) && return "XFAIL refused: " * first_line
    return "GAP refused: " * first_line
end

function run_function(MOD, T::Table, INV, f::AF.Func)
    want_call = !isempty(f.params) && wanted("call/" * f.name)
    want_ret = f.ret !== nothing && wanted("ret/" * f.name)
    (want_call || want_ret) || return
    report(call, ret) = (want_call && probe("call/" * f.name, call); want_ret && probe("ret/" * f.name, ret))
    sym = get(PLAN.mangled, f.name, "")
    w = isempty(sym) ? nothing : get(INV, sym, nothing)
    if w === nothing
        v = "GAP not wrapped (symbol $(isempty(sym) ? "absent from metadata" : sym))"
        gap("unwrapped", "$(f.name) ($(sym))")
        return report(v, v)
    end
    if w.kind === :trap
        v = refusal_verdict(f, w.msg)
        startswith(v, "GAP") && gap("trap", "$(f.name): " * clip(first(split(w.msg, '\n')), 300))
        return report(v, v)
    end
    w.kind === :callable || (gap("unwrapped", "$(f.name): $(w.name) makes no foreign call"); return report("GAP no foreign call", "GAP no foreign call"))
    fn = getfield(MOD, w.name)
    # arguments
    slots = AF.call_slots(U, f)
    vals = [s.leaf === nothing ?
            AF.arg_value(U, f, s.ord, f.params[s.param].ty, -1,
                         f.params[s.param].ty isa AF.EnumT || (f.params[s.param].ty isa AF.Prim && AF.prim_signed(f.params[s.param].ty.k)),
                         width(T, f.params[s.param].ty)) :
            AF.arg_value(U, f, s.ord, s.leaf.ty, s.leaf.bits, s.leaf.signed, width(T, s.leaf.ty))
            for s in slots]
    args = Any[]
    notes = String[]
    for (pi, p) in enumerate(f.params)
        idx = [i for (i, s) in enumerate(slots) if s.param == pi]
        if p.ty isa AF.RecT || AF.isptrparam(p.ty)
            r = p.ty isa AF.RecT ? p.ty.r : ((p.ty::AF.ArrT).elem::AF.RecT).r
            JT = julia_type(MOD, r)
            if JT === nothing
                v = "GAP blocked: no Julia type for $(U.records[r].name)"
                return report(v, v)
            end
            (x, img, nts) = build_record(MOD, T, r, JT, vals[idx])
            append!(notes, nts)
            x === nothing && return report("FAIL cannot build $(nameof(JT)): " * join(nts, "; "),
                                           "FAIL cannot build argument")
            for d in setter_diffs(T, r, x, img)
                push!(notes, "setter: $(p.name)" * d)
            end
            push!(args, p.ty isa AF.RecT ? x : Ref(x))
        else
            push!(args, scalar_arg(MOD, T, p.ty, vals[idx[1]]))
        end
    end
    # A wrapper that declares a concrete type for a parameter (`::UInt8` for a
    # C `char`) rejects the C-natural Julia value (`Cchar`) with a MethodError.
    # Pass what it declares, and report the narrower signature.
    ms = [m for m in methods(fn) if m.nargs - 1 == length(args)]
    if length(ms) == 1 && !Base.invokelatest(applicable, fn, args...)
        sig = Base.unwrap_unionall(ms[1].sig)
        for (pi, p) in enumerate(f.params)
            DT = fieldtype(sig, pi + 1)
            args[pi] isa DT && continue
            (isconcretetype(DT) && (DT <: Number || DT <: Enum || DT <: Ptr)) || continue
            (p.ty isa AF.Prim || p.ty isa AF.EnumT) || continue
            idx = findfirst(s -> s.param == pi, slots)
            args[pi] = try
                to_julia(DT, vals[idx])
            catch
                args[pi]
            end
            gap("param-type", string(f.name, "(", p.name, "): declared ::", DT, " for a C `",
                                     AF.type_spell(U, p.ty), "`, so a ", typeof(scalar_arg(MOD, T, p.ty, vals[idx])),
                                     " argument has no method"))
        end
    end
    n = length(slots)
    for i in 1:n+8
        unsafe_store!(P_REC, 0xA5A5A5A5A5A5A5A5, i)
    end
    unsafe_store!(P_RECN, Cint(-1)); unsafe_store!(P_RECFN, Cint(-1))
    result = try
        Base.invokelatest(fn, args...)
    catch e
        e
    end
    GC.gc(false)
    if result isa Exception
        msg = sprint(showerror, result)
        if occursin("ABI Safety Trap", msg)
            v = refusal_verdict(f, msg)
            startswith(v, "GAP") && gap("trap", "$(f.name): " * clip(first(split(msg, '\n')), 300))
            return report(v, v)
        end
        v = "FAIL threw: " * clip(msg, 500)
        return report(v, v)
    end
    # the call: what the callee recorded
    callv = "PASS"
    if want_call
        bad = String[]
        fid = unsafe_load(P_RECFN); rn = unsafe_load(P_RECN)
        fid == f.id || push!(bad, "callee did not run (af_rec_fn = $fid, want $(f.id))")
        if fid == f.id
            rn == n || push!(bad, "callee recorded $rn values, want $n")
            for (i, s) in enumerate(slots)
                p = f.params[s.param]
                (ty, bits, sg) = s.leaf === nothing ?
                    (p.ty, -1, p.ty isa AF.EnumT || (p.ty isa AF.Prim && AF.prim_signed(p.ty.k))) :
                    (s.leaf.ty, s.leaf.bits, s.leaf.signed)
                want = recorded(T, ty, bits, sg, vals[i])
                got = unsafe_load(P_REC, i)
                if got != want
                    nm = s.leaf === nothing ? p.name : p.name * s.leaf.cpath
                    push!(bad, "$nm = 0x$(string(got; base = 16)) want 0x$(string(want; base = 16))")
                end
            end
        end
        if !isempty(bad)
            callv = "FAIL " * join(first(bad, 10), "; ") * (length(bad) > 10 ? "; … $(length(bad)) in all" : "") *
                    (isempty(notes) ? "" : " [" * join(first(notes, 6), "; ") * "]")
        end
    end
    # the return: each leaf through the wrapper's getters, and the raw bytes
    retv = "PASS"
    if want_ret
        retv = check_return(MOD, T, f, result)
    end
    report(callv, retv)
end

function check_return(MOD, T::Table, f::AF.Func, result)
    rt = f.ret
    if rt isa AF.RecT
        r = rt.r
        JT = julia_type(MOD, r)
        JT === nothing && return "GAP no Julia type for $(U.records[r].name)"
        result isa JT || return "FAIL returned a $(typeof(result)), not $(nameof(JT))"
        ls = AF.leaves(U, r)
        bad = String[]
        raw = value_bytes(result)
        rawok = raw !== nothing && length(raw) == T.size[r]
        for (i, l) in enumerate(ls)
            v = AF.ret_value(U, f, i - 1, l.ty, l.bits, l.signed, width(T, l.ty))
            wb = value_bits(T, l.ty, l.bits, v)
            got = getpath(MOD, result, l.steps)
            rb = rawok ? image_bits(T, raw, r, l) : nothing
            if got isa NoAccess
                gap("no-getter", string(nameof(JT), l.cpath, ": ", got.why))
                rawok && rb != wb && push!(bad, string(l.cpath, ": bytes hold 0x", string(rb; base = 16),
                                                       " want 0x", string(wb; base = 16)))
                continue
            elseif got isa AccessError
                push!(bad, string(l.cpath, ": ", got.why,
                                  rawok ? (rb == wb ? " (bytes right: getter)" : " (bytes wrong)") : ""))
                continue
            end
            gb = julia_bits(T, l.ty, l.bits, got)
            if gb != wb
                push!(bad, string(l.cpath, " = ", gb isa String ? gb : "0x" * string(gb; base = 16),
                                  " want 0x", string(wb; base = 16),
                                  rawok ? (rb == wb ? " (bytes right: getter)" : " (bytes wrong)") : ""))
            end
        end
        return isempty(bad) ? "PASS" : "FAIL " * join(first(bad, 10), "; ") * (length(bad) > 10 ? "; … $(length(bad)) in all" : "")
    end
    sg = rt isa AF.EnumT || (rt isa AF.Prim && AF.prim_signed(rt.k))
    v = AF.ret_value(U, f, 0, rt, -1, sg, width(T, rt))
    wb = value_bits(T, rt, -1, v)
    gb = julia_bits(T, rt, -1, result)
    gb == wb && return "PASS"
    return string("FAIL returned ", gb isa String ? gb : "0x" * string(gb; base = 16), " want 0x", string(wb; base = 16),
                  " (", repr(result), ")")
end

function main()
    running("load")
    if !isempty(LOAD_ERROR)
        probe("load", "FAIL wrapper did not load: " * LOAD_ERROR)
        out("PROBE_DONE")
        return
    end
    MOD = getfield(Main, Symbol(PLAN.modname))
    T = read_table()
    if !isempty(T.ok)
        probe("load", "FAIL harness: the C table does not match the plan: " * T.ok)
        out("PROBE_DONE")
        return
    end
    INV = inventory(PLAN.wrapper)
    ncall = count(f -> haskey(INV, get(PLAN.mangled, f.name, "")) &&
                       INV[PLAN.mangled[f.name]].kind === :callable, U.funcs)
    ntrap = count(f -> haskey(INV, get(PLAN.mangled, f.name, "")) &&
                       INV[PLAN.mangled[f.name]].kind === :trap, U.funcs)
    probe("load", string("PASS loaded in ", round(LOAD_TIME; digits = 1), "s; ", length(U.funcs), " functions: ",
                         ncall, " callable, ", ntrap, " refused, ", length(U.funcs) - ncall - ntrap, " other"))
    for r in AF.named_records(U)
        nm = U.records[r].name
        if wanted("layout/" * nm)
            running("layout/" * nm)
            v = try check_layout(MOD, T, r) catch e; "FAIL check threw: " * errstr(e) end
            probe("layout/" * nm, v)
        end
        if !isempty(AF.leaves(U, r)) && wanted("access/" * nm)
            running("access/" * nm)
            v = try check_access(MOD, T, r) catch e; "FAIL check threw: " * errstr(e) end
            probe("access/" * nm, v)
        end
    end
    for f in U.funcs
        (wanted("call/" * f.name) || wanted("ret/" * f.name)) || continue
        running("fn/" * f.name)
        try
            run_function(MOD, T, INV, f)
        catch e
            v = "FAIL check threw: " * errstr(e)
            isempty(f.params) || !wanted("call/" * f.name) || probe("call/" * f.name, v)
            f.ret === nothing || !wanted("ret/" * f.name) || probe("ret/" * f.name, v)
        end
    end
    GC.gc()
    out("PROBE_DONE")
end

main()
