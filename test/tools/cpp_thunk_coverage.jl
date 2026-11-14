#!/usr/bin/env julia
# =============================================================================
# cpp_thunk_coverage — how much of each Hub C++ library a wrapper can call
# =============================================================================
#
# A HAND-RUN DEVELOPER TOOL, NOT A TEST. The twin of c_ccall_coverage.jl for
# the C++ generator. It reads the Hub (beside this repo, or the path given),
# which no test may do, and is wired into no suite. It builds and wraps
# nothing: it measures the wrappers, metadata and manifests already on disk,
# AS THEY ARE, so their `BUILD_GENERATOR` is the version measured.
#
#     julia --project=. test/tools/cpp_thunk_coverage.jl [<hub>] [--all] [--list]
#
# `--list` prints every missed symbol under its cause instead of the first few.
#
# For every C++ package in the Hub's index.toml (`--all`: every package whose
# replibuild.toml says `language = "cpp"`, indexed or not), and for every
# `[graph]` package's C++ libraries (one row per library, indexed or not; its
# C libraries are the C tool's and are skipped with a note):
#
#   surface   the functions in `compilation_metadata.json`, one per mangled
#             symbol (DWARF lists a definition once per compilation unit),
#             less what the recipe's `[wrap] exclude_symbols` drops, matched
#             on name, demangled and mangled with RepliBuild's own
#             `_glob_matcher` as wrap matches them, less functions the
#             metadata marks not exported, and, in a graph, less the
#             functions an earlier library's wrapper already binds (its
#             `type_manifest.json` `bound_symbols`; the generator drops them
#             before wrapping, and they are callable through that module).
#   tier 3    surface symbols that are a `ccall`/`@ccall` target somewhere in
#             the wrapper's PARSED code, as the C tool counts them: a
#             docstring never counts, a trap stub calls nothing.
#   tier 2    surface symbols whose thunk the parsed code invokes:
#             `JITManager.invoke("_mlir_ciface_<m>_thunk", …)`, an AOT slot
#             (`invoke_aot_ptr(_FPTR_x[], …)`, resolved through the emitted
#             `_THUNK_SLOTS` table) or an AOT virtual `ccall((:thunk_<m>,
#             THUNKS_LIBRARY_PATH), …)`. A thunk listed in
#             `thunk_manifest.json` that nothing invokes is not bound.
#   trap      a stub whose body raises the generator's refusal, split by the
#             reason the message carries (GeneratorCpp's trap emitters and
#             `Utils._unresolved_return_trap_msg` / `_x87_trap_body`):
#               by-value record with no layout    "…layout is not in this library's DWARF"
#               SysV refusal                       "…so no thunk could call it" (the reason after "by value, ")
#               conflicted type name               `_CONFLICT_TRAP_NOTE`
#               unresolved return type             "…no Julia type was resolved for that return"
#               long double by value               `_x87_trap_body`
#               unmapped type (FFI trap)           BRANCH 0's "FFI Safety Trap"
#             and, for the record traps, which record blocks how many functions.
#   expected  in the surface, not bound, and meant: several C++ entry points a
#             Julia caller cannot tell apart stand behind one Julia method, so
#             these are set aside, not missed:
#               destructor variant       D0/D1/D2 of a class another variant of which
#                                        is bound: one Julia method by construction
#               const/non-const pair     same parameters, receiver constness only
#               rvalue twin              a `T&&` overload beside its `const T&` twin,
#                                        which the generator does not emit
#               one width, two spellings `set_value(long)` beside `(long long)` on LP64
#   dropped   in the surface, neither bound, trapped nor expected: a distinct
#             entry point lost, by what the generator did:
#               overload set collapsed   another overload of the C++ name is bound and
#                                        the two became one `::Any` signature
#               name derivation          another C++ function spells to the same Julia
#                                        name (`operator<` and `operator>` are both `_`)
#               dispatch collision       its thunk is in the manifest, so a chunk was
#                                        emitted, and `_dedup_method_chunks` dropped it
#               varargs                  variadic and not declared in [wrap.varargs]
#               not emitted              nothing in the wrapper names it
#   types     the records the wrapper defines: with fields, as a byte blob
#             (`_data::NTuple{N, UInt8}`), as a byte record (a `mutable`
#             union or bitfield record), as an opaque `struct X end`, and
#             the ones imported from an earlier library of a graph (`const X
#             = Base.parentmodule(@__MODULE__).Owner.X`). `Managed` owners
#             and factory structs are counted apart.
#
# Prints a Markdown table per library, the totals, the misses by cause
# (sorted by count, the first symbols of each), the expected drops, the
# trapped records, and one summary line over the callable surface (the
# surface less the expected drops):
#   cpp coverage: <bound>/<callable> (<pct>%), traps <n>, dropped <n>, expected <n>
#
# Vacuous green is refused: at least one library must be measured and the
# total surface must be nonzero.
# =============================================================================

using TOML
using RepliBuild

const ARGV = filter(a -> !startswith(a, "--"), ARGS)
const HUB = isempty(ARGV) ? normpath(joinpath(@__DIR__, "..", "..", "..", "RepliBuild-Hub")) : ARGV[1]
isfile(joinpath(HUB, "index.toml")) || error("no index.toml under $HUB: pass the Hub's path")

const INDEX = TOML.parsefile(joinpath(HUB, "index.toml"))
const INDEXED = Set(k for (k, v) in INDEX if get(v, "language", "") == "cpp")

_toml(p) = joinpath(HUB, "packages", p, "replibuild.toml")
_lang(p) = String(get(get(TOML.parsefile(_toml(p)), "wrap", Dict()), "language", ""))
_is_graph(p) = haskey(TOML.parsefile(_toml(p)), "graph")

"The packages to measure: indexed C++ ones, every graph package, and with --all every C++ TOML."
function cpp_packages()
    pkgs = Set(collect(INDEXED))
    for d in readdir(joinpath(HUB, "packages"))
        isfile(_toml(d)) || continue
        _is_graph(d) && push!(pkgs, d)
        "--all" in ARGS && _lang(d) == "cpp" && push!(pkgs, d)
    end
    return sort!(collect(pkgs))
end

# ─── wrapper parsing ─────────────────────────────────────────────────────────

_callee(x) = x isa QuoteNode ? string(x.value) : x isa Symbol ? string(x) : x isa String ? x : nothing

"A literal string, or the literal parts of an interpolated one."
_lit(x) = x isa String ? x :
          x isa Expr && x.head == :string ? join(a for a in x.args if a isa String) : nothing

const _THUNK_RE = r"^_mlir_ciface_(.+)_thunk$"

"""
What one function body does: the `ccall` targets it names (with the library
symbol they are paired with), the thunks it invokes (JIT string, AOT slot,
AOT virtual), and the first refusal message it raises.
"""
mutable struct Body
    ccalls::Vector{Tuple{String,String}}   # (symbol, library spelling)
    thunks::Set{String}                    # mangled
    slots::Set{String}                     # _FPTR_* read through `[]`
    trap::Union{Nothing,String}
end
Body() = Body(Tuple{String,String}[], Set{String}(), Set{String}(), nothing)

function scan_body!(b::Body, ex)
    if ex isa String
        m = match(_THUNK_RE, ex)
        m === nothing || push!(b.thunks, String(m.captures[1]))
        return b
    end
    ex isa Expr || return b
    if ex.head == :call && ex.args[1] === :ccall && length(ex.args) >= 2
        t = ex.args[2]
        if t isa Expr && t.head == :tuple && length(t.args) >= 2
            n = _callee(t.args[1]); lib = _callee(t.args[2])
            n === nothing || push!(b.ccalls, (n, something(lib, "?")))
        else
            n = _callee(t)
            n === nothing || push!(b.ccalls, (n, "?"))
        end
    elseif ex.head == :macrocall && ex.args[1] === Symbol("@ccall")
        for a in ex.args[2:end]
            c = a isa Expr && a.head == :(::) ? a.args[1] : a
            if c isa Expr && c.head == :call
                f = c.args[1]
                if f isa Expr && f.head == :.
                    n = _callee(f.args[2])
                    n === nothing || push!(b.ccalls, (n, something(_callee(f.args[1]), "?")))
                else
                    n = _callee(f)
                    n === nothing || push!(b.ccalls, (n, "?"))
                end
            end
        end
    elseif ex.head == :ref && length(ex.args) == 1 && ex.args[1] isa Symbol && startswith(String(ex.args[1]), "_FPTR_")
        push!(b.slots, String(ex.args[1]))
    elseif ex.head == :call && length(ex.args) >= 2 && b.trap === nothing &&
           ex.args[1] isa Expr && ex.args[1].head == :. && ex.args[1].args[1] === :Base &&
           ex.args[1].args[2] == QuoteNode(:error)
        s = _lit(ex.args[2])
        if s !== nothing && (startswith(s, "ABI Safety Trap") || startswith(s, "FFI Safety Trap"))
            b.trap = s
        end
    end
    foreach(a -> scan_body!(b, a), ex.args)
    return b
end

"The name a `function` definition's signature binds, or `nothing`."
function fname(sig)
    while sig isa Expr && sig.head in (:where, :(::))
        sig = sig.args[1]
    end
    sig isa Expr && sig.head == :call || return nothing
    f = sig.args[1]
    f isa Expr && f.head == :(::) && (f = f.args[1])   # (f::T)(…) callable struct
    return f isa Symbol ? String(f) : nothing
end

struct Def
    name::String
    mangled::Union{Nothing,String}   # from the docstring's "Mangled symbol: `…`"
    body::Body
end

struct StructDef
    name::String
    mutable::Bool
    fields::Vector{Tuple{String,String}}   # (field, type text)
end

"Every function definition, struct definition and AOT slot of a parsed wrapper."
function collect_defs!(defs::Vector{Def}, structs::Vector{StructDef}, slots::Dict{String,String}, ex, doc = nothing)
    ex isa Expr || return
    if ex.head == :macrocall && length(ex.args) >= 4 &&
       (ex.args[1] == GlobalRef(Core, Symbol("@doc")) || ex.args[1] === Symbol("@doc"))
        d = _lit(ex.args[3])
        collect_defs!(defs, structs, slots, ex.args[4], d)
        return
    elseif ex.head == :function && length(ex.args) == 2
        n = fname(ex.args[1])
        m = doc === nothing ? nothing : match(r"Mangled symbol: `([^`]+)`", doc)
        n === nothing || push!(defs, Def(n, m === nothing ? nothing : String(m.captures[1]),
                                         scan_body!(Body(), ex.args[2])))
        return
    elseif ex.head == :struct
        nm = ex.args[2]
        nm isa Expr && nm.head == :<: && (nm = nm.args[1])
        nm isa Expr && nm.head == :curly && (nm = nm.args[1])
        fields = Tuple{String,String}[]
        for a in ex.args[3].args
            if a isa Symbol
                push!(fields, (String(a), "Any"))
            elseif a isa Expr && a.head == :(::) && a.args[1] isa Symbol
                push!(fields, (String(a.args[1]), string(a.args[2])))
            end
        end
        nm isa Symbol && push!(structs, StructDef(String(nm), ex.args[1], fields))
        return
    elseif ex.head == :const && ex.args[1] isa Expr && ex.args[1].head == :(=) &&
           ex.args[1].args[1] === :_THUNK_SLOTS
        rhs = ex.args[1].args[2]
        if rhs isa Expr && rhs.head == :ref
            for t in rhs.args[2:end]
                if t isa Expr && t.head == :tuple && length(t.args) == 2 && t.args[1] isa Symbol
                    m = _lit(t.args[2]) === nothing ? nothing : match(_THUNK_RE, _lit(t.args[2]))
                    m === nothing || (slots[String(t.args[1])] = String(m.captures[1]))
                end
            end
        end
        return
    end
    foreach(a -> collect_defs!(defs, structs, slots, a), ex.args)
end

# ─── trap and type classification ────────────────────────────────────────────

"The cause a refusal message names, in the order the generator appends them."
function trap_cause(msg::AbstractString)
    occursin("FFI Safety Trap", msg) && return "unmapped type (FFI trap)"
    occursin("no Julia type was resolved for that return", msg) && return "unresolved return type"
    # The x87 scalar: BRANCH 0's own stub, or the Tier 2 layout trap, which
    # sees `long double` as a record with no layout (it is no record).
    (occursin("80-bit x87", msg) || "long double" in trap_records(msg)) && return "long double by value"
    occursin("layout is not in this library's DWARF", msg) && return "by-value record with no layout"
    if occursin("so no thunk could call it", msg)
        m = match(r"by value, ([^;.]*?), so no thunk could call it", msg)
        return "SysV refusal: " * (m === nothing ? "?" : String(m.captures[1]))
    end
    occursin("keys types by their bare DWARF name", msg) && return "conflicted type name"
    return "other refusal"
end

"The types a refusal message says the function takes or returns by value."
trap_records(msg::AbstractString) =
    unique(String(m.captures[1]) for m in eachmatch(r"(?:takes|returns) `([^`]+)`(?: \w+)? by value", msg))

"""
The `const`/non-`const` twin of a method's mangled symbol (`_ZNK…` ↔ `_ZN…`),
or `nothing`. Two such overloads become one Julia signature
(`(this::Any, …)`), so `_dedup_method_chunks` keeps one: a distinct entry
point, not a destructor variant.
"""
function const_twin(m::AbstractString)
    startswith(m, "_ZNK") && return "_ZN" * m[5:end]
    startswith(m, "_ZN") && !startswith(m, "_ZNK") && return "_ZNK" * m[4:end]
    return nothing
end

_is_bytes(t) = occursin(r"^NTuple\{\d+, ?UInt8\}$", t)

function type_kind(s::StructDef)
    isempty(s.fields) && return :opaque
    (startswith(s.name, "Managed") || any(f -> f[1] == "handle", s.fields)) && s.mutable && return :owner
    if length(s.fields) == 1 && s.fields[1][1] in ("_data", "data") && _is_bytes(s.fields[1][2])
        return s.mutable ? :byte_record : :blob
    end
    return :fields
end

_dtor_group(m) = (x = match(r"^(.*?)D[0-2]Ev$", m); x === nothing ? nothing : String(x.captures[1]))

# ─── one library ─────────────────────────────────────────────────────────────

"""
The libraries of a package: itself, or each C++ library of its graph with the
`bound_symbols` of the libraries wrapped before it.
"""
function libraries(p)
    dir = joinpath(HUB, "packages", p)
    toml = _toml(p)
    out = NamedTuple[]
    if _is_graph(p)
        gpath = joinpath(dir, "julia", "graph.toml")
        isfile(gpath) || return [(; row = p, note = "graph not built (no julia/graph.toml)")]
        g = TOML.parsefile(gpath)
        earlier = Set{String}()
        skipped = String[]
        for lib in g["libraries"]
            ldir = joinpath(dir, "julia", dirname(lib["wrapper"]))
            if lib["bucket"] != "cpp"
                push!(skipped, lib["name"])
            else
                push!(out, (; row = p * "/" * lib["name"], toml, wrapper = joinpath(dir, "julia", lib["wrapper"]),
                            metadata = joinpath(dir, "julia", lib["metadata"]),
                            manifest = joinpath(ldir, "thunk_manifest.json"),
                            earlier = copy(earlier), indexed = p in INDEXED || get(INDEX, p, nothing) !== nothing))
            end
            tm = joinpath(ldir, "type_manifest.json")
            isfile(tm) && union!(earlier, String.(get(RepliBuild._read_json(tm), "bound_symbols", String[])))
        end
        isempty(skipped) || push!(out, (; row = p, note = "C libraries skipped (the C tool's): " * join(skipped, ", ")))
        return out
    end
    wrapper = joinpath(dir, "julia", RepliBuild.ConfigurationManager.wrapper_module_name(toml) * ".jl")
    return [(; row = p, toml, wrapper, metadata = joinpath(dir, "julia", "compilation_metadata.json"),
             manifest = joinpath(dir, "julia", "thunk_manifest.json"), earlier = Set{String}(),
             indexed = p in INDEXED)]
end

function measure(lib)
    haskey(lib, :note) && return (; row = lib.row, ok = false, note = lib.note)
    isfile(lib.wrapper) || return (; row = lib.row, ok = false, note = "no wrapper at julia/$(basename(lib.wrapper))")
    isfile(lib.metadata) || return (; row = lib.row, ok = false, note = "no compilation_metadata.json")

    src = read(lib.wrapper, String)
    ast = Meta.parseall(src)
    any(e -> e isa Expr && e.head in (:incomplete, :error), ast.args) &&
        return (; row = lib.row, ok = false, note = "wrapper does not parse")
    defs = Def[]; structs = StructDef[]; slots = Dict{String,String}()
    collect_defs!(defs, structs, slots, ast)

    # What the parsed code binds, over every definition (finalizers included).
    ccall_targets = Set{String}()
    thunks = Set{String}()
    for d in defs, (n, l) in d.body.ccalls
        if l == "THUNKS_LIBRARY_PATH" && startswith(n, "thunk_")
            push!(thunks, n[7:end])
        else
            push!(ccall_targets, n)
        end
    end
    for d in defs
        union!(thunks, d.body.thunks)
        for s in d.body.slots
            haskey(slots, s) && push!(thunks, slots[s])
        end
    end

    md = RepliBuild._read_json(lib.metadata)
    byman = Dict{String,Any}()               # one entry per mangled symbol
    for f in get(md, "functions", [])
        m = String(get(f, "mangled", ""))
        isempty(m) && continue
        haskey(byman, m) || (byman[m] = f)
    end
    patterns = String[String(x) for x in get(get(TOML.parsefile(lib.toml), "wrap", Dict()), "exclude_symbols", String[])]
    (matches, _) = RepliBuild._glob_matcher(patterns)
    excluded = 0; unexported = 0; earlier = 0
    surface = String[]
    for (m, f) in byman
        if matches(get(f, "name", nothing)) || matches(get(f, "demangled", nothing)) || matches(m)
            excluded += 1
        elseif get(f, "exported", true) == false
            unexported += 1
        elseif m in lib.earlier
            earlier += 1
        else
            push!(surface, m)
        end
    end
    sort!(surface)

    manifest = isfile(lib.manifest) ?
        Set(String.(get(RepliBuild._read_json(lib.manifest), "function_thunks", String[]))) : Set{String}()
    # Trap stubs by mangled symbol (their docstring), else by the demangled
    # name the message quotes.
    trap_by_m = Dict{String,String}()
    trap_by_dem = Dict{String,String}()
    for d in defs
        d.body.trap === nothing && continue
        d.mangled === nothing || (trap_by_m[d.mangled] = d.body.trap)
        q = match(r"cannot call '[^']*' \((.*?)\)\. It ", d.body.trap)
        q === nothing || (trap_by_dem[String(q.captures[1])] = d.body.trap)
    end

    tier3 = String[]; tier2 = String[]
    traps = Dict{String,Vector{String}}()          # cause => symbols
    dropped = Dict{String,Vector{String}}()        # cause => symbols: misses
    expected = Dict{String,Vector{String}}()       # cause => symbols: drops the generator means
    blocked = Dict{String,Int}()                   # record => functions it traps
    bound = Set{String}()
    for m in surface
        (m in ccall_targets || m in thunks) && push!(bound, m)
    end
    # The C++ name without its parameter list: two overloads of one name
    # become one Julia name, and `::Any` parameters one signature.
    _cxx_name(m) = (d = String(get(byman[m], "demangled", "")); i = findfirst('(', d);
                    i === nothing ? d : d[1:prevind(d, i)])
    bound_names = Set(_cxx_name(m) for m in bound)
    # The generator's own name for a function (`_julia_function_name`), so a
    # drop can be told from a name two C++ functions share once spelled.
    _jl_name(f) = begin
        nm = String(get(f, "name", "")); cls = String(get(f, "class", ""))
        RepliBuild.Wrapper._julia_function_name(
            get(f, "is_method", false) == true && !isempty(cls) ? "$(cls)_$nm" : nm, String(get(f, "mangled", nm)))
    end
    _arity(f) = length(get(f, "parameters", Any[]))
    bound_jl = Dict{Tuple{String,Int},Vector{String}}()
    for b in bound
        push!(get!(bound_jl, (_jl_name(byman[b]), _arity(byman[b])), String[]), b)
    end
    # Parameter shapes a Julia caller cannot tell apart: the C types with
    # `const` and `&&`→`&` normalised (an rvalue beside its lvalue twin), and
    # with every integer at the Julia type it names on this host (two C
    # spellings of one width).
    _norm(c) = replace(replace(String(c), r"\b(const|volatile)\b\s*" => ""), r"\s+" => "")
    _lv_shape(f) = [replace(_norm(get(p, "c_type", "")), "&&" => "&") for p in get(f, "parameters", Any[])]
    _has_rvalue(f) = any(p -> endswith(_norm(get(p, "c_type", "")), "&&"), get(f, "parameters", Any[]))
    _width_shape(f) = [(jt = String(get(p, "julia_type", "")); jt in RepliBuild.Wrapper._INTEGER_CTYPES ?
                        RepliBuild.Wrapper._concrete_ctype(jt) : _norm(get(p, "c_type", "")))
                       for p in get(f, "parameters", Any[])]
    for m in surface
        f = byman[m]
        if m in ccall_targets
            push!(tier3, m)
        elseif m in thunks
            push!(tier2, m)
        elseif haskey(trap_by_m, m) || haskey(trap_by_dem, String(get(f, "demangled", "")))
            msg = get(trap_by_m, m, get(trap_by_dem, String(get(f, "demangled", "")), ""))
            push!(get!(traps, trap_cause(msg), String[]), m)
            for r in trap_records(msg)
                r == "long double" && continue      # a scalar, counted by its cause
                blocked[r] = get(blocked, r, 0) + 1
            end
        else
            g = _dtor_group(m)
            twin = const_twin(m)
            same_name = get(bound_jl, (_jl_name(f), _arity(f)), String[])
            cxx = _cxx_name(m)
            siblings = [b for b in same_name if _cxx_name(b) == cxx]
            # Expected: what the generator drops on purpose, one Julia method
            # standing for several C++ entry points a Julia caller cannot tell
            # apart. Dropped: a distinct, callable entry point lost.
            if g !== nothing && any(b -> _dtor_group(b) == g, bound)
                push!(get!(expected, "destructor variant (D0/D1/D2: one Julia method per class)", String[]), m)
            elseif twin !== nothing && twin in bound
                push!(get!(expected, "const/non-const method pair (receiver constness; one kept)", String[]), m)
            elseif _has_rvalue(f) && any(b -> !_has_rvalue(byman[b]) && _lv_shape(byman[b]) == _lv_shape(f), siblings)
                push!(get!(expected, "rvalue-reference overload beside its lvalue twin (not emitted)", String[]), m)
            elseif any(b -> _width_shape(byman[b]) == _width_shape(f), siblings)
                push!(get!(expected, "integer overload of one width in another C spelling (one kept)", String[]), m)
            else
                cause = if get(f, "is_vararg", false) == true
                    "varargs not declared in [wrap.varargs]"
                elseif !isempty(siblings)
                    "overload set collapsed to one ::Any signature (another overload is bound)"
                elseif !isempty(same_name)
                    "Julia name shared with another C++ function (name derivation: `$(_jl_name(f))`)"
                elseif m in manifest
                    "dispatch collision (thunk in manifest, chunk dropped by dedup)"
                else
                    "not emitted"
                end
                push!(get!(dropped, cause, String[]), m)
            end
        end
    end

    kinds = Dict(k => 0 for k in (:fields, :blob, :byte_record, :opaque, :owner))
    for s in structs
        kinds[type_kind(s)] += 1
    end
    imported = count(_ -> true, eachmatch(r"^const \w+ = Base\.parentmodule\(@__MODULE__\)\."m, src))
    gen = match(r"BUILD_GENERATOR = \"([^\"]*)\"", src)
    unused_thunks = length(setdiff(manifest, thunks))
    foreign_thunks = length(setdiff(thunks, keys(byman)))   # STL members and the like

    return (; row = lib.row, ok = true, surface = length(surface), tier3 = length(tier3), tier2 = length(tier2),
            ntrap = sum(length, values(traps); init = 0), ndrop = sum(length, values(dropped); init = 0),
            nexpected = sum(length, values(expected); init = 0),
            traps, dropped, expected, blocked, kinds, imported, excluded, unexported, earlier, unused_thunks,
            foreign_thunks, byman, generator = gen === nothing ? "?" : gen.captures[1], indexed = lib.indexed)
end

# ─── report ──────────────────────────────────────────────────────────────────

rows = NamedTuple[]
for p in cpp_packages(), lib in libraries(p)
    push!(rows, measure(lib))
end
ok = [r for r in rows if r.ok]
pct(a, b) = b == 0 ? "—" : string(round(100a / b; digits = 2))
_short(s, n = 90) = length(s) > n ? s[1:prevind(s, n)] * "…" : s
_dem(r, m) = _short(String(get(r.byman[m], "demangled", m)))

println("| Library | Bound / callable | % | Tier 3 | Tier 2 | Traps | Dropped | Expected | Excluded / earlier | Types f/b/u/o/i | Generator |")
println("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows
    if !r.ok
        println("| $(r.row) | — | — | | | | | | $(r.note) | | |")
        continue
    end
    note = r.indexed ? "" : " (not indexed)"
    b = r.tier3 + r.tier2
    callable = r.surface - r.nexpected
    k = r.kinds
    println("| $(r.row)$note | $b/$callable | $(pct(b, callable)) | $(r.tier3) | $(r.tier2) | $(r.ntrap) | ",
            "$(r.ndrop) | $(r.nexpected) | $(r.excluded == 0 ? "" : r.excluded)$(r.earlier == 0 ? "" : " / $(r.earlier) earlier") | ",
            "$(k[:fields])/$(k[:blob])/$(k[:byte_record])/$(k[:opaque])/$(r.imported) | $(r.generator) |")
end

surface = sum(r -> r.surface, ok; init = 0)
tier3 = sum(r -> r.tier3, ok; init = 0)
tier2 = sum(r -> r.tier2, ok; init = 0)
ntrap = sum(r -> r.ntrap, ok; init = 0)
ndrop = sum(r -> r.ndrop, ok; init = 0)
nexpected = sum(r -> r.nexpected, ok; init = 0)
bound = tier3 + tier2
callable = surface - nexpected
println()
println("libraries measured: $(length(ok)) of $(length(rows)) rows")
println("surface (unique mangled, net of exclusions, not-exported and earlier-bound): $surface")
println("expected drops (several C++ entry points behind one Julia method, by design): $nexpected")
println("callable (surface less expected): $callable")
println("bound:   $bound/$callable = $(pct(bound, callable))%  (tier 3: $tier3, tier 2: $tier2)")
println("traps:   $ntrap   dropped: $ndrop   unexported: $(sum(r -> r.unexported, ok; init = 0))")
println("thunks in a manifest nothing invokes: $(sum(r -> r.unused_thunks, ok; init = 0))" *
        "   thunks invoked outside the surface (STL members): $(sum(r -> r.foreign_thunks, ok; init = 0))")
k = Dict(x => sum(r -> r.kinds[x], ok; init = 0) for x in (:fields, :blob, :byte_record, :opaque, :owner))
println("types:   fields $(k[:fields]), blobs $(k[:blob]), byte records (unions/bitfields) $(k[:byte_record]), " *
        "opaque $(k[:opaque]), imported $(sum(r -> r.imported, ok; init = 0)), owners (Managed/factory) $(k[:owner])")

# Misses by cause, over every library, most first.
causes = Dict{String,Vector{Tuple{String,String}}}()   # cause => (library, mangled)
for r in ok, (c, ms) in r.traps, m in ms
    push!(get!(causes, "trap: " * c, Tuple{String,String}[]), (r.row, m))
end
for r in ok, (c, ms) in r.dropped, m in ms
    push!(get!(causes, "dropped: " * c, Tuple{String,String}[]), (r.row, m))
end
byrow = Dict(r.row => r for r in ok)
println("\nmisses by cause (count, then the first symbols):")
for (c, xs) in sort!(collect(causes); by = x -> (-length(x[2]), x[1]))
    println("  $(length(xs))  $c")
    perlib = Dict{String,Int}()
    for (row, _) in xs
        perlib[row] = get(perlib, row, 0) + 1
    end
    println("        per library: ", join(("$k $v" for (k, v) in sort!(collect(perlib); by = x -> -x[2])), ", "))
    for (row, m) in xs[1:("--list" in ARGS ? end : min(4, end))]
        println("        $row  ", _dem(byrow[row], m))
    end
end

exp_causes = Dict{String,Vector{Tuple{String,String}}}()
for r in ok, (c, ms) in r.expected, m in ms
    push!(get!(exp_causes, c, Tuple{String,String}[]), (r.row, m))
end
if !isempty(exp_causes)
    println("\nexpected drops (not misses):")
    for (c, xs) in sort!(collect(exp_causes); by = x -> (-length(x[2]), x[1]))
        println("  $(length(xs))  $c")
        for (row, m) in xs[1:("--list" in ARGS ? end : min(3, end))]
            println("        $row  ", _dem(byrow[row], m))
        end
    end
end

blocked = Dict{String,Int}()
for r in ok, (rec, n) in r.blocked
    blocked[rec] = get(blocked, rec, 0) + n
end
if !isempty(blocked)
    println("\ntrapped records (functions each blocks):")
    for (rec, n) in sort!(collect(blocked); by = x -> (-x[2], x[1]))[1:min(15, end)]
        println("  $n  $(_short(rec, 70))")
    end
end

println()
println("cpp coverage: $bound/$callable ($(pct(bound, callable))%), traps $ntrap, dropped $ndrop, expected $nexpected")

isempty(ok) && error("no C++ library was measured: nothing to report")
surface > 0 || error("the measured surface is empty: the metadata lists no functions")
