# =============================================================================
# NAME SCREEN
# =============================================================================
#
# A generated module is a namespace the LIBRARY populates: each of its
# functions, records, enumerators and globals becomes a top-level binding,
# beside text the generators wrote against Base, Core, Libdl and RepliBuild.
# A library identifier that is also a name of that text changes what the text
# means. `int String(int)` became a method of `Base.String` in every module of
# the process, `struct Integer` turned every `x::Integer` signature into the
# struct, `int nothing(int)` made `return nothing` return a function, and
# `int eval(int)` or `int LIBRARY_PATH(int)` cost the whole module
# (2026-11-17).
#
# This file is the one screen every such name passes: functions and method
# proxies through `_julia_function_name`, global accessors through
# `_screened_function_name`, records and enums through `_screen_records`,
# enumerators through `_render_enum_chunks`, and the module's own name through
# `_assert_module_name_free`. It asks, never lists:
#
#   * what Julia says about the name: whether a function may be called that at
#     all, and what the name already means in a module that has bound nothing
#     (`_fresh_binding`: Base's and Core's exports, and `eval`/`include`);
#   * what the wrapper's own text says: which names it references and which it
#     binds, read off the emitted text (`_name_uses`).
#
# The second question has no answer before the text exists, and in a finished
# text a reference to `open` does not say whose `open` it means. So the first
# generation is a PROBE: every library name that already means something in a
# fresh module is moved to its alternative, and then any reference or binding
# of the original name left in the text is the wrapper's own. Names nothing
# uses go back to the library's spelling in a second generation
# (`_screen_verdict`). A library with no such name is generated once, and a
# name the wrapper's text does not use keeps its spelling: `error`, `length`
# and `stat` stay module-local functions, withheld from `export` as before.
#
# So what a library keeps is decided by how the generators write. Text that
# calls a function of Base by its bare name (`isfile(p)`) takes that name from
# every library that has one; text that calls `Base.isfile(p)` takes nothing.
# The header every wrapper starts with is written the second way.

import Base.CoreLogging

# ── What Julia says about a name ────────────────────────────────────────────

const _FRESH_MODULE = Ref{Union{Nothing,Module}}(nothing)
const _FRESH_LOCK = ReentrantLock()

# A module like a wrapper before the wrapper binds anything: `using Base`, and
# its own `eval` and `include`, which `Module(:name)` does not give, so it is
# written as a wrapper is, into a host nothing else sees. Created on first use,
# never at precompile.
function _fresh_module()::Module
    m = _FRESH_MODULE[]
    m === nothing || return m
    return lock(_FRESH_LOCK) do
        _FRESH_MODULE[] === nothing &&
            (_FRESH_MODULE[] = Core.eval(Module(:RepliBuildNameScreen), :(module FreshModule end)))
        _FRESH_MODULE[]::Module
    end
end

"""
    _fresh_binding(name) -> Union{Nothing,Some}

What `name` already means in a module that has bound nothing: `nothing` when
it is free, else `Some(value)`. Asked of the running Julia, so a name a newer
release exports is screened on that release alone.
"""
function _fresh_binding(name::AbstractString)
    Base.isidentifier(name) || return nothing
    s = Symbol(name)
    m = _fresh_module()
    (isdefined(m, s) && s !== nameof(m)) || return nothing
    (Base.isdeprecated(Base, s) || Base.isdeprecated(Core, s)) && return Some(nothing)
    return Some(getglobal(m, s))
end

# `eval` and `include`: bindings every module owns, so a library cannot.
function _every_module_binds(name::AbstractString)::Bool
    _fresh_binding(name) === nothing && return false
    m = _fresh_module()
    return Base.binding_module(m, Symbol(name)) === m
end

const _DEFINABLE = Dict{String,Bool}()

# Whether Julia lets a function be called `name`: the parser refuses a reserved
# word and lowering refuses `ccall` and `cglobal`. Asked, since the list moves
# with the release (and `Base.isidentifier("end")` is true).
function _definable_function_name(name::AbstractString)::Bool
    n = String(name)
    return lock(_FRESH_LOCK) do
        get!(_DEFINABLE, n) do
            ex = try
                Meta.parse("function $n(x) end"; raise = false)
            catch
                return false
            end
            (ex isa Expr && ex.head === :function && ex.args[1] isa Expr &&
             ex.args[1].head === :call && ex.args[1].args[1] === Symbol(n)) || return false
            low = try
                Meta.lower(_fresh_module(), ex)
            catch
                return false
            end
            return !(low isa Expr && low.head === :error)
        end
    end
end

"""
    _function_name_refusal(name) -> Union{Nothing,Symbol}

Why a library function can never be bound as `name`, whatever the wrapper's
text says: `:syntax` (Julia has no function of that name), `:every_module`
(`eval`, `include`), or `:base_type` (the name is a type of Base or Core, and a
method written `function String(x)` in a module that does not own `String` is
a method of Base's type, for every module of the process).
"""
function _function_name_refusal(name::AbstractString)::Union{Nothing,Symbol}
    _definable_function_name(name) || return :syntax
    b = _fresh_binding(name)
    b === nothing && return nothing
    _every_module_binds(name) && return :every_module
    something(b) isa Type && return :base_type
    return nothing
end

# The name a refused or colliding function takes: the reserved words' `_`.
function _function_alternative(name::AbstractString)::String
    to = String(name) * "_"
    while _function_name_refusal(to) !== nothing || _fresh_binding(to) !== nothing
        to *= "_"
    end
    return to
end

_screen_why(why::Symbol, name::AbstractString) =
    why === :syntax       ? "Julia has no function named `$name`" :
    why === :every_module ? "every module binds its own `$name`" :
    why === :base_type    ? "`$name` is a type of Base or Core, and a function of that name would be a method of it in every module" :
    why === :own_use      ? "the wrapper's own text uses `$name`" :
    why === :own_binding  ? "the wrapper itself binds `$name`" :
    why === :type_constructor ?
        "the type `$name` defines a method of the same signature itself (a constructor such as its " *
        "zero-initializer `$name()`), and keeps it" :
                            "the wrapper's own text may use `$name`"

# ── The screen of one wrap ──────────────────────────────────────────────────

struct _NameMove
    kind::Symbol                    # :function, :type or :enumerator
    from::String
    to::String
    why::Symbol
    source::String                  # the symbol, record or enum behind it
end

"""
    _NameScreen

The screen of one generation of one wrapper. `probe` is the first generation:
every candidate (a name `_fresh_binding` knows) moves. A later generation
moves exactly `move`, what the verdict on the text before it found. `met` are
the candidates this generation met and `judged` the ones a probe has ruled on;
`kept` are the names it bound as the library spells them, by kind; `moves` is
what it moved, for the report.
"""
mutable struct _NameScreen
    probe::Bool
    move::Dict{Tuple{Symbol,String},Symbol}
    judged::Set{Tuple{Symbol,String}}
    met::Set{Tuple{Symbol,String}}
    kept::Dict{Symbol,Set{String}}
    moves::Vector{_NameMove}
end

_NameScreen(; probe::Bool = true,
            move = Dict{Tuple{Symbol,String},Symbol}(),
            judged = Set{Tuple{Symbol,String}}()) =
    _NameScreen(probe, move, judged, Set{Tuple{Symbol,String}}(),
                Dict{Symbol,Set{String}}(k => Set{String}() for k in (:function, :type, :enumerator)),
                _NameMove[])

# The screen of the generation in progress. `wrap_introspective` installs it
# (`with_name_screen`); a generator called on its own runs with none and gets
# the refusals alone.
const _SCREEN = Ref{Union{Nothing,_NameScreen}}(nothing)

function with_name_screen(f, sc::_NameScreen)
    _SCREEN[] !== nothing && error("RepliBuild: with_name_screen nested; a wrap runs one generator at a time")
    _SCREEN[] = sc
    try
        return f()
    finally
        _SCREEN[] = nothing
    end
end

# Whether this generation moves `name` bound as `kind`, and why. Records what
# it met. `refusal` is the kind's own answer that holds in any text.
function _screen_decide!(sc::Union{Nothing,_NameScreen}, kind::Symbol, name::String,
                         refusal::Union{Nothing,Symbol})::Union{Nothing,Symbol}
    refusal === nothing || return refusal
    sc === nothing && return nothing
    key = (kind, name)
    why = get(sc.move, key, nothing)
    if _fresh_binding(name) !== nothing
        push!(sc.met, key)
        # A candidate no probe has ruled on moves: a probe moves them all, and
        # one a later generation is the first to meet has no verdict yet.
        (why === nothing && (sc.probe || !(key in sc.judged))) && (why = :probe)
    end
    why === nothing && push!(sc.kept[kind], name)
    return why
end

"""
    _screened_function_name(name, source = "") -> String

`name` as a top-level function may be bound: itself, or its alternative
(`String_`) when Julia refuses it (`_function_name_refusal`) or the screen of
the wrap in progress moves it. `_julia_function_name` ends here; a global's
accessor, which is not derived there, calls it directly.
"""
function _screened_function_name(name::AbstractString, source::AbstractString = "")::String
    n = String(name)
    sc = _SCREEN[]
    why = _screen_decide!(sc, :function, n, _function_name_refusal(n))
    why === nothing && return n
    to = _function_alternative(n)
    sc === nothing || push!(sc.moves, _NameMove(:function, n, to, why, String(source)))
    return to
end

"""
    _screen_enumerator(base) -> Union{Nothing,Symbol}

Why the enumerator `base` cannot be bound under its own name, for
`_render_enum_chunks`, which gives it `<enum>_<base>` as it does one that
collides with another binding of the wrapper.
"""
function _screen_enumerator(base::AbstractString)::Union{Nothing,Symbol}
    n = String(base)
    return _screen_decide!(_SCREEN[], :enumerator, n, _every_module_binds(n) ? :every_module : nothing)
end

_screen_note_enumerator!(base::AbstractString, to::AbstractString, why::Symbol, enum::AbstractString) =
    (sc = _SCREEN[]; sc === nothing || push!(sc.moves, _NameMove(:enumerator, String(base), String(to), why, String(enum))); nothing)

# ── Records ─────────────────────────────────────────────────────────────────

_plain_identifier(s::AbstractString) = occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", s)

# `name` as a whole word of a C or Julia type spelling, and not the head of a
# Julia one: in `Ptr{Ptr}`, the pointer to a record named `Ptr`, only the
# second is the record.
_name_token(name::AbstractString) = Regex("(?<![A-Za-z0-9_])" * name * "(?![A-Za-z0-9_{])")

"""
    _screen_records(sc, metadata, sanitize) -> metadata

The metadata the generators read under screen `sc`: `metadata` itself when no
record moves, else a copy in which each moved record carries its alternative
name (`c_Bool`), as if the library had declared it so.

A record cannot be moved where its name is spelled, in the type sanitizers:
those run over Julia spellings too, and the metadata spells a library
`struct Bool` and C's `bool` alike (`julia_type: "Bool"`), as it does `Cvoid`,
`UInt8` or `Cstring`. Only the C spelling tells them apart (`Bool` against
`bool`), so the move is made here, where both are in hand: the record's key,
and every C spelling of it, take the new name, and a `julia_type` follows only
when the `c_type` beside it named the record. The library's symbols and the
demangled text are untouched, and the thunk side reads the file on disk.

A candidate is a record or enum key that is a plain identifier the sanitizer
leaves alone (so the names `_sanitize_cpp_type_name` already moves stay its
own) and that `_fresh_binding` knows.
"""
function _screen_records(sc::_NameScreen, metadata::AbstractDict, sanitize)
    defs = get(metadata, "struct_definitions", nothing)
    names = Set{String}()
    defs isa AbstractDict && for k in keys(defs)
        push!(names, replace(String(k), r"^__enum__" => ""))
    end
    henums = get(metadata, "header_enums", nothing)
    henums isa AbstractDict && union!(names, String.(keys(henums)))

    plan = Pair{String,String}[]
    for name in sort!(collect(names))
        (_plain_identifier(name) && String(sanitize(name)) == name) || continue
        why = _screen_decide!(sc, :type, name, nothing)
        why === nothing && continue
        to = "c_" * name
        while to in names || _fresh_binding(to) !== nothing
            to *= "_"
        end
        push!(plan, name => to)
        push!(sc.moves, _NameMove(:type, name, to, why, name))
    end
    isempty(plan) && return metadata

    moved = deepcopy(metadata)
    for (from, to) in plan
        _rename_record!(moved, from, to)
    end
    return moved
end

# Give the record or enum `from` the name `to` throughout `metadata`.
function _rename_record!(metadata::AbstractDict, from::String, to::String)
    re = _name_token(from)
    tok(s) = s isa AbstractString ? replace(String(s), re => to) : s
    # An enum's key is `__enum__<name>`, where the name follows a `_`.
    rekey!(d) = d isa AbstractDict && for k in collect(keys(d))
        nk = startswith(k, "__enum__") ? "__enum__" * tok(k[9:end]) : tok(k)
        nk == k || (d[nk] = pop!(d, k))
    end

    # A method's class, and the constructor and destructor named after it.
    for f in get(metadata, "functions", Any[])
        f isa AbstractDict || continue
        cls = get(f, "class", nothing)
        if cls isa AbstractString && occursin(re, cls)
            nm = get(f, "name", nothing)
            nm isa AbstractString && (nm == from || nm == "~" * from) && (f["name"] = tok(nm))
            f["class"] = tok(cls)
        end
    end
    for key in ("struct_definitions", "record_abi", "header_enums", "stl_methods", "type_registry")
        rekey!(get(metadata, key, nothing))
    end
    # A registry entry is keyed by its C spelling and may carry no `c_type`.
    reg = get(metadata, "type_registry", nothing)
    reg isa AbstractDict && for (k, v) in reg
        (v isa AbstractDict && !haskey(v, "c_type") && occursin(_name_token(to), String(k))) || continue
        haskey(v, "julia_type") && (v["julia_type"] = tok(v["julia_type"]))
    end
    sub = get(metadata, "graph_subclassed", nothing)
    sub isa AbstractVector && (metadata["graph_subclassed"] = Any[tok(x) for x in sub])
    _respell_record!(metadata, re, tok)
    return metadata
end

# C spellings by the field that holds them. `julia_type` is not one: it follows
# its `c_type`.
const _C_SPELLING_FIELDS = ("record", "container", "scope")

function _respell_record!(x, re::Regex, tok)
    if x isa AbstractDict
        ct = get(x, "c_type", nothing)
        if ct isa AbstractString && occursin(re, ct)
            x["c_type"] = tok(ct)
            jt = get(x, "julia_type", nothing)
            jt isa AbstractString && (x["julia_type"] = tok(jt))
        end
        for f in _C_SPELLING_FIELDS
            haskey(x, f) && x[f] isa AbstractString && (x[f] = tok(x[f]))
        end
        sc = get(x, "scopes", nothing)
        sc isa AbstractVector && (x["scopes"] = Any[tok(s) for s in sc])
        # A base class and a template argument are spelled under `type`.
        for f in ("base_classes", "template_params")
            v = get(x, f, nothing)
            v isa AbstractVector || continue
            for e in v
                e isa AbstractDict && get(e, "type", nothing) isa AbstractString && (e["type"] = tok(e["type"]))
            end
        end
        for v in values(x)
            (v isa AbstractDict || v isa AbstractVector) && _respell_record!(v, re, tok)
        end
    elseif x isa AbstractVector
        for v in x
            (v isa AbstractDict || v isa AbstractVector) && _respell_record!(v, re, tok)
        end
    end
    return x
end

"""
    _library_named_manifest!(manifest, metadata, moves)

A type manifest keyed and digested as the library names its types. The
manifest is derived from the metadata the generators read, in which a moved
record carries its alternative name; the libraries wrapped after this one
match on the C++ name and on digests of the entry (`_resolve_type_imports`),
so both come from `metadata`, the library's own. The Julia name stays the one
the text binds.
"""
function _library_named_manifest!(manifest::AbstractDict, metadata::AbstractDict,
                                  moves::AbstractVector{_NameMove})
    types = get(manifest, "types", nothing)
    types isa AbstractDict || return manifest
    defs = get(metadata, "struct_definitions", Dict{String,Any}())
    back = Dict{String,String}(m.to => m.from for m in moves if m.kind === :type)
    isempty(back) && return manifest
    strong_index = _strong_member_index(metadata)
    for key in sort!(collect(String.(keys(types))))
        entry = types[key]
        entry isa AbstractDict || continue
        prefix = startswith(key, "__enum__") ? "__enum__" : ""
        bare = key[ncodeunits(prefix)+1:end]
        orig = prefix * get(back, bare, bare)
        info = get(defs, orig, nothing)
        info isa AbstractDict || continue          # a placeholder, keyed by its Julia name
        entry["scopes"] = get(info, "scopes", nothing)
        entry["layout_signature"] = _layout_signature(info)
        entry["strong_members"] = _strong_members(metadata, orig; index = strong_index)
        entry["entry_sha"] = _sha256_hex(_canonical_text(info))
        if orig != key
            delete!(types, key)
            types[orig] = entry
        end
    end
    return manifest
end

# ── What a text binds and references ────────────────────────────────────────

"""
    _NameUses

`bound`: every name a module text binds at top level, with the kinds it is
bound as, in source order (`:function`, `:struct`, `:type`, `:const`,
`:global`, `:enum`, `:enumerator`, and, which `_defined_bindings` does not
report, `:import` and `:module`). `refs`: every name it references at module
scope, by Julia's scope rules: a parameter, a local, a `let` or `for` variable,
a field name, a keyword and a quoted symbol are not references, and
`Base.error` references `Base`.
"""
struct _NameUses
    bound::Dict{String,Vector{Symbol}}
    refs::Set{String}
end

"""
    _name_uses(text) -> Union{Nothing,_NameUses}

Read `_NameUses` off the parsed text; `nothing` when it does not parse, which
`_assert_wrapper_parses` reports.

Deliberately its own walk of the surface syntax, the same on every Julia, not
a reading of lowered code. Where it has to guess it guesses towards a
reference: an unknown form has every symbol in it counted, since a reference
that is not one costs a rename, and a missed one costs the module.
"""
function _name_uses(text::AbstractString)::Union{Nothing,_NameUses}
    ex = try
        Meta.parseall(text)
    catch
        return nothing
    end
    u = _NameUses(Dict{String,Vector{Symbol}}(), Set{String}())
    bad = Ref(false)
    _uses!(u, ex, _NO_LOCALS, true, bad)
    return bad[] ? nothing : u
end

const _NO_LOCALS = Set{Symbol}()

_nu_bind!(u::_NameUses, n, kind::Symbol) =
    (n isa Symbol && push!(get!(() -> Symbol[], u.bound, String(n)), kind); nothing)

function _nu_ref!(u::_NameUses, s::Symbol, locals::Set{Symbol})
    (s in locals || s === :end || s === :begin) && return
    str = String(s)
    (isempty(str) || !Base.is_id_start_char(first(str))) && return       # an operator
    push!(u.refs, str)
    return
end

_nu_assigns(h::Symbol) = h === :(=) ||
    (h in (:+=, :-=, :*=, :/=, ://=, :\=, :^=, :÷=, :%=, :<<=, :>>=, :>>>=, :|=, :&=, :⊻=, :$=))

# The name a signature defines: a Symbol for `f(…)`, an Expr for `Base.f(…)`
# or `(x::T)(…)`, `nothing` for an anonymous function.
function _nu_fn_name(sig)
    sig isa Symbol && return sig
    sig isa Expr || return nothing
    if sig.head === :where || (sig.head === :(::) && length(sig.args) == 2)
        return _nu_fn_name(sig.args[1])
    elseif sig.head === :call
        f = sig.args[1]
        f isa Expr && f.head === :curly && return f.args[1]
        return f
    end
    return nothing
end

# Whether an assignment target defines a method: `f(x) = …`, with any `where`
# and return type.
function _nu_is_method(lhs)::Bool
    lhs isa Expr || return false
    lhs.head === :call && return true
    (lhs.head === :where || (lhs.head === :(::) && length(lhs.args) == 2)) && return _nu_is_method(lhs.args[1])
    return false
end

function _nu_where_names!(out::Set{Symbol}, params)
    for p in params
        p isa Symbol ? push!(out, p) :
        (p isa Expr && p.head in (:(<:), :(>:)) && !isempty(p.args) && p.args[1] isa Symbol) && push!(out, p.args[1])
    end
    return out
end

function _nu_param_names!(out::Set{Symbol}, a)
    if a isa Symbol
        push!(out, a)
    elseif a isa Expr
        if a.head === :(::)
            length(a.args) == 2 && _nu_param_names!(out, a.args[1])
        elseif a.head in (:kw, :(=), :...)
            _nu_param_names!(out, a.args[1])
        elseif a.head in (:parameters, :tuple)
            foreach(b -> _nu_param_names!(out, b), a.args)
        end
    end
    return out
end

# Parameter and `where` names of a signature.
function _nu_sig_names!(out::Set{Symbol}, sig)
    if sig isa Symbol
        return out                               # `function f end`
    elseif !(sig isa Expr)
        return out
    elseif sig.head === :where
        _nu_where_names!(out, sig.args[2:end])
        return _nu_sig_names!(out, sig.args[1])
    elseif sig.head === :(::) && length(sig.args) == 2 && _nu_is_method(sig.args[1])
        return _nu_sig_names!(out, sig.args[1])
    elseif sig.head === :call
        f = sig.args[1]
        f isa Expr && f.head === :(::) && length(f.args) == 2 && f.args[1] isa Symbol && push!(out, f.args[1])
        foreach(a -> _nu_param_names!(out, a), sig.args[2:end])
    elseif sig.head === :tuple || sig.head === :block
        foreach(a -> _nu_param_names!(out, a), sig.args)
    else
        _nu_param_names!(out, sig)
    end
    return out
end

function _nu_lhs_names!(out::Set{Symbol}, lhs)
    if lhs isa Symbol
        push!(out, lhs)
    elseif lhs isa Expr
        if lhs.head in (:tuple, :parameters)
            foreach(a -> _nu_lhs_names!(out, a), lhs.args)
        elseif lhs.head === :(::) && length(lhs.args) == 2
            _nu_lhs_names!(out, lhs.args[1])
        elseif lhs.head in (:(=), :kw, :...)
            _nu_lhs_names!(out, lhs.args[1])
        elseif _nu_is_method(lhs)
            nm = _nu_fn_name(lhs)
            nm isa Symbol && push!(out, nm)
        end
    end
    return out
end

# Every name `ex` assigns. Julia's rule for a function body: assigned anywhere
# in it means local to it.
function _nu_assigned!(out::Set{Symbol}, ex)
    ex isa Expr || return out
    h = ex.head
    (h === :quote || h === :inert) && return out
    if _nu_assigns(h)
        _nu_lhs_names!(out, ex.args[1])
        foreach(a -> _nu_assigned!(out, a), ex.args[2:end])
        return out
    elseif h === :local
        foreach(a -> (_nu_lhs_names!(out, a); _nu_assigned!(out, a)), ex.args)
        return out
    elseif h === :try
        length(ex.args) >= 2 && ex.args[2] isa Symbol && push!(out, ex.args[2])
    elseif h === :function || h === :->
        nm = _nu_fn_name(ex.args[1])
        (h === :function && nm isa Symbol) && push!(out, nm)
        h === :-> && ex.args[1] isa Symbol ? push!(out, ex.args[1]) : _nu_sig_names!(out, ex.args[1])
    elseif h in (:tuple, :parameters, :call, :macrocall, :ref, :curly, :vect)
        # `=` in one of these is a keyword or a named field, not an assignment
        for a in ex.args
            if a isa Expr && a.head in (:(=), :kw)
                foreach(b -> _nu_assigned!(out, b), a.args[2:end])
            else
                _nu_assigned!(out, a)
            end
        end
        return out
    end
    foreach(a -> _nu_assigned!(out, a), ex.args)
    return out
end

# The target of an assignment at top level binds; `a.b = …` and `a[i] = …`
# use `a`.
function _nu_bind_lhs!(u::_NameUses, lhs, kind::Symbol, locals::Set{Symbol}, top::Bool, bad)
    if lhs isa Symbol
        (top && !(lhs in locals)) && _nu_bind!(u, lhs, kind)
    elseif lhs isa Expr
        if lhs.head in (:tuple, :parameters)
            foreach(a -> _nu_bind_lhs!(u, a, kind, locals, top, bad), lhs.args)
        elseif lhs.head === :(::) && length(lhs.args) == 2
            _nu_bind_lhs!(u, lhs.args[1], kind, locals, top, bad)
            _uses!(u, lhs.args[2], locals, top, bad)
        elseif lhs.head === :curly && lhs.args[1] isa Symbol
            top && _nu_bind!(u, lhs.args[1], kind)               # `const T{P} = …`
        else
            _uses!(u, lhs, locals, top, bad)
        end
    end
    return
end

# The types and defaults a signature evaluates.
function _nu_param_uses!(u::_NameUses, a, locals, bad)
    a isa Expr || return
    if a.head === :(::)
        _uses!(u, a.args[end], locals, false, bad)
        length(a.args) == 2 && _nu_param_uses!(u, a.args[1], locals, bad)
    elseif a.head in (:kw, :(=))
        _nu_param_uses!(u, a.args[1], locals, bad)
        _uses!(u, a.args[2], locals, false, bad)
    elseif a.head in (:parameters, :tuple, :...)
        foreach(b -> _nu_param_uses!(u, b, locals, bad), a.args)
    end
    return
end

function _nu_sig_uses!(u::_NameUses, sig, locals::Set{Symbol}, bad)
    sig isa Expr || return
    if sig.head === :where
        for p in sig.args[2:end]
            p isa Expr && p.head in (:(<:), :(>:)) && length(p.args) == 2 && _uses!(u, p.args[2], locals, false, bad)
        end
        _nu_sig_uses!(u, sig.args[1], locals, bad)
    elseif sig.head === :(::) && length(sig.args) == 2 && _nu_is_method(sig.args[1])
        _nu_sig_uses!(u, sig.args[1], locals, bad)
        _uses!(u, sig.args[2], locals, false, bad)
    elseif sig.head === :call
        f = sig.args[1]
        if f isa Expr
            f.head === :. ? _uses!(u, f, locals, false, bad) :
            f.head === :curly ? foreach(a -> _uses!(u, a, locals, false, bad), f.args[2:end]) :
            f.head === :(::) ? _uses!(u, f.args[end], locals, false, bad) : nothing
        end
        foreach(a -> _nu_param_uses!(u, a, locals, bad), sig.args[2:end])
    elseif sig.head === :tuple || sig.head === :block
        foreach(a -> _nu_param_uses!(u, a, locals, bad), sig.args)
    else
        _nu_param_uses!(u, sig, locals, bad)
    end
    return
end

function _nu_function!(u::_NameUses, sig, body, locals::Set{Symbol}, top::Bool, bad)
    nm = _nu_fn_name(sig)
    (nm isa Symbol && top && !(nm in locals)) && _nu_bind!(u, nm, :function)
    inner = copy(locals)
    _nu_sig_names!(inner, sig)
    body === nothing || _nu_assigned!(inner, body)
    _nu_sig_uses!(u, sig, inner, bad)
    body === nothing || _uses!(u, body, inner, false, bad)
    return
end

# `T`, `T{P}`, `T <: S`: binds `T`, returns its parameters.
function _nu_type_head!(u::_NameUses, head, kind::Symbol, locals, top::Bool, bad)::Set{Symbol}
    tp = Set{Symbol}()
    if head isa Symbol
        top && _nu_bind!(u, head, kind)
    elseif head isa Expr && head.head === :(<:) && length(head.args) == 2
        tp = _nu_type_head!(u, head.args[1], kind, locals, top, bad)
        _uses!(u, head.args[2], union(locals, tp), top, bad)
    elseif head isa Expr && head.head === :curly
        head.args[1] isa Symbol && top && _nu_bind!(u, head.args[1], kind)
        for p in head.args[2:end]
            if p isa Symbol
                push!(tp, p)
            elseif p isa Expr && p.head in (:(<:), :(>:)) && length(p.args) == 2 && p.args[1] isa Symbol
                push!(tp, p.args[1])
                _uses!(u, p.args[2], locals, top, bad)
            end
        end
    end
    return tp
end

function _nu_struct!(u::_NameUses, head, body, locals, top::Bool, bad)
    inner = union(locals, _nu_type_head!(u, head, :struct, locals, top, bad))
    push!(inner, :new)
    body isa Expr || return
    for f in body.args
        if f isa Expr && f.head === :(::) && length(f.args) == 2 && f.args[1] isa Symbol
            _uses!(u, f.args[2], inner, false, bad)                   # field::Type
        elseif f isa Expr && f.head === :const
            for g in f.args
                g isa Expr && g.head === :(::) && length(g.args) == 2 && _uses!(u, g.args[2], inner, false, bad)
            end
        elseif f isa Expr && (f.head === :function || (f.head === :(=) && _nu_is_method(f.args[1])))
            _nu_function!(u, f.args[1], f.args[2], inner, false, bad)  # an inner constructor binds nothing here
        elseif f isa Expr
            _uses!(u, f, inner, false, bad)
        end                                                            # a bare field name, a line number
    end
    return
end

function _nu_import!(u::_NameUses, a)
    a isa Expr || return
    leaf(b) = b isa Expr && (b.head === :as ? _nu_bind!(u, b.args[2], :import) :
                             b.head === :. ? _nu_bind!(u, b.args[end], :import) : nothing)
    if a.head === :(:)
        foreach(leaf, a.args[2:end])                 # `import M: a, b`
    else
        leaf(a)                                      # `using M`, `import M`, `import M.x`
    end
    return
end

function _nu_macrocall!(u::_NameUses, ex::Expr, locals, top::Bool, bad)
    mac = ex.args[1]
    macname = mac isa Symbol ? mac :
              mac isa GlobalRef ? mac.name :
              (mac isa Expr && mac.head === :. && mac.args[end] isa QuoteNode) ? mac.args[end].value : nothing
    mac isa Expr && mac.head === :. && _uses!(u, mac.args[1], locals, top, bad)   # `GC.@preserve`
    rest = Any[a for a in ex.args[2:end] if !(a isa LineNumberNode)]
    if macname === Symbol("@enum") && !isempty(rest)
        h = rest[1]
        if h isa Expr && h.head === :(::)
            top && _nu_bind!(u, h.args[1], :enum)
            _uses!(u, h.args[2], locals, top, bad)
        else
            top && _nu_bind!(u, h, :enum)
        end
        for m in rest[2:end], s in (m isa Expr && m.head === :block ? m.args : (m,))
            if s isa Expr && s.head === :(=)
                top && _nu_bind!(u, s.args[1], :enumerator)
                _uses!(u, s.args[2], locals, top, bad)
            else
                top && _nu_bind!(u, s, :enumerator)
            end
        end
        return
    end
    is_doc = macname === Symbol("@doc")
    for a in rest
        if a isa Expr && a.head === :(=) && a.args[1] isa Symbol && !is_doc
            _uses!(u, a.args[2], locals, top, bad)     # `@warn "…" key=value`
        else
            _uses!(u, a, locals, top, bad)
        end
    end
    return
end

# An iteration specification of a `for` or a generator: `x = iter` names a
# variable of that scope, and only `iter` is evaluated.
function _nu_iteration!(u::_NameUses, a, locals::Set{Symbol}, bad)
    if a isa Expr && a.head === :(=)
        _uses!(u, a.args[2], locals, false, bad)
    elseif a isa Expr && a.head === :block
        foreach(b -> _nu_iteration!(u, b, locals, bad), a.args)       # `for a in x, b in y`
    elseif a isa Expr && a.head === :filter
        _uses!(u, a.args[1], locals, false, bad)                      # `… for x in xs if cond`
        foreach(b -> _nu_iteration!(u, b, locals, bad), a.args[2:end])
    else
        _uses!(u, a, locals, false, bad)
    end
    return
end

function _uses!(u::_NameUses, ex, locals::Set{Symbol}, top::Bool, bad)
    if ex isa Symbol
        _nu_ref!(u, ex, locals)
        return
    end
    ex isa Expr || return
    h = ex.head
    args = ex.args
    if h in (:quote, :inert, :export, :public, :meta, :inbounds, :line, :boundscheck, :loopinfo)
        return
    elseif h === :error || h === :incomplete
        bad[] = true
    elseif h === :module
        _nu_bind!(u, args[2], :module)
        _uses!(u, args[3], _NO_LOCALS, true, bad)
    elseif h === :toplevel || h === :block || h === :if || h === :elseif
        foreach(a -> _uses!(u, a, locals, top, bad), args)
    elseif h === :using || h === :import
        foreach(a -> _nu_import!(u, a), args)
    elseif h === :const || h === :global
        kind = h === :const ? :const : :global
        for a in args
            if a isa Symbol
                top && _nu_bind!(u, a, kind)
            elseif a isa Expr && a.head === :(=) && !_nu_is_method(a.args[1])
                _nu_bind_lhs!(u, a.args[1], kind, locals, top, bad)
                _uses!(u, a.args[2], locals, top, bad)
            else
                _uses!(u, a, locals, top, bad)
            end
        end
    elseif h === :function || h === :macro
        _nu_function!(u, args[1], length(args) >= 2 ? args[2] : nothing, locals, top, bad)
    elseif h === :->
        inner = copy(locals)
        args[1] isa Symbol ? push!(inner, args[1]) : _nu_sig_names!(inner, args[1])
        _nu_assigned!(inner, args[2])
        args[1] isa Symbol || _nu_sig_uses!(u, args[1], inner, bad)
        _uses!(u, args[2], inner, false, bad)
    elseif h === :(=)
        if _nu_is_method(args[1])
            _nu_function!(u, args[1], args[2], locals, top, bad)
        else
            _nu_bind_lhs!(u, args[1], :global, locals, top, bad)
            _uses!(u, args[2], locals, top, bad)
        end
    elseif _nu_assigns(h)
        args[1] isa Symbol ? _nu_ref!(u, args[1], locals) : _uses!(u, args[1], locals, top, bad)
        foreach(a -> _uses!(u, a, locals, top, bad), args[2:end])
    elseif h === :struct
        _nu_struct!(u, args[2], args[3], locals, top, bad)
    elseif h === :abstract || h === :primitive
        tp = _nu_type_head!(u, args[1], :type, locals, top, bad)
        foreach(a -> _uses!(u, a, union(locals, tp), top, bad), args[2:end])
    elseif h === :macrocall
        _nu_macrocall!(u, ex, locals, top, bad)
    elseif h === :.
        _uses!(u, args[1], locals, top, bad)
        # `f.(x)` is a call; `a.b` names a field
        length(args) >= 2 && args[2] isa Expr && args[2].head === :tuple && _uses!(u, args[2], locals, top, bad)
    elseif h === :kw
        length(args) >= 2 && _uses!(u, args[2], locals, top, bad)
    elseif h in (:call, :tuple, :parameters, :ref, :curly, :vect, :braces)
        for (i, a) in enumerate(args)
            if a isa Expr && a.head in (:(=), :kw) && !(h === :call && i == 1)
                foreach(b -> _uses!(u, b, locals, top, bad), a.args[2:end])    # a keyword, a named field
            else
                _uses!(u, a, locals, top, bad)
            end
        end
    elseif h === :let
        inner = copy(locals)
        binds = args[1]
        for b in (binds isa Expr && binds.head === :block ? binds.args : (binds,))
            if b isa Expr && b.head === :(=)
                _uses!(u, b.args[2], inner, false, bad)
                _nu_lhs_names!(inner, b.args[1])
            elseif b isa Symbol
                push!(inner, b)
            end
        end
        _nu_assigned!(inner, args[2])
        _uses!(u, args[2], inner, false, bad)
    elseif h in (:for, :while, :try, :generator, :filter, :comprehension, :typed_comprehension, :do, :where)
        # A scope of its own. At top level what it assigns is its own too; in
        # a function those names are the function's already.
        inner = copy(locals)
        _nu_assigned!(inner, ex)
        h === :where && _nu_where_names!(inner, args[2:end])
        if h === :for
            _nu_iteration!(u, args[1], inner, bad)
            _uses!(u, args[2], inner, false, bad)
        elseif h === :generator || h === :filter
            _uses!(u, args[1], inner, false, bad)
            foreach(a -> _nu_iteration!(u, a, inner, bad), args[2:end])
        else
            for a in args
                (h === :try && a isa Symbol) && continue          # the `catch` variable
                _uses!(u, a, inner, false, bad)
            end
        end
    elseif h === :local
        foreach(a -> a isa Symbol || _uses!(u, a, locals, top, bad), args)
    else
        foreach(a -> _uses!(u, a, locals, top, bad), args)
    end
    return
end

# ── The verdict on a generation ─────────────────────────────────────────────

# A probe and the generation its verdict asks for; one more when a name turns
# up that the probe never met. Decisions only accumulate, so more than this is
# a bug in the screen.
const _NAME_SCREEN_GENERATIONS = 4

# Whether the text binds `name` as something the library's own bindings of it
# do not explain: an import, the module, or one more type, constant or global
# than the record and the enumerator the library gave that name. Methods
# beside a type are its constructors, and methods beside methods an overload
# set, as `_assert_no_duplicate_toplevel_names` reads them; a method of a
# signature the type's own constructors define is `_constructor_clashes`'.
function _own_binding_beside(kinds::Vector{Symbol}, name::String, sc::_NameScreen)::Bool
    (:import in kinds || :module in kinds) && return true
    explained = (name in sc.kept[:type]) + (name in sc.kept[:enumerator])
    return count(k -> k !== :function, kinds) > explained
end

"""
    _constructor_clashes(sc, uses, text) -> Set{String}

The functions `sc` kept that stand beside a type of their name and define a
method the text defines twice: a constructor the wrapper emits for the type
has that signature. C gives a struct tag a namespace of its own, so
`struct conf` and the global `conf` (or `struct pool *pool(void)`) are two
names in C and one in Julia, and the global's accessor `conf()` and the
struct's zero-initializer `conf()` are one method: included as a file, the
later replaced the earlier, and a precompiled package refused the overwrite
(curl's `Curl_ssl`, 2026-11-18; `_assert_no_duplicate_methods`). The type's
constructor keeps the name, since every record promises `T()`; the function
moves like any screened name.

Read off the text (`_method_definitions`, the guard's own reading), and only
for a name the text binds as a type, so a library without the shape pays
nothing.
"""
function _constructor_clashes(sc::_NameScreen, uses::_NameUses, text::AbstractString)::Set{String}
    out = Set{String}()
    beside = Set{String}(n for n in sc.kept[:function]
                         if any(k -> k === :struct || k === :type, get(uses.bound, n, Symbol[])))
    isempty(beside) && return out
    seen = Set{String}()
    for (sig, _) in _method_definitions(text)
        if sig in seen
            name = sig[1:prevind(sig, something(findfirst('(', sig), lastindex(sig) + 1))]
            name in beside && push!(out, name)
        else
            push!(seen, sig)
        end
    end
    return out
end

"""
    _screen_verdict(sc, text) -> Union{Nothing,_NameScreen}

Judge the text a generation under `sc` produced. `nothing` when it stands:
every name `sc` moved had to move, and nothing it kept collides. Else the
screen of the generation to run next, which moves exactly what had to.

A candidate the probe moved had to move when the text still references or
binds the original name: with the library's own binding out of the way, that
use is the wrapper's. A name kept as the library spells it has to move when
the text binds it as something else too (`_own_binding_beside`): the module's
own constant, an import, the module; and a function kept beside a type of its
name has to move when one of its methods is a constructor the type defines
already (`_constructor_clashes`). Decisions only accumulate, so the
generations end.
"""
function _screen_verdict(sc::_NameScreen, text::AbstractString)::Union{Nothing,_NameScreen}
    uses = _name_uses(text)
    uses === nothing && return nothing            # unparseable: `_assert_wrapper_parses` owns it
    want = copy(sc.move)
    judged = copy(sc.judged)
    for key in sc.met
        key in judged && continue                 # an earlier probe ruled, and `move` holds the answer
        push!(judged, key)
        name = key[2]
        name in uses.refs ? (want[key] = :own_use) :
        haskey(uses.bound, name) ? (want[key] = :own_binding) : nothing
    end
    for kind in (:function, :type, :enumerator), name in sc.kept[kind]
        kinds = get(uses.bound, name, nothing)
        kinds === nothing && continue
        _own_binding_beside(kinds, name, sc) && (want[(kind, name)] = :own_binding)
    end
    for name in _constructor_clashes(sc, uses, text)
        haskey(want, (:function, name)) || (want[(:function, name)] = :type_constructor)
    end
    moved = Set{Tuple{Symbol,String}}((m.kind, m.from) for m in sc.moves if m.why !== :syntax &&
                                      m.why !== :every_module && m.why !== :base_type)
    # Nothing moved that did not have to, and nothing has to move that this
    # generation was not already told to (a name it was told to move and did
    # not meet is not a reason to generate again).
    if all(k -> haskey(want, k), moved) && all(k -> k in moved || haskey(sc.move, k), keys(want))
        # The text stands. The reasons a probe could not give are in `want`.
        for (i, m) in enumerate(sc.moves)
            m.why === :probe && (sc.moves[i] = _NameMove(m.kind, m.from, m.to, want[(m.kind, m.from)], m.source))
        end
        return nothing
    end
    return _NameScreen(probe = false, move = want, judged = judged)
end

"""
    _screened_generations(generate, metadata, sanitize; what) -> (result, screen, wrap_metadata)

Run `generate(wrap_metadata)` until the screen lets its text stand. `generate`
returns `(text, rest)`; it is called under a `_NameScreen` with the metadata
that screen's records are named in (`_screen_records`), first as the probe and
then, when the verdict asks, with exactly the names that have to move. Returns
the result of the generation that stood, its screen (what moved, for the
report) and the metadata it read.

What a generation logs is held and released only for the one that stood, so a
wrap that generates twice warns once.
"""
function _screened_generations(generate, metadata::AbstractDict, sanitize;
                               what::AbstractString = "the wrapper")
    screen = _NameScreen()
    generations = 0
    while true
        generations += 1
        wrap_metadata = _screen_records(screen, metadata, sanitize)
        (result, held) = _holding_logs(() -> with_name_screen(() -> generate(wrap_metadata), screen))
        next = _screen_verdict(screen, String(first(result)))
        if next === nothing
            _release_logs(held)
            return (result, screen, wrap_metadata)
        end
        generations >= _NAME_SCREEN_GENERATIONS &&
            error("RepliBuild: the name screen did not settle on $what in $generations generations " *
                  "(to move: $(join(sort!(["$(k[1]) `$(k[2])`" for k in keys(next.move)]), ", "))). " *
                  "This is a generator bug, not a configuration problem.")
        back = sort!(unique!(String[m.from for m in screen.moves
                                    if m.why === :probe && !haskey(next.move, (m.kind, m.from))]))
        println("  names: generating again: ",
                isempty(back) ? "" : "$(length(back)) library name(s) the wrapper's own text does not use " *
                                     "keep their spelling ($(join(first(back, 8), ", "))" *
                                     (length(back) > 8 ? ", …); " : "); "),
                length(next.move), " collide with it and move")
        screen = next
    end
end

"""
    _assert_module_name_free(text, module_name, config_file)

Refuse a module whose own name is one its text uses or binds. Inside
`module String`, `String` is the module, so `::Union{String,Nothing}` on every
string return names it; a module called `Libdl` or `RepliBuild` takes the name
the wrapper imports. A project directory named after a Julia type got such a
module (CamelCase of `[project] name`) and one that failed to load or to call.

The name is not changed here: the registry, a scaffolded package and `use`
find the wrapper's file by it. The error says what to set, as graph mode's
does for a graph module. A library identifier of that name is not this case:
the screen has moved it by now.
"""
function _assert_module_name_free(text::AbstractString, module_name::AbstractString,
                                  config_file::AbstractString = "replibuild.toml")
    uses = _name_uses(text)
    uses === nothing && return nothing
    name = String(module_name)
    kinds = filter(!=(:module), get(uses.bound, name, Symbol[]))
    used = name in uses.refs
    (isempty(kinds) && !used) && return nothing
    what = !isempty(kinds) ? "binds `$name` itself (" * join(unique(_describe_binding.(kinds)), ", ") * ")" :
                             "uses `$name` for what Julia means by it"
    error("""
    The wrapper module would be named `$name`, and the generated wrapper $what.
    Inside `module $name` that name is the module, so the wrapper would not load,
    or would not call.

    Set `module_name` under `[wrap]` in $config_file to a name of your own
    (`module_name = "$(name)Wrapper"`). Without it the module is named after
    `[project] name`, in CamelCase.""")
end

# ── The report ──────────────────────────────────────────────────────────────

_screen_kind(kind::Symbol) = kind === :function ? "function" : kind === :type ? "type" : "enumerator"

"""
    _screen_report(sc) -> Vector{String}

One line per name the generation under `sc` moved, sorted, for the wrap's
output: what it was, the name it is bound under now, and why.
"""
function _screen_report(sc::_NameScreen)::Vector{String}
    seen = Set{Tuple{Symbol,String,String}}()
    rows = Tuple{String,String,String,String}[]
    for m in sort(sc.moves; by = m -> (String(m.kind), m.from, m.to))
        (m.kind, m.from, m.to) in seen && continue
        push!(seen, (m.kind, m.from, m.to))
        push!(rows, (_screen_kind(m.kind), m.from, m.to, _screen_why(m.why, m.from)))
    end
    return String["$kind `$from` is bound as `$to`: $why" for (kind, from, to, why) in rows]
end

# ── Log records of a generation that may not be the last ────────────────────

struct _HeldLogs <: CoreLogging.AbstractLogger
    parent::CoreLogging.AbstractLogger
    records::Vector{Any}
end
CoreLogging.min_enabled_level(l::_HeldLogs) = CoreLogging.min_enabled_level(l.parent)
CoreLogging.shouldlog(l::_HeldLogs, args...) = CoreLogging.shouldlog(l.parent, args...)
CoreLogging.catch_exceptions(l::_HeldLogs) = CoreLogging.catch_exceptions(l.parent)
CoreLogging.handle_message(l::_HeldLogs, args...; kwargs...) = (push!(l.records, (args, kwargs)); nothing)

# Run `f`, holding what it logs. A wrap that generates twice says each thing
# once: the caller releases the records of the generation that stood.
function _holding_logs(f)
    held = _HeldLogs(CoreLogging.current_logger(), Any[])
    try
        return (CoreLogging.with_logger(f, held), held)
    catch
        _release_logs(held)             # a generation that threw says what it said
        rethrow()
    end
end

_release_logs(held::_HeldLogs) =
    foreach(((args, kwargs),) -> CoreLogging.handle_message(held.parent, args...; kwargs...), held.records)
