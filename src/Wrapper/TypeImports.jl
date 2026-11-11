# Wrapper/TypeImports.jl — cross-library type identity in graph mode
# (docs/updates/CROSS_LIBRARY_TYPES.md, Option 1).
#
# Several libraries built from one project graph each get a wrapper, and
# library B's API uses types library A defines. Each wrapper used to emit its
# own copy of every type its DWARF held, so B had a second Julia type for each
# of A's, often an empty one, and passing A's object into B's function was a
# MethodError. Now B's wrapper IMPORTS the types an earlier library of the
# graph emitted and does not emit them: `const X = Base.parentmodule(@__MODULE__).A.X`,
# a reference into the graph module, whose submodules are included in
# topological order (`..A.X` is import syntax, not an expression).
#
# Who owns a type is decided from the graph plus what each wrapper EMITTED
# (`type_manifest.json`, written by every wrap from the final text), never from
# header paths or symbol names alone: the first library in topological order
# whose wrapper emitted the type as its own owns it, when the identity check
# passes (the scopes meet; equal layouts where both have one; no member of the
# type defined strongly in both libraries). A single-library wrap has no
# earlier manifests, so every type is its own and the wrapper is byte-identical
# to one wrapped before this existed.
#
# Three pieces, all shared by GeneratorC and GeneratorCpp ("two generators, one
# answer"):
#   * `_resolve_type_imports` / `_apply_type_imports!`: the pre-pass
#     `wrap_introspective` runs after its other metadata filters. It marks each
#     imported entry `"imported" => Dict("module", "name", …)` and replaces its
#     body with the owner's, so every layout reader sees the owner's definition.
#   * `_emits_type`: the one predicate every per-type emission loop asks.
#   * `_type_manifest` and `_type_imports_file`: what a wrap leaves beside the
#     library for the libraries after it and for the thunk side
#     (`JLCSIRGenerator.apply_type_imports!`).
# And one write-time guard, `_assert_no_imported_redefinition`: a loop that
# forgets the predicate is a refused wrap, not a load-time redefinition.

import ..Compiler: _layout_signature, _canonical_text

"The manifest of the types a wrapper emitted as its own, beside the library."
const TYPE_MANIFEST = "type_manifest.json"
"The owners' definitions a wrapper imported, beside the library, for the thunk side."
const TYPE_IMPORTS = "type_imports.json"

# ─────────────────────────────────────────────────────────────────────────────
# Predicates and spellings
# ─────────────────────────────────────────────────────────────────────────────

"""
    _emits_type(key, info) -> Bool

Whether this wrapper defines the type `key` itself. `false` for an entry the
pre-pass marked imported: the type is one `const` naming the owner's
(`_type_imports_chunk`) and nothing else in the wrapper may define it. Every loop that emits something per
type (a forward declaration, a struct, an enum, a `Managed` type, a factory
struct, setters, the layout table, an upcast) asks this.
"""
_emits_type(key, info) = !(info isa AbstractDict && haskey(info, "imported"))

"The pre-pass's record on an imported entry, or `nothing`."
_import_of(info) = info isa AbstractDict ? get(info, "imported", nothing) : nothing

"The sanitizer each generator spells a type name with."
_type_sanitizer(language::Symbol) = language === :c ? _sanitize_c_type_name : _sanitize_cpp_type_name

"""
    _type_julia_spelling(key, sanitize) -> String

The Julia name a generator binds a metadata key to: the sanitizer's answer
with the three replacements both struct emitters apply (`*` to `Ptr`, `&` to
`Ref`, anything else to `_`). An enum key loses its `__enum__` prefix first.
"""
function _type_julia_spelling(key::AbstractString, sanitize)::String
    k = replace(String(key), r"^__enum__" => "")
    s = String(sanitize(k))
    s = replace(s, "*" => "Ptr")
    s = replace(s, "&" => "Ref")
    return replace(s, r"[^a-zA-Z0-9_]" => "_")
end

"""
    _imported_type_names(dwarf_structs, sanitize) -> Vector{String}

The Julia names this wrapper imports rather than defines, sorted: each
imported entry under this wrapper's own spelling of its key.
"""
function _imported_type_names(dwarf_structs, sanitize)::Vector{String}
    names = String[_type_julia_spelling(k, sanitize) for (k, v) in dwarf_structs if !_emits_type(k, v)]
    return sort!(unique!(names))
end

"""
    _imported_value_type_names(dwarf_structs, sanitize) -> Vector{String}

The imported names a pointer or reference return may be typed with: every
imported type the owner emitted as something other than a `mutable struct`
(`_typed_return_names` has the same rule for the wrapper's own types).
"""
function _imported_value_type_names(dwarf_structs, sanitize)::Vector{String}
    names = String[]
    for (k, v) in dwarf_structs
        imp = _import_of(v)
        imp === nothing && continue
        get(imp, "kind", "") in ("mutable", "idiomatic", "placeholder") && continue
        push!(names, _type_julia_spelling(k, sanitize))
    end
    return sort!(unique!(names))
end

# Julia spellings that are never a record of the library (the generators' own
# list, in one place for the reach walk).
const _BUILTIN_JULIA_SPELLINGS = Set{String}([
    "Cvoid", "Cint", "Cuint", "Cintptr_t", "Cuintptr_t", "Clong", "Culong", "Cshort", "Cushort",
    "Cchar", "Cuchar", "Cfloat", "Cdouble", "Bool", "UInt8", "Int8", "UInt16", "Int16", "UInt32",
    "Int32", "UInt64", "Int64", "Csize_t", "Clonglong", "Culonglong", "Cptrdiff_t", "Cssize_t",
    "Cwchar_t", "Cstring", "Float32", "Float64", "ComplexF32", "ComplexF64", "Int128", "UInt128",
    "Any", "Nothing", "Cvoid", "Ptr", "Ref"])

"The record name a `julia_type` spelling reaches through `Ptr{}`, `Ref{}` and `NTuple{}`, or `nothing`."
function _reached_type_name(julia_type::AbstractString)
    t = String(julia_type)
    while true
        if startswith(t, "Ptr{") && endswith(t, "}")
            t = t[5:end-1]
        elseif startswith(t, "Ref{") && endswith(t, "}")
            t = t[5:end-1]
        elseif startswith(t, "NTuple{")
            inner = peel_container_arg(t)
            inner === nothing && return nothing
            t = inner
        else
            break
        end
    end
    t = String(strip(t))
    (isempty(t) || t in _BUILTIN_JULIA_SPELLINGS) && return nothing
    occursin(r"[^A-Za-z0-9_:<>, ]", t) && return nothing
    return t
end

"The bare record name a C spelling points at or names (`const Leaf *`, `Shape &`, `ns::T*`), or `nothing`."
function _reached_c_type_name(c_type::AbstractString)
    t = String(strip(c_type))
    t = replace(t, r"^(?:(?:const|volatile|struct|class|enum|union)\s+)+" => "")
    t = replace(t, r"(?:\s*(?:\*|&|const|volatile))+$" => "")
    t = String(strip(t))
    (isempty(t) || t in ("void", "unknown")) && return nothing
    occursin(r"[*&(\[]", t) && return nothing
    occursin(r"^[A-Za-z_][A-Za-z0-9_:<>, ]*$", t) || return nothing
    return _bare_class_key(t)
end

"""
    _referenced_record_names(metadata) -> Vector{String}

The record names the members and signatures of `metadata` reach without a
definition in `struct_definitions`, sorted: the names GeneratorC and
GeneratorCpp turn into forward declarations, read off each `julia_type` and,
for a pointer whose Julia type degraded to `Ptr{Cvoid}` because the record was
unknown to this library's parser, off the C spelling. These are candidates for
an import too (a class another library of the graph homes).
"""
function _referenced_record_names(metadata)::Vector{String}
    defs = get(metadata, "struct_definitions", Dict{String,Any}())
    found = Set{String}()
    note(n) = begin
        n === nothing && return
        (haskey(defs, n) || haskey(defs, "__enum__" * n)) && return
        n in _INTERNAL_TYPE_BLOCKLIST && return
        isempty(n) && return
        push!(found, String(n))
    end
    both(x) = begin
        note(_reached_type_name(String(get(x, "julia_type", "Any"))))
        ct = get(x, "c_type", nothing)
        ct isa AbstractString && occursin(r"[*&]", ct) && note(_reached_c_type_name(ct))
    end
    for (k, info) in defs
        info isa AbstractDict || continue
        for m in get(info, "members", Any[])
            m isa AbstractDict && both(m)
        end
    end
    for f in get(metadata, "functions", Any[])
        for p in get(f, "parameters", Any[])
            p isa AbstractDict && both(p)
        end
        ret = get(f, "return_type", nothing)
        ret isa AbstractDict && both(ret)
    end
    return sort!(collect(found))
end

# ─────────────────────────────────────────────────────────────────────────────
# Sorted JSON
# ─────────────────────────────────────────────────────────────────────────────

"""
    _print_json_sorted(io, x)

`x` as JSON with every object's keys sorted, so a manifest is a function of its
contents on every Julia release and under either JSON major.
"""
function _print_json_sorted(io::IO, x)
    if x isa AbstractDict
        print(io, "{")
        first = true
        for k in sort!(collect(keys(x)); by = string)
            first || print(io, ",")
            first = false
            JSON.print(io, string(k))
            print(io, ":")
            _print_json_sorted(io, x[k])
        end
        print(io, "}")
    elseif x isa AbstractSet
        _print_json_sorted(io, sort!(collect(x); by = string))
    elseif x isa AbstractVector || x isa Tuple
        print(io, "[")
        first = true
        for v in x
            first || print(io, ",")
            first = false
            _print_json_sorted(io, v)
        end
        print(io, "]")
    elseif x isa Symbol
        JSON.print(io, string(x))
    else
        JSON.print(io, x)
    end
    return nothing
end

_sha256_hex(s::AbstractString) = bytes2hex(SHA.sha256(Vector{UInt8}(String(s))))

# ─────────────────────────────────────────────────────────────────────────────
# The manifest a wrap leaves behind
# ─────────────────────────────────────────────────────────────────────────────

"""
    _emitted_type_kinds(text) -> Dict{String,String}

What the wrapper's text defines per type name: `"struct"`, `"blob"` (a struct
whose one field is `_data`), `"mutable"`, `"idiomatic"` (a factory struct, one
`handle::Ptr{Cvoid}` field), `"managed"` (a `Managed<X>`), `"enum"` (a
`primitive type`, an `@enum`, or a `const` alias of one) or `"placeholder"`
(`struct X end`). Read off the final text, like `DISPATCH_TIER`: a manifest
says what was emitted, never what was intended.
"""
function _emitted_type_kinds(text::AbstractString)::Dict{String,String}
    kinds = Dict{String,String}()
    for m in eachmatch(r"^struct\s+([A-Za-z_][A-Za-z0-9_]*)\s+end\s*$"m, text)
        kinds[m.captures[1]] = "placeholder"
    end
    for m in eachmatch(r"^struct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\n(\s*)(\S+)"m, text)
        name = m.captures[1]
        haskey(kinds, name) && continue
        kinds[name] = startswith(m.captures[3], "_data::NTuple{") ? "blob" : "struct"
    end
    for m in eachmatch(r"^mutable\s+struct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\n\s*(\S+)"m, text)
        name = m.captures[1]
        kinds[name] = startswith(name, "Managed") ? "managed" :
                      m.captures[2] == "handle::Ptr{Cvoid}" ? "idiomatic" : "mutable"
    end
    for re in (r"^primitive\s+type\s+([A-Za-z_][A-Za-z0-9_]*)"m, r"^@enum\s+([A-Za-z_][A-Za-z0-9_]*)"m)
        for m in eachmatch(re, text)
            kinds[m.captures[1]] = "enum"
        end
    end
    for m in eachmatch(r"^const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*$"m, text)
        get(kinds, m.captures[2], "") == "enum" && (kinds[m.captures[1]] = "enum")
    end
    return kinds
end

"The mangled symbols of `metadata` defined strongly (`nm` type `T`) whose demangled class is `key`."
function _strong_members(metadata, key::AbstractString)::Vector{String}
    strong = Set{String}(String(s["mangled"]) for s in get(metadata, "symbols", Any[])
                         if s isa AbstractDict && get(s, "type", "") == "T" && haskey(s, "mangled"))
    out = String[]
    for f in get(metadata, "functions", Any[])
        f isa AbstractDict || continue
        cls = String(get(f, "class", ""))
        isempty(cls) && continue
        (cls == key || _bare_class_key(cls) == key) || continue
        m = String(get(f, "mangled", ""))
        m in strong && push!(out, m)
    end
    return sort!(unique!(out))
end

"A class spelling's bare key: the last `::` component, template arguments kept."
function _bare_class_key(cls::AbstractString)::String
    depth = 0
    cut = 0
    s = String(cls)
    i = firstindex(s)
    while i <= lastindex(s)
        c = s[i]
        if c == '<'
            depth += 1
        elseif c == '>'
            depth -= 1
        elseif c == ':' && depth == 0 && i < lastindex(s) && s[nextind(s, i)] == ':'
            cut = nextind(s, i)
            i = nextind(s, i)
        end
        i = nextind(s, i)
    end
    return cut == 0 ? s : s[nextind(s, cut):end]
end

"""
    _type_manifest(metadata, text, facts; module_name, language, library) -> Dict

What this wrapper emitted as its own, for the libraries wrapped after it. Per
type key of `struct_definitions` this wrapper defines (imported entries are
left out): the Julia name as this generator spelled it, the kind
(`_emitted_type_kinds`), size, `scopes`, `decl_files`, the layout signature
the identity check compares, the layout facts a dependent's layout proof needs
to embed it (`facts["layouts"]`: the generator's `jl_layouts` or
`resolved_layouts` entry, size and alignment as a field), whether the type
carries a register-class risk (`facts["class_risk"]`: a byte blob standing in
for a float-bearing record), the `Managed` type if one was emitted, the
strongly defined members (the ownership guard), and a digest of the metadata
entry. Forward declarations the text emitted (`struct X end`) are listed as
`placeholder`s under their name, so a later owner can add their conversions.
`bound_symbols` lists every mangled symbol the wrapper binds, so a later
wrapper drops the functions it would bind again.
"""
function _type_manifest(metadata, text::AbstractString, facts::AbstractDict;
                        module_name::AbstractString, language::Symbol, library::AbstractString)
    sanitize = _type_sanitizer(language)
    defs = get(metadata, "struct_definitions", Dict{String,Any}())
    kinds = _emitted_type_kinds(text)
    layouts = get(facts, "layouts", Dict{String,Any}())
    class_risk = get(facts, "class_risk", Set{String}())
    types = Dict{String,Any}()
    covered = Set{String}()
    for key in sort!(collect(keys(defs)); by = string)
        info = defs[key]
        info isa AbstractDict || continue
        _emits_type(key, info) || continue
        jn = _type_julia_spelling(key, sanitize)
        kind = get(kinds, jn, nothing)
        kind === nothing && continue           # not emitted: STL-internal, blocklisted, a dedup loser
        kind == "placeholder" && continue      # a definition that fell to a forward declaration is not owned
        push!(covered, jn)
        layout = get(layouts, jn, nothing)
        types[String(key)] = Dict{String,Any}(
            "julia_name" => jn,
            "kind" => kind,
            "byte_size" => get(info, "byte_size", nothing),
            "scopes" => get(info, "scopes", nothing),
            "decl_files" => get(info, "decl_files", nothing),
            "layout_signature" => _layout_signature(info),
            "layout" => layout === nothing ? nothing : Any[layout[1], layout[2]],
            "class_risk" => jn in class_risk,
            "managed" => get(kinds, "Managed" * jn, "") == "managed" ? "Managed" * jn : nothing,
            "strong_members" => _strong_members(metadata, String(key)),
            "entry_sha" => _sha256_hex(_canonical_text(info)))
    end
    for (name, kind) in kinds
        kind == "placeholder" || continue
        name in covered && continue
        haskey(types, name) && continue
        types[name] = Dict{String,Any}("julia_name" => name, "kind" => "placeholder")
    end
    bound = sort!(unique!(String[String(get(f, "mangled", "")) for f in get(metadata, "functions", Any[])
                                  if f isa AbstractDict && !isempty(get(f, "mangled", ""))]))
    return Dict{String,Any}("version" => 1, "module" => String(module_name),
                            "language" => string(language), "library" => basename(String(library)),
                            "types" => types, "bound_symbols" => bound)
end

"Write `manifest` as `TYPE_MANIFEST` under `dir`, sorted; returns its path."
function _write_type_manifest(dir::AbstractString, manifest::AbstractDict)::String
    mkpath(dir)
    path = joinpath(dir, TYPE_MANIFEST)
    open(path, "w") do io
        _print_json_sorted(io, manifest)
    end
    return path
end

"The manifest under `dir`, or `nothing` when there is none."
function _read_type_manifest(dir::AbstractString)
    path = joinpath(dir, TYPE_MANIFEST)
    return isfile(path) ? _read_json(path) : nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# The ownership rule
# ─────────────────────────────────────────────────────────────────────────────

"""
    _earlier_library(mod, manifest, metadata, library) -> NamedTuple

One library wrapped before the one being wrapped: its module name (`mod`), the
manifest its wrap wrote (`_read_type_manifest`), its metadata as built
(`compilation_metadata.json`, never the pre-pass's copy) and the path of its
library file. `GraphMode.wrap_graph_project` builds the list in topological
order.
"""
_earlier_library(mod::AbstractString, manifest::AbstractDict, metadata::AbstractDict,
                 library::AbstractString) =
    (mod = String(mod), manifest = manifest, metadata = metadata, library = String(library),
     manifest_path = joinpath(dirname(String(library)), TYPE_MANIFEST))

"The `functions` rows of `metadata` that are constructors or destructors of class `key`."
function _raii_rows(metadata, key::AbstractString)::Vector{Any}
    rows = Any[]
    for f in get(metadata, "functions", Any[])
        f isa AbstractDict || continue
        get(f, "is_method", false) || continue
        cls = String(get(f, "class", ""))
        (cls == key || _bare_class_key(cls) == key) || continue
        demangled = String(get(f, "demangled", ""))
        (occursin("~", demangled) || String(get(f, "name", "")) == _bare_class_key(cls)) || continue
        push!(rows, f)
    end
    sort!(rows; by = f -> String(get(f, "mangled", "")))
    return rows
end

"""
    _resolve_type_imports(metadata, earlier; pe = Sys.iswindows()) -> (imports, reports)

Who owns each type this wrapper's metadata defines or reaches, against the
libraries wrapped before it (`_earlier_library`, topological order):

1. The candidates are every key of `struct_definitions` and every record name
   reached without a definition (`_referenced_record_names`), in sorted order.
2. The owner is the first earlier library whose manifest emitted the key as its
   own (never a placeholder). Two earlier libraries that both emitted it are
   siblings: the first is taken and the rest reported.
3. With a definition here, the owner's must be the same type: the `scopes`
   meet (unknown on either side: own, reported); equal `layout_signature`s
   where this definition has one (unequal: an error naming both libraries,
   since two views of one type is an ODR violation between them); and no
   member of the type defined strongly in both (an error: two libraries claim
   it). The guard does not run on PE, where the export directory does not tell
   a COMDAT copy from a definition. A name reached without a definition has
   nothing to check.
4. No owner: the type is this wrapper's own when it has a definition, else it
   stays a forward declaration.

`imports` maps each imported key to its owner's module and Julia name, the
owner's kind and layout facts, the owner's metadata entry and `record_abi`
facts, its constructor and destructor rows (the thunk side's RAII), and the
owner's manifest digest. `reports` are the lines the wrap prints. Every step
iterates sorted keys or the given order.
"""
function _resolve_type_imports(metadata, earlier::AbstractVector; pe::Bool = Sys.iswindows())
    imports = Dict{String,Any}()
    reports = String[]
    isempty(earlier) && return (imports, reports)
    defs = get(metadata, "struct_definitions", Dict{String,Any}())
    candidates = sort!(unique!(vcat(String[String(k) for k in keys(defs)], _referenced_record_names(metadata))))
    for key in candidates
        owners = Any[]
        for e in earlier
            types = get(e.manifest, "types", Dict{String,Any}())
            t = get(types, key, nothing)
            t isa AbstractDict || continue
            get(t, "kind", "") == "placeholder" && continue
            push!(owners, (e, t))
        end
        isempty(owners) && continue
        (owner, otype) = owners[1]
        if length(owners) > 1
            push!(reports, "$key: emitted by $(join([o[1].mod for o in owners], " and ")); " *
                           "$(owner.mod), first in load order, is taken as its owner")
        end
        here = get(defs, key, nothing)
        if here isa AbstractDict
            hs = get(here, "scopes", nothing)
            os = get(otype, "scopes", nothing)
            if !(hs isa AbstractVector) || !(os isa AbstractVector)
                push!(reports, "$key: scope unknown on $(hs isa AbstractVector ? owner.mod : "this library")" *
                               " (metadata from before `scopes`; rebuild), so it stays this library's own")
                continue
            end
            if isempty(intersect(Set(String.(hs)), Set(String.(os))))
                push!(reports, "$key: a different type from $(owner.mod)'s " *
                               "(scopes $(join(hs, ", ")) here, $(join(os, ", ")) there); stays this library's own")
                continue
            end
            hsig = _layout_signature(here)
            osig = String(something(get(otype, "layout_signature", nothing), ""))
            if !isempty(hsig) && !isempty(osig) && hsig != osig
                error("graph types: `$key` has one scope ($(join(hs, ", "))) and two layouts: " *
                      "$(owner.mod) built it as $osig, this library as $hsig. The two libraries " *
                      "were compiled with different views of one type (defines, packing or flags); " *
                      "fix the build rather than wrapping both.")
            end
            if !pe
                mine = Set(_strong_members(metadata, key))
                theirs = Set(String.(get(otype, "strong_members", String[])))
                both = sort!(collect(intersect(mine, theirs)))
                isempty(both) || error("graph types: `$key` has members defined strongly in both " *
                                       "$(owner.mod) and this library: $(join(first(both, 4), ", ")). " *
                                       "Two libraries claim to define it; neither can own it.")
            end
        end
        oentry = get(get(owner.metadata, "struct_definitions", Dict{String,Any}()), key, nothing)
        oentry isa AbstractDict || continue        # the owner's metadata moved from under its manifest
        imports[key] = Dict{String,Any}(
            "module" => owner.mod,
            "julia_name" => String(otype["julia_name"]),
            "kind" => String(get(otype, "kind", "struct")),
            "layout" => get(otype, "layout", nothing),
            "class_risk" => get(otype, "class_risk", false) == true,
            "managed" => get(otype, "managed", nothing),
            "owner_entry" => deepcopy(oentry),
            "record_abi" => deepcopy(get(get(owner.metadata, "record_abi", Dict{String,Any}()), key, nothing)),
            "raii" => deepcopy(_raii_rows(owner.metadata, key)),
            "owner_library" => owner.library,
            "manifest_path" => owner.manifest_path,
            "manifest_sha" => isfile(owner.manifest_path) ?
                              bytes2hex(open(SHA.sha256, owner.manifest_path)) : "")
    end
    return (imports, reports)
end

"""
    _apply_type_imports!(metadata, imports) -> metadata

The pre-pass: each imported key's entry in `struct_definitions` becomes the
owner's definition marked `"imported"` (module, name, kind, layout facts), and
is added where this metadata had none, so every reader of layouts
(`flatten_struct_members`, the layout proofs, `is_ccall_safe`,
`by_value_crossing`) sees the owner's definition without a second lookup path.
The owner's `record_abi` facts are added where this metadata lacks them.
`metadata["type_imports"]` keeps the per-owner summary the import chunk
emits (`_type_imports_chunk`).
"""
function _apply_type_imports!(metadata, imports::AbstractDict)
    isempty(imports) && return metadata
    defs = get!(metadata, "struct_definitions", Dict{String,Any}())
    abi = get!(metadata, "record_abi", Dict{String,Any}())
    for key in sort!(collect(keys(imports)); by = string)
        imp = imports[key]
        entry = deepcopy(imp["owner_entry"])
        entry["imported"] = Dict{String,Any}("module" => imp["module"], "name" => imp["julia_name"],
                                             "kind" => imp["kind"], "layout" => imp["layout"],
                                             "class_risk" => imp["class_risk"])
        defs[key] = entry
        if imp["record_abi"] !== nothing && !haskey(abi, key)
            abi[key] = deepcopy(imp["record_abi"])
        end
    end
    owners = Dict{String,Any}()
    for imp in values(imports)
        owners[String(imp["module"])] = Dict{String,Any}("manifest_path" => imp["manifest_path"],
                                                          "manifest_sha" => imp["manifest_sha"],
                                                          "library" => imp["owner_library"])
    end
    metadata["type_imports"] = owners
    return metadata
end

"""
    _drop_functions_bound_earlier!(metadata, earlier) -> Int

Remove from `metadata["functions"]` every function an earlier library's wrapper
binds (its manifest's `bound_symbols`): an inline member or free function of
an owner's headers, emitted weak in this library too. One Julia function per
C++ function, and no two thunks of one name behind the thunk cache's
first-engine-wins. Returns how many were dropped.
"""
function _drop_functions_bound_earlier!(metadata, earlier::AbstractVector)::Int
    bound = Set{String}()
    for e in earlier
        union!(bound, String.(get(e.manifest, "bound_symbols", String[])))
    end
    (isempty(bound) || !haskey(metadata, "functions")) && return 0
    before = length(metadata["functions"])
    filter!(f -> !(String(get(f, "mangled", "")) in bound), metadata["functions"])
    return before - length(metadata["functions"])
end

"""
    _earlier_placeholders(metadata, earlier) -> Dict{String,Vector{Vector{String}}}

For each type this library defines as its own, the earlier libraries whose
wrapper holds only a forward declaration of it (`struct K end`, a `placeholder`
in their manifest): `key => [[module, julia_name], …]`, in load order. That
happens when a library reaches a class by declaration and the class's owner is
wrapped after it (no link dependency, and the sorted-id tie-break put the owner
later). The owner then emits the pointer conversions that let its owning types
pass into the earlier library's functions (`_placeholder_conversions_chunk`).
"""
function _earlier_placeholders(metadata, earlier::AbstractVector)
    defs = get(metadata, "struct_definitions", Dict{String,Any}())
    out = Dict{String,Vector{Vector{String}}}()
    for e in earlier
        for (name, t) in get(e.manifest, "types", Dict{String,Any}())
            t isa AbstractDict && get(t, "kind", "") == "placeholder" || continue
            info = get(defs, name, nothing)
            (info isa AbstractDict && _emits_type(name, info)) || continue
            push!(get!(() -> Vector{String}[], out, String(name)), String[e.mod, String(get(t, "julia_name", name))])
        end
    end
    return out
end

"""
    _placeholder_conversions_chunk(placeholders, handle_types) -> String

The owner's side of `_earlier_placeholders`: for each of its classes an earlier
library holds as `struct K end`, and for each owning type this wrapper emits
for the class (`handle_types`: the class's Julia name to its `Managed<K>` or
factory struct, both with a `handle` field), a `Base.unsafe_convert` to
`Ptr{<Earlier>.K}`, so an owned object passes into the earlier library's
functions, whose ccall types name its placeholder. Sorted by class, then
module. Raw pointers convert already; a by-value use is impossible through a
placeholder, and user code dispatching on `Ptr{<Earlier>.K}` does not match
`Ptr{<Owner>.K}`, which is documented, not fixed.
"""
function _placeholder_conversions_chunk(placeholders::AbstractDict, handle_types::AbstractDict)::String
    lines = String[]
    for key in sort!(collect(keys(placeholders)); by = string)
        hts = get(handle_types, key, String[])
        isempty(hts) && continue
        for (mod, jn) in sort(placeholders[key])
            for ht in sort(collect(hts))
                target = "Ptr{Base.parentmodule(@__MODULE__).$(mod).$(jn)}"
                push!(lines, "Base.unsafe_convert(::Type{$target}, o::$ht) = $target(o.handle)")
            end
        end
    end
    isempty(lines) && return ""
    return """
    # ── Earlier libraries' placeholders of this library's classes ────────────
    # A library wrapped before this one reached these classes by declaration
    # alone, so it holds `struct K end` and its functions take a pointer to
    # that. This library owns the classes: its owning types convert to those
    # pointers, so an object made here passes into that library's functions.
    $(join(lines, "\n"))

    """
end

"""
    _graph_subclassed(metadatas) -> Vector{String}

The base class names every record of `metadatas` derives from, sorted: what
the D0 factory-struct rule must also know about (`has_subclass` is otherwise
this library's view alone, and a dependent library deriving from a class
breaks the direct deleting-destructor call the same way).
"""
function _graph_subclassed(metadatas)::Vector{String}
    out = Set{String}()
    for md in metadatas
        for (_, info) in get(md, "struct_definitions", Dict{String,Any}())
            info isa AbstractDict || continue
            for b in get(info, "base_classes", Any[])
                b isa AbstractDict || continue
                t = replace(String(get(b, "type", "")), r"^(class|struct)\s+" => "")
                isempty(t) || push!(out, t)
            end
        end
    end
    return sort!(collect(out))
end

# ─────────────────────────────────────────────────────────────────────────────
# What the generators emit for imports
# ─────────────────────────────────────────────────────────────────────────────

"""
    _type_imports_chunk(dwarf_structs, sanitize, metadata, output_dir) -> String

The import section of a wrapper: one
`const <name> = Base.parentmodule(@__MODULE__).<Owner>.<name>` per imported type, sorted by owner then name, then `IMPORTED_FROM` (each owner's
manifest, relative to this wrapper, and its digest as imported) and
`_check_type_imports`, which `__init__` calls to warn when an owner was
re-wrapped since. Empty when nothing is imported, so a single-library wrapper
is unchanged.
"""
function _type_imports_chunk(dwarf_structs, sanitize, metadata, output_dir::AbstractString)::String
    lines = Tuple{String,String,String}[]
    for (k, v) in dwarf_structs
        imp = _import_of(v)
        imp === nothing && continue
        push!(lines, (String(imp["module"]), _type_julia_spelling(k, sanitize), String(imp["name"])))
    end
    isempty(lines) && return ""
    sort!(unique!(lines))
    io = IOBuffer()
    print(io, """
        # ── Types owned by other libraries of this graph ─────────────────────────
        # Each is defined once, by the wrapper of the library that homes it, and
        # imported here so one C++ type is one Julia type across the graph: a
        # pointer, a `Managed` object or a value from that module passes straight
        # into this one's functions. This file loads only through the graph module,
        # which includes the owners first.
        """)
    for (mod, name, owner_name) in lines
        # `..Owner.X` is import syntax, not an expression: the parent module is
        # the graph module, which included the owner before this file.
        println(io, "const ", name, " = Base.parentmodule(@__MODULE__).", mod, ".", owner_name)
    end
    owners = get(metadata, "type_imports", Dict{String,Any}())
    print(io, "\nconst IMPORTED_FROM = Dict{Symbol,Tuple{String,String}}(\n")
    for mod in sort!(collect(keys(owners)); by = string)
        o = owners[mod]
        rel = replace(relpath(String(o["manifest_path"]), String(output_dir)), '\\' => '/')
        println(io, "    :", mod, " => (", repr(rel), ", ", repr(String(o["manifest_sha"])), "),")
    end
    print(io, """
        )

        # An owner re-wrapped since this wrapper was generated may spell or lay out
        # a type differently. Only a warning: the graph module re-wraps every
        # library together, so this fires only when files were moved by hand.
        function _check_type_imports()
            for (mod, (rel, sha)) in IMPORTED_FROM
                path = joinpath(@__DIR__, rel)
                isfile(path) || continue
                actual = try
                    bytes2hex(open(RepliBuild.SHA.sha256, path))
                catch
                    ""
                end
                if !isempty(actual) && actual != sha
                    @warn "\$(nameof(@__MODULE__)): the types imported from \$mod were generated against another wrap of it; re-run wrap on the graph." *
                          "\\n  manifest: \$path"
                end
            end
            return nothing
        end

        """)
    return String(take!(io))
end

"""
    _type_imports_file(imports) -> Dict

The sidecar the thunk side reads (`JLCSIRGenerator.apply_type_imports!`): per
imported key, the owner's module and Julia name, its metadata entry with its
`conflict` marks, its `record_abi` facts, its constructor and destructor rows,
and its library's path.
"""
function _type_imports_file(imports::AbstractDict)
    out = Dict{String,Any}()
    for (key, imp) in imports
        out[String(key)] = Dict{String,Any}(
            "module" => imp["module"], "julia_name" => imp["julia_name"],
            "owner_entry" => imp["owner_entry"], "record_abi" => imp["record_abi"],
            "raii" => imp["raii"], "owner_library" => imp["owner_library"])
    end
    return Dict{String,Any}("version" => 1, "imports" => out)
end

"""
    _write_type_imports(dir, imports)

Write `TYPE_IMPORTS` under `dir` for a wrap that imported something, and remove
a stale one otherwise.
"""
function _write_type_imports(dir::AbstractString, imports::AbstractDict)
    path = joinpath(dir, TYPE_IMPORTS)
    if isempty(imports)
        rm(path; force = true)
        return nothing
    end
    mkpath(dir)
    open(path, "w") do io
        _print_json_sorted(io, _type_imports_file(imports))
    end
    return path
end

# ─────────────────────────────────────────────────────────────────────────────
# The write-time guard
# ─────────────────────────────────────────────────────────────────────────────

"""
    _assert_no_imported_redefinition(wrapper_content, module_name, imported)

Refuse a wrapper whose text defines a type it imports: a `struct`, `mutable
struct`, `@enum`, `primitive type` or `Managed<X>` for any name in `imported`.
Checked on the text, like the other write-time guards: an emission loop that
forgot `_emits_type` is a refused wrap here, not Julia's "invalid redefinition
of constant" when a consumer loads the graph.
"""
function _assert_no_imported_redefinition(wrapper_content::AbstractString, module_name::AbstractString,
                                          imported::AbstractVector)
    isempty(imported) && return nothing
    offenders = String[]
    for name in imported
        n = Regex("^\\s*(?:mutable\\s+)?struct\\s+$(name)\\b", "m")
        e = Regex("^\\s*(?:@enum|primitive\\s+type)\\s+$(name)\\b", "m")
        g = Regex("^\\s*mutable\\s+struct\\s+Managed$(name)\\b", "m")
        occursin(n, wrapper_content) && push!(offenders, "$name (a struct)")
        occursin(e, wrapper_content) && push!(offenders, "$name (an enum)")
        occursin(g, wrapper_content) && push!(offenders, "Managed$name")
    end
    isempty(offenders) && return nothing
    error("""
    Refusing to write wrapper '$module_name': $(length(offenders)) imported type(s) are \
    also defined by the module, which would redefine a constant when the graph module loads.

      $(join(offenders, "\n  "))

    An imported type is one `const` naming the owner's and nothing else; every loop that \
    emits something per type asks `_emits_type`. One of them did not.
    """)
end
