# =============================================================================
# HIGH-LEVEL WRAPPER API
# =============================================================================

"""
    wrap_library(config::RepliBuildConfig, library_path::String;
                 headers::Vector{String}=String[],
                 generate_tests::Bool=false,
                 generate_docs::Bool=true)

Generate Julia wrapper for compiled library.

Always uses introspective (DWARF metadata) wrapping when metadata is available,
otherwise falls back to basic symbol-only extraction with conservative types.

# Arguments
- `config`: RepliBuildConfig with wrapper settings
- `library_path`: Path to compiled library (.so, .dylib, .dll)
- `headers`: Optional header files (currently unused, reserved for future)
- `generate_tests`: Generate test file (default: false, TODO)
- `generate_docs`: Include comprehensive documentation (default: true)

# Returns
Path to generated Julia wrapper file
"""
function wrap_library(config::RepliBuildConfig, library_path::String;
                     headers::Vector{String}=String[],
                     generate_tests::Bool=false,
                     generate_docs::Bool=true)


    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    # A [wrap.varargs] table is a human claim about upstream's variadic API that
    # no rebuild can re-derive. Enforce its `proven_at` pin HERE rather than at
    # config load: wrapping is the only stage that consumes those signatures, so
    # this is where a stale table turns into wrong wrappers.
    enforce_varargs_provenance(config)

    # Check for metadata (DWARF + symbol info from compilation)
    metadata_file = joinpath(dirname(library_path), "compilation_metadata.json")
    has_metadata = isfile(metadata_file)

    if !has_metadata
        @warn "No compilation metadata found. Did you compile with -g flag?"
        @warn "Falling back to basic symbol-only wrapper (conservative types, limited safety)"
        return wrap_basic(config, library_path, generate_docs=generate_docs)
    end

    return wrap_introspective(config, library_path, headers, generate_docs=generate_docs)
end

"""
    _write_wrapper(output_file, wrapper_content, module_name) -> String

Guard, rotate, write. The single write path for every generated wrapper.

**Rotation.** A regenerated wrapper overwrites the only copy of what the
generator produced last time, and the `julia/` output directories are
gitignored — so before this existed there was no way to answer "what did this
change?" except moving the file aside by hand beforehand, which only works if
you already suspected something. Regenerating is the moment the previous
version becomes interesting, and also the moment it is destroyed.

So the prior generation is kept beside the new one as `<Module>.jl.prev` —
exactly one, never a growing pile. `.prev` does not end in `.jl`, so nothing
that scans for wrappers picks it up.

**Only rotated when the content actually differs**, ignoring the wrap-time
timestamp in the header. Re-wrapping with no source or generator change would
otherwise overwrite `.prev` with an effectively identical copy and destroy the
last real diff — the failure you notice only when you need it. And the
timestamp alone makes EVERY regeneration differ, so comparing raw bytes would
have rotated unconditionally and quietly made the feature useless: `.prev`
would always be the wrap from thirty seconds ago. `generated_at` in `METADATA`
is deliberately NOT normalized — it comes from the build, not the wrap, so a
change there is real news.

This is deliberately NOT version control. It answers one question — "what did
regenerating just change?" — which is the question a wrapper developer has
several times an hour, and which `BUILD_ID` complements from the other side:
that names the library a wrapper was generated FROM, this holds what the
generator said about it last time.
"""
function _write_wrapper(output_file::AbstractString, wrapper_content::AbstractString,
                        module_name::AbstractString)
    _assert_wrapper_loadable(wrapper_content, module_name)

    if isfile(output_file)
        previous = try
            read(output_file, String)
        catch
            nothing
        end
        if previous !== nothing && _wrapper_differs(previous, wrapper_content)
            prev_file = output_file * ".prev"
            try
                # cp, not mv: the current wrapper stays valid on disk until the
                # new one has been written over it.
                cp(output_file, prev_file; force = true)
                @info "wrap: previous generation kept at $(basename(prev_file)) " *
                      "(diff it against $(basename(output_file)) to see what regenerating changed)"
            catch e
                # Losing the backup must never lose the wrap.
                @warn "wrap: could not keep the previous wrapper generation" path=prev_file exception=e
            end
        end
    end

    write(output_file, wrapper_content)
    return output_file
end

# The header carries the wrap-time clock, which changes on every run and means
# nothing. Comparing raw bytes would rotate `.prev` unconditionally and leave it
# holding the wrap from moments ago instead of the last generation that differed.
_normalize_wrapper(s::AbstractString) =
    replace(s, r"^# Generated: .*$"m => "# Generated: <n>")

_wrapper_differs(a::AbstractString, b::AbstractString) =
    _normalize_wrapper(a) != _normalize_wrapper(b)

"""
    _assert_wrapper_loadable(wrapper_content, module_name)

Refuse to write a wrapper that would raise `UndefVarError` at include time.

This is the worst failure the pipeline can produce: not a degraded function or
a wrong value, but every function in the module gone at once, reported as an
error naming a type the user never mentioned. miniaudio hit it live on
2026-08-02 — `ma_fopen(FILE**, …)` put `Ptr{Ptr{_IO_FILE}}` in a ccall
signature while `_IO_FILE`, being on `_INTERNAL_TYPE_BLOCKLIST`, was never
declared; all 1178 functions were dead on one missing declaration.

Deliberately checks the generated TEXT rather than the generator's own record
of what it defined. The bug was precisely that those two disagreed, so a guard
built on the same bookkeeping would have agreed with the bug.
"""
function _assert_wrapper_loadable(wrapper_content::AbstractString, module_name::AbstractString)
    _assert_wrapper_parses(wrapper_content, module_name)
    _assert_base_calls_qualified(wrapper_content, module_name)
    _assert_exports_defined(wrapper_content, module_name)
    _assert_cstring_policy(wrapper_content, module_name)
    _assert_no_shadowed_ccall_types(wrapper_content, module_name)
    _assert_no_any_ccall_return(wrapper_content, module_name)

    undefined_types = _undefined_ccall_types(wrapper_content)
    isempty(undefined_types) && return nothing

    detail = join(("  $t  — used by $fn" for (t, fn) in undefined_types), "\n")
    error("""
    Refusing to write wrapper '$module_name': $(length(undefined_types)) type(s) \
    appear in foreign-call signatures but are never declared by the module. \
    Writing it would produce a file that raises UndefVarError at include time, \
    disabling every function in it.

    $detail

    A type reaches this state when a signature uses it but the struct emitters \
    skip it — typically a system type on _INTERNAL_TYPE_BLOCKLIST (e.g. _IO_FILE \
    behind a `FILE*` parameter). Fix by degrading the use to Ptr{Cvoid}, which is \
    ABI-identical, via _resolve_forward_ptr — not by widening the blocklist, \
    which suppresses declarations without suppressing uses and is what caused \
    this in the first place.
    """)
end

"""
    _assert_no_shadowed_ccall_types(wrapper_content, module_name)

Refuse a wrapper where a PARAMETER shadows a type named in its own ccall tuple.

Sibling of `_undefined_ccall_types`, and the same catastrophe from the other
direction: there the name resolves to nothing, here it resolves to the *wrong
thing* — a local. `ccall` evaluates its argument tuple eagerly at method
definition, so either way the module dies at include and takes every function
with it. That symmetry is why this belongs beside it rather than in a generator.

The C idiom is `struct bufq *bufq`, a parameter named after its own type, which
libcurl uses 26 times across 5 types. `_undefined_ccall_types` cannot see it:
`bufq` IS declared by the module, so nothing is undefined — it is merely
unreachable from inside that one function.

Checked on the emitted TEXT, like its sibling, because the bug is precisely the
generator's bookkeeping disagreeing with what it wrote.
"""
function _assert_no_shadowed_ccall_types(wrapper_content::AbstractString,
                                         module_name::AbstractString)
    offenders = Tuple{String,String}[]   # (function, shadowed type)

    # Split into function blocks FIRST, then match inside each one.
    #
    # This was a single regex spanning the whole file, which needed
    # `(?:(?!^function )[\s\S])*?` to stop at the next definition — a negative
    # lookahead evaluated at every character of every body. A function with no
    # `ccall` makes that scan run all the way to the next `function` and
    # backtrack, and a Tier-2 wrapper is mostly such functions. llamacpp
    # (3686 functions, 3.1 MB) exhausted PCRE outright:
    # `PCRE.exec error: JIT stack limit reached`, refusing to wrap at all.
    #
    # Splitting on the anchor is linear and bounds every subsequent match to one
    # body, so cost grows with the wrapper instead of with the wrapper squared.
    heads  = collect(eachmatch(r"^function ([A-Za-z_]\w*)\(([^)]*)\)"m, wrapper_content))
    isempty(heads) && return nothing
    # `ccall((:sym, LIB), R, (types,), args)` — the argument-type tuple.
    ccall_pat = r"ccall\(\([^)]*\),\s*[^,]+,\s*\(([^)]*)\)"

    for (i, h) in enumerate(heads)
        stop = i < length(heads) ? prevind(wrapper_content, heads[i+1].offset) :
                                   lastindex(wrapper_content)
        body = SubString(wrapper_content, h.offset, stop)
        cm = match(ccall_pat, body)
        cm === nothing && continue
        fn, params, types = h.captures[1], h.captures[2], cm.captures[1]
        pnames = Set{String}()
        for p in _split_toplevel_commas(params)
            nm = strip(first(split(p, "::"; limit = 2)))
            isempty(nm) || push!(pnames, String(nm))
        end
        isempty(pnames) && continue
        for tm in eachmatch(r"[A-Za-z_][A-Za-z0-9_]*", types)
            t = String(tm.match)
            t in pnames && push!(offenders, (String(fn), t))
        end
    end
    isempty(offenders) && return nothing

    uniq = unique(offenders)
    detail = join(("  $t  — shadowed by the parameter of $fn" for (fn, t) in first(uniq, 20)), "\n")
    more = length(uniq) > 20 ? "\n  … (+$(length(uniq) - 20) more)" : ""
    error("""
    Refusing to write wrapper '$module_name': $(length(uniq)) ccall argument type(s) \
    are shadowed by a parameter of the same name. `ccall` resolves its argument \
    tuple eagerly at method definition, so Julia rejects the method with "could not \
    evaluate ccall argument type (it might depend on a local variable)" and the \
    whole module dies at include.

    $detail$more

    This is the `struct foo *foo` idiom — a C parameter named after its own type. \
    Fix by renaming the PARAMETER at the emission site (the type name must survive; \
    it is a module binding the ccall tuple has to reach), not by renaming the type.
    """)
end

"""
    _assert_no_any_ccall_return(wrapper_content, module_name)

Refuse a wrapper that declares a foreign call's return type as `Any`.

Unlike the two guards above this one is not about resolution — `Any` resolves
fine. It is about MEANING: in a foreign call `Any` declares that the callee
returns a `jl_value_t*`, so Julia takes whatever integer came back and treats
it as a pointer to a Julia object. The crash lands inside method dispatch on a
LATER call, with a stack that names neither the wrapper nor the library.

`Any` reaches a return position when the type mapper could not name the type.
That is a fine signal internally — the C emission loop uses `julia_return_type
== "Any"` as its struct-return branch — but it must never survive into emitted
text. libcurl shipped 18: every `curl_easy_setopt` / `curl_multi_setopt` /
`curl_share_setopt` / `curl_easy_getinfo` overload, the whole configuration API,
each one a segfault on first call.

Checked on the emitted text, and deliberately NOT on `::Any` in a Julia
signature — `f(x::Any)` is ordinary and correct. Only the foreign-call return
position is wrong.
"""
function _assert_no_any_ccall_return(wrapper_content::AbstractString,
                                     module_name::AbstractString)
    offenders = Tuple{String,String}[]
    fnames = [(m.offset, String(m.captures[1]))
              for m in eachmatch(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_!]*)"m, wrapper_content)]
    enclosing(pos) = begin
        best = "<module scope>"
        for (o, n) in fnames; o < pos ? (best = n) : break; end
        best
    end
    # @ccall form: `@ccall LIB.var"sym"(…)::Any`
    for m in eachmatch(r"@ccall\s+[A-Za-z_][\w.]*\.var\"[^\"]*\"\([^)]*\)::Any\b", wrapper_content)
        push!(offenders, (enclosing(m.offset), "@ccall"))
    end
    # classic form: `ccall((:sym, LIB), Any, (…), …)`
    for m in eachmatch(r"ccall\(\([^)]*\)\s*,\s*Any\s*,", wrapper_content)
        push!(offenders, (enclosing(m.offset), "ccall"))
    end
    isempty(offenders) && return nothing

    uniq = unique(offenders)
    detail = join(("  $fn  ($form)" for (fn, form) in first(uniq, 20)), "\n")
    more = length(uniq) > 20 ? "\n  … (+$(length(uniq) - 20) more)" : ""
    error("""
    Refusing to write wrapper '$module_name': $(length(uniq)) foreign call(s) declare \
    their RETURN type as `Any`. That tells Julia the callee returns a jl_value_t*, so \
    the returned value is dereferenced as a Julia object — a segfault inside dispatch, \
    far from the call site.

    $detail$more

    `Any` in a return position means the type mapper could not name the type. Resolve \
    it to the real one (an enum's name usually sits in the metadata's `c_type`), or \
    degrade to `Cvoid` — discarding a value is recoverable, corrupting one is not.
    """)
end

# A foreign call whose RETURN type is Cstring, in any of the shapes the
# generators emit: plain ccall, the AOT-thunk ccall, the variadic `@ccall`
# semicolon form, JIT dispatch, and a Tier-1 sliced llvmcall.
#
# Every pattern is anchored on the RETURN position, and both anchors were
# earned. `[wrap.cstring_owned]`'s own free call
# (`ccall((:free, LIBRARY_PATH), Cvoid, (Cstring,), ptr)`) names Cstring in an
# argument tuple; and a variadic overload taking a string
# (`@ccall LIB.var"b2Log"(fmt::Cstring; va_1::Cstring)::Cvoid`) names it in an
# argument ANNOTATION — an unanchored `::Cstring` flagged all four of box2d's
# b2Log/b2Dump overloads, which return void.
const _RAW_CSTRING_CALL = (
    r"ccall\(\(:[^)]*\)\s*,\s*Cstring\b",          # ccall((:sym, LIB), Cstring, …)
    r"@ccall\s.*\)::Cstring\s*$",                   # @ccall LIB.var"sym"(…;…)::Cstring
    r"invoke\(\"[^\"]*\"\s*,\s*Cstring\b",          # JITManager.invoke("…", Cstring, …)
    r"llvmcall\(\([^)]*\)\s*,\s*Cstring\b",         # Base.llvmcall((ir, "sym"), Cstring, …)
)

"""
    _assert_cstring_policy(wrapper_content, module_name)

Refuse to write a wrapper in which a `char*` return escapes as a bare `Cstring`.

Every `char*`-returning function must be presented one of exactly two ways: the
policy wrapper (`::Union{String,Nothing}` — NULL is a value, the buffer is
copied, and a `[wrap.cstring_owned]` deallocator runs), or its raw `<name>_ptr`
sibling, which is *named* for the fact that it hands back an unmanaged pointer.
Anything else is the caller receiving a pointer they did not ask for, with no
NULL check and no free.

**This guard exists because the tier decided the presentation.** The C++
generator's MLIR-dispatch branch `continue`s past the ccall path, and the
`Cstring` policy lived only on that path — so 77 functions across five Hub
packages (imgui 31, tinyxml2 22, pugixml 14, llamacpp 9, hello_world 1) returned
a raw pointer with no `_ptr` sibling, and any `cstring_owned` declaration on one
of them was **silently ignored**, which is the leak version of the same bug.
Found 2026-08-12 from a user calling `hello_message()` and getting
`Cstring(0x7fb608c37000)` back.

Checks the emitted TEXT, like its siblings: the defect was a code path that
never consulted the policy, so a guard built on the generator's own bookkeeping
would have agreed with it. Negative-checkable — run it over any pre-fix C++
wrapper and it names every offender.
"""
function _assert_cstring_policy(wrapper_content::AbstractString, module_name::AbstractString)
    offenders = String[]
    current = ""
    policy_typed = false

    for line in eachsplit(wrapper_content, '\n')
        m = match(r"^function\s+([A-Za-z_][A-Za-z0-9_!]*)\s*\(", line)
        if m !== nothing
            current = String(m.captures[1])
            # The annotation can only be on the signature line the generators
            # emit; a `_ptr` sibling is exempt by name, which is the contract.
            policy_typed = occursin("::Union{String,Nothing}", line) || endswith(current, "_ptr")
            continue
        end
        isempty(current) && continue
        line == "end" && (current = ""; continue)
        policy_typed && continue
        if any(p -> occursin(p, line), _RAW_CSTRING_CALL)
            current in offenders || push!(offenders, current)
        end
    end

    isempty(offenders) && return nothing

    shown = join(("  " * f for f in first(offenders, 20)), "\n")
    more = length(offenders) > 20 ? "\n  … and $(length(offenders) - 20) more" : ""
    error("""
    Refusing to write wrapper '$module_name': $(length(offenders)) function(s) return \
    a raw `Cstring` without the char* return policy and without being a `_ptr` variant.

    $shown$more

    A char* return must be emitted through _cstring_wrapper_pair (Wrapper/Utils.jl), \
    which produces the `::Union{String,Nothing}` wrapper AND the raw `<name>_ptr` \
    sibling from one derivation. Reaching this state means an emission path built \
    the call itself and skipped that helper — the dispatch tier decides how a \
    function is CALLED, never how its result is presented. Note a bare Cstring also \
    silently discards any `[wrap.cstring_owned]` deallocator declared for it.
    """)
end

# Base functions the generated code calls on its own behalf, whose names a C
# library is free to take. Anything here MUST be emitted `Base.`-qualified.
const _MUST_QUALIFY_IN_WRAPPER = ("error",)

"""
    _assert_base_calls_qualified(wrapper_content, module_name)

Refuse to write a wrapper that calls one of `_MUST_QUALIFY_IN_WRAPPER`
unqualified.

A generated module is a namespace the LIBRARY populates, so any bare Base name
the generator relies on is one C symbol away from meaning something else. Found
live on llama.cpp: libstdc++'s `std::codecvt_base::result` has an enum member
named `error`, so `@enum result::Cuint begin … error = 2 … end` rebound `error`
for the whole module and every failure path in the wrapper — including the
long-standing `getproperty` "no field" branch — raised
`MethodError: objects of type result are not callable` instead of its message.

The check is on the emitted TEXT, not on a list of names the generator believes
it avoided, for the same reason `_assert_wrapper_loadable` is: a guard sharing
the bug's bookkeeping agrees with the bug. Refusing to emit the library's own
`error = 2` is not an option — that is real API surface; qualifying our own
calls is.
"""
function _assert_base_calls_qualified(wrapper_content::AbstractString, module_name::AbstractString)
    offenders = String[]
    for name in _MUST_QUALIFY_IN_WRAPPER
        # A call position not already qualified (`Base.error(`), not part of a
        # longer identifier (`_error(`, `error_code(`), and not a macro (`@error`).
        # Matched on a STRING-LITERAL first argument, which is what distinguishes
        # the generator's own diagnostics from the library's. A library is
        # entitled to own the name: cJSON has a `struct error`, so its wrapper
        # legitimately emits `struct error`, a zero-arg `function error()`, and
        # `return error(Ptr{UInt8}())` inside it — all correct, none of them a
        # call the generator made on its own behalf. Requiring the argument to
        # be a literal separates the two with no false positives, and every
        # emission site the generator has passes one.
        pat = Regex("(?<![\\w.@])" * name * "\\s*\\(\\s*\"")
        for (i, line) in enumerate(eachsplit(wrapper_content, '\n'))
            s = strip(line)
            startswith(s, "#") && continue     # comment
            occursin(pat, line) && push!(offenders, "  line $i: $s")
        end
    end
    isempty(offenders) && return nothing
    shown = join(first(offenders, 10), "\n")
    more = length(offenders) > 10 ? "\n  … and $(length(offenders) - 10) more" : ""
    error("""
    Refusing to write wrapper '$module_name': $(length(offenders)) unqualified call(s) \
    to $(join(("`" * n * "`" for n in _MUST_QUALIFY_IN_WRAPPER), ", ")) in generated code.

    $shown$more

    The module's namespace belongs to the library, so a bare Base name can be \
    rebound by any C symbol or enum member that happens to share it. Emit \
    `Base.<name>(...)` instead.
    """)
end

"""
    _assert_exports_defined(wrapper_content, module_name)

Refuse to write a wrapper whose `export` line names a binding the module never
makes.

Julia does not validate export targets at load, so this costs nothing until
something walks `names(Mod)` — and then it is an `UndefVarError` on a name the
module itself advertised. REPL tab-completion is the shortest path to it: type
`get_Val<TAB>` against the lua wrapper and Julia suggests `get_Value_gc`,
which does not exist. Doc generators and introspection passes hit it on their
first iteration.

Measured across the Hub the day this guard was written: **102 undefined
exports in 5 of 18 packages** — sqlite 64, llamacpp 22, lua 12, miniaudio 2,
zlib 2 — from two derivations that had drifted from the definitions:

  - union accessors screened OUT of the definitions (`defined_struct_names`)
    but still pushed onto the export list
  - enum members exported under the raw C spelling while `@enum` binds the
    sanitized one: `__RLIMIT_NICE`/`_RLIMIT_NICE` (leading underscores
    collapsed), `ma_dr_wav__metadata_…` (interior), zlib's `COPY_`/`COPY`
    (trailing rstrip) — three transforms, one class

`_export_statement` now filters against the emitted body, so this should never
fire from that path. It exists for the paths that do not go through it and for
the next derivation that drifts — the same reason `_assert_wrapper_loadable`
reads the emitted TEXT rather than the generator's record of what it defined.
A guard sharing the bug's bookkeeping agrees with the bug.
"""
function _assert_exports_defined(wrapper_content::AbstractString, module_name::AbstractString)
    exported = _exported_names(wrapper_content)
    isempty(exported) && return nothing

    defined = _defined_names(wrapper_content)
    # An empty set means the body did not parse — `_assert_wrapper_parses` runs
    # first and owns that diagnosis, so do not pile a bogus 100%-undefined
    # report on top of it.
    isempty(defined) && return nothing

    missing_names = sort!(filter(n -> !(n in defined), exported))
    isempty(missing_names) && return nothing

    shown = join(("  " * n for n in first(missing_names, 15)), "\n")
    more = length(missing_names) > 15 ? "\n  … and $(length(missing_names) - 15) more" : ""
    error("""
    Refusing to write wrapper '$module_name': $(length(missing_names)) exported \
    name(s) are never defined by the module.

    $shown$more

    `using` this module would succeed and then hand the caller names that raise \
    UndefVarError — including through tab-completion, which reads the export \
    list. Fix by deriving the export list from what was EMITTED rather than from \
    the candidate set (see `_export_statement`), not by defining the missing \
    names: they are usually screened out on purpose, or bound under a different \
    spelling because the definition path sanitizes and the export path does not.
    """)
end

"""
    _assert_wrapper_parses(wrapper_content, module_name)

Refuse to write a wrapper that is not valid Julia **syntax**.

`_assert_wrapper_loadable` checks that ccall signatures only name declared
types, but its scope is deliberately semantic and deliberately narrow. A
malformed *identifier* never gets that far: the file dies in the parser, and one
bad character kills the entire module.

Found on llama.cpp. DWARF spells a lambda's type as
`(lambda at ./src/llama-model-loader.cpp:1538:79)`; the parentheses, slashes and
colons reached a struct field, and all **98,094 lines** of the wrapper were a
syntax error — every one of 4,990 functions dead, discovered only when a user
tried to `include` it. Parsing the text we are about to write costs a couple of
seconds and turns that into a generation-time failure naming the exact line.

Deliberately a syntax check only: it makes no claim about whether the module
runs, just that a parser accepts it.
"""
function _assert_wrapper_parses(wrapper_content::AbstractString, module_name::AbstractString)
    function fail(detail)
        # The rejected source is the only way to find the offending line, and the
        # guard's whole point is that it never reaches the wrapper path. Dump it
        # so the emitter bug is debuggable instead of merely reported.
        dump_path = joinpath(tempdir(), "replibuild_rejected_$(module_name).jl")
        try
            write(dump_path, wrapper_content)
        catch
            dump_path = "<could not write dump>"
        end
        # Truncated: an embedded error node renders the surrounding expression,
        # which on a 98k-line module ran to 340 KB and buried the message.
        detail_str = first(string(detail), 1200)
        error("""
            Refusing to write wrapper '$module_name': the generated source is not \
            valid Julia syntax. Writing it would produce a file that fails at parse \
            time, disabling every function in the module.

            $detail_str

            Rejected source written to:
              $dump_path

            This is an emitter bug, not a configuration problem — some C++ spelling \
            reached an identifier position without being sanitized. Fix the emitter \
            (see _sanitize_cpp_type_name / _sanitize_c_type_name), not the library.
            """)
    end

    parsed = try
        Meta.parseall(wrapper_content; filename = "$(module_name).jl")
    catch e
        e isa Base.Meta.ParseError || rethrow()
        fail(sprint(showerror, e))
    end

    # Older/alternative parsers embed the failure instead of raising it.
    stack = Any[parsed]
    while !isempty(stack)
        node = pop!(stack)
        node isa Expr || continue
        if node.head === :error || node.head === :incomplete
            # args[1] is the ParseError / message; the node itself carries the
            # whole surrounding expression and is useless in a log line.
            fail(isempty(node.args) ? node.head : node.args[1])
        end
        append!(stack, node.args)
    end
    return nothing
end

# =============================================================================
# TIER 1: BASIC WRAPPER (Symbol-Only)
# =============================================================================

"""
    wrap_basic(config::RepliBuildConfig, library_path::String; generate_docs::Bool=true)

Generate basic Julia wrapper from binary symbols only (no headers required).

Quality: ~40% - Conservative types, placeholder signatures, requires manual refinement.
Use when: Headers not available, quick prototyping, binary-only distribution.
"""
function wrap_basic(config::RepliBuildConfig, library_path::String; generate_docs::Bool=true)

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    # Create type registry
    registry = create_type_registry(config)

    # Extract symbols
    symbols = extract_symbols(library_path, registry, demangle=true, method=:nm)

    if isempty(symbols)
        @warn "No symbols found in library"
        return nothing
    end

    # Filter functions and data
    functions = filter(s -> s.symbol_type == :function, symbols)
    data_symbols = filter(s -> s.symbol_type == :data, symbols)

    # Generate wrapper module
    module_name = get_module_name(config)
    wrapper_content = generate_basic_module(config, library_path, functions, data_symbols,
                                           module_name, registry, generate_docs)

    # Write to file
    output_dir = get_output_path(config)
    mkpath(output_dir)
    output_file = joinpath(output_dir, "$(module_name).jl")

    _write_wrapper(output_file, wrapper_content, module_name)

    return output_file
end

"""
Generate basic wrapper module content.
"""
function generate_basic_module(config::RepliBuildConfig, lib_path::String,
                               functions::Vector{SymbolInfo}, data_symbols::Vector{SymbolInfo},
                               module_name::String, registry::TypeRegistry, generate_docs::Bool)

    # Header with metadata
    header = """
    # Auto-generated Julia wrapper for $(config.project.name)
    # Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    # Generator: RepliBuild Wrapper (Basic: Symbol extraction)
    # Library: $(basename(lib_path))

    """

    content = header

    # Module declaration
    content *= "module $module_name\n\n"
    content *= "const Cintptr_t = Int\n"
    content *= "const Cuintptr_t = UInt\n\n"
    content *= "using Libdl\n\n"

    # Library management
    content *= """
    # =============================================================================
    # LIBRARY MANAGEMENT
    # =============================================================================

    const _LIB_PATH = raw"$(abspath(lib_path))"
    const _LIB = Ref{Ptr{Nothing}}(C_NULL)
    const _LOAD_ERRORS = String[]

    function __init__()
        try
            _LIB[] = Libdl.dlopen(_LIB_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
        catch e
            push!(_LOAD_ERRORS, string(e))
            @error "Failed to load library $(basename(lib_path))" exception=e
        end
    end

    \"""
        is_loaded()

    Check if the library is successfully loaded.
    \"""
    is_loaded() = _LIB[] != C_NULL

    \"""
        get_load_errors()

    Get any errors that occurred during library loading.
    \"""
    get_load_errors() = copy(_LOAD_ERRORS)

    \"""
        get_lib_path()

    Get the path to the underlying library.
    \"""
    get_lib_path() = _LIB_PATH

    # Safety check macro
    macro check_loaded()
        quote
            if !is_loaded()
                error("Library not loaded. Errors: ", join(get_load_errors(), "; "))
            end
        end
    end

    """

    # Function wrappers
    content *= """
    # =============================================================================
    # FUNCTION WRAPPERS
    # =============================================================================

    """

    function_count = 0
    exports = String["is_loaded", "get_load_errors", "get_lib_path"]

    for func in functions
        if function_count >= 50
            break  # Limit to avoid huge files
        end

        func_wrapper = generate_basic_function_wrapper(func, registry, generate_docs)
        if !isnothing(func_wrapper)
            content *= func_wrapper * "\n"
            push!(exports, func.julia_name)
            function_count += 1
        end
    end

    if length(functions) > 50
        content *= """
        # ... and $(length(functions) - 50) more functions omitted
        # Regenerate with headers for complete wrapping:
        #   RepliBuild.wrap("$lib_path", headers=["your_header.h"])

        """
    end

    # Library info function
    content *= """
    # =============================================================================
    # METADATA
    # =============================================================================

    \"""
        library_info()

    Get information about the wrapped library.
    \"""
    function library_info()
        return Dict{Symbol,Any}(
            :name => "$(config.project.name)",
            :path => _LIB_PATH,
            :loaded => is_loaded(),
            :tier => :basic,
            :type_safety => "40% (conservative placeholders)",
            :functions_wrapped => $function_count,
            :functions_total => $(length(functions)),
            :data_symbols => $(length(data_symbols))
        )
    end

    """

    push!(exports, "library_info")

    # Exports — filtered against the body emitted so far. `content` at this point
    # is the entire module apart from its closing `end`, which parseall must see
    # or the whole body reads as unparseable and no filtering happens.
    content *= "# Exports\n"
    content *= _export_statement(exports, content * "\nend\n")

    content *= "end # module $module_name\n"

    return content
end

"""
Generate wrapper for a single function (basic tier).
"""
function generate_basic_function_wrapper(func::SymbolInfo, registry::TypeRegistry, generate_docs::Bool)
    if isempty(func.julia_name)
        return nothing
    end

    # For basic tier, we use conservative Any types since we don't have parameter info
    wrapper = ""

    if generate_docs
        wrapper *= """
        \"""
            $(func.julia_name)(args...)

        Wrapper for C/C++ function `$(func.demangled_name)`.

        Signature uses placeholder types. Actual types unknown without headers.
        Return type and parameters may need manual adjustment.

        # C/C++ Symbol
        `$(func.name)`
        \"""
        """
    end

    wrapper *= """
    function $(func.julia_name)(args...)
        @check_loaded()
        ccall((:$(func.name), _LIB[]), Any, (), args...)
    end

    """

    return wrapper
end

# =============================================================================
# TIER 2: ADVANCED WRAPPER (Header-Aware via Clang.jl)
# =============================================================================

"""
    wrap_with_clang(config::RepliBuildConfig, library_path::String, headers::Vector{String}; generate_docs::Bool=true)

Generate advanced Julia wrapper using Clang.jl for type-aware binding generation.

Quality: ~85% - Accurate types from headers, production-ready with minor tweaks.
Use when: Headers available, need type safety, production deployment.
"""
function wrap_with_clang(config::RepliBuildConfig, library_path::String, headers::Vector{String};
                        generate_docs::Bool=true)

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    if isempty(headers)
        error("Headers required for advanced wrapping")
    end

    # Verify headers exist
    for header in headers
        if !isfile(header)
            @warn "Header not found: $header"
        end
    end

    # Build config for ClangJLBridge
    clang_config = Dict(
        "project" => Dict("name" => config.project.name),
        "compile" => Dict("include_dirs" => config.compile.include_dirs),
        "binding" => Dict(
            "use_ccall_macro" => false,
            "add_doc_strings" => generate_docs,
            "use_julia_native_enum" => true
        )
    )

    output_file = ClangJLBridge.generate_bindings_clangjl(clang_config, library_path, headers)

    if isnothing(output_file)
        error("Binding generation failed")
    end

    return output_file
end

# =============================================================================
# TIER 3: INTROSPECTIVE WRAPPER (Metadata-Rich)
# =============================================================================

"""
    wrap_introspective(config::RepliBuildConfig, library_path::String, headers::Vector{String}; generate_docs::Bool=true)

Generate introspective Julia wrapper using compilation metadata for perfect type accuracy.

Quality: ~95% - Exact types from compilation, language-agnostic, zero manual configuration.
Use when: Metadata available from RepliBuild compilation, need perfect bindings.

This is the culmination of RepliBuild's vision: automatic, accurate, language-agnostic wrapping.
"""
function wrap_introspective(config::RepliBuildConfig, library_path::String, headers::Vector{String};
                           generate_docs::Bool=true)

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    # Load compilation metadata
    metadata_file = joinpath(dirname(library_path), "compilation_metadata.json")
    if !isfile(metadata_file)
        error("Compilation metadata not found: $metadata_file\nRun RepliBuild.build() first to generate metadata")
    end

    metadata = JSON.parsefile(metadata_file)

    if !haskey(metadata, "functions")
        error("Invalid metadata: missing 'functions' key")
    end

    functions = metadata["functions"]

    # Extract supplementary types from headers (enums, unused types, etc.)
    # Rust has no C headers — skip Clang.jl header extraction entirely
    # use_clang_jl = false skips AST-based extraction (DWARF-only path)
    include_dirs = get(metadata, "include_dirs", String[])
    empty_header_types = Dict("enums" => Dict(), "constants" => Dict(), "typedefs" => Dict(), "structs" => String[])
    header_types = if !config.wrap.use_clang_jl
        empty_header_types
    elseif !isempty(headers)
        try
            ClangJLBridge.extract_header_types(headers, include_dirs)
        catch e
            @warn "Clang.jl header extraction failed, falling back to DWARF-only wrapping" exception=e
            empty_header_types
        end
    else
        # Auto-discover headers from include directories
        discovered_headers = String[]
        for inc_dir in include_dirs
            if isdir(inc_dir)
                append!(discovered_headers, ClangJLBridge.discover_headers(inc_dir, recursive=false))
            end
        end
        if !isempty(discovered_headers)
            try
                ClangJLBridge.extract_header_types(discovered_headers, include_dirs)
            catch e
                @warn "Clang.jl header extraction failed, falling back to DWARF-only wrapping" exception=e
                empty_header_types
            end
        else
            empty_header_types
        end
    end

    # Merge header types into metadata
    if !isempty(header_types["enums"])
        if !haskey(metadata, "header_enums")
            metadata["header_enums"] = header_types["enums"]
        else
            merge!(metadata["header_enums"], header_types["enums"])
        end
    end

    # Store function pointer typedefs for callback documentation
    if haskey(header_types, "function_pointers") && !isempty(header_types["function_pointers"])
        metadata["function_pointer_typedefs"] = header_types["function_pointers"]
    end

    # Create type registry with metadata + typedef resolution table
    typedef_table = get(metadata, "typedef_table", Dict{String,Any}())
    # Convert to String,String for custom_types merge
    typedef_custom = Dict{String,String}(String(k) => String(v) for (k, v) in typedef_table if v != "Any")
    registry = create_type_registry(config, custom_types=typedef_custom)

    # AOT Compilation Pass
    thunks_lib_path = ""
    if config.compile.aot_thunks
        output_dir = get_output_path(config)
        lib_name = basename(library_path)
        thunks_name = replace(lib_name, ".so" => "_thunks.so", ".dylib" => "_thunks.dylib", ".dll" => "_thunks.dll")
        thunks_so = joinpath(output_dir, thunks_name)
        if isfile(thunks_so)
            thunks_lib_path = abspath(thunks_so)
        else
            @warn "AOT thunks enabled but companion library not found at $thunks_so"
        end
    end

    # Generate wrapper module
    module_name = get_module_name(config)

    (wrapper_content, needed_thunks) = if registry.language == :c
        # C path: no DAG — Julia's internal LLVM handles C ABI correctly.
        # All dispatch goes through is_c_lto_safe() heuristics alone.
        (generate_introspective_module_c(config, library_path, metadata,
                                         module_name, registry, generate_docs, thunks_lib_path), nothing)
    else
        # C++ path: DAG catches transitive layout drift that per-function heuristics miss
        dag_result = DAGDiff.dag_diff(metadata)

        if config.wrap.dag
            dag_dir = joinpath(config.project.root, "dag")
            mkpath(dag_dir)
            DAGDiff.render_html(dag_result, joinpath(dag_dir, "index.html"))
            DAGDiff.export_dot(dag_result, joinpath(dag_dir, "diff.dot"))
        end

        generate_introspective_module_cpp(config, library_path, metadata,
                                          module_name, registry, generate_docs, thunks_lib_path;
                                          dag_result=dag_result)
    end

    # Write to file
    output_dir = get_output_path(config)
    mkpath(output_dir)
    output_file = joinpath(output_dir, "$(module_name).jl")

    _write_wrapper(output_file, wrapper_content, module_name)

    # Write thunk manifest for dead-thunk elimination.
    # JITManager reads this to skip generating MLIR thunks for functions
    # that are dispatched via ccall (Tier 1) and never need a thunk.
    if needed_thunks !== nothing
        manifest = Dict{String,Any}(
            "function_thunks" => sort!(collect(needed_thunks)),
            "version" => 1
        )
        manifest_path = joinpath(output_dir, "thunk_manifest.json")
        open(manifest_path, "w") do io
            JSON.print(io, manifest, 2)
        end
    end

    println("  wrap: $(basename(output_file))")

    return output_file
end


"""
    generate_vararg_wrappers(func_name, mangled, julia_name, params, return_type, overloads, generate_docs, demangled) -> (code, export_names)

Generate typed overload wrappers for a variadic C function.
Julia's `ccall` requires fixed signatures, so we generate:
- A base wrapper with only the fixed (non-variadic) parameters
- Typed overloads for each signature listed in `overloads`
All varargs wrappers use the `@ccall` semicolon form (never JIT/thunks): the
`;` marks where varargs begin, so the call lowers to a true variadic
foreigncall. On x86-64 SysV that makes codegen set AL to the number of vector
registers used before the call — the callee's `va_start` prologue gates its
XMM spill on AL, so a non-variadic call site (flat type tuple) only reads
float varargs correctly when leftover AL happens to be nonzero.
"""
# Types a [wrap.varargs] overload entry may name. These strings are written
# verbatim into generated signatures and @ccall annotations, so an unchecked
# typo ("Cnit") surfaces as an UndefVarError when the WRAPPER loads — naming
# neither the TOML entry nor the function. Validate here, at wrap time, with
# an error that names both.
const _VARARG_ALLOWED_TYPES = Set([
    "Any", "Cstring", "Cwstring",
    "Cint", "Cuint",
    "Clong", "Culong", "Clonglong", "Culonglong",
    "Cintmax_t", "Cuintmax_t", "Csize_t", "Cssize_t", "Cptrdiff_t", "Cwchar_t",
    "Cdouble", "Cintptr_t", "Cuintptr_t",
    "Int32", "Int64", "Int128", "Int",
    "UInt32", "UInt64", "UInt128", "UInt",
    "Float64",
])

# C DEFAULT ARGUMENT PROMOTION (C11 6.5.2.2p6-7). In the variadic part of a call
# `float` becomes `double` and every integer type of rank below `int` becomes
# `int`, so these types NEVER occupy a variadic slot — there is no ABI in which
# a caller writes 4 bytes into a slot the callee reads with `va_arg(ap, double)`.
#
# `@ccall f(fixed…; va_1::Cfloat)` writes exactly Cfloat: Julia does not promote,
# and nothing else in this generator does either. The callee then reads 8 bytes
# where 4 were written and formats whatever followed — WRONG OUTPUT, NO CRASH,
# which is the worst failure mode a printf wrapper has. It cannot be caught
# downstream: the wrapper loads, the ccall is well-formed, and only the rendered
# text is wrong. So it is rejected here, at wrap time, naming the type to use.
const _VARARG_PROMOTED_TO = Dict(
    "Cfloat" => "Cdouble", "Float32" => "Cdouble", "Float16" => "Cdouble",
    "Cchar"  => "Cint", "Cuchar"  => "Cint",
    "Cshort" => "Cint", "Cushort" => "Cint",
    "Bool"   => "Cint",
    "Int8"   => "Cint", "UInt8"   => "Cint",
    "Int16"  => "Cint", "UInt16"  => "Cint",
)

function _validate_vararg_type(func_name::String, t::String)
    if haskey(_VARARG_PROMOTED_TO, t)
        promoted = _VARARG_PROMOTED_TO[t]
        error("""
            RepliBuild: '$t' cannot name a variadic argument in [wrap.varargs.$func_name].
            C default argument promotion converts it to `$promoted` before the callee
            ever sees it, so a slot declared '$t' is the wrong WIDTH — the call
            succeeds and the callee reads garbage. Use "$promoted" instead:
                $func_name = [["$promoted"]]
            (If the C prototype says `$t`, that is the pre-promotion declaration;
            the variadic slot still holds a $promoted.)
            """)
    end
    t in _VARARG_ALLOWED_TYPES && return
    m = match(r"^Ptr\{([A-Za-z_][A-Za-z0-9_]*)\}$", t)
    m !== nothing && return   # Ptr{AnyIdentifier} — struct pointers included
    error("""
        RepliBuild: invalid type '$t' in [wrap.varargs.$func_name].
        Each overload entry lists the VARIADIC argument types as strings, e.g.
            $func_name = [["Cstring"], ["Cint", "Cdouble"]]
        Allowed: $(join(sort!(collect(_VARARG_ALLOWED_TYPES)), ", ")), or Ptr{...}.
        """)
end

# The type a variadic slot ACCEPTS in the generated signature. The @ccall keeps
# the declared type, where cconvert/unsafe_convert run and their results stay
# rooted for the call — so widening here changes nothing about the ABI and is
# what lets `f_Cint(fmt, 42)` work at all (`42` is Int64; Cint is Int32, and a
# `::Cint` signature rejects it before the ccall is ever reached).
#
# Safe against ambiguity by construction: every overload is its own named
# function (`f_Cint`, `f_Cdouble`), so no two of these signatures compete.
function _vararg_sig_type(t::String)
    (t == "Cstring" || t == "Cwstring") && return "Union{AbstractString,$t}"
    startswith(t, "Ptr{") && return "Any"
    t == "Any" && return "Any"
    t in ("Cdouble", "Float64") && return "Real"
    return "Integer"   # every remaining allowed type is an integer type
end

# Overload names are derived from the type list; Ptr{Cvoid} would otherwise
# produce `f_Ptr{Cvoid}` — a syntax error in the generated wrapper.
_vararg_name_part(t::String) = replace(t, r"[{}]" => "")

function generate_vararg_wrappers(func_name::String, mangled::String, julia_name::String,
                                  params::Vector, return_type,
                                  overloads::Vector{Vector{String}},
                                  generate_docs::Bool, demangled::String, lang::Symbol;
                                  cstring_free::String="")
    code = ""
    export_names = String[]

    for va_types in overloads, t in va_types
        _validate_vararg_type(func_name, t)
    end
    if length(unique(overloads)) != length(overloads)
        error("RepliBuild: duplicate overload signature in [wrap.varargs.$func_name] — " *
              "each entry must list a distinct variadic type tuple.")
    end

    # Build fixed parameter info
    fixed_param_names = String[]
    fixed_param_types = String[]  # Julia/ccall types
    fixed_julia_sig_types = String[]  # Ergonomic types for signature
    fixed_needs_conversion = Bool[]

    for param in params
        # Skip the varargs placeholder — compare the raw name: sanitization
        # mangles it to "varargs_", so checking the safe name never matched
        # and the placeholder leaked into signatures as `varargs_::`
        if param["name"] == "varargs..."
            continue
        end
        safe_name = lang == :c ? make_c_identifier(param["name"]) : make_cpp_identifier(param["name"])
        push!(fixed_param_names, safe_name)

        julia_type = param["julia_type"]
        push!(fixed_param_types, julia_type)

        # Ergonomic type mapping (same logic as main wrapper gen)
        if julia_type in ["Cint", "Clong", "Cshort"]
            push!(fixed_julia_sig_types, "Integer")
            push!(fixed_needs_conversion, true)
        elseif startswith(julia_type, "Ptr{")
            push!(fixed_julia_sig_types, "Any")
            push!(fixed_needs_conversion, false)
        else
            push!(fixed_julia_sig_types, julia_type)
            push!(fixed_needs_conversion, false)
        end
    end

    julia_return_type = get(return_type, "julia_type", "Cvoid")
    c_return_type = String(get(return_type, "c_type", ""))

    # An `Any` return whose C type is an aggregate is a by-value struct return
    # the mapper could not NAME, and no ABI-correct call can be emitted for it.
    #
    # `Any` in a foreign return position tells Julia the callee returned a
    # `jl_value_t*`, so the result is dereferenced as a Julia object — a segfault
    # inside dispatch, far from the call site. Degrading to `Cvoid` is not the
    # escape either: a MEMORY-class aggregate comes back through a hidden sret
    # pointer, so dropping the value leaves the ABI wrong rather than merely
    # lossy.
    #
    # Live case: llama.cpp's `format[abi:cxx11]` returns `std::basic_string`,
    # correctly absent from `struct_definitions` because cf09702 drops STL types,
    # so nothing can name it. The `::Any` ccall shipped for as long as the
    # function has existed; `_assert_no_any_ccall_return` is what finally refused
    # it, and refusing the whole wrapper takes 5,614 working functions with it.
    #
    # So refuse at the CALL SITE instead, exactly as the C++ generator already
    # does for unmappable parameters: the module loads, everything else works,
    # and this one function explains itself to anyone who calls it.
    if julia_return_type == "Any" && !isempty(c_return_type) &&
       c_return_type != "unknown" &&
       !occursin("*", c_return_type) && !occursin("void", c_return_type)
        trap = """
        \"\"\"
            $julia_name(args...)

        **Unavailable.** `$func_name` returns `$c_return_type` by value, a type
        RepliBuild could not map, so no ABI-correct call can be emitted. Calling
        this raises rather than corrupting memory.
        \"\"\"
        function $julia_name(args...)
            Base.error(\"\"\"
            FFI Safety Trap: cannot call '$julia_name'.

            It returns $c_return_type by value, and that type could not be
            mapped to Julia. Declaring the return as a boxed value would make
            Julia dereference the result as an object; declaring it void would
            leave the hidden return pointer unaccounted for. Both corrupt
            memory, so neither was emitted.

            If you need this function, give the type a mapping in
            replibuild.toml so it can be named.
            \"\"\")
        end
        """
        return (trap, [julia_name])
    end

    # Cstring returns get the shared policy (NULL → nothing, String copy,
    # [wrap.cstring_owned] free) — printf-family varargs like sqlite3_mprintf
    # return malloc'd buffers, so this is where the policy matters most.
    is_cstring_ret = julia_return_type == "Cstring"
    sig_return_type = is_cstring_ret ? "Union{String,Nothing}" : julia_return_type

    # Build fixed parameter signature
    fixed_sig_parts = ["$(n)::$(t)" for (n, t) in zip(fixed_param_names, fixed_julia_sig_types)]
    fixed_sig = join(fixed_sig_parts, ", ")

    # Build conversion code for fixed params
    fixed_conversion = ""
    fixed_ccall_names = String[]
    for (name, ctype, needs_conv) in zip(fixed_param_names, fixed_param_types, fixed_needs_conversion)
        if needs_conv
            converted = "$(name)_c"
            push!(fixed_ccall_names, converted)
            fixed_conversion *= "    $converted = $ctype($name)\n"
        else
            push!(fixed_ccall_names, name)
        end
    end

    # --- Base wrapper (fixed args only) ---
    # The callee is still variadic even when no varargs are passed, so the call
    # site must lower as a variadic foreigncall (trailing `;` in @ccall) — that
    # is what makes codegen set AL on x86-64 SysV instead of leaving garbage in it.
    # `var"…"` keeps the symbol position safe for any C identifier.
    fixed_atccall = join(["$(n)::$(t)" for (n, t) in zip(fixed_ccall_names, fixed_param_types)], ", ")

    doc = ""
    if generate_docs
        doc = """
        \"\"\"
            $julia_name($fixed_sig) -> $sig_return_type

        Wrapper for variadic C function: `$demangled` (base call with fixed args only)
        \"\"\"
        """
    end

    if is_cstring_ret
        code *= """
        $doc
        function $julia_name($fixed_sig)::Union{String,Nothing}
        $fixed_conversion    ptr = @ccall LIBRARY_PATH.var"$mangled"($fixed_atccall;)::Cstring
        $(_cstring_policy_lines(cstring_free))
        end

        """
    else
        code *= """
        $doc
        function $julia_name($fixed_sig)::$julia_return_type
        $fixed_conversion    return @ccall LIBRARY_PATH.var"$mangled"($fixed_atccall;)::$julia_return_type
        end

        """
    end
    push!(export_names, julia_name)

    # --- Typed overloads ---
    for va_types in overloads
        # Build overload function name: fname_Type1_Type2 (Ptr{X} → PtrX)
        type_suffix = join(map(_vararg_name_part, va_types), "_")
        overload_name = "$(julia_name)_$(type_suffix)"

        # Build variadic parameter names and types
        va_param_names = ["va_$(i)" for i in 1:length(va_types)]
        # Signature takes the WIDENED type, the @ccall below keeps the DECLARED
        # one — see `_vararg_sig_type`. Splitting these is what makes the tail
        # as ergonomic as the fixed params without touching the ABI.
        va_sig_parts = ["$(n)::$(_vararg_sig_type(t))" for (n, t) in zip(va_param_names, va_types)]

        # Full signature = fixed + variadic
        all_sig_parts = vcat(fixed_sig_parts, va_sig_parts)
        all_sig = join(all_sig_parts, ", ")

        # @ccall varargs: everything after `;` is variadic, each with its own
        # declared type. Lowers to a variadic foreigncall, so AL is set correctly.
        va_atccall = join(["$(n)::$(t)" for (n, t) in zip(va_param_names, va_types)], ", ")

        overload_doc = ""
        if generate_docs
            overload_doc = """
            \"\"\"
                $overload_name($all_sig) -> $sig_return_type

            Typed variadic overload for: `$demangled`
            Variadic types: $(join(va_types, ", "))
            \"\"\"
            """
        end

        if is_cstring_ret
            code *= """
            $overload_doc
            function $overload_name($all_sig)::Union{String,Nothing}
            $fixed_conversion    ptr = @ccall LIBRARY_PATH.var"$mangled"($fixed_atccall; $va_atccall)::Cstring
            $(_cstring_policy_lines(cstring_free))
            end

            """
        else
            code *= """
            $overload_doc
            function $overload_name($all_sig)::$julia_return_type
            $fixed_conversion    return @ccall LIBRARY_PATH.var"$mangled"($fixed_atccall; $va_atccall)::$julia_return_type
            end

            """
        end
        push!(export_names, overload_name)
    end

    return (code, export_names)
end

"""
Generate introspective wrapper module content using compilation metadata.
"""

