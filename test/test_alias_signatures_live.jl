# =============================================================================
# Symbols no DIE describes, live (2026-11-17)
#
# `int aliased_name(int) __attribute__((alias("aliased_real")))` has no DIE of
# its own. The metadata guessed its signature from its name and the wrapper
# bound `aliased_name()::Cint`: a call with no arguments, whose callee read a
# stale register. The build now gives such a symbol the signature of the
# described function at its address, and one with no such function is a
# refusal that says there is no debug information, in both generators.
#
# test/alias_signature_test (C, the in-process toolchain) has an alias, a weak
# alias, an alias returning a record and one taking a stack argument, a K&R
# function, and a `nodebug` function. test/alias_signature_cpp_test (C++, the
# system compiler) has a C-linkage alias, a mangled one, two `nodebug`
# functions, and an alias whose aliasee is `static`, which is refused too: a
# local name need not be one function's, so it is no key into the DWARF table.
# Each is built, wrapped and called from a child. The pieces:
# test_alias_signatures.jl.
# =============================================================================

using Test
using Libdl
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const ALIAS_SIG_DIR = joinpath(@__DIR__, "alias_signature_test")
const ALIAS_SIG_CPP_DIR = joinpath(@__DIR__, "alias_signature_cpp_test")
const ALIAS_SIG_PROBE = joinpath(ALIAS_SIG_DIR, "probe_alias_signatures.jl")

function _alias_sig_build(dir, name, lang, source)
    for d in ("build", "julia", ".replibuild_cache", ".debug")
        p = joinpath(dir, d)
        ispath(p) && RepliBuild._rm_tree(p)
    end
    toml_path = joinpath(dir, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "$name"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(dir))"

    [compile]
    flags = ["-O1", "-fPIC"]
    source_files = ["$(escape_string(joinpath(dir, source)))"]
    include_dirs = ["$(escape_string(dir))"]

    [link]
    enable_lto = false
    optimization_level = "1"

    [binary]
    type = "shared"

    [wrap]
    language = "$lang"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    redirect_stdout(devnull) do
        RepliBuild.build(toml_path)
    end
    RepliBuild.wrap(toml_path)
    jdir = joinpath(dir, "julia")
    meta = RepliBuild._read_json(joinpath(jdir, "compilation_metadata.json"))
    return (jdir, meta, Dict(String(f["mangled"]) => f for f in meta["functions"]))
end

_alias_sig_body(code, f) = (m = match(Regex("^function $(f)\\(.*?^end\$", "ms"), code);
                            m === nothing ? "" : String(m.match))

function _alias_sig_live(wrapper, kind, names)
    r = run_isolated(`$ALIAS_SIG_PROBE $wrapper $kind`)
    v = probe_verdicts(r.output)
    @test r.ok
    @test haskey(v, "PROBE_DONE")
    for n in names
        @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
    end
    @test count(n -> haskey(v, n), names) == length(names)
end

@testset "Symbols no DIE describes: C, live" begin
    (jdir, meta, fns) = _alias_sig_build(ALIAS_SIG_DIR, "aliassig", "c", "alias_api.c")
    wrapper = joinpath(jdir, "Aliassig.jl")
    @test isfile(wrapper)
    code = read(wrapper, String)

    # A weak alias is a symbol of an ELF library's dynamic table. MinGW's lld
    # leaves one out of a DLL's export directory (it is `T` in the COFF symbol
    # table and nothing can bind it), so on Windows `al_scaled` is not the
    # library's API and has no row and no wrapper. What the loader binds decides.
    lib = joinpath(jdir, "libaliassig." * Libdl.dlext)
    exported = RepliBuild.Compiler._runtime_symbol_names(lib)
    @test exported !== nothing && "al_name" in exported
    weak_exported = exported !== nothing && "al_scaled" in exported
    Sys.islinux() && @test weak_exported
    @test haskey(fns, "al_scaled") == weak_exported

    @testset "the metadata: an alias has its aliasee's signature" begin
        @test meta["alias_signatures"] === true
        checked = 0
        for (alias, real) in (("al_name", "al_real"), ("al_scaled", "al_scaled_real"),
                              ("al_pair", "al_pair_real"), ("al_wide", "al_wide_real"))
            (alias == "al_scaled" && !weak_exported) && continue
            a, r = fns[alias], fns[real]
            @test (alias, get(a, "alias_of", nothing)) == (alias, real)
            @test (alias, a["return_type_source"], a["parameters_source"]) == (alias, "dwarf", "dwarf")
            @test (alias, a["return_type"]) == (alias, r["return_type"])
            @test (alias, a["parameters"]) == (alias, r["parameters"])
            @test !isempty(a["parameters"])
            # Its own identity is untouched.
            @test (a["name"], a["mangled"]) == (alias, alias)
            @test !haskey(r, "alias_of")
            checked += 1
        end
        @test checked == 3 + weak_exported
        # A DIE with no prototype: the return type is DWARF's.
        @test fns["al_old_style"]["return_type_source"] == "dwarf"
        @test !haskey(fns["al_old_style"], "alias_of")
        # No DIE and no function at its address: still inferred.
        @test fns["al_undescribed"]["return_type_source"] == "inferred"
        @test !haskey(fns["al_undescribed"], "alias_of")
    end

    @testset "the wrapper: a typed call for each alias, a refusal for the rest" begin
        @test occursin("ccall((:al_name, LIBRARY_PATH), Cint, (Cint,), x_c)", _alias_sig_body(code, "al_name"))
        if weak_exported
            @test occursin("ccall((:al_scaled, LIBRARY_PATH), Cdouble, (Cdouble, Cint,)", _alias_sig_body(code, "al_scaled"))
        else
            @test !occursin("al_scaled,", code) && !occursin("function al_scaled(", code)
        end
        @test occursin("ccall((:al_pair, LIBRARY_PATH), AlPair, (Cdouble, Cdouble,)", _alias_sig_body(code, "al_pair"))
        @test occursin("ccall((:al_old_style, LIBRARY_PATH), Cint, (), )", _alias_sig_body(code, "al_old_style"))
        u = _alias_sig_body(code, "al_undescribed")
        @test startswith(u, "function al_undescribed(args...)\n")
        @test occursin("has no debug information for it", u)
        @test !occursin("ccall((:al_undescribed", code)
    end

    @testset "every function, live" begin
        cases = ["alias_takes_its_argument", "weak_alias_floating_and_integer",
                 "alias_returns_a_record", "alias_with_a_stack_argument",
                 "no_prototype_still_called", "no_debug_information_is_refused"]
        weak_exported || filter!(!=("weak_alias_floating_and_integer"), cases)
        _alias_sig_live(wrapper, "c", cases)
    end
end

if requires("symbols no DIE describes, C++", :libJLCS, :clangxx)
@testset "Symbols no DIE describes: C++, live" begin
    (jdir, meta, fns) = _alias_sig_build(ALIAS_SIG_CPP_DIR, "aliassigx", "cpp", "alias_xapi.cpp")
    wrapper = joinpath(jdir, "Aliassigx.jl")
    @test isfile(wrapper)
    code = read(wrapper, String)

    @testset "the metadata: an alias has its aliasee's signature" begin
        checked = 0
        for (alias, real) in (("ax_name", "ax_real"), ("_Z10ax_mangleddi", "_Z15ax_mangled_realdi"))
            a = fns[alias]
            @test (alias, get(a, "alias_of", nothing)) == (alias, real)
            @test (alias, a["return_type_source"], a["parameters_source"]) == (alias, "dwarf", "dwarf")
            @test (alias, length(a["parameters"])) == (alias, alias == "ax_name" ? 1 : 2)
            checked += 1
        end
        @test checked == 2
        @test fns["_Z10ax_mangleddi"]["return_type"]["c_type"] == "double"
        # The aliasee's definition is the alias's: `ax_real` cannot throw, so
        # its C-linkage alias keeps a plain ccall.
        @test fns["ax_name"]["nounwind"] === true
        # `ax_via_static`'s aliasee is a local symbol: not looked up by name.
        for m in ("ax_undescribed", "_Z18ax_undescribed_cxxi", "ax_via_static")
            @test (m, fns[m]["return_type_source"]) == (m, "inferred")
            @test !haskey(fns[m], "alias_of")
        end
    end

    @testset "the wrapper: a typed call for each alias, a refusal for the rest" begin
        @test occursin("ccall((:ax_name, LIBRARY_PATH), Cint, (Cint,), x_c)", _alias_sig_body(code, "ax_name"))
        # A mangled function that is not `noexcept` takes its thunk, under its own symbol.
        @test occursin("JITManager.invoke(\"_mlir_ciface__Z10ax_mangleddi_thunk\", Cdouble,", _alias_sig_body(code, "ax_mangled"))
        for f in ("ax_undescribed", "ax_undescribed_cxx", "ax_via_static")
            u = _alias_sig_body(code, f)
            @test (f, startswith(u, "function $f(args...)\n")) == (f, true)
            @test (f, occursin("has no debug information for it", u)) == (f, true)
        end
        @test !occursin("ccall((:ax_undescribed", code) && !occursin("ccall((:ax_via_static", code)
        @test !occursin("ax_undescribed_cxxi_thunk", code)
    end

    @testset "every function, live" begin
        _alias_sig_live(wrapper, "cpp", ["c_linkage_alias", "mangled_alias", "no_debug_information_is_refused",
                                         "alias_of_a_static_function_is_refused"])
    end
end
end # requires
