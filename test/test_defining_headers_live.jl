# =============================================================================
# A header that defines functions, and the forced unit, live (2026-11-21)
#
# test_forced_unit_definitions.jl checks the readers. This builds
# test/defining_header_test, which has both shapes the Hub's fmt and clipper2
# failed to build with on the 4.6.0 release sweep:
#
#   impl.h     declares a class AND defines a function and a variable with
#              external linkage (fmt's `format-inl.h`). The forced unit
#              includes it because the class is declared there.
#   exports.h  declares no class and defines C entry points (clipper2's
#              `clipper.export.h`). The recipe lists it in `[wrap]
#              shim_headers` after the header it needs, so both are first in
#              every generated unit. It does not compile by itself, which is
#              what made the first version of this fix blame the clean
#              headers for it on the real clipper2.
#
# `impl.cpp` is the one unit meant to include them. With either in the forced
# unit the relink is "multiple definition of", and the build failed.
#
# Checked: the build succeeds; both headers are left out of the unit and
# named; the class declared in impl.h is skipped with the reason; the classes
# declared in clean headers still get their constructors and member functions,
# exported. And, in a child with the detection switched off, the relink fails
# as it did and the build still comes through, without the forced unit and
# with every row saying why.
#
# Needs clang++ (the build of a C++ library).
# =============================================================================

using Test
using RepliBuild
using Libdl

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const DH_DIR = joinpath(@__DIR__, "defining_header_test")

# Written by the test and by the child: the same project either way.
const DH_PROJECT = """
using RepliBuild
function dh_project(dir)
    for d in ("build", "julia", ".replibuild_cache", ".debug")
        p = joinpath(dir, d)
        ispath(p) && RepliBuild._rm_tree(p)
    end
    toml_path = joinpath(dir, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, \"\"\"
    [project]
    name = "defining_header"
    root = "\$(escape_string(dir))"

    [compile]
    flags = ["-std=c++17", "-O1", "-fPIC", "-fstandalone-debug"]
    source_files = ["\$(escape_string(joinpath(dir, "impl.cpp")))"]
    include_dirs = ["\$(escape_string(dir))"]

    [link]
    enable_lto = false
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    language = "cpp"
    shim_headers = ["widget.h", "exports.h"]

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    \"\"\")
    return toml_path
end
"""
include_string(@__MODULE__, DH_PROJECT)

dh_classes(plan, key) = sort!(String[String(f["class"]) for f in get(plan, key, Any[])])
dh_reason(plan, class) = something((String(f["reason"]) for f in get(plan, "skipped", Any[]) if f["class"] == class)..., "")
# Opened and closed again: the child below builds the same project, and Windows
# deletes neither a loaded DLL nor the folder it is in.
function dh_exports(lib, sym)
    h = Libdl.dlopen(lib)
    try
        return Libdl.dlsym(h, Symbol(sym); throw_error = false) !== nothing
    finally
        Libdl.dlclose(h)
    end
end

if requires("a header that defines functions and the forced unit, live", :clangxx)
@testset "A header that defines functions, and the forced unit, live" begin
    toml_path = dh_project(DH_DIR)
    out = Pipe()
    lib = withenv("REPLIBUILD_JULIA_ENV" => "none") do
        redirect_stdout(out) do
            RepliBuild.build(toml_path)
        end
    end
    close(out.in)
    log = read(out, String)
    @test isfile(lib)
    md = RepliBuild._read_json(joinpath(DH_DIR, "julia", "compilation_metadata.json"))
    ctors = md["inline_constructors"]
    methods = md["inline_methods"]

    @testset "the two defining headers are left out, and named" begin
        @test occursin("forced unit: left out 2 header(s)", log)
        @test occursin("exports.h, impl.h", log)
        unit = read(joinpath(DH_DIR, ".replibuild_cache", "replibuild_ctors.cpp"), String)
        @test !occursin("exports.h", unit)
        @test !occursin("impl.h\"", unit)
        @test occursin("shape.h", unit) && occursin("widget.h", unit)
        # The umbrella header before the export header stays, and stays first.
        @test findfirst("widget.h", unit).start < findfirst("shape.h", unit).start
    end

    @testset "the class declared in the defining header is skipped with the reason" begin
        @test "Gadget" in dh_classes(ctors, "skipped")
        @test "Gadget" in dh_classes(methods, "skipped")
        why = dh_reason(ctors, "Gadget")
        @test occursin("its header defines functions the library's own unit defines", why)
        @test occursin("impl.h", why)
        # The symbol named is one of the two the header defines.
        @test occursin("gadget_total(Gadget const&)", why) || occursin("gadget_count", why)
    end

    @testset "the classes in clean headers are forced, and exported" begin
        @test dh_classes(ctors, "forced") == ["Shape", "Widget"]
        @test dh_classes(methods, "forced") == ["Shape", "Widget"]
        @test dh_exports(lib, "_ZN5ShapeC2Ed") || dh_exports(lib, "_ZN5ShapeC1Ed")
        @test dh_exports(lib, "_ZN6WidgetC2Ei") || dh_exports(lib, "_ZN6WidgetC1Ei")
        @test dh_exports(lib, "_ZNK5Shape4areaEv")
        @test dh_exports(lib, "_ZNK6Widget5twiceEv")
        # The library's own definitions are there, once.
        @test dh_exports(lib, "export_version")
        @test dh_exports(lib, "_Z12gadget_totalRK6Gadget")
        @test !dh_exports(lib, "_ZN6GadgetC2Ei") && !dh_exports(lib, "_ZN6GadgetC1Ei")
    end

    @testset "with the detection off, the build falls back to the library without the unit" begin
        code = DH_PROJECT * """
        # The detection off, by its exact method: the relink then fails as it
        # did before, and the build has to come through on the fallback.
        @eval RepliBuild.Compiler _foreign_strong_definitions(ir_path::AbstractString)::Vector{String} = String[]
        using Libdl
        probe(name, ok, detail = "") = (println("PROBE ", name, ": ", ok ? "PASS" : "FAIL " * string(detail)); flush(stdout))
        toml = dh_project($(repr(DH_DIR)))
        lib = try
            redirect_stdout(devnull) do
                RepliBuild.build(toml)
            end
        catch e
            probe("built", false, first(split(sprint(showerror, e), '\\n')))
            nothing
        end
        if lib !== nothing
            probe("built", isfile(lib))
            md = RepliBuild._read_json(joinpath($(repr(DH_DIR)), "julia", "compilation_metadata.json"))
            c = md["inline_constructors"]; m = md["inline_methods"]
            probe("nothing_forced", isempty(c["forced"]) && isempty(m["forced"]), (c["forced"], m["forced"]))
            classes = sort!(String[String(f["class"]) for f in c["skipped"]])
            probe("all_reported", classes == ["Gadget", "Shape", "Widget"], classes)
            reasons = unique(String[String(f["reason"]) for f in vcat(c["skipped"], m["skipped"])])
            # The linker's own words: GNU ld's "multiple definition", lld's "duplicate symbol".
            probe("reason", length(reasons) == 1 && startswith(reasons[1], "the library did not link with the forced unit: ") &&
                            (occursin("multiple definition", reasons[1]) || occursin("duplicate symbol", reasons[1])), reasons)
            h = Libdl.dlopen(lib)
            has(s) = Libdl.dlsym(h, Symbol(s); throw_error = false) !== nothing
            probe("library_as_first_linked", has("export_version") && has("_Z12gadget_totalRK6Gadget") &&
                                             !has("_ZN5ShapeC2Ed") && !has("_ZNK5Shape4areaEv"))
        end
        println("PROBE_DONE")
        """
        r = withenv("REPLIBUILD_JULIA_ENV" => "none") do
            run_isolated(`-e $code`; timeout = 900)
        end
        v = probe_verdicts(r.output)
        @test r.ok && haskey(v, "PROBE_DONE")
        for name in ("built", "nothing_forced", "all_reported", "reason", "library_as_first_linked")
            @test (name, startswith(get(v, name, "no PROBE line"), "PASS")) == (name, true)
        end
    end
end
end # requires
