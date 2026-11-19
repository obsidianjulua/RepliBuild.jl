# =============================================================================
# C++ globals outside the global namespace, live (B19, 2026-11-18)
#
# A namespace member (`cfg::depth`) and a static data member (`Stat::count`)
# are exported under mangled symbols, and the metadata looked them up by their
# bare DWARF names: the wrap dropped them as "not in the library's dynamic
# symbol table", and static data members were not recorded at all. This builds
# test/cpp_globals_test at DWARF 5 and at DWARF 4 (a static member is
# declared as a DW_TAG_variable in the one and a DW_TAG_member in the other),
# wraps it, and reads and writes the globals from a child. On Linux it also
# reads each library with libdw and asks for the readelf answer.
# test_cpp_globals.jl (unit) holds the parser and the emitted text.
# Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const CPP_GLOBALS_DIR = joinpath(@__DIR__, "cpp_globals_test")

if requires("C++ globals outside the global namespace, live", :libJLCS, :clangxx)
@testset "C++ globals outside the global namespace, live" begin
    toml_path = joinpath(CPP_GLOBALS_DIR, "replibuild.toml")
    src = joinpath(CPP_GLOBALS_DIR, "src", "cpp_globals.cpp")
    inc = joinpath(CPP_GLOBALS_DIR, "include")
    ran = 0
    for (label, dwarf) in (("DWARF 5", "-gdwarf-5"), ("DWARF 4", "-gdwarf-4"))
        @testset "$label" begin
            # Regenerated each run: it holds absolute, machine-local paths.
            write(toml_path, """
            [project]
            name = "cppglobals"
            # escape_string on every interpolated path: TOML basic strings take escapes.
            root = "$(escape_string(CPP_GLOBALS_DIR))"

            [compile]
            flags = ["-std=c++17", "-fPIC", "$dwarf"]
            source_files = ["$(escape_string(src))"]
            include_dirs = ["$(escape_string(inc))"]

            [link]
            enable_lto = false

            [binary]
            type = "shared"

            [wrap]
            language = "cpp"

            [types]
            strictness = "warn"
            allow_unknown_structs = true

            [cache]
            enabled = false
            """)
            redirect_stdout(devnull) do
                RepliBuild.build(toml_path)
            end
            out = mktemp() do path, io
                redirect_stdout(io) do
                    RepliBuild.wrap(toml_path)
                end
                close(io)
                read(path, String)
            end
            wrapper = joinpath(CPP_GLOBALS_DIR, "julia", "Cppglobals.jl")
            @test isfile(wrapper)

            @testset "the metadata keys each global by its symbol, with its scope" begin
                md = RepliBuild._read_json(joinpath(CPP_GLOBALS_DIR, "julia", "compilation_metadata.json"))
                g = md["globals"]
                scope_of(sym) = haskey(g, sym) ? (g[sym]["name"], g[sym]["scope"]) : nothing
                @test scope_of("_ZN1a1xE") == ("x", "a")
                @test scope_of("_ZN1b1xE") == ("x", "b")
                @test scope_of("_ZN3cfg5depthE") == ("depth", "cfg")
                @test scope_of("_ZN4Stat5countE") == ("count", "Stat")
                @test scope_of("_ZN3geo4Vec310live_countE") == ("live_count", "geo::Vec3")
                @test scope_of("g_top") == ("g_top", "")
                # The metadata keeps a private member and says so; the wrap drops it.
                @test get(get(g, "_ZN3geo7Counter6total_E", Dict()), "accessibility", nothing) == "private"
                @test get(get(g, "_ZN4Stat5prot_E", Dict()), "accessibility", nothing) == "protected"
                @test md["global_constants"] == ["Stat::K"]
            end

            @testset "the wrap says what it passed over" begin
                @test occursin("2 private or protected static data member(s) get no accessor", out)
                @test occursin("1 static member constant(s) have a value and no symbol", out)
            end

            @testset "used from a child" begin
                r = run_isolated(`$(joinpath(CPP_GLOBALS_DIR, "probe_cpp_globals.jl")) $wrapper`)
                v = probe_verdicts(r.output)
                @test r.ok
                @test haskey(v, "PROBE_DONE")
                cases = ["namespaced", "same_name_two_scopes", "static_member", "live_count",
                         "string_pointer_only", "not_offered"]
                for n in cases
                    @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
                end
                @test count(n -> haskey(v, n), cases) == length(cases)
            end

            if Sys.islinux() && RepliBuild.LibDW.available()
                @testset "libdw reads the same globals" begin
                    lib = joinpath(CPP_GLOBALS_DIR, "julia", "libcppglobals.so")
                    read_with(reader) = withenv(() -> RepliBuild.Compiler.extract_dwarf_return_types(lib)[3],
                                                "REPLIBUILD_DWARF" => reader)
                    from_readelf = read_with("readelf")
                    @test haskey(from_readelf, "_ZN3geo4Vec310live_countE")
                    @test from_readelf == read_with("libdw")
                end
            end
            ran += 1
        end
    end
    @test ran == 2
end
end # requires
