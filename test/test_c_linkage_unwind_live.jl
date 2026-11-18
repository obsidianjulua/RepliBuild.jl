# =============================================================================
# C linkage in a C++ library, live (2026-11-17)
#
# `extern "C" int c_linkage_throws(int x)` in a C++ library was bound as a
# plain ccall because its name is unmangled, and its `throw` ended the process:
# "terminate called after throwing an instance of 'std::runtime_error'",
# SIGABRT. A C-linkage function now reaches ccall only when its definition is
# `nounwind` in the linked IR; any other takes a Tier 2 thunk, and the throw
# arrives as `RepliBuild.JITManager.CxxException`.
#
# This builds test/c_linkage_unwind_test with the system compiler: six
# `extern "C"` functions that throw on a negative argument (an int, a void, a
# `const char*`, a record by value and a function-pointer caller among them),
# two that cannot throw, and one defined in a `.c` file. It is built at link
# level 0 and at 2, since the optimizer proves more definitions `nounwind` and
# must never prove these. Each wrapper is called from a child, throws included;
# on the old routing the child aborts at its first throw. Needs clang++ and
# libJLCS. The routing rule itself: test_c_linkage_unwind.jl.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const CL_UNWIND_DIR = joinpath(@__DIR__, "c_linkage_unwind_test")

if requires("C linkage in a C++ library, live", :libJLCS, :clangxx)
@testset "C linkage in a C++ library, live" begin
    toml_path = joinpath(CL_UNWIND_DIR, "replibuild.toml")
    throwing = ["cl_apply", "cl_checked", "cl_checked_void", "cl_name", "cl_point", "cl_throws_int"]
    quiet = ["cl_add", "cl_calls", "cl_plain_c"]

    for level in ("0", "2")
        @testset "link optimization_level = $level" begin
            for d in ("build", "julia", ".replibuild_cache", ".debug")
                p = joinpath(CL_UNWIND_DIR, d)
                ispath(p) && RepliBuild._rm_tree(p)
            end
            # Regenerated each run: it holds absolute, machine-local paths.
            write(toml_path, """
            [project]
            name = "clunwind"
            # escape_string on every interpolated path: TOML basic strings take escapes.
            root = "$(escape_string(CL_UNWIND_DIR))"

            [compile]
            flags = ["-fPIC", "-O$level"]
            source_files = ["$(escape_string(joinpath(CL_UNWIND_DIR, "unwind_api.cpp")))",
                            "$(escape_string(joinpath(CL_UNWIND_DIR, "unwind_plain.c")))"]
            include_dirs = ["$(escape_string(CL_UNWIND_DIR))"]

            [link]
            enable_lto = false
            optimization_level = "$level"

            [binary]
            type = "shared"

            [wrap]
            language = "cpp"

            [types]
            strictness = "warn"
            allow_unknown_structs = true
            allow_function_pointers = true

            [cache]
            enabled = false
            """)
            redirect_stdout(devnull) do
                RepliBuild.build(toml_path)
            end
            RepliBuild.wrap(toml_path)
            wrapper = joinpath(CL_UNWIND_DIR, "julia", "Clunwind.jl")
            @test isfile(wrapper)

            # The compiler's answer, as the metadata recorded it.
            meta = RepliBuild._read_json(joinpath(CL_UNWIND_DIR, "julia", "compilation_metadata.json"))
            proof = Dict(String(f["mangled"]) => get(f, "nounwind", nothing) for f in meta["functions"])
            for name in throwing
                @test (name, get(proof, name, :absent)) == (name, false)
            end
            for name in quiet
                @test (name, get(proof, name, :absent)) == (name, true)
            end

            # What the wrapper does with it: a thunk for each function that may
            # throw, a plain ccall for each that cannot.
            txt = read(wrapper, String)
            body(name) = (m = match(Regex("^function $(name)(?:_ptr)?\\(.*?^end\$", "ms"), txt);
                          m === nothing ? "" : String(m.match))
            for name in throwing
                @test occursin("JITManager.invoke(\"_mlir_ciface_$(name)_thunk\"", body(name))
                @test !occursin("ccall((:$name,", txt)
            end
            for name in quiet
                @test occursin("ccall((:$name, LIBRARY_PATH)", body(name))
            end
            manifest = RepliBuild._read_json(joinpath(CL_UNWIND_DIR, "julia", "thunk_manifest.json"))
            @test sort!(String.(manifest["function_thunks"])) == throwing

            # Every call, throws included, from a child: the old routing ends
            # it with SIGABRT at the first throw.
            r = run_isolated(`$(joinpath(CL_UNWIND_DIR, "probe_c_linkage_unwind.jl")) $wrapper`)
            v = probe_verdicts(r.output)
            @test r.ok
            @test haskey(v, "PROBE_DONE")
            names = ["throw_arrives_as_CxxException", "returns_through_the_thunk", "void_return",
                     "cstring_return", "record_return", "function_pointer_argument",
                     "throw_of_a_non_exception", "proven_nounwind_keeps_ccall", "alive_after_many_throws"]
            for n in names
                @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
            end
            @test count(n -> haskey(v, n), names) == length(names)
        end
    end
end
end # requires
