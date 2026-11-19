# test_rust_wrap_live.jl — a Rust crate used from Julia through its generated module.
#
# test/rust_test/crate is built and wrapped (`RepliBuild.build`, `RepliBuild.wrap`),
# and test/rust_test/verify_module.jl loads the module in a child process and
# calls it as a user would: Julia values in, Julia values out, a Rust panic as
# a Julia exception, and a stub that says why for what is not wrapped yet.
# Unoptimized and optimized: the two builds route the same functions
# differently (nothing is proved not to panic in the first).
#
# Then the registry: `register` builds, wraps and vendors the project, `use`
# loads the vendored module, and an edit to the crate's source is a new build.

using Test
using Libdl
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const _RWL_DIR = joinpath(@__DIR__, "rust_test")
const _RWL_TOML = joinpath(_RWL_DIR, "replibuild.toml")

_rwl_quiet(f) = withenv(() -> redirect_stdout(f, devnull), "REPLIBUILD_JULIA_ENV" => "none")

if requires("Rust wrapper (live)", :libJLCS, :cargo, :rustc, :nm)

@testset "Rust wrapper (live)" begin
    for opt in (0, 3)
        @testset "opt-level $opt" begin
            # test/*/*.toml is untracked, so the project file is written here.
            write(_RWL_TOML, "[project]\nname = \"rb_fix\"\n\n[rust]\nmanifest = \"crate/Cargo.toml\"\nopt_level = $opt\n")
            lib = _rwl_quiet(() -> RepliBuild.build(_RWL_TOML))
            wrapper = _rwl_quiet(() -> RepliBuild.wrap(_RWL_TOML))
            @test wrapper == joinpath(dirname(lib), "RbFix.jl") && isfile(wrapper)
            md = RepliBuild._read_json(joinpath(dirname(lib), "compilation_metadata.json"))
            text = read(wrapper, String)
            @test occursin("module RbFix", text) && occursin("const LIBRARY_SHA = \"$(md["library_sha"])\"", text)
            # Every function of the library is in the module, wrapped or with its reason.
            (_, report) = RepliBuild.GeneratorRust.generate(md; module_name = "RbFix", library_path = lib)
            @test length(report) == length(md["functions"])
            unsupported = sort!([r["name"] for r in report if r["tier"] == "unsupported"])
            # What is left: two calls the build refused, a trait object, and
            # the one type no path outside the crate names.
            @test unsupported == ["cpoint_norm", "dyn_area", "located", "secret"]
            # Unoptimized, every Rust function goes through the catching frame;
            # optimized, the ones that cannot panic do not.
            tiers = Dict(r["name"] => r["tier"] for r in report)
            @test tiers["checked_div"] == "guarded"
            @test tiers["add"] == (opt == 0 ? "guarded" : "ccall")
            @test tiers["maybe"] == (opt == 0 ? "guarded" : "thunk")

            r = run_isolated(`$(joinpath(_RWL_DIR, "verify_module.jl")) $wrapper`; timeout = 600)
            v = probe_verdicts(r.output)
            @test r.ok
            @test haskey(v, "PROBE_DONE")
            expected = ["scalars", "integer_arguments_are_converted_or_refused", "strings_and_slices",
                        "tuples_come_back_as_tuples", "fixed_size_arrays", "a_slice_of_records",
                        "option_is_a_value_or_nothing", "records_by_value_and_by_reference",
                        "two_scalar_records", "fieldless_enums", "enums_with_data", "one_bit_values",
                        "methods_take_self_first", "optional_reference", "a_panic_is_an_exception",
                        "unsupported_functions_say_why", "how_each_function_calls",
                        "strings_are_copied_out_and_built_in", "a_string_rust_changes_in_place", "a_string_must_be_utf8",
                        "vectors_of_plain_data", "a_vector_rust_changes_in_place", "vectors_of_strings",
                        "result_is_a_value_or_an_exception", "option_and_tuple_of_owned_values",
                        "a_box_is_held_and_moved", "an_enum_that_owns", "a_record_that_owns", "dropped_exactly_once",
                        "a_raw_pointer_is_not_ownership", "what_the_module_says_it_owns"]
            for name in expected
                @test (name, startswith(get(v, name, "no PROBE line"), "PASS") ? "PASS" : get(v, name, "no PROBE line")) ==
                      (name, "PASS")
            end
            @test length(v) == length(expected) + 1
        end
    end

    @testset "register and use" begin
        mktempdir() do dir
            cp(joinpath(_RWL_DIR, "crate"), joinpath(dir, "crate"))
            toml = joinpath(dir, "replibuild.toml")
            write(toml, "[project]\nname = \"rb_fix\"\n\n[rust]\nmanifest = \"crate/Cargo.toml\"\nopt_level = 3\n")
            home = joinpath(dir, "home")
            code = """
            using RepliBuild
            RepliBuild.register($(repr(toml)))
            M = RepliBuild.use("rb_fix")
            println("PROBE use: ", (M.add(2, 3), M.maybe(41), M.hello()) == (Int32(5), Int32(42), "Hello, World!") ? "PASS" : "FAIL")
            println("PROBE owned: ", (M.make_greeting("hub"), M.squares(3), RepliBuild.RustRuntime.isowned(M.boxed(1))) ==
                                     ("hello hub", [0.0, 1.0, 4.0], true) ? "PASS" : "FAIL")
            println("PROBE vendored: ", startswith(M.LIBRARY_PATH, $(repr(joinpath(home, "builds")))) ? "PASS" : "FAIL \$(M.LIBRARY_PATH)")
            e = try; M.checked_div(1, 0); nothing; catch err; err; end
            println("PROBE panic: ", e isa RepliBuild.RustRuntime.RustPanic ? "PASS" : "FAIL \$e")
            println("PROBE_DONE")
            """
            run_use() = withenv("REPLIBUILD_HOME" => home, "REPLIBUILD_JULIA_ENV" => "none") do
                run_isolated(`-e $code`; timeout = 900)
            end
            r = run_use()
            v = probe_verdicts(r.output)
            @test r.ok && haskey(v, "PROBE_DONE")
            for name in ("use", "owned", "vendored", "panic")
                @test (name, get(v, name, "no PROBE line")) == (name, "PASS")
            end
            # One vendored build (beside it, `.debug` holds the thunk module's text for a debugger).
            vendored() = filter(f -> !startswith(f, "."), readdir(joinpath(home, "builds")))
            builds = vendored()
            @test length(builds) == 1
            @test sort!(filter(f -> !startswith(f, "."), readdir(joinpath(home, "builds", builds[1])))) ==
                  ["RbFix.jl", "compilation_metadata.json", "librb_fix.$(Libdl.dlext)", "rust_metadata.key"]
            # The crate's source is the project's input: an edit is another build.
            cfg() = RepliBuild.DependencyResolver.resolve_dependencies(RepliBuild.ConfigurationManager.load_config(toml))
            before = RepliBuild.PackageRegistry.hash_config(cfg())
            @test before == builds[1]
            src = joinpath(dir, "crate", "src", "lib.rs")
            write(src, replace(read(src, String), "\"Hello, World!\"" => "\"Hello, Julia!\""))
            @test RepliBuild.PackageRegistry.hash_config(cfg()) != before
            code2 = replace(code, "\"Hello, World!\"" => "\"Hello, Julia!\"", "RepliBuild.register($(repr(toml)))\n" => "")
            r2 = withenv("REPLIBUILD_HOME" => home, "REPLIBUILD_JULIA_ENV" => "none") do
                run_isolated(`-e $code2`; timeout = 900)
            end
            v2 = probe_verdicts(r2.output)
            @test r2.ok && get(v2, "use", "no PROBE line") == "PASS"
            @test length(vendored()) == 2
        end
    end
end

end # requires
