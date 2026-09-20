# test_exclude_symbols.jl — `[wrap] exclude_symbols`
#
# The wrapper's surface is otherwise "every exported symbol the binary has", and
# for a library that links in GENERATED or vendored C++ that is the wrong set.
# llamacpp is the motivating case: enabling ggml's Vulkan backend adds 136
# generated translation units whose only contents are SPIR-V byte arrays —
# ~1,958 `matmul_*_data` / `matmul_*_len` globals in the largest one alone —
# plus the whole `vk::*Error` class family out of vulkan.hpp, vtables and
# non-virtual thunks included. None of it is API.
#
# Before this existed the only lever was `-fvisibility=hidden`, which is blunt
# and conditional: it works only if upstream annotated its exports AND the
# annotation macro is active in the build. ggml gates GGML_API behind
# `#ifdef GGML_SHARED`, so the flag alone hid llamacpp's ENTIRE public API —
# a 101 MB .so, a wrapper larger than the previous run, exit code 0, and an
# UndefVarError at the first call. `verify_wrap_surface` R1 now refuses that
# build; this filter removes the reason to reach for the flag in the first place.
#
# Pure Julia — no toolchain, no binary. Lives in runtests.jl.

using Test
using RepliBuild
using TOML

const W  = RepliBuild.Wrapper
const CM = RepliBuild.ConfigurationManager

@testset "[wrap] exclude_symbols" begin

    @testset "glob compilation" begin
        # `*` and `?` are the only wildcards.
        @test occursin(W._glob_to_regex("matmul_*"), "matmul_f32_data")
        @test occursin(W._glob_to_regex("matmul_*"), "matmul_")
        @test occursin(W._glob_to_regex("foo?"), "foo1")
        @test !occursin(W._glob_to_regex("foo?"), "foo")
        @test !occursin(W._glob_to_regex("foo?"), "foo12")

        # ANCHORED at both ends. A pattern names a whole symbol, so a short
        # pattern cannot silently take out a longer unrelated name.
        @test !occursin(W._glob_to_regex("vk"), "vkontakte_init")
        @test !occursin(W._glob_to_regex("matmul_"), "matmul_f32_data")

        # Every other metacharacter is a LITERAL. These are written against nm
        # output, where `(`, `+`, `.`, `<` and `[` are ordinary symbol text; if
        # the input were treated as a regex, `operator+` would compile to "one or
        # more `operator`" and match nothing, with no error to read.
        @test occursin(W._glob_to_regex("operator+"), "operator+")
        @test !occursin(W._glob_to_regex("operator+"), "operator++")
        @test occursin(W._glob_to_regex("std::vector<int>"), "std::vector<int>")
        @test !occursin(W._glob_to_regex("std::vector<int>"), "std::vector<long>")
        @test occursin(W._glob_to_regex("a.b"), "a.b")
        @test !occursin(W._glob_to_regex("a.b"), "axb")
        @test occursin(W._glob_to_regex("f(int)"), "f(int)")
        @test occursin(W._glob_to_regex("x[0]"), "x[0]")
        @test occursin(W._glob_to_regex("a|b"), "a|b")
        @test !occursin(W._glob_to_regex("a|b"), "a")
    end

    # A function is matched on any of its three spellings: which one a user can
    # actually see depends on the symbol. A C function is all three at once; a
    # C++ method is legible only demangled and addressable only mangled.
    _md() = Dict{String,Any}(
        "functions" => Any[
            Dict{String,Any}("name" => "llama_decode", "demangled" => "llama_decode",
                             "mangled" => "llama_decode"),
            Dict{String,Any}("name" => "~DeviceLostError",
                             "demangled" => "vk::DeviceLostError::~DeviceLostError()",
                             "mangled" => "_ZN2vk15DeviceLostErrorD2Ev"),
            Dict{String,Any}("name" => "ggml_init", "demangled" => "ggml_init",
                             "mangled" => "ggml_init"),
        ],
        "globals" => Dict{String,Any}("matmul_f32_data" => 1, "matmul_f32_len" => 2,
                                      "llama_max_devices" => 3),
        "function_count" => 3,
    )

    @testset "filters functions and globals" begin
        md = _md()
        r = W._apply_symbol_filter!(md, ["vk::*", "matmul_*"])

        @test r.functions_dropped == 1
        @test r.globals_dropped == 2
        @test isempty(r.unmatched)
        @test [f["name"] for f in md["functions"]] == ["llama_decode", "ggml_init"]
        @test sort(collect(keys(md["globals"]))) == ["llama_max_devices"]

        # function_count is what downstream reads for the empty-surface guard;
        # leaving it stale would make the guard compare against the pre-filter
        # number and never fire.
        @test md["function_count"] == length(md["functions"]) == 2
    end

    @testset "matches on mangled spelling too" begin
        md = _md()
        r = W._apply_symbol_filter!(md, ["_ZN2vk*"])
        @test r.functions_dropped == 1
        @test isempty(r.unmatched)
        @test !any(f -> f["name"] == "~DeviceLostError", md["functions"])
    end

    @testset "no patterns is a no-op" begin
        md = _md()
        r = W._apply_symbol_filter!(md, String[])
        @test r.functions_dropped == 0 && r.globals_dropped == 0
        @test length(md["functions"]) == 3
        @test length(md["globals"]) == 3
    end

    # A pattern matching nothing is the stale-filter failure: the symbol it named
    # was renamed or dropped upstream, or the glob was mistyped. Either way the
    # author believes something is excluded that is not, and the wrapper quietly
    # regrows a surface someone deliberately cut. The caller turns this into an
    # error; what is asserted here is that it is REPORTED rather than swallowed.
    @testset "stale patterns are reported" begin
        md = _md()
        r = W._apply_symbol_filter!(md, ["vk::*", "nope_*", "also_absent"])
        @test sort(r.unmatched) == ["also_absent", "nope_*"]
        @test r.functions_dropped == 1   # the live pattern still applied
    end

    # Hit counting must not short-circuit: if `matches` stopped at the first
    # pattern that fired, a broader glob listed earlier would make a narrower one
    # look stale purely because of ordering.
    @testset "overlapping patterns each count a hit" begin
        md = _md()
        r = W._apply_symbol_filter!(md, ["matmul_*", "matmul_f32_len"])
        @test isempty(r.unmatched)
        @test r.globals_dropped == 2
    end

    @testset "config parse and round-trip" begin
        mktempdir() do dir
            toml = joinpath(dir, "replibuild.toml")
            write(toml, """
            [project]
            name = "xs"
            root = "."

            [wrap]
            language = "cpp"
            exclude_symbols = ["matmul_*", "vk::*"]
            """)
            cfg = CM.load_config(toml)
            @test cfg.wrap.exclude_symbols == ["matmul_*", "vk::*"]

            # Round-trip: save_config rewrites config.config_file in place. A
            # filter that does not survive that is silently dropped the next time
            # anything regenerates the manifest.
            CM.save_config(cfg)
            @test TOML.parsefile(toml)["wrap"]["exclude_symbols"] == ["matmul_*", "vk::*"]
            @test CM.load_config(toml).wrap.exclude_symbols == ["matmul_*", "vk::*"]
        end
    end

    @testset "absent key defaults to empty" begin
        mktempdir() do dir
            toml = joinpath(dir, "replibuild.toml")
            write(toml, """
            [project]
            name = "xs"
            root = "."

            [wrap]
            language = "c"
            """)
            cfg = CM.load_config(toml)
            @test cfg.wrap.exclude_symbols == String[]
            # An absent filter must not appear in the serialized form either —
            # an empty `exclude_symbols = []` in every generated manifest is noise.
            CM.save_config(cfg)
            @test !haskey(get(TOML.parsefile(toml), "wrap", Dict()), "exclude_symbols")
        end
    end
end
