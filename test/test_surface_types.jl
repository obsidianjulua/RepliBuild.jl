#!/usr/bin/env julia
# test/test_surface_types.jl — [wrap] surface_types_only (2026-09-20,
# onednn 3.11.3 GENERATOR-wrap-all-dwarf-types.md).
#
# A wrapper's type set was "every type DWARF carries", which for any C++ library
# that compiles vendored internals into the same .so is a wildly different set
# from the one its API names. onednn: 2974 struct_definitions, 79 reachable.
# The other ~2900 are `dnnl::impl` and nGEN GPU-JIT types, and they do not
# merely bloat the module — they BREAK it (`const Integer = Pipe(3)` shadowing
# `Base.Integer`, duplicate `_` fields on nGEN instruction structs), so the
# wrapper does not load at all.
#
# OPT-IN, and these tests pin why it must stay opt-in: md4c's whole API is SAX
# callbacks, a `function_ptr(int)*` member erases its signature types, and its
# reachable set is genuinely 2 of 36 while it ships 231/231 on the other 34.
#
# Library-free: synthetic metadata, no toolchain.

using Test
using RepliBuild

const W = RepliBuild.Wrapper

# A library shaped like the real failure: an API function, a type reachable only
# through a member, a type reachable only from a global, and an internal the
# surface never names.
function fixture()
    Dict{String,Any}(
        "functions" => Any[
            Dict{String,Any}(
                "name" => "lib_open",
                "return_type" => Dict{String,Any}("julia_type" => "lib_status_t", "c_type" => "lib_status_t"),
                "parameters" => Any[
                    Dict{String,Any}("name" => "h", "julia_type" => "Ptr{Ptr{lib_handle}}", "c_type" => "lib_handle**"),
                ],
                "class" => "",
            ),
        ],
        "globals" => Dict{String,Any}(
            "lib_default_cfg" => Dict{String,Any}("name" => "lib_default_cfg",
                                                  "julia_type" => "lib_config", "c_type" => "const lib_config"),
            "internal_tbl"    => Dict{String,Any}("name" => "internal_tbl",
                                                  "julia_type" => "NTuple{4, Pipe}", "c_type" => "const Pipe[4]"),
        ),
        "struct_definitions" => Dict{String,Any}(
            "lib_handle"          => Dict{String,Any}("kind" => "struct", "members" => Any[
                                        Dict{String,Any}("name" => "d", "julia_type" => "lib_detail", "c_type" => "lib_detail")]),
            "lib_detail"          => Dict{String,Any}("kind" => "struct", "members" => Any[]),
            "__enum__lib_status_t"=> Dict{String,Any}("kind" => "enum", "enumerators" => Any[]),
            "lib_config"          => Dict{String,Any}("kind" => "struct", "members" => Any[]),
            "__enum__Pipe"        => Dict{String,Any}("kind" => "enum", "enumerators" => Any[]),
            "InternalNoise"       => Dict{String,Any}("kind" => "class", "members" => Any[]),
            "__enum__ErrCodes"    => Dict{String,Any}("kind" => "enum", "enumerators" => Any[]),
        ),
    )
end

@testset "[wrap] surface_types_only" begin

    @testset "reaches params, returns, and transitively through members" begin
        md = fixture()
        # Only lib_default_cfg resolves at runtime; internal_tbl does not.
        rd = Set(["lib_open", "lib_default_cfg"])
        r = W._apply_surface_type_filter!(md, String[]; runtime_defined=rd)
        sd = md["struct_definitions"]

        @test haskey(sd, "lib_handle")            # a parameter type
        @test haskey(sd, "__enum__lib_status_t")  # a return type, via __enum__ key
        @test haskey(sd, "lib_detail")            # ONLY reachable through a member
        @test haskey(sd, "lib_config")            # only reachable from a bindable global
        @test r.dropped == 3
        @test r.kept == 4
    end

    @testset "globals are gated on what the loader can resolve" begin
        # THE onednn BUG. `internal_tbl` is in the metadata but not in the .so's
        # dynamic symbol table — it is an internal hidden by -fvisibility=hidden.
        # Seeding from it drags in `Pipe`, whose `const Integer = Pipe(3)` alias
        # is the Base.Integer shadowing failure. Measured on the real package:
        # 332 globals recorded, 0 in dynsym, 58 extra types pulled in.
        md = fixture()
        W._apply_surface_type_filter!(md, String[]; runtime_defined=Set(["lib_open", "lib_default_cfg"]))
        @test !haskey(md["struct_definitions"], "__enum__Pipe")

        # And the converse: a global that DOES resolve keeps its type. box2d
        # binds all 16 of its globals and would break without this.
        md2 = fixture()
        W._apply_surface_type_filter!(md2, String[]; runtime_defined=Set(["lib_open", "lib_default_cfg", "internal_tbl"]))
        @test haskey(md2["struct_definitions"], "__enum__Pipe")
        @test haskey(md2["struct_definitions"], "lib_config")
    end

    @testset "unknown runtime symbols fail SAFE" begin
        # `nothing` = could not look (no .so, nm missing, PE without exports).
        # It must NOT be read as "nothing is bindable" — that would silently
        # drop types the wrapper still emits. Seed from every global instead.
        md = fixture()
        W._apply_surface_type_filter!(md, String[]; runtime_defined=nothing)
        @test haskey(md["struct_definitions"], "__enum__Pipe")   # over-included, deliberately
        @test haskey(md["struct_definitions"], "lib_config")
    end

    @testset "keep-list rescues what reachability cannot see" begin
        # The md4c class: a constant-only enum is never named by any signature
        # because the function returns plain `int`.
        md = fixture()
        r = W._apply_surface_type_filter!(md, ["ErrCodes"]; runtime_defined=Set(["lib_open"]))
        # Matched by BARE name even though the key carries the __enum__ prefix.
        @test haskey(md["struct_definitions"], "__enum__ErrCodes")
        @test isempty(r.unmatched)

        md2 = fixture()
        r2 = W._apply_surface_type_filter!(md2, ["Internal*"]; runtime_defined=Set(["lib_open"]))
        @test haskey(md2["struct_definitions"], "InternalNoise")
        @test isempty(r2.unmatched)
    end

    @testset "a keep-pattern matching nothing is reported, not swallowed" begin
        # Worse than a stale exclude: this list exists to HOLD ON to types, so a
        # silent miss drops API. The caller turns this into a hard error.
        md = fixture()
        r = W._apply_surface_type_filter!(md, ["NoSuchType*"]; runtime_defined=Set(["lib_open"]))
        @test r.unmatched == ["NoSuchType*"]
    end

    @testset "internals with no path from the surface are dropped" begin
        md = fixture()
        W._apply_surface_type_filter!(md, String[]; runtime_defined=Set(["lib_open"]))
        @test !haskey(md["struct_definitions"], "InternalNoise")
        @test !haskey(md["struct_definitions"], "__enum__ErrCodes")
    end

    @testset "a C++ method's receiver class seeds the walk" begin
        # Without this `std::vector<int>` is unreachable from its own methods,
        # because a method records its receiver in `class`, not as a parameter.
        # Measured on the Hub: box2d3 41.8% -> 80.4%, glfw 49.3% -> 90.4%.
        md = Dict{String,Any}(
            "functions" => Any[Dict{String,Any}(
                "name" => "size", "class" => "vector<int>",
                "return_type" => Dict{String,Any}("julia_type" => "Csize_t", "c_type" => "size_t"),
                "parameters" => Any[])],
            "globals" => Dict{String,Any}(),
            "struct_definitions" => Dict{String,Any}(
                "vector<int>" => Dict{String,Any}("kind" => "class", "members" => Any[]),
                "Unrelated"   => Dict{String,Any}("kind" => "class", "members" => Any[])),
        )
        W._apply_surface_type_filter!(md, String[]; runtime_defined=Set(["size"]))
        @test haskey(md["struct_definitions"], "vector<int>")
        @test !haskey(md["struct_definitions"], "Unrelated")
    end

    @testset "config plumbing" begin
        # Both keys must survive a TOML round trip and be PRESERVED across a
        # forced re-discovery — losing them regenerates a wrapper that is broken
        # in a way the build never reports.
        @test ("wrap", "surface_types_only") in RepliBuild.Discovery.PRESERVED_TOML_KEYS
        @test ("wrap", "surface_types_extra") in RepliBuild.Discovery.PRESERVED_TOML_KEYS

        dir = mktempdir()
        try
            toml = joinpath(dir, "replibuild.toml")
            write(toml, """
            [project]
            name = "probe"
            version = "0.1.0"

            [wrap]
            language = "c"
            surface_types_only = true
            surface_types_extra = ["MD_*_DETAIL", "ErrCodes"]
            """)
            cfg = RepliBuild.ConfigurationManager.load_config(toml)
            @test cfg.wrap.surface_types_only === true
            @test cfg.wrap.surface_types_extra == ["MD_*_DETAIL", "ErrCodes"]

            # Round trip through the serializer: a key that parses but is never
            # written back is lost on the next save, which is the same silent
            # class as a key that is parsed and never read.
            RepliBuild.ConfigurationManager.save_config(cfg)
            cfg2 = RepliBuild.ConfigurationManager.load_config(toml)
            @test cfg2.wrap.surface_types_only === true
            @test cfg2.wrap.surface_types_extra == ["MD_*_DETAIL", "ErrCodes"]
        finally
            rm(dir; recursive=true, force=true)
        end
    end

    @testset "default is OFF, and that is load-bearing" begin
        # md4c is the reason. Its reachable set is 2 of 36 — every MD_*_DETAIL
        # struct a user needs to write a callback is invisible, because
        # `function_ptr(int)*` members erase their signature types. Flipping this
        # default on would silently gut a package that ships 231/231 today.
        dir = mktempdir()
        try
            toml = joinpath(dir, "replibuild.toml")
            write(toml, """
            [project]
            name = "probe"
            version = "0.1.0"

            [wrap]
            language = "c"
            """)
            cfg = RepliBuild.ConfigurationManager.load_config(toml)
            @test cfg.wrap.surface_types_only === false
            @test isempty(cfg.wrap.surface_types_extra)
        finally
            rm(dir; recursive=true, force=true)
        end
    end

    @testset "globals absent from the symbol table get no accessor" begin
        # NOT behind surface_types_only, and that is the point: an accessor for
        # a symbol that is not in the library is a guaranteed failure at the
        # call site, never a policy choice about how much surface to expose.
        #
        # Verified against the real packages before this was written. onednn
        # emitted 656 such functions — two thirds of its wrapper — and llamacpp
        # 4099, uniformly: 0 of 4099 are in its dynsym, no exceptions, so this
        # is not a hit-and-miss heuristic. Both fail today, two ways depending
        # on whether the type survived alongside the name:
        #   ggml_table_gelu_f16() -> could not load symbol "ggml_table_gelu_f16"
        #   unicode_ranges_nfd()  -> UndefVarError: initializer_list_range_nfd
        md = fixture()
        n = W._drop_unresolvable_globals!(md, Set(["lib_open", "lib_default_cfg"]))
        @test n == 1
        @test haskey(md["globals"], "lib_default_cfg")   # resolves -> kept
        @test !haskey(md["globals"], "internal_tbl")     # absent  -> dropped

        # Nothing to drop when every global resolves: box2d 16/16, curl 95/95,
        # pcre2 31/31 are untouched by this.
        md2 = fixture()
        @test W._drop_unresolvable_globals!(md2, Set(["lib_open", "lib_default_cfg", "internal_tbl"])) == 0
        @test length(md2["globals"]) == 2
    end

    @testset "unreadable symbol table drops NOTHING" begin
        # `nothing` = could not look (PE with no export directory, no library on
        # disk, nm unavailable). A global wrongly kept costs one broken
        # accessor; a global wrongly dropped is missing API, so the safe
        # direction is to keep everything.
        md = fixture()
        @test W._drop_unresolvable_globals!(md, nothing) == 0
        @test length(md["globals"]) == 2
    end

    @testset "dropping a global also stops it seeding a type" begin
        # Ordering: the global drop runs BEFORE the type filter, so an
        # unresolvable global cannot pull its type into the module. This is how
        # onednn's `Pipe` leaves — it is reachable ONLY through `pipeMap`.
        md = fixture()
        rd = Set(["lib_open", "lib_default_cfg"])
        W._drop_unresolvable_globals!(md, rd)
        W._apply_surface_type_filter!(md, String[]; runtime_defined=rd)
        @test !haskey(md["struct_definitions"], "__enum__Pipe")
        @test haskey(md["struct_definitions"], "lib_config")
    end
end
