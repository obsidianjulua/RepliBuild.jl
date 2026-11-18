#!/usr/bin/env julia
# A symbol no DIE describes: an alias takes its aliasee's signature, anything
# else is refused.
#
#     int aliased_real(int x) { return x + 1; }
#     int aliased_name(int x) __attribute__((alias("aliased_real")));
#
# clang writes no DIE for `aliased_name`, so its metadata entry was inferred
# from the demangled name, which for a C symbol is the name: `"parameters":
# []`, `Cint`, `parameters_source` and `return_type_source` both `inferred`.
# The wrapper bound `aliased_name()::Cint`, a live call that passed nothing and
# returned what the callee made of a stale register (2026-11-17).
#
# The symbol table has both names at one address, and the code there is one
# function's: `extract_compilation_metadata` gives a symbol with no DIE the
# signature of a described function at its address (`_alias_signature_key`,
# recorded as `alias_of`). One with no such function stays `inferred`, and both
# generators emit a refusal for it that says there is no debug information
# (`_lacks_debug_info`, `_no_debug_info_stub`), never a call. A function with a
# DIE and no prototype (K&R `int old_style()`) has its return type from DWARF
# and is called as before. Toolchain-free; the live half is
# test_alias_signatures_live.jl.
#
# Negative checks: without the `_lacks_debug_info` branch in either generator
# the "refused" testsets find a ccall; with `_alias_signature_key` returning
# `nothing` the key testset fails.

using Test
using RepliBuild
using Libdl

const ALS_C = RepliBuild.Compiler
const ALS_W = RepliBuild.Wrapper

_als_param(name, c, jl) = Dict{String,Any}("name" => name, "c_type" => c, "julia_type" => jl)
_als_fn(name, mangled; params = Any[], ret = ("int", "Cint"), rsrc = "dwarf", psrc = "dwarf", extra...) =
    merge(Dict{String,Any}(
        "name" => name, "mangled" => mangled, "demangled" => startswith(mangled, "_Z") ? "$name(int)" : name,
        "class" => "", "is_method" => false, "is_vararg" => false, "is_noexcept" => false, "exported" => true,
        "parameters" => params,
        "return_type" => Dict{String,Any}("c_type" => ret[1], "julia_type" => ret[2], "size" => 4),
        "return_type_source" => rsrc, "parameters_source" => psrc),
        Dict{String,Any}(String(k) => v for (k, v) in extra))

@testset "A symbol no DIE describes: an alias's signature, or a refusal" begin

    @testset "_alias_signature_key: the described function at the address" begin
        by = Dict{String,Vector{String}}(
            "2dc0" => ["aliased_real", "aliased_name"],
            "3000" => ["lonely"],
            "3100" => ["zeta_real", "alpha_real", "twin_of_two"],
            "3200" => ["__rb_lib_impl", "via_promoted"],
            "3300" => ["_ZN3BoxC1Ei", "_ZN3BoxC2Ei", "box_make"],
            "3400" => ["impl_local", "via_static"])
        dwarf = Dict{String,Any}("aliased_real" => 1, "zeta_real" => 1, "alpha_real" => 1,
                                 "_ZN3BoxC4Ei" => 1, "impl_local" => 1, "impl" => 1)
        key(m, a) = ALS_C._alias_signature_key(m, a, by, dwarf)
        @test key("aliased_name", "2dc0") == ("aliased_real", "aliased_real")
        # Nothing else at the address, or nothing there that a DIE describes.
        @test key("lonely", "3000") === nothing
        @test key("via_promoted", "3200") === nothing
        # Several described names at one address are one piece of code: the
        # smallest name, whatever order the symbol table lists them in.
        @test key("twin_of_two", "3100") == ("alpha_real", "alpha_real")
        # A constructor variant is described under the unified C4 key.
        @test key("box_make", "3300") == ("_ZN3BoxC1Ei", "_ZN3BoxC4Ei")
        # A local (`static`) function's name is in the table too.
        @test key("via_static", "3400") == ("impl_local", "impl_local")
        # An address the table does not have, or none.
        @test key("aliased_name", "ffff") === nothing
        @test key("aliased_name", "") === nothing
    end

    @testset "an alias takes its aliasee's byval record" begin
        fns = [_als_fn("take_al", "take_al"; alias_of = "take_real"), _als_fn("take_real", "take_real"),
               _als_fn("other", "other")]
        byval = Dict{String,Vector{Dict{String,Any}}}(
            "take_real" => [Dict{String,Any}("record" => "Wide", "align" => 16)])
        ALS_C._mark_byval_params!(fns, byval)
        @test fns[1]["byval_params"] == Any[Dict{String,Any}("record" => "Wide", "align" => 16)]
        @test fns[2]["byval_params"] == fns[1]["byval_params"]
        @test !haskey(fns[3], "byval_params")
    end

    @testset "_lacks_debug_info: no DIE at all" begin
        @test ALS_W._lacks_debug_info(_als_fn("f", "f"; rsrc = "inferred", psrc = "inferred"))
        # K&R `int old_style()` and `int f(void)`: a DIE with no parameters.
        @test !ALS_W._lacks_debug_info(_als_fn("old_style", "old_style"; psrc = "inferred"))
        @test !ALS_W._lacks_debug_info(_als_fn("f", "f"))
        # Function dictionaries that never said (synthetic metadata).
        @test !ALS_W._lacks_debug_info(Dict{String,Any}("name" => "f", "mangled" => "f"))
    end

    dir = mktempdir()
    config(language) = begin
        toml = joinpath(dir, "replibuild_$language.toml")
        write(toml, """
        [project]
        name = "aliassynth"
        root = "$(escape_string(dir))"
        [link]
        enable_lto = false
        [wrap]
        language = "$language"
        [types]
        strictness = "warn"
        allow_unknown_structs = true
        [cache]
        enabled = false
        """)
        RepliBuild.ConfigurationManager.load_config(toml)
    end
    libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    x = Any[_als_param("x", "int", "Cint")]
    # What the build writes for each: the alias with its aliasee's signature
    # under its OWN symbol, the K&R function with no parameters, and the
    # symbol nothing describes with the guess from its name.
    functions(; cpp) = Any[
        _als_fn("aliased_real", "aliased_real"; params = x, nounwind = true),
        _als_fn("aliased_name", "aliased_name"; params = x, nounwind = true, alias_of = "aliased_real"),
        _als_fn("old_style", "old_style"; psrc = "inferred", nounwind = true),
        _als_fn("undescribed", "undescribed"; rsrc = "inferred", psrc = "inferred", nounwind = true),
        (cpp ? (merge(_als_fn("poke", "_ZN3Box4pokeEi"; params = x, rsrc = "inferred", psrc = "inferred"),
                      Dict{String,Any}("demangled" => "Box::poke(int)", "class" => "Box", "is_method" => true)),) : ())...]
    meta(language; marker = true) = begin
        m = Dict{String,Any}("functions" => functions(cpp = language == "c++"), "globals" => Dict{String,Any}(),
                             "struct_definitions" => Dict{String,Any}(), "record_abi" => Dict{String,Any}(),
                             "language" => language)
        marker && (m["alias_signatures"] = true)
        m
    end
    body(out, name) = (m = match(Regex("^function $(name)\\(.*?^end\$", "ms"), out); m === nothing ? "" : String(m.match))
    generate(language; marker = true) = begin
        cfg = config(language == "c++" ? "cpp" : "c")
        gen = language == "c++" ? ALS_W.generate_introspective_module_cpp : ALS_W.generate_introspective_module_c
        res = gen(cfg, libref, meta(language; marker), language == "c++" ? "AliasSynthX" : "AliasSynthC",
                  ALS_W.create_type_registry(cfg), true)
        res isa Tuple ? res : (res, nothing)
    end

    for language in ("c", "c++")
        @testset "the $language generator" begin
            (out, thunks) = generate(language)
            # The alias is called under its own symbol with the aliasee's signature.
            @test occursin("ccall((:aliased_name, LIBRARY_PATH), Cint, (Cint,), x_c)", body(out, "aliased_name"))
            @test occursin("ccall((:aliased_real, LIBRARY_PATH), Cint, (Cint,), x_c)", body(out, "aliased_real"))
            # A DIE with no prototype is still a call.
            @test occursin("ccall((:old_style, LIBRARY_PATH), Cint, (), )", body(out, "old_style"))
            # No DIE: refused, whatever it is called with, and never called.
            u = body(out, "undescribed")
            @test startswith(u, "function undescribed(args...)\n")
            @test occursin("Base.error(\"ABI Safety Trap: cannot call 'undescribed'. The library exports " *
                           "`undescribed` and has no debug information for it", u)
            @test !occursin("ccall((:undescribed", out) && !occursin("_mlir_ciface_undescribed", out)
            @test ALS_W._is_trap_chunk(u)
            @test occursin(r"^export [^\n]*\bundescribed\b"m, out)
            # This metadata looked for an alias, so the refusal does not send the reader to a rebuild.
            @test !occursin("rebuild the package", u)
            if language == "c++"
                # A member function is refused under the name its wrapper would have had.
                p = body(out, "Box_poke")
                @test startswith(p, "function Box_poke(args...)\n")
                @test occursin("The library exports `_ZN3Box4pokeEi` and has no debug information", p)
                @test !occursin("_ZN3Box4pokeEi_thunk", out)
                @test isempty(thunks)
            end

            # The module loads, the refusal is an error at the call, and the
            # other functions are defined beside it.
            holder = Module(Symbol("AliasHolder", language == "c" ? "C" : "X"))
            Base.include_string(holder, out)
            M = getfield(holder, language == "c" ? :AliasSynthC : :AliasSynthX)
            for args in ((), (1,), (1, 2.0, "three"))
                err = try
                    Base.invokelatest(getfield(M, :undescribed), args...)
                    nothing
                catch e
                    e
                end
                @test err isa ErrorException && occursin("no debug information", err.msg)
            end
            @test hasmethod(getfield(M, :aliased_name), Tuple{Int})
            @test hasmethod(getfield(M, :old_style), Tuple{})
        end

        @testset "$language: metadata written before aliases were looked up" begin
            # Its `inferred` functions may be aliases: the refusal names the rebuild.
            (out, _) = generate(language; marker = false)
            u = body(out, "undescribed")
            @test occursin("no debug information for it", u)
            @test occursin("rebuild the package (delete .replibuild_cache/project_hash) and wrap again", u)
            @test !occursin("ccall((:undescribed", out)
        end
    end
end
