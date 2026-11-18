# =============================================================================
# How a C/C++ type becomes a Julia type, live (2026-11-17)
#
# test_type_mapping.jl checks the mappers and what both generators emit. This
# builds the two fixtures, wraps them, and CALLS every function from a child:
#
#   test/typemap_c_test (C, Tier 3)
#     - `void out_small(enum Small *out)` was bound as `out_small(out::Small)`
#       with `ccall(..., (Small,), out)`: C wrote through the enum's value as
#       an address, and the child died. A pointer to an enum is `Ptr{E}` in
#       every position, so a `Ref`, a `Vector` or a `Ptr` passes and a bare
#       value is a MethodError.
#     - one `unsigned __int128` in a signature refused the whole wrap. It is
#       `UInt128`, and `ccall` passes one as C does only while both halves get
#       registers: `u128_after5` and `u128_after7` read a wrong value through
#       a plain ccall on x86-64 SysV, and are refused there.
#     - `_Float16` has no Julia type in the map: one trap stub, and the rest
#       of the library is callable.
#
#   test/typemap_cpp_test (C++, Tier 2)
#     - `bool color_is(const Color &c, Color d)` got `UInt8(c)` where its thunk
#       loads a pointer, and `color_is(Blue, Blue)` was false.
#     - `char8_t`, `char16_t`, `char32_t` and `std::byte` by value were traps.
#     - a 128-bit integer by value takes the thunk, which places it as C does.
#
# The probes state the invariant every host must hold (the right value, or a
# refusal); which functions a host refuses is asserted here per convention.
# Needs the C bucket; the C++ half needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const TYPEMAP_C_DIR = joinpath(@__DIR__, "typemap_c_test")
const TYPEMAP_CPP_DIR = joinpath(@__DIR__, "typemap_cpp_test")

# One function's body in a generated wrapper (its `function name(` up to `end`).
function typemap_body(wsrc::AbstractString, name::AbstractString)
    m = match(Regex("^function $(name)\\(.*?^end\$", "ms"), wsrc)
    return m === nothing ? "" : String(m.match)
end
typemap_is_trap(wsrc, name) = (b = typemap_body(wsrc, name);
                               occursin("ABI Safety Trap", b) && !occursin("ccall(", b))

function typemap_toml(dir, name, language, flags, src)
    toml_path = joinpath(dir, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "$name"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(dir))"

    [compile]
    flags = [$(join(repr.(flags), ", "))]
    source_files = ["$(escape_string(src))"]
    include_dirs = ["$(escape_string(dir))"]

    [link]
    enable_lto = false
    optimization_level = "2"

    [binary]
    type = "shared"

    [wrap]
    language = "$language"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    return toml_path
end

@testset "Type mapping, live: C (enum pointers, 128-bit integers, an unmapped scalar)" begin
    toml_path = typemap_toml(TYPEMAP_C_DIR, "typemap_c", "c", ["-O2", "-fPIC"],
                             joinpath(TYPEMAP_C_DIR, "typemap.c"))
    redirect_stdout(devnull) do
        RepliBuild.build(toml_path)
    end
    # One unmapped scalar used to refuse the wrapper here.
    RepliBuild.wrap(toml_path)
    wrapper = joinpath(TYPEMAP_C_DIR, "julia", "TypemapC.jl")
    @test isfile(wrapper)
    wsrc = isfile(wrapper) ? read(wrapper, String) : ""

    @testset "what was emitted" begin
        @test occursin("ccall((:out_small, LIBRARY_PATH), Cvoid, (Ptr{Small},), out)", wsrc)
        @test occursin("(Ptr{Small}, Ptr{Small},), a, b)", typemap_body(wsrc, "small_swap"))
        @test occursin(r"^function smalls_static\(\)::Ptr\{Small\}"m, wsrc)
        @test occursin(r"^function smalls_handle\(\)::Ptr\{Ptr\{Small\}\}"m, wsrc)
        @test occursin(r"^    kinds::Ptr\{Small\}$"m, wsrc)
        @test occursin("(Ptr{Perm},), out)", typemap_body(wsrc, "out_perm"))
        @test occursin("ccall((:u128_echo, LIBRARY_PATH), UInt128, (UInt128,), v)", wsrc) ||
              typemap_is_trap(wsrc, "u128_echo")
        # The scalar the map does not know: a stub naming it, on every host.
        @test typemap_is_trap(wsrc, "f16_echo") && occursin("`_Float16`", typemap_body(wsrc, "f16_echo"))
        @test typemap_is_trap(wsrc, "f16_bits")
        @test !typemap_is_trap(wsrc, "plain_add")
        @test !occursin(r"ccall\([^\n]*\bAny\b", wsrc)
    end

    @testset "which 128-bit crossings this host refuses" begin
        # x86-64 SysV: a parameter that does not get two registers. Win64: a
        # return (XMM0, where ccall expects a hidden pointer). AAPCS64: none.
        abi = RepliBuild.host_record_abi()
        returning = ["u128_echo", "u128_add", "u128_make", "u128_after4", "u128_after_double",
                     "u128_after5", "u128_after7", "i128_after5", "u128_then_long"]
        refused = abi === :sysv ? ["u128_after5", "u128_after7", "i128_after5", "u128_then_long"] :
                  abi === :win64 ? returning : String[]
        @info "128-bit integers through ccall" abi refused
        for f in vcat(returning, ["u128_hi", "u128_lo", "u128_out"])
            @test (f, typemap_is_trap(wsrc, f)) == (f, f in refused)
        end
    end

    @testset "every function, from a child" begin
        r = run_isolated(`$(joinpath(TYPEMAP_C_DIR, "probe_typemap_c.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["enum_out_param", "enum_bare_value_refused", "enum_array", "enum_swap", "enum_pointer_return",
                 "enum_pointer_member", "enum_pointer_global", "bitmask_enum_pointer",
                 "u128_right_or_refused", "u128_pointer_and_halves", "unmapped_scalar_is_one_stub"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end

if requires("type mapping through a thunk, live", :libJLCS, :clangxx)
@testset "Type mapping, live: C++ (enum references, character types, std::byte, 128-bit integers)" begin
    toml_path = typemap_toml(TYPEMAP_CPP_DIR, "typemap_cpp", "cpp", ["-std=c++20", "-fPIC", "-O2"],
                             joinpath(TYPEMAP_CPP_DIR, "typemap.cpp"))
    redirect_stdout(devnull) do
        RepliBuild.build(toml_path)
    end
    RepliBuild.wrap(toml_path)
    wrapper = joinpath(TYPEMAP_CPP_DIR, "julia", "TypemapCpp.jl")
    @test isfile(wrapper)
    wsrc = isfile(wrapper) ? read(wrapper, String) : ""

    @testset "what was emitted" begin
        @test occursin(r"^function tmap_color_out\(c::Ref\{Color\}\)"m, wsrc)
        @test occursin(r"^function tmap_color_is\(c::Ref\{Color\}, d::Any\)"m, wsrc)
        # The reference is passed as an address (`pointer_arg`), the by-value
        # enum as its integer.
        @test occursin("Bool, RepliBuild.JITManager.pointer_arg(Ptr{Color}, c), UInt8(d))", typemap_body(wsrc, "tmap_color_is"))
        @test !occursin("UInt8(c)", wsrc)
        @test occursin(r"^primitive type byte <: RepliBuild\.OpenEnums\.OpenEnum\{UInt8\} 8 end"m, wsrc)
        @test occursin(r"^primitive type Handle <: RepliBuild\.OpenEnums\.OpenEnum\{UInt32\} 32 end"m, wsrc)
        @test !occursin(r"^struct (byte|Handle|char8_t|char16_t|char32_t)\b"m, wsrc)
        @test occursin(r"^    colors::Ptr\{Color\}$"m, wsrc)
        # No function of the fixture is a trap, but for a 128-bit integer by
        # value under Win64, which a thunk cannot pass as MinGW's callee reads it.
        abi = RepliBuild.host_record_abi()
        fns = sort!(unique!([String(m.captures[1]) for m in eachmatch(r"^function (tmap_\w+)\("m, wsrc)]))
        u128 = ["tmap_u128_after5", "tmap_u128_after7", "tmap_u128_echo", "tmap_u128_hi"]
        @test length(fns) == 22 && issubset(u128, fns)
        @test (abi, filter(f -> typemap_is_trap(wsrc, f), fns)) == (abi, abi === :win64 ? u128 : String[])
        # The metadata keeps `std::byte` as the toolchain's, and no other toolchain enum.
        meta = RepliBuild.Wrapper._read_json(joinpath(TYPEMAP_CPP_DIR, "julia", "compilation_metadata.json"))
        sd = meta["struct_definitions"]
        @test get(get(sd, "__enum__byte", Dict()), "toolchain", false) == true
        @test sort!([k for (k, d) in sd if startswith(k, "__enum__") && get(d, "toolchain", false) == true]) ==
              ["__enum__byte"]
        @test isempty(get(get(sd, "__enum__Handle", Dict("enumerators" => [1])), "enumerators", [1]))
    end

    @testset "every function, from a child" begin
        r = run_isolated(`$(joinpath(TYPEMAP_CPP_DIR, "probe_typemap_cpp.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["enum_reference_out", "enum_const_reference", "enum_bare_value_refused", "enum_pointer_param",
                 "enum_pointer_return", "enum_pointer_member", "char_types", "std_byte", "empty_library_enum",
                 "declared_only_enum_pointer", "u128_through_a_thunk"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end
end # requires
