# =============================================================================
# Nested-struct-member ABI trace test (library-free)
#
# Traces the C generator through compile → DWARF → wrap → live by-value
# crossings for structs with struct-typed members.
#
# SysV: a 16-byte all-float struct travels in XMM registers; representing it
# as NTuple{16,UInt8} silently reads/writes integer registers. Win64: size is
# the only criterion and aggregates never reach XMM — that struct is sret /
# memory, so a blob can round-trip by accident. `fields_resolved` /
# `xform_not_blob` are the actual guard on both ABIs; the live round-trips
# still prove the crossing works.
# The probe runs in a subprocess so an ABI break can never take down the
# test session (see also: a packed float struct must refuse loudly on SysV and
# cross by value on Win64 and AAPCS64, where size alone places a non-HFA).
# =============================================================================

using Test
using RepliBuild
import Libdl

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const ABI_TEST_DIR = joinpath(@__DIR__, "abi_nested_test")

@testset "Nested-struct ABI resolution" begin
    toml_path = joinpath(ABI_TEST_DIR, "replibuild.toml")

    # Regenerate the config each run (absolute paths, machine-local)
    write(toml_path, """
    [project]
    name = "abi_nested"
    # escape_string on every interpolated path: TOML basic strings take escapes,
    # so a Windows `C:\\Users\\...` would parse as `\\U` and fail.
    root = "$(escape_string(ABI_TEST_DIR))"

    [compile]
    flags = ["-O2", "-fPIC"]
    source_files = ["$(escape_string(joinpath(ABI_TEST_DIR, "src", "nested.c")))"]
    include_dirs = ["$(escape_string(joinpath(ABI_TEST_DIR, "include")))"]

    [link]
    # LTO off: trace the pure Tier-3 ccall path (Hub production configuration)
    enable_lto = false
    optimization_level = "2"

    [binary]
    type = "shared"

    [wrap]
    language = "c"

    [types]
    strictness = "warn"
    allow_unknown_structs = true
    allow_function_pointers = true

    [cache]
    enabled = false
    """)

    RepliBuild.build(toml_path)
    RepliBuild.wrap(toml_path)

    wrapper = joinpath(ABI_TEST_DIR, "julia", "AbiNested.jl")
    @test isfile(wrapper)

    # Subprocess-isolated probe (an ABI break is a crash): one PROBE line per case
    probe = joinpath(ABI_TEST_DIR, "probe_abi_nested.jl")
    r = run_isolated(`$probe $wrapper`)
    v = probe_verdicts(r.output)

    @test r.ok
    @test haskey(v, "PROBE_DONE")                  # probe ran to completion (no abort)
    for probe_name in ["fields_resolved", "xform_not_blob", "xform_return", "xform_byvalue_arg",
                       "mass_roundtrip", "disc_roundtrip", "poly_memory_class",
                       "nestint_roundtrip", "packed_float_byvalue",
                       "float_param_loosening", "with_helper"]
        @test startswith(get(v, probe_name, "no PROBE line"), "PASS")
    end

    @testset "PackedFV's crossing per convention, from this host's metadata" begin
        # The live probe sees one host. The same metadata generated for each
        # convention pins what the others emit for the 9-byte packed float
        # blob: SysV refuses the argument (MEMORY class, INTEGER blob) and
        # returns through an explicit sret; Win64 passes it by reference to a
        # copy and returns through the hidden pointer Julia's ccall supplies;
        # AAPCS64 puts a 9-byte non-HFA in two general registers each way.
        # Both are plain calls for the blob.
        W = RepliBuild.Wrapper
        cfg = RepliBuild.ConfigurationManager.load_config(toml_path)
        meta = RepliBuild._read_json(joinpath(ABI_TEST_DIR, "julia", "compilation_metadata.json"))
        libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
        body(code, name) = let i = findfirst("function $name(", code)
            i === nothing ? "" : (r = SubString(code, first(i)); j = findfirst(r"\nend\b", r);
                                  j === nothing ? String(r) : String(r[1:last(j)]))
        end
        checked = 0
        for abi in (:sysv, :win64, :aapcs64)
            code = W.generate_introspective_module_c(cfg, libref, meta, "AbiNestedX", W.create_type_registry(cfg), false;
                                                     abi = abi)
            take = body(code, "packedfv_sum")
            make = body(code, "make_packedfv")
            if abi === :sysv
                @test (abi, occursin("ABI Safety Trap", take)) == (abi, true)
            else
                @test (abi, occursin("ccall((:packedfv_sum, LIBRARY_PATH), Cfloat, (PackedFV,), s)", take)) == (abi, true)
            end
            if abi === :sysv
                @test (abi, occursin("(Ptr{PackedFV}, ), ret_buf", make)) == (abi, true)
            else
                @test (abi, occursin("ccall((:make_packedfv, LIBRARY_PATH), PackedFV, (), )", make)) == (abi, true)
            end
            # The field-resolved records cross plainly on every convention.
            @test (abi, occursin("ccall((:xform_p, LIBRARY_PATH), Vec2, (XForm,), t)", body(code, "xform_p"))) == (abi, true)
            checked += 1
        end
        @test checked == 3
    end
end
