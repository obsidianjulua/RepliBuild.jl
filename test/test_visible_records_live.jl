# =============================================================================
# Records made visible, live (2026-11-15)
#
# test_visible_records.jl (unit) checks the key, the unit's text and the name
# lookup. This builds test/visible_record_test (a header-only class nothing in
# the .cpp mentions, as Bullet's btDefaultMotionState is) four ways: not
# listed (absent, as before), listed (a type, and its inline constructor
# forced and callable), listed under plain -g (named in a warning), and a
# misspelled or declared-only entry (a build error naming it).
# Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const VISIBLE_RECORD_DIR = joinpath(@__DIR__, "visible_record_test")

if requires("[types] records, live", :libJLCS, :clangxx)
@testset "[types] records, live" begin
    toml_path = joinpath(VISIBLE_RECORD_DIR, "replibuild.toml")
    src = joinpath(VISIBLE_RECORD_DIR, "src", "visible_record.cpp")
    inc = joinpath(VISIBLE_RECORD_DIR, "include")
    md_path = joinpath(VISIBLE_RECORD_DIR, "julia", "compilation_metadata.json")
    wrapper = joinpath(VISIBLE_RECORD_DIR, "julia", "Visiblerecord.jl")

    # Regenerated per build: it holds absolute, machine-local paths.
    function write_toml(; debug::String, records::Vector{String})
        write(toml_path, """
        [project]
        name = "visiblerecord"
        # escape_string on every interpolated path: TOML basic strings take escapes.
        root = "$(escape_string(VISIBLE_RECORD_DIR))"

        [compile]
        flags = ["-std=c++17", "-fPIC", "-O2", "$debug"]
        source_files = ["$(escape_string(src))"]
        include_dirs = ["$(escape_string(inc))"]

        [link]
        enable_lto = false
        optimization_level = "2"

        [binary]
        type = "shared"

        [wrap]
        language = "cpp"
        shim_headers = ["visible_record.h"]

        [types]
        strictness = "warn"
        allow_unknown_structs = true
        records = [$(join(repr.(records), ", "))]

        [cache]
        enabled = false
        """)
    end
    function build_logged()
        mktemp() do path, io
            redirect_stdout(io) do
                RepliBuild.build(toml_path)
            end
            close(io)
            read(path, String)
        end
    end
    unit = joinpath(VISIBLE_RECORD_DIR, ".replibuild_cache", "replibuild_records.cpp")

    @testset "not listed: the class is in no unit's DWARF, as before" begin
        rm(unit; force = true)
        write_toml(debug = "-fstandalone-debug", records = String[])
        log = build_logged()
        md = RepliBuild._read_json(md_path)
        @test haskey(md["struct_definitions"], "Used")
        @test !haskey(md["struct_definitions"], "Unused") && !haskey(md["struct_definitions"], "Inner")
        @test !occursin("ctors:", log)
        @test !isfile(unit)
    end

    @testset "listed: a definition, then its inline constructor forced" begin
        write_toml(debug = "-fstandalone-debug", records = ["vr::Unused", "vr::Outer::Inner"])
        log = build_logged()
        @test isfile(unit)
        text = read(unit, String)
        @test occursin("#include \"visible_record.h\"", text)
        @test occursin("template void replibuild_force::record<vr::Unused>(vr::Unused*);", text)
        md = RepliBuild._read_json(md_path)
        @test haskey(md["struct_definitions"], "Unused") && haskey(md["struct_definitions"], "Inner")
        # The inline-constructor pass sees the records the unit made visible.
        @test occursin(r"ctors: 2 inline constructor\(s\) of 2 class\(es\)", log)
        emitted = sort([String(f["class"]) for f in md["inline_constructors"]["forced"] if f["emitted"] == true])
        @test emitted == ["Inner", "Unused"]
        mangled = Set(String(f["mangled"]) for f in md["functions"])
        @test "_ZN2vr6UnusedC2Ei" in mangled
        # The unit's helper is not API.
        @test !any(f -> occursin("replibuild_force", String(get(f, "demangled", ""))), md["functions"])
        RepliBuild.wrap(toml_path)
        code = read(wrapper, String)
        @test occursin("\nstruct Unused\n    v::Cint\nend\n", code)
        @test occursin(r"\nfunction vr_Unused_Unused\(this::Any, x::Integer\)\n", code)
        r = run_isolated(`$(joinpath(VISIBLE_RECORD_DIR, "probe_visible_record.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["listed_record_constructs", "nested_record_constructs", "used_record_untouched"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end

    @testset "listed under plain -g: constructor homing keeps it a declaration, and the build says so" begin
        write_toml(debug = "-g", records = ["vr::Unused"])
        @test_logs (:warn, r"\[types\] records: 1 class\(es\).*no definition in its debug info: vr::Unused\..*-fstandalone-debug"s) match_mode = :any begin
            redirect_stdout(devnull) do
                RepliBuild.build(toml_path)
            end
        end
        @test !haskey(RepliBuild._read_json(md_path)["struct_definitions"], "Unused")
    end

    @testset "an entry the unit cannot compile is a build error that names it" begin
        write_toml(debug = "-fstandalone-debug", records = ["vr::Unused", "vr::Unusde", "vr::Incomplete"])
        err = try
            redirect_stdout(devnull) do
                RepliBuild.build(toml_path)
            end
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        if err !== nothing
            @test occursin("[types] records: 2 entries do not compile", err)
            @test occursin("vr::Unusde:", err) && occursin("no member named 'Unusde'", err)
            @test occursin("vr::Incomplete:", err) && occursin("incomplete type", err)
            @test !occursin("vr::Unused:", err)                 # the good entry is not blamed
            @test occursin("visible_record.h", err)             # the headers it was tried against
        end
    end
end
end # requires
