# test_rust_build_live.jl — `RepliBuild.build` on a Rust crate, with cargo.
#
# test/rust_test/crate is plain Rust: nothing in it is written for RepliBuild
# (no `extern "C"` on its API, no `#[repr(C)]`, no `#[inline(never)]`). This
# builds it, reads `compilation_metadata.json`, and holds the metadata to the
# library: test/rust_test/verify.jl calls the crate's functions from Julia in
# a child process, with the signatures and layouts the metadata states.
#
# It also holds what the build promises about itself: nothing is extracted
# twice for one library, an optimized build keeps every function its
# unoptimized build has, and a function whose code LLVM merged into another's
# is still there.

using Test
using Libdl
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const _RBL_DIR = joinpath(@__DIR__, "rust_test")
const _RBL_TOML = joinpath(_RBL_DIR, "replibuild.toml")

# test/*/*.toml is untracked, so the project file is written here.
_rbl_write(opt::Int) = write(_RBL_TOML, """
    [project]
    name = "rb_fix"

    [rust]
    manifest = "crate/Cargo.toml"
    opt_level = $opt
    """)

function _rbl_build()
    log = tempname()
    lib = open(log, "w") do io
        redirect_stdout(io) do
            withenv("REPLIBUILD_JULIA_ENV" => "none") do
                RepliBuild.build(_RBL_TOML)
            end
        end
    end
    out = read(log, String)
    rm(log; force = true)
    return (lib, out)
end

_rbl_error(f) = try
    f()
    ""
catch e
    sprint(showerror, e)
end

if requires("Rust front end (live)", :cargo, :rustc, :nm)

@testset "Rust front end (live)" begin
    # From cold: another run's cargo output is not this run's.
    for d in ("build", "julia")
        rm(joinpath(_RBL_DIR, d); recursive = true, force = true)
    end
    _rbl_write(0)
    (lib, out) = _rbl_build()
    meta_path = joinpath(dirname(lib), "compilation_metadata.json")
    key_path = joinpath(dirname(lib), "rust_metadata.key")
    md = RepliBuild._read_json(meta_path)
    fn = Dict(f["path"] => f for f in md["functions"])
    refused(md) = sort!([f["path"] for f in md["functions"] if f["refusal"] !== nothing])

    @testset "the build" begin
        @test isfile(lib) && basename(lib) == "librb_fix." * Libdl.dlext
        @test occursin("cargo: rb_fix built", out) && occursin("callable", out)
        @test md["language"] == "rust" && md["metadata_version"] == RepliBuild.RustBuild.METADATA_VERSION
        # The package is `rb-fix`; the crate, and every name derived from it, `rb_fix`.
        @test md["crate"]["name"] == "rb_fix" && md["crate"]["package"] == "rb-fix"
        @test !isempty(md["crate"]["disambiguator"])
        @test md["library"] == basename(lib)
        @test md["library_sha"] == bytes2hex(open(RepliBuild.SHA.sha256, lib))
        @test md["build"]["opt_level"] == 0 && !isempty(md["toolchain"]["rustc"]) && !isempty(md["toolchain"]["llvm"])
        # cargo's target directory is under the build directory, not in the crate.
        @test isdir(joinpath(_RBL_DIR, "build", "rust", "target")) && !isdir(joinpath(_RBL_DIR, "crate", "target"))
        @test isfile(key_path)
    end

    @testset "the surface is the crate's own functions" begin
        # Named, not counted: the standard library's functions outnumber these 40 to 1.
        for path in ("rb_fix::add", "rb_fix::rec_make", "rb_fix::slice_sum", "rb_fix::maybe", "rb_fix::make_greeting",
                     "<rb_fix::Acc>::new", "<rb_fix::Acc>::push", "<rb_fix::Shape as rb_fix::Area>::area",
                     "rb_fix::dyn_area", "rb_fix::make_area", "rb_fix::checked_div", "c_add", "cpoint_norm")
            @test haskey(fn, path)
        end
        @test length(md["functions"]) >= 40
        @test all(f -> startswith(f["path"], "rb_fix::") || startswith(f["path"], "<rb_fix::") || f["no_mangle"],
                  md["functions"])
        @test issorted([(f["path"], f["symbol"]) for f in md["functions"]])
        @test isempty(md["skipped"])
        # What the crate instantiated of the standard library is in the library and is not its API.
        @test md["foreign_definitions"] > 0
        @test !any(f -> occursin("core::", f["path"][1:min(end, 8)]) || startswith(f["path"], "alloc::"), md["functions"])
        # A generic function exists as the instantiation the crate made.
        @test fn["rb_fix::largest::<f64>"]["generic"] && !fn["rb_fix::largest_f64"]["generic"]
        @test fn["c_add"]["no_mangle"] && !fn["rb_fix::add"]["no_mangle"]
        # A static is not a function.
        @test !any(f -> occursin("COUNTER", f["path"]), md["functions"])
    end

    @testset "each call is the compiler's own" begin
        pass(path) = [(p["name"], p["pass"], p["ir"]) for p in fn[path]["parameters"]]
        @test pass("rb_fix::slice_sum") == [("xs", "pair", [1, 2])]
        @test pass("rb_fix::rec_sum") == [("r", "indirect", [1])] && pass("rb_fix::rec_sum_ref") == [("r", "direct", [1])]
        @test fn["rb_fix::rec_make"]["return"]["pass"] == "indirect"
        @test fn["rb_fix::maybe"]["return"]["pass"] == "pair" && fn["rb_fix::maybe"]["ir"]["return"]["type"] == "{ i32, i32 }"
        @test fn["rb_fix::hello"]["return"]["pass"] == "pair" && fn["<rb_fix::Acc>::new"]["return"]["pass"] == "pair"
        @test fn["rb_fix::make_area"]["return"]["pass"] == "pair"          # Box<dyn Area>: data and vtable
        @test pass("rb_fix::dyn_area") == [("a", "pair", [1, 2])]
        @test pass("rb_fix::b3_rot") == [("x", "cast", [1])]
        @test pass("rb_fix::string_len_owned") == [("s", "indirect", [1])]
        @test fn["rb_fix::make_greeting"]["return"]["type"] == "alloc::string::String"
        @test fn["rb_fix::boxed"]["return"]["pass"] == "direct"
        @test md["types"][fn["rb_fix::boxed"]["return"]["type"]]["pointer"] == "box"
        @test md["types"][fn["rb_fix::unbox_d"]["parameters"][1]["type"]]["pointer"] == "box"
        @test fn["<rb_fix::Acc>::push"]["owner"] == "rb_fix::Acc"
        @test md["types"]["rb_fix::Rec"]["size"] == 16 && md["types"]["rb_fix::Shape"]["kind"] == "enum"
        # Exactly the two that cannot be matched, each for its own reason.
        @test refused(md) == ["cpoint_norm", "rb_fix::located"]
        @test occursin("track_caller", fn["rb_fix::located"]["refusal"])
        # An unoptimized build proves nothing about unwinding for the crate's own code.
        @test !fn["rb_fix::checked_div"]["nounwind"]
    end

    @testset "the library does what the metadata says" begin
        r = run_isolated(`$(joinpath(_RBL_DIR, "verify.jl")) $lib $meta_path`; timeout = 300)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        expected = ["add", "c_add", "flag", "char_code", "level_next", "checked_div", "count_static", "slice_sum",
                    "str_len_substring", "largest_f64", "acc_push", "acc_mean", "box_roundtrip", "opt_ref_some_none",
                    "slot_niche"]
        Sys.ARCH === :x86_64 && append!(expected, ["rec_make_layout", "shape_discriminant", "string_out_and_in"])
        for name in expected
            @test (name, startswith(get(v, name, "no PROBE line"), "PASS") ? "PASS" : get(v, name, "no PROBE line")) ==
                  (name, "PASS")
        end
        @test length(v) == length(expected) + 1        # every probe that ran is one that was expected
    end

    @testset "nothing is extracted twice for one library" begin
        key = read(key_path, String)
        stamp = md["timestamp"]
        (lib2, out2) = _rbl_build()
        @test lib2 == lib
        @test occursin("up to date", out2) && occursin("metadata: cached", out2)
        @test read(key_path, String) == key
        @test RepliBuild._read_json(meta_path)["timestamp"] == stamp
        # A missing key is "not extracted": the metadata is written again.
        rm(key_path)
        (_, out3) = _rbl_build()
        @test !occursin("metadata: cached", out3) && isfile(key_path) && read(key_path, String) == key
        again = RepliBuild._read_json(meta_path)
        @test again["timestamp"] != stamp
        @test [(f["symbol"], f["refusal"]) for f in again["functions"]] ==
              [(f["symbol"], f["refusal"]) for f in md["functions"]]
        @test again["types"] == md["types"]
    end

    @testset "an optimized build keeps the crate's functions" begin
        _rbl_write(3)
        (lib3, out3) = _rbl_build()
        @test occursin("opt-level 3", out3)
        md3 = RepliBuild._read_json(joinpath(dirname(lib3), "compilation_metadata.json"))
        fn3 = Dict(f["path"] => f for f in md3["functions"])
        @test md3["build"]["opt_level"] == 3
        # rustc gives a small `pub fn` no symbol in an optimized build unless
        # incremental compilation is on, which this build keeps on: every
        # function that is not an instantiation is still there.
        plain = sort!([f["path"] for f in md["functions"] if !f["generic"]])
        @test length(plain) >= 39
        @test [p for p in plain if !haskey(fn3, p)] == String[]
        @test refused(md3) == ["cpoint_norm", "rb_fix::located"]
        # The same calls, read from other define lines and other DIEs.
        for path in plain
            fn[path]["refusal"] === nothing || continue
            @test (path, [(p["type"], p["pass"], p["ir"]) for p in fn3[path]["parameters"]], fn3[path]["return"]["pass"]) ==
                  (path, [(p["type"], p["pass"], p["ir"]) for p in fn[path]["parameters"]], fn[path]["return"]["pass"])
        end
        # Optimized, the compiler proves what cannot unwind, and says what can.
        @test fn3["rb_fix::add"]["nounwind"] && !fn3["rb_fix::checked_div"]["nounwind"]
        # A function with the same code as another is kept as an alias of it, and is still callable.
        @test fn3["c_add"]["refusal"] === nothing
        haskey(fn3["c_add"], "alias_of") && @test fn3["c_add"]["alias_of"] == fn3["rb_fix::add"]["symbol"]

        r = run_isolated(`$(joinpath(_RBL_DIR, "verify.jl")) $lib3 $(joinpath(dirname(lib3), "compilation_metadata.json"))`;
                         timeout = 300)
        v = probe_verdicts(r.output)
        @test r.ok && haskey(v, "PROBE_DONE")
        failed = sort!([k * ": " * val for (k, val) in v if k != "PROBE_DONE" && !startswith(val, "PASS")])
        @test failed == String[]
        @test length(v) >= 16
    end

    @testset "a Rust project stays out of the C and C++ pipeline" begin
        cfg = RepliBuild.ConfigurationManager.load_config(_RBL_TOML)
        @test occursin("cargo builds", _rbl_error(() -> RepliBuild.Compiler.compile_project(cfg)))
        @test occursin("[rust] project", _rbl_error(() -> RepliBuild.Wrapper.wrap_library(cfg, lib)))
        msg = _rbl_error(() -> RepliBuild.wrap(_RBL_TOML))
        @test occursin("not built yet", msg) && occursin("compilation_metadata.json", msg)
    end

    @testset "what cargo cannot build is cargo's error, in full" begin
        mktempdir() do dir
            cp(joinpath(_RBL_DIR, "crate"), joinpath(dir, "crate"))
            write(joinpath(dir, "crate", "src", "lib.rs"), "pub fn broken() -> i32 { \"not a number\" }\n")
            toml = joinpath(dir, "replibuild.toml")
            write(toml, "[project]\nname = \"rb_fix\"\n\n[rust]\nmanifest = \"crate/Cargo.toml\"\n")
            msg = _rbl_error(() -> redirect_stdout(() -> RepliBuild.build(toml), devnull))
            @test occursin("cargo could not build", msg) && occursin("mismatched types", msg)
            @test !isfile(joinpath(dir, "julia", "rust_metadata.key"))
            # A manifest that is not there is said so before cargo is asked anything.
            write(toml, "[project]\nname = \"rb_fix\"\n\n[rust]\nmanifest = \"nowhere/Cargo.toml\"\n")
            @test occursin("does not exist", _rbl_error(() -> redirect_stdout(() -> RepliBuild.build(toml), devnull)))
        end
    end
end

end # requires
