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
    # Generic functions exist only for the types something uses them with.
    instantiate = ["rb_fix::midpoint::<f64>", "rb_fix::Pair::<i32>::new", "rb_fix::Pair::<i32>::sum",
                   "(rb_fix::into_f64 as fn(f32) -> f64)"]
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

# Every file under `dir` with its content's hash: what a build must leave alone.
_rbl_tree(dir) = sort!([relpath(joinpath(r, f), dir) => bytes2hex(open(RepliBuild.SHA.sha256, joinpath(r, f)))
                        for (r, _, fs) in walkdir(dir) for f in fs])

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
    crate_tree = _rbl_tree(joinpath(_RBL_DIR, "crate"))
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
        # The library is the generated root crate's, which carries the crate.
        @test md["crate"]["root"] == "rb_fix_rb"
        @test !isempty(md["crate"]["disambiguator"])
        @test md["library"] == basename(lib)
        @test md["library_sha"] == bytes2hex(open(RepliBuild.SHA.sha256, lib))
        @test md["build"]["opt_level"] == 0 && !isempty(md["toolchain"]["rustc"]) && !isempty(md["toolchain"]["llvm"])
        # Everything cargo writes is under the build directory: the root crate,
        # its lock file, the target directory. The crate's own tree is as it was.
        @test isdir(joinpath(_RBL_DIR, "build", "rust", "target"))
        root = joinpath(_RBL_DIR, "build", "rust", "root")
        @test isfile(joinpath(root, "Cargo.toml")) && isfile(joinpath(root, "lib.rs")) && isfile(joinpath(root, "Cargo.lock"))
        # ...and the per-type lines, written from what the build found, with their record.
        @test isfile(joinpath(root, "rb_glue.rs")) && isfile(joinpath(root, "rb_glue.json"))
        @test sort!(readdir(joinpath(_RBL_DIR, "crate"))) == ["Cargo.lock", "Cargo.toml", "src"]
        @test crate_tree == _rbl_tree(joinpath(_RBL_DIR, "crate"))
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
        # One byte that is only ever 0 or 1 is `i1` in the call, whatever its type.
        @test fn["rb_fix::flip"]["ir"]["return"]["type"] == "i1" && pass("rb_fix::flip") == [("s", "direct", [1])]
        @test fn["rb_fix::check"]["ir"]["return"]["type"] == "i1" && fn["rb_fix::check"]["return"]["pass"] == "direct"
        @test fn["rb_fix::opts"]["ir"]["return"]["type"] == "{ double, i1 }" && fn["rb_fix::opts"]["return"]["pass"] == "pair"
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
            @test (path, [(p["pass"], p["ir"]) for p in fn3[path]["parameters"]], fn3[path]["return"]["pass"]) ==
                  (path, [(p["pass"], p["ir"]) for p in fn[path]["parameters"]], fn[path]["return"]["pass"])
            # ...and the same types, wherever the function still has debug info of its own.
            fn3[path]["own_debug_info"] || continue
            @test (path, [p["type"] for p in fn3[path]["parameters"]], fn3[path]["return"]["type"]) ==
                  (path, [p["type"] for p in fn[path]["parameters"]], fn[path]["return"]["type"])
        end
        @test all(f -> f["own_debug_info"], md["functions"])            # unoptimized, nothing is merged
        # Optimized, LLVM merges functions whose code is the same: `flip` takes
        # an enum and `flag` a bool, and they are one function once compiled,
        # as are `slot_make` and `opt_box`, and `string_ref_len` and `secret_len`.
        if !occursin("building without them", out3)
            # The build's two LLVM options keep each a function of its own, so
            # each has its own `define` line and its own debug info: its own types.
            @test all(f -> f["own_debug_info"], md3["functions"])
            @test !any(f -> haskey(f, "alias_of"), md3["functions"])
            @test fn3["rb_fix::flip"]["parameters"][1]["type"] == "rb_fix::Side" && fn3["rb_fix::flip"]["return"]["type"] == "rb_fix::Side"
            @test fn3["rb_fix::flag"]["parameters"][1]["type"] == "bool"
            @test fn3["rb_fix::slot_make"]["return"]["type"] == "rb_fix::Slot"
            @test fn3["rb_fix::string_ref_len"]["parameters"][1]["type"] == "&alloc::string::String"
        else
            # A rustc whose LLVM refuses those options: a merged function is an
            # alias with no debug info. Its call is its target's; its types are
            # reported as its target's and marked as not its own.
            @info "this rustc's LLVM does not take the options that keep merged functions apart"
            for f in md3["functions"]
                f["own_debug_info"] && continue
                @test haskey(f, "alias_of") && f["refusal"] === nothing
                target = only(g for g in md3["functions"] if g["symbol"] == f["alias_of"])
                @test [p["type"] for p in f["parameters"]] == [p["type"] for p in target["parameters"]]
            end
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

    @testset "what a value owns is the compiler's answer" begin
        RG = RepliBuild.RustGlue
        md3 = RepliBuild._read_json(meta_path)                       # the optimized build, which is the one on disk
        @test md3["build"]["opt_level"] == 3
        for m in (md, md3)
            g = m["glue"]
            owns(k) = g[k]["owns"]
            @test owns("alloc::string::String") && owns("alloc::vec::Vec<f64, alloc::alloc::Global>")
            @test owns("alloc::boxed::Box<rb_fix::Rec, alloc::alloc::Global>") && owns("rb_fix::Named") && owns("rb_fix::Slot")
            # One integer, and `impl Drop`: nothing in its layout says it must be dropped.
            @test owns("rb_fix::Guard")
            # A raw pointer, and no drop code: nothing in its layout says it need not be.
            @test !owns("rb_fix::View")
            @test !owns("rb_fix::Rec") && !owns("rb_fix::Shape") && !owns("core::option::Option<i32>")
            # Defined in a private module, exported from the crate's root: found under the path that is public.
            @test g["rb_fix::inner::Token"]["spelling"] == "rb_fix::Token" && owns("rb_fix::inner::Token")
            @test g["alloc::vec::Vec<f64, alloc::alloc::Global>"]["spelling"] == "alloc::vec::Vec<f64>"
            @test haskey(g["alloc::vec::Vec<f64, alloc::alloc::Global>"], "vec") && !haskey(g["alloc::string::String"], "vec")
            # No path outside the crate spells `Secret` (public in a private
            # module) or `Fault` (exported only under another name). A function
            # returns each, and that function has a path: the compiler takes
            # the type from it, and says the type is dropped.
            for (key, from) in (("rb_fix::hidden::Secret", "what `rb_fix::secret` returns"),
                                ("rb_fix::hidden::Fault", "the `Err` of what `rb_fix::parse_byte` returns"),
                                ("core::result::Result<u8, rb_fix::hidden::Fault>", "what `rb_fix::parse_byte` returns"))
                @test (key, get(get(g, key, Dict()), "spelling", "refused")) == (key, from)
                @test g[key]["by_address"] === true && owns(key) && startswith(g[key]["drop"], "__rb_dropfn_")
            end
            @test !haskey(g["rb_fix::Named"], "by_address")
            # Nothing returns a `Sealed`: said, with the compiler's own reason and that one.
            @test collect(keys(m["glue_refused"])) == ["rb_fix::hidden::Sealed"]
            @test occursin("module `hidden` is private", m["glue_refused"]["rb_fix::hidden::Sealed"]) &&
                  occursin("no function with a public path returns one", m["glue_refused"]["rb_fix::hidden::Sealed"])
            @test sort(m["glue_helpers"]) == ["__rb_string_new", "__rb_string_view"]
            # Every type the functions move was asked about, and no other.
            @test Set(RG.wanted(m["functions"], m["types"])) == union(Set(keys(g)), Set(keys(m["glue_refused"])))
            @test allunique(e["drop"] for e in values(g))
        end
        @test occursin("ownership: ", out) && occursin("no drop function (1)", out)
        # One line of the file per answered type; the crate's own tree as it was.
        root = joinpath(_RBL_DIR, "build", "rust", "root")
        glue_lines = readlines(joinpath(root, "rb_glue.rs"))
        spelled = [e for e in values(md3["glue"]) if !get(e, "by_address", false)]
        @test count(l -> startswith(l, "#[no_mangle] pub unsafe extern \"C\" fn __rb_drop_"), glue_lines) == length(spelled)
        @test count(l -> startswith(l, "#[no_mangle] pub static __rb_dropfn_"), glue_lines) == length(md3["glue"]) - length(spelled) >= 3
        @test crate_tree == _rbl_tree(joinpath(_RBL_DIR, "crate"))
        # Each drop function is in the library and is its own: asked with no
        # value it says which it is, so none was merged into another. A copy
        # is opened, so this process holds no handle on the path a later
        # build writes to.
        exported = RepliBuild.Compiler._runtime_symbol_names(lib)
        @test all(e -> e["drop"] in exported, values(md3["glue"]))
        copy = tempname() * "." * Libdl.dlext
        cp(lib, copy)
        h = Libdl.dlopen(copy)
        asked = 0
        for e in values(md3["glue"])
            if get(e, "by_address", false)
                # The static holds where the function the compiler picked is; with no value it does nothing.
                fp = unsafe_load(Ptr{Ptr{Cvoid}}(Libdl.dlsym(h, e["drop"])))
                @test fp != C_NULL && ccall(fp, UInt, (Ptr{Cvoid}, Ptr{Cvoid}), C_NULL, C_NULL) == 0
            else
                n = parse(UInt, match(r"(\d+)$", e["drop"]).captures[1])
                @test ccall(Libdl.dlsym(h, e["drop"]), UInt, (Ptr{Cvoid}, Ptr{Cvoid}), C_NULL, C_NULL) == n
            end
            asked += 1
        end
        @test asked == length(md3["glue"]) >= 25
    end

    @testset "functions the library has no symbol for" begin
        md3 = RepliBuild._read_json(meta_path)
        exported = RepliBuild.Compiler._runtime_symbol_names(lib)
        for m in (md, md3)
            f = Dict(r["path"] => r for r in m["functions"])
            # An `#[inline]` function is compiled into its callers and has no
            # symbol; the build finds it in the crate and has it compiled in.
            for path in ("rb_fix::small", "rb_fix::sum_diff", "rb_fix::triple", "rb_fix::greet", "rb_fix::divide",
                         "<rb_fix::Rec>::dsum", "<rb_fix::Rec>::with_d", "<rb_fix::Named>::id", "<rb_fix::Named>::rename")
                @test (path, get(get(f, path, Dict()), "instance", "absent")) == (path, "inline")
                @test f[path]["refusal"] === nothing && f[path]["own_debug_info"]
            end
            # What `derive` writes is inline too.
            @test f["<rb_fix::Tag as core::clone::Clone>::clone"]["instance"] == "inline"
            @test f["<rb_fix::Tag as core::cmp::PartialEq>::eq"]["instance"] == "inline"
            # A generic function for the types the project file asks for.
            for path in ("rb_fix::midpoint::<f64>", "<rb_fix::Pair<i32>>::new", "<rb_fix::Pair<i32>>::sum", "rb_fix::into_f64::<f32>")
                @test (path, get(get(f, path, Dict()), "instance", "absent")) == (path, "instantiate")
                @test f[path]["refusal"] === nothing
            end
            @test f["rb_fix::midpoint::<f64>"]["generic"] && f["rb_fix::into_f64::<f32>"]["generic"]
            # It is described like any function: the same join, on its own `define` line.
            @test f["rb_fix::sum_diff"]["return"]["pass"] == "pair" && f["rb_fix::greet"]["return"]["type"] == "alloc::string::String"
            @test [p["type"] for p in f["<rb_fix::Named>::rename"]["parameters"]] == ["&mut rb_fix::Named", "&str"]
            # A function that is not public is not the crate's API, and is not there.
            @test !haskey(f, "rb_fix::private_helper") && m["instances"]["not_public"] >= 1
            @test m["instances"]["instantiate"] == 4 && m["instances"]["inline"] >= 12
            @test isempty(m["instances_refused"])
            # A function the library exports is reached by its name, as before.
            @test !haskey(f["rb_fix::add"], "address") && !haskey(f["rb_fix::uses_helper"], "address")
            @test allunique(r["symbol"] for r in m["functions"])
        end
        @test occursin("no symbol of their own", out)
        # The loader binds none of these by name: each is reached through an
        # exported static that holds where it is.
        f3 = Dict(r["path"] => r for r in md3["functions"])
        inst = [r for r in md3["functions"] if haskey(r, "address")]
        @test length(inst) >= 16
        @test all(r -> r["address"] in exported, inst) && allunique(r["address"] for r in inst)
        @test !(f3["rb_fix::small"]["symbol"] in exported)
        copy = tempname() * "." * Libdl.dlext
        cp(lib, copy)
        h = Libdl.dlopen(copy)
        at(path) = unsafe_load(Ptr{Ptr{Cvoid}}(Libdl.dlsym(h, f3[path]["address"])))
        @test ccall(at("rb_fix::small"), Int32, (Int32,), 41) == 42
        @test ccall(at("rb_fix::triple"), UInt64, (UInt64,), 5) == 15
        @test ccall(at("rb_fix::midpoint::<f64>"), Float64, (Float64, Float64), 1.0, 4.0) == 2.5
        @test ccall(at("rb_fix::into_f64::<f32>"), Float64, (Float32,), 2.5f0) == 2.5
        # The compile that lists the crate's functions is a second IR file of
        # the crate; the build's own is untouched, and nothing is linked from it.
        deps = joinpath(_RBL_DIR, "build", "rust", "target", "debug", "deps")
        @test count(n -> startswith(n, "rb_fix-") && endswith(n, ".ll"), readdir(deps)) >= 2
        @test crate_tree == _rbl_tree(joinpath(_RBL_DIR, "crate"))
    end

    @testset "a crate whose whole API has no symbol, and an entry that is not a function" begin
        mktempdir() do dir
            cp(joinpath(_RBL_DIR, "crate"), joinpath(dir, "crate"))
            src = joinpath(dir, "crate", "src", "lib.rs")
            toml = joinpath(dir, "replibuild.toml")
            project(extra) = write(toml, "[project]\nname = \"rb_fix\"\n\n[rust]\nmanifest = \"crate/Cargo.toml\"\n" * extra)
            build() = withenv("REPLIBUILD_JULIA_ENV" => "none") do
                redirect_stdout(() -> RepliBuild.build(toml), devnull)
            end
            # Nothing exported at all: one inline function and one generic one.
            write(src, "#[inline] pub fn only(x: i32) -> i32 { x + 1 }\npub fn twice<T: Copy + core::ops::Add<Output = T>>(x: T) -> T { x + x }\n")
            project("instantiate = [\"rb_fix::twice::<u8>\"]\n")
            m = RepliBuild._read_json(joinpath(dirname(build()), "compilation_metadata.json"))
            @test sort!([r["path"] for r in m["functions"]]) == ["rb_fix::only", "rb_fix::twice::<u8>"]
            @test all(r -> haskey(r, "address") && r["refusal"] === nothing, m["functions"])
            # An entry that names no function is the project file's own mistake: an error that names it.
            project("instantiate = [\"rb_fix::no_such::<f64>\"]\n")
            msg = _rbl_error(build)
            @test occursin("[rust] instantiate", msg) && occursin("rb_fix::no_such::<f64>", msg)
            # The entry alone changed, not the crate: the build still acted on it.
            # And it is not left with the line: the next build succeeds.
            project("")
            m2 = RepliBuild._read_json(joinpath(dirname(build()), "compilation_metadata.json"))
            @test [r["path"] for r in m2["functions"]] == ["rb_fix::only"]
            # A crate with nothing to wrap is an error, as an empty C library is.
            write(src, "pub fn twice<T: Copy + core::ops::Add<Output = T>>(x: T) -> T { x + x }\n")
            @test occursin("has no function to wrap", _rbl_error(build))
        end
    end

    @testset "the per-type lines follow the crate" begin
        mktempdir() do dir
            cp(joinpath(_RBL_DIR, "crate"), joinpath(dir, "crate"))
            src = joinpath(dir, "crate", "src", "lib.rs")
            toml = joinpath(dir, "replibuild.toml")
            write(toml, "[project]\nname = \"rb_fix\"\n\n[rust]\nmanifest = \"crate/Cargo.toml\"\n")
            build() = withenv("REPLIBUILD_JULIA_ENV" => "none") do
                redirect_stdout(() -> RepliBuild.build(toml), devnull)
            end
            glue(lib) = RepliBuild._read_json(joinpath(dirname(lib), "compilation_metadata.json"))["glue"]
            write(src, "pub struct A { pub s: String }\npub fn a() -> A { A { s: String::new() } }\n")
            g1 = glue(build())
            @test g1["rb_fix::A"]["owns"] && !haskey(g1, "rb_fix::B")
            # `A` is gone: the line written for it no longer compiles. The build
            # starts over without the old lines and writes the ones this crate needs.
            write(src, "pub struct B { pub v: Vec<u8> }\npub fn b() -> B { B { v: Vec::new() } }\n")
            g2 = glue(build())
            @test g2["rb_fix::B"]["owns"] && !haskey(g2, "rb_fix::A")
            @test !occursin("rb_fix::A", read(joinpath(dir, "build", "rust", "root", "rb_glue.rs"), String))
            # A crate that moves nothing owned has no line at all.
            write(src, "pub fn plain(x: i32) -> i32 { x + 1 }\n")
            g3 = glue(build())
            @test isempty(g3)
            @test read(joinpath(dir, "build", "rust", "root", "rb_glue.rs"), String) == RepliBuild.RustGlue.glue_text(RepliBuild.RustGlue.Row[])[1]
        end
    end

    @testset "a Rust project stays out of the C and C++ pipeline" begin
        cfg = RepliBuild.ConfigurationManager.load_config(_RBL_TOML)
        @test occursin("cargo builds", _rbl_error(() -> RepliBuild.Compiler.compile_project(cfg)))
        @test occursin("[rust] project", _rbl_error(() -> RepliBuild.Wrapper.wrap_library(cfg, lib)))
    end

    @testset "a no_std crate that denies its own warnings, from a dependency" begin
        # The two things the first real library (libm) did, and the Hub's
        # shape: the crate is a [dependencies] entry's tree, not the project's.
        mktempdir() do dir
            cp(joinpath(_RBL_DIR, "nostd"), joinpath(dir, "upstream"))
            before = _rbl_tree(joinpath(dir, "upstream"))
            toml = joinpath(dir, "replibuild.toml")
            write(toml, """
                [project]
                name = "nostd_demo"

                [dependencies.upstream]
                type = "local"
                path = "upstream"

                [rust]
                dependency = "upstream"
                opt_level = 3
                """)
            lib2 = withenv("REPLIBUILD_JULIA_ENV" => "none") do
                redirect_stdout(() -> RepliBuild.build(toml), devnull)
            end
            md2 = RepliBuild._read_json(joinpath(dirname(lib2), "compilation_metadata.json"))
            f2 = Dict(f["path"] => f for f in md2["functions"])
            @test md2["crate"]["name"] == "rb_nostd" && md2["crate"]["root"] == "rb_nostd_rb"
            @test sort!(collect(keys(f2))) == ["rb_nostd::split", "rb_nostd::sum", "rb_nostd::triple"]
            @test all(f -> f["refusal"] === nothing, md2["functions"])
            @test f2["rb_nostd::split"]["return"]["pass"] == "pair" && f2["rb_nostd::split"]["ir"]["return"]["type"] == "{ double, i32 }"
            # The dependency's tree is as it was: no lock file, no target directory.
            @test _rbl_tree(joinpath(dir, "upstream")) == before
            @test isfile(joinpath(dir, "build", "rust", "root", "Cargo.lock"))
            h = Libdl.dlopen(lib2)
            @test ccall(Libdl.dlsym(h, f2["rb_nostd::triple"]["symbol"]), Cint, (Cint,), 14) == 42
            xs = [1.0, 2.0, 3.5]
            @test ccall(Libdl.dlsym(h, f2["rb_nostd::sum"]["symbol"]), Cdouble, (Ptr{Cdouble}, Csize_t), xs, 3) == 6.5
        end
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
