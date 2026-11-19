# A `[rust]` project's configuration, and the parts of the cargo driver that
# need no cargo: the command line, the choice of crate out of `cargo
# metadata`'s answer, the library out of cargo's JSON messages.

using Test
using RepliBuild
using TOML
using Libdl

const RCM = RepliBuild.ConfigurationManager
const RB = RepliBuild.RustBuild

function rust_project(dir::AbstractString, body::AbstractString)
    mkpath(dir)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, "[project]\nname = \"demo\"\n\n" * body)
    return toml
end
rust_error(f) = try
    f()
    ""
catch e
    sprint(showerror, e)
end

@testset "[rust]: the section selects the Rust front end" begin
    mktempdir() do dir
        cfg = RCM.load_config(rust_project(dir, "[rust]\nmanifest = \"crate/Cargo.toml\"\n"))
        rc = cfg.rust
        @test rc !== nothing
        # Paths are absolute and `/`-separated; cargo's target directory is under [paths] build.
        @test rc.manifest == replace(joinpath(realpath(dir), "crate", "Cargo.toml"), '\\' => '/') ||
              rc.manifest == replace(joinpath(abspath(dir), "crate", "Cargo.toml"), '\\' => '/')
        @test endswith(rc.target_dir, "/build/rust/target") && isabspath(rc.target_dir)
        @test rc.package == "" && rc.opt_level == 0 && isempty(rc.features) && rc.default_features
        # The language follows from the section.
        @test cfg.wrap.language === :rust
        @test cfg.graph === nothing && cfg.ingest === nothing

        # A project without the section is what it was.
        plain = RCM.load_config(rust_project(joinpath(dir, "plain"), ""))
        @test plain.rust === nothing && plain.wrap.language === :cpp

        # Every key, read.
        full = RCM.load_config(rust_project(joinpath(dir, "full"), """
            [rust]
            manifest = "Cargo.toml"
            package = "inner"
            opt_level = 3
            features = ["a", "b", "a"]
            default_features = false
            """))
        @test full.rust.package == "inner" && full.rust.opt_level == 3
        @test full.rust.features == ["a", "b"] && !full.rust.default_features
    end
end

@testset "[rust]: a wrong section is an error that names it" begin
    mktempdir() do dir
        load(body) = rust_error(() -> RCM.load_config(rust_project(dir, body)))
        @test occursin("[rust] target: unknown key", load("[rust]\ntarget = \"x\"\n"))
        @test occursin("opt_level", load("[rust]\nopt_level = 4\n"))
        @test occursin("opt_level", load("[rust]\nopt_level = \"3\"\n"))
        @test occursin("opt_level", load("[rust]\nopt_level = true\n"))
        @test occursin("features", load("[rust]\nfeatures = \"a\"\n"))
        @test occursin("default_features", load("[rust]\ndefault_features = 1\n"))
        @test occursin("manifest", load("[rust]\nmanifest = \"\"\n"))
        @test occursin("manifest must be a string", load("[rust]\nmanifest = 3\n"))
        # One mode per project.
        @test occursin("[ingest] or [graph]", load("[rust]\n\n[graph]\nsource = \".\"\n"))
        @test occursin("[ingest] or [graph]", load("[rust]\n\n[ingest]\nlibrary = \"libx.so\"\n"))
        # The language cannot disagree with the section, either way.
        @test occursin("language", load("[rust]\n\n[wrap]\nlanguage = \"c\"\n"))
        @test occursin("no [rust] section", load("[wrap]\nlanguage = \"rust\"\n"))
        @test load("[rust]\n\n[wrap]\nlanguage = \"rust\"\n") == ""
    end
end

@testset "[rust]: the section survives a save and a forced discover" begin
    mktempdir() do dir
        toml = rust_project(dir, """
            [rust]
            manifest = "crate/Cargo.toml"
            package = "inner"
            opt_level = 2
            features = ["simd"]
            default_features = false
            """)
        cfg = RCM.load_config(toml)
        RCM.save_config(cfg)
        saved = TOML.parsefile(toml)
        # Relative again, and nothing derived: a tracked TOML holds no machine path.
        @test saved["rust"] == Dict("manifest" => "crate/Cargo.toml", "package" => "inner", "opt_level" => 2,
                                    "features" => ["simd"], "default_features" => false)
        @test saved["wrap"]["language"] == "rust"
        again = RCM.load_config(toml)
        @test again.rust.manifest == cfg.rust.manifest && again.rust.target_dir == cfg.rust.target_dir
        @test (again.rust.package, again.rust.opt_level, again.rust.features, again.rust.default_features) ==
              ("inner", 2, ["simd"], false)
        # Defaults are not written out as settings.
        RCM.save_config(RCM.load_config(rust_project(joinpath(dir, "min"), "[rust]\n")))
        @test TOML.parsefile(joinpath(dir, "min", "replibuild.toml"))["rust"] ==
              Dict("manifest" => "Cargo.toml", "opt_level" => 0)

        @test ("rust", "*") in RepliBuild.Discovery.PRESERVED_TOML_KEYS
        kept = RepliBuild.Discovery._collect_preserved_sections(toml)
        @test kept[("rust", "manifest")] == "crate/Cargo.toml" && kept[("rust", "features")] == ["simd"]

        # Every copy of a config carries the section along.
        @test RCM.merge_compile_flags(cfg, ["-DX"]).rust === cfg.rust
        @test RCM.with_source_files(cfg, String[]).rust === cfg.rust
        @test RCM.with_include_dirs(cfg, String[]).rust === cfg.rust
        @test RepliBuild.DependencyResolver.resolve_dependencies(cfg).rust === cfg.rust
    end
end

@testset "[rust]: wrap says the generator is not built" begin
    mktempdir() do dir
        toml = rust_project(dir, "[rust]\n")
        # `wrap` says what exists and what does not, and does not call it a failure of the build.
        # (The guards inside `compile_project` and `wrap_library` are held by
        # test_rust_build_live.jl, which has those two compiled anyway.)
        msg = rust_error(() -> RepliBuild.wrap(toml))
        @test occursin("not built yet", msg) && occursin("compilation_metadata.json", msg)
    end
end

@testset "cargo: the command line" begin
    rc(opt; package = "", features = String[], default_features = true) =
        RCM.RustConfig("/p/crate/Cargo.toml", package, opt, features, default_features, "/p/build/rust/target")
    crate = RB.Crate("rb-fix", "rb_fix", "/p/crate/Cargo.toml", "/p/crate")
    args = RB.cargo_args(rc(0), crate; locked = true, windows = false)
    @test args[1] == "rustc"
    config(args) = [args[i+1] for i in eachindex(args) if args[i] == "--config"]
    for needed in (["--manifest-path", "/p/crate/Cargo.toml"], ["--package", "rb-fix"], ["--crate-type", "dylib"],
                   ["--target-dir", "/p/build/rust/target"])
        i = findfirst(==(needed[1]), args)
        @test i !== nothing && args[i+1] == needed[2]
    end
    @test "--lib" in args && "--locked" in args && "--message-format=json-render-diagnostics" in args
    @test !("--features" in args) && !("--no-default-features" in args)
    # The profile is stated whole: a debug build at 0, with one IR file.
    @test "profile.dev.opt-level=0" in config(args) && "profile.dev.debug=2" in config(args)
    @test "profile.dev.codegen-units=1" in config(args) && "profile.dev.panic=\"unwind\"" in config(args)
    @test "profile.dev.incremental=false" in config(args)
    @test "profile.dev.debug-assertions=true" in config(args) && "profile.dev.overflow-checks=true" in config(args)
    # What reaches rustc for the crate alone comes after `--`.
    after = args[findfirst(==("--"), args)+1:end]
    @test after[1] == "--emit=llvm-ir,link"
    @test "symbol-mangling-version=v0" in after
    @test "link-arg=-Wl,-Bsymbolic" in after                      # ELF: the library binds itself
    @test !any(a -> occursin("no-insert-timestamp", a), after)
    @test all(a -> a in ("-C",) || !startswith(a, "-C"), after)   # each `-C` has its own value

    # Above opt-level 0: a release build that keeps incremental on, which is
    # what keeps a small `pub fn` its symbol.
    args3 = RB.cargo_args(rc(3), crate; locked = false, windows = false)
    @test "profile.dev.opt-level=3" in config(args3) && "profile.dev.incremental=true" in config(args3)
    @test "profile.dev.debug-assertions=false" in config(args3) && "profile.dev.overflow-checks=false" in config(args3)
    @test "profile.dev.debug=2" in config(args3) && !("--locked" in args3)

    # PE: no self-binding flag (a DLL never imports its own symbols), and no clock in the header.
    pe = RB.cargo_args(rc(0), crate; locked = false, windows = true)
    pe_after = pe[findfirst(==("--"), pe)+1:end]
    @test "link-arg=-Wl,--no-insert-timestamp" in pe_after && !("link-arg=-Wl,-Bsymbolic" in pe_after)

    feat = RB.cargo_args(rc(0; features = ["a", "b"], default_features = false), crate; locked = false, windows = false)
    @test feat[findfirst(==("--features"), feat)+1] == "a,b" && "--no-default-features" in feat
    # Flags for cargo come before `--`.
    @test findfirst(==("--features"), feat) < findfirst(==("--"), feat)
end

@testset "cargo: which crate" begin
    metadata(pkgs) = Dict{String,Any}("workspace_root" => "/w", "packages" => pkgs)
    pkg(name, manifest, targets) = Dict{String,Any}("name" => name, "manifest_path" => manifest, "targets" => targets)
    lib(name) = Dict{String,Any}("name" => name, "kind" => ["lib"], "crate_types" => ["lib"])
    bin(name) = Dict{String,Any}("name" => name, "kind" => ["bin"], "crate_types" => ["bin"])
    rc(manifest; package = "") = RCM.RustConfig(manifest, package, 0, String[], true, "/w/build/rust/target")

    one = metadata([pkg("rb-fix", "/w/Cargo.toml", [bin("tool"), lib("rb_fix")])])
    c = RB.select_crate(one, rc("/w/Cargo.toml"))
    @test (c.package, c.name, c.manifest, c.workspace_root) == ("rb-fix", "rb_fix", "/w/Cargo.toml", "/w")
    # A library target named with a `-` is a crate named with `_`.
    dashed = metadata([pkg("my-lib", "/w/Cargo.toml", [lib("my-lib")])])
    @test RB.select_crate(dashed, rc("/w/Cargo.toml")).name == "my_lib"

    ws = metadata([pkg("a", "/w/a/Cargo.toml", [lib("a")]), pkg("b", "/w/b/Cargo.toml", [lib("b")])])
    @test RB.select_crate(ws, rc("/w/Cargo.toml"; package = "b")).name == "b"
    @test RB.select_crate(ws, rc("/w/a/Cargo.toml")).name == "a"
    # A workspace manifest names no package: the error lists what it has.
    msg = rust_error(() -> RB.select_crate(ws, rc("/w/Cargo.toml")))
    @test occursin("workspace", msg) && occursin("[rust] package", msg) && occursin("a, b", msg)
    msg = rust_error(() -> RB.select_crate(ws, rc("/w/Cargo.toml"; package = "c")))
    @test occursin("no such package", msg) && occursin("a, b", msg)
    # A binary crate has no API.
    only_bin = metadata([pkg("tool", "/w/Cargo.toml", [bin("tool")])])
    @test occursin("no library target", rust_error(() -> RB.select_crate(only_bin, rc("/w/Cargo.toml"))))
    proc = metadata([pkg("mac", "/w/Cargo.toml", [Dict{String,Any}("name" => "mac", "kind" => ["proc-macro"],
                                                                   "crate_types" => ["proc-macro"])])])
    @test occursin("no library target", rust_error(() -> RB.select_crate(proc, rc("/w/Cargo.toml"))))
end

@testset "cargo: which file is the library" begin
    crate = RB.Crate("rb-fix", "rb_fix", "/w/Cargo.toml", "/w")
    so = "/t/debug/librb_fix." * Libdl.dlext
    artifact(name, types, files; fresh = false) =
        "{\"reason\":\"compiler-artifact\",\"target\":{\"name\":\"$name\",\"crate_types\":$(repr(types))," *
        "\"kind\":$(repr(types))},\"filenames\":$(repr(files)),\"fresh\":$fresh}"
    messages = join([
        "warning: a line that is not JSON",
        "{\"reason\":\"compiler-message\",\"message\":{\"rendered\":\"x\"}}",
        artifact("dep", ["lib"], ["/t/debug/deps/libdep.rlib"]),
        artifact("rb_fix", ["dylib"], ["/t/debug/librb_fix.rlib", so]; fresh = true),
        "{\"reason\":\"build-finished\",\"success\":true}"], "\n")
    a = RB.find_artifact(messages, crate)
    @test a.path == so && a.fresh
    @test !RB.find_artifact(replace(messages, "\"fresh\":true" => "\"fresh\":false"), crate).fresh
    # The crate built as something else is not its dynamic library.
    wrong = join([artifact("rb_fix", ["rlib"], ["/t/debug/librb_fix.rlib"])], "\n")
    msg = rust_error(() -> RB.find_artifact(wrong, crate))
    @test occursin("no dynamic library", msg) && occursin("rb_fix", msg)
    @test occursin("no artifact at all", rust_error(() -> RB.find_artifact("", crate)))
end

@testset "cargo: the tools are the user's, or the ones named" begin
    @test occursin("not `cargo` or `rustc`", rust_error(() -> RB.tool("gcc")))
    withenv("REPLIBUILD_TOOL_CARGO" => "/opt/rust/bin/cargo", "REPLIBUILD_TOOL_RUSTC" => nothing) do
        @test RB.tool("cargo") == "/opt/rust/bin/cargo"
        @test RB.tool("rustc") == something(Sys.which("rustc"), "rustc")
    end
    # A tool that is not there is named, with what to do about it.
    withenv("REPLIBUILD_TOOL_RUSTC" => joinpath(tempdir(), "no-such-rustc-$(getpid())")) do
        msg = rust_error(() -> RB.toolchain(tempdir()))
        @test occursin("could not be started", msg) && occursin("REPLIBUILD_TOOL_RUSTC", msg)
    end
end

@testset "a path's scope and name" begin
    @test RB._path_scope_name("lean::add") == (["lean"], "add")
    @test RB._path_scope_name("lean::deep::er::f") == (["lean", "deep", "er"], "f")
    @test RB._path_scope_name("lean::largest::<f64>") == (["lean"], "largest")
    @test RB._path_scope_name("c_add") == (String[], "c_add")
    @test RB._path_scope_name("<lean::Acc>::push") == (String[], "push")
    @test RB._path_scope_name("<lean::Shape as lean::Area>::area") == (String[], "area")
end
