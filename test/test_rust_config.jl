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
            instantiate = ["geo::largest::<f64>", " geo::Pair::<i32>::sum ", "geo::largest::<f64>"]
            """))
        @test full.rust.package == "inner" && full.rust.opt_level == 3
        @test full.rust.features == ["a", "b"] && !full.rust.default_features
        # Generic functions to compile for particular types: the one thing no compile of the crate can say.
        @test full.rust.instantiate == ["geo::largest::<f64>", "geo::Pair::<i32>::sum"]
        @test isempty(cfg.rust.instantiate)
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
        # An entry is one Rust expression: not a list of statements, not empty, not a table.
        @test occursin("instantiate", load("[rust]\ninstantiate = \"geo::f::<u8>\"\n"))
        @test occursin("instantiate", load("[rust]\ninstantiate = [\"\"]\n"))
        @test occursin("instantiate", load("[rust]\ninstantiate = [\"a::f; std::process::exit(1)\"]\n"))
        @test occursin("instantiate", load("[rust]\ninstantiate = [\"{ a::f }\"]\n"))
        @test occursin("instantiate", load("[rust]\ninstantiate = [3]\n"))
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
            instantiate = ["geo::largest::<f64>"]
            """)
        cfg = RCM.load_config(toml)
        RCM.save_config(cfg)
        saved = TOML.parsefile(toml)
        # Relative again, and nothing derived: a tracked TOML holds no machine path.
        @test saved["rust"] == Dict("manifest" => "crate/Cargo.toml", "package" => "inner", "opt_level" => 2,
                                    "features" => ["simd"], "default_features" => false,
                                    "instantiate" => ["geo::largest::<f64>"])
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

@testset "[rust] dependency: the crate is a dependency's checkout" begin
    git_dep = """
        [dependencies.upstream]
        type = "git"
        url = "https://example.invalid/upstream.git"
        tag = "v1.0.0"
        commit = "0123456789abcdef0123456789abcdef01234567"
        """
    mktempdir() do dir
        toml = rust_project(dir, "[rust]\ndependency = \"upstream\"\nmanifest = \"./crates//core/Cargo.toml\"\n\n" * git_dep)
        cfg = RCM.load_config(toml)
        # Relative to the checkout, normalised, and joined to it only when asked.
        @test cfg.rust.dependency == "upstream" && cfg.rust.manifest == "crates/core/Cargo.toml"
        root = RepliBuild.DependencyResolver.checkout_root(cfg, "upstream")
        @test RB.manifest_path(cfg) == replace(joinpath(root, "crates", "core", "Cargo.toml"), '\\' => '/')
        @test occursin(".replibuild_cache", RB.manifest_path(cfg))
        located = RB._located(cfg)
        @test located.manifest == RB.manifest_path(cfg) && isempty(located.dependency) &&
              located.target_dir == cfg.rust.target_dir
        # The default manifest is the checkout's own.
        plain = RCM.load_config(rust_project(joinpath(dir, "plain"), "[rust]\ndependency = \"upstream\"\n\n" * git_dep))
        @test plain.rust.manifest == "Cargo.toml"
        # Without a dependency the manifest is where the project says, as before.
        own = RCM.load_config(rust_project(joinpath(dir, "own"), "[rust]\nmanifest = \"crate/Cargo.toml\"\n"))
        @test RB.manifest_path(own) == own.rust.manifest && isabspath(own.rust.manifest)

        # Saved as written: relative to the checkout, never a machine path.
        RCM.save_config(cfg)
        saved = TOML.parsefile(toml)
        @test saved["rust"]["dependency"] == "upstream" && saved["rust"]["manifest"] == "crates/core/Cargo.toml"
        @test RCM.load_config(toml).rust.manifest == "crates/core/Cargo.toml"

        load(body) = rust_error(() -> RCM.load_config(rust_project(joinpath(dir, "bad"), body)))
        @test occursin("names no [dependencies] entry", load("[rust]\ndependency = \"nowhere\"\n\n" * git_dep))
        @test occursin("upstream", load("[rust]\ndependency = \"nowhere\"\n\n" * git_dep))
        @test occursin("no [dependencies] section", load("[rust]\ndependency = \"upstream\"\n"))
        @test occursin("must be relative to the checkout",
                       load("[rust]\ndependency = \"upstream\"\nmanifest = \"/abs/Cargo.toml\"\n\n" * git_dep))
        @test occursin("leaves the checkout",
                       load("[rust]\ndependency = \"upstream\"\nmanifest = \"../Cargo.toml\"\n\n" * git_dep))
        @test occursin("no source tree",
                       load("[rust]\ndependency = \"sys\"\n\n[dependencies.sys]\ntype = \"system\"\n"))
    end
end

@testset "[rust]: wrap needs the build, and the registry hashes the crate" begin
    mktempdir() do dir
        toml = rust_project(dir, "[rust]\nmanifest = \"crate/Cargo.toml\"\n")
        # `wrap` is generated from what `build` wrote.
        msg = rust_error(() -> RepliBuild.wrap(toml))
        @test occursin("Library not found", msg) && occursin("RepliBuild.build", msg)
        @test occursin("does not apply to a [rust] project", rust_error(() -> RepliBuild.wrap(toml; headers = ["x.h"])))

        # What a build of the crate reads is what the registry hashes: every
        # file of the crate's tree, and nothing RepliBuild or cargo wrote.
        mkpath(joinpath(dir, "crate", "src")); mkpath(joinpath(dir, "crate", "target", "debug")); mkpath(joinpath(dir, "crate", ".git"))
        write(joinpath(dir, "crate", "Cargo.toml"), "[package]\nname = \"demo\"\n")
        write(joinpath(dir, "crate", "src", "lib.rs"), "pub fn f() {}\n")
        write(joinpath(dir, "crate", "target", "debug", "junk"), "x")
        write(joinpath(dir, "crate", ".git", "HEAD"), "x")
        cfg = RCM.load_config(toml)
        @test RB.source_root(cfg) == replace(joinpath(realpath(dir), "crate"), '\\' => '/') ||
              RB.source_root(cfg) == replace(joinpath(abspath(dir), "crate"), '\\' => '/')
        @test [first(p) for p in RB.input_files(cfg)] == ["Cargo.toml", "src/lib.rs"]
        @test RB.source_identity(cfg) == ""
        h = RepliBuild.PackageRegistry.hash_config
        before = h(cfg)
        write(joinpath(dir, "crate", "target", "debug", "junk"), "y")          # cargo's own output
        @test h(cfg) == before
        write(joinpath(dir, "crate", "src", "lib.rs"), "pub fn f() -> i32 { 1 }\n")
        @test h(cfg) != before
        # A crate at the project's root: the build and output directories are not its source.
        root = rust_project(joinpath(dir, "flat"), "[rust]\n")
        mkpath(joinpath(dir, "flat", "src")); mkpath(joinpath(dir, "flat", "build")); mkpath(joinpath(dir, "flat", "julia"))
        write(joinpath(dir, "flat", "Cargo.toml"), ""); write(joinpath(dir, "flat", "src", "lib.rs"), "")
        write(joinpath(dir, "flat", "build", "x"), ""); write(joinpath(dir, "flat", "julia", "y"), "")
        @test [first(p) for p in RB.input_files(RCM.load_config(root))] == ["Cargo.toml", "replibuild.toml", "src/lib.rs"]
    end
    # From a dependency, the pin is part of what the source is.
    mktempdir() do dir
        dep(commit) = "[rust]\ndependency = \"up\"\n\n[dependencies.up]\ntype = \"git\"\nurl = \"https://example.invalid/up.git\"\n" *
                      "tag = \"v1\"\ncommit = \"$commit\"\n"
        a = RCM.load_config(rust_project(joinpath(dir, "a"), dep("0"^40)))
        b = RCM.load_config(rust_project(joinpath(dir, "b"), dep("1"^40)))
        @test occursin("0"^40, RB.source_identity(a)) && RB.source_identity(a) != RB.source_identity(b)
        @test endswith(RB.source_root(a), ".replibuild_cache/deps/up") && isempty(RB.input_files(a))   # not fetched: no files, no error
    end
end

@testset "cargo: the generated root crate" begin
    rc(; features = String[], default_features = true) =
        RCM.RustConfig("/p/crate/Cargo.toml", "", "", 0, features, default_features, "/p/build/rust/target")
    crate = RB.Crate("rb-fix", "rb_fix", "/p/up/crate/Cargo.toml", "/p/up")
    @test RB.root_name(crate) == "rb_fix_rb"
    @test RB.root_dir(rc()) == "/p/build/rust/root"         # beside cargo's target directory, never in the crate
    files = RB.root_files(rc(), crate)
    @test sort!(collect(keys(files))) == ["Cargo.toml", "lib.rs"]
    manifest = TOML.parse(files["Cargo.toml"])
    @test manifest["package"]["name"] == "rb_fix_rb" && manifest["lib"]["crate-type"] == ["dylib"]
    @test manifest["lib"]["path"] == "lib.rs"
    # The crate is a path dependency, under its PACKAGE name, from its own directory.
    @test manifest["dependencies"] == Dict("rb-fix" => Dict("path" => "/p/up/crate"))
    # Its own workspace: a workspace in a directory above does not claim it.
    @test manifest["workspace"] == Dict()
    # One line makes the crate part of the library, under its CRATE name...
    @test occursin("\nextern crate rb_fix;\n", files["lib.rs"])
    # ...and the rest is RepliBuild's own: the frame that ends a panic, which
    # only Rust may do, and what drops, reads and builds a value Rust owns.
    # Nothing else of the crate is named or touched.
    code = join((l for l in split(files["lib.rs"], '\n') if !startswith(strip(l), "//")), "\n")
    @test count("rb_fix", code) == 1
    @test occursin("pub extern \"C\" fn __rb_catch(", code) && occursin("catch_unwind(AssertUnwindSafe(|| thunk(args, ret)))", code)
    @test occursin("pub extern \"C\" fn __rb_panic_message(buf: *mut u8, cap: usize) -> usize", code)
    # A drop that panics ends inside the library too.
    @test occursin("unsafe fn __rb_drop<T>(p: *mut T) -> usize", code) &&
          occursin("catch_unwind(AssertUnwindSafe(|| core::ptr::drop_in_place(p)))", code)
    # A `String` is built from text that was checked, and read by the standard library's own functions.
    @test occursin("pub unsafe extern \"C\" fn __rb_string_new(data: *const u8, len: usize, out: *mut String) -> u8", code)
    @test occursin("core::str::from_utf8(bytes)", code)
    @test occursin("pub unsafe extern \"C\" fn __rb_string_view(s: *const String, data: *mut *const u8, len: *mut usize)", code)
    @test occursin("unsafe fn __rb_vec_view<T>(", code) && occursin("unsafe fn __rb_vec_new<T>(", code)
    # What holds the address of a function the root crate compiled because a line named it.
    @test occursin("pub struct __RbAddr(*const ());", code) && occursin("unsafe impl Sync for __RbAddr {}", code)
    # A type no path spells: the compiler picks its drop function from a function that returns one,
    # and which of the two it picks is its answer to "is this type dropped".
    @test occursin("const fn __rb_pick<T>() -> __RbDrop<T>", code) &&
          occursin("if core::mem::needs_drop::<T>() { __rb_drop_owned::<T> } else { __rb_drop_plain::<T> }", code)
    @test occursin("__rb_by_return!(__rb_ret0 __rb_ok0 __rb_err0 __rb_some0 : );", code)
    @test occursin("__rb_by_return!(__rb_ret$(RepliBuild.RustGlue.MAX_ARITY) ", code) &&
          !occursin("__rb_ret$(RepliBuild.RustGlue.MAX_ARITY + 1)", code)
    @test count("#[no_mangle]", code) == 4
    # The per-type lines are a file of their own, which the build writes from what it found.
    @test endswith(rstrip(code), "include!(\"rb_glue.rs\");") && !haskey(files, "rb_glue.rs")
    @test occursin("\nextern crate alloc;\n", code)
    # A crate attribute comes before every item.
    @test findfirst("#![allow(improper_ctypes_definitions)]", code).start < findfirst("extern crate", code).start
    # A panic outside that frame still goes to whatever hook was there before.
    @test occursin("previous(info)", code)
    feat = TOML.parse(RB.root_files(rc(; features = ["a", "b"], default_features = false), crate)["Cargo.toml"])
    @test feat["dependencies"]["rb-fix"] == Dict("path" => "/p/up/crate", "features" => ["a", "b"], "default-features" => false)
    # The same inputs give the same text: a rewrite that changes nothing must not make cargo rebuild.
    @test RB.root_files(rc(), crate) == files

    mktempdir() do dir
        up = joinpath(dir, "up"); mkpath(up)
        r = RCM.RustConfig(joinpath(up, "Cargo.toml"), "", "", 0, String[], true, joinpath(dir, "build", "rust", "target"))
        c = RB.Crate("rb-fix", "rb_fix", joinpath(up, "Cargo.toml"), up)
        root = RB._write_root(r, c)
        @test root == RB.root_dir(r) && isfile(joinpath(root, "Cargo.toml")) && isfile(joinpath(root, "lib.rs"))
        # `lib.rs` includes it, so it is there from the first build on, empty.
        @test read(joinpath(root, "rb_glue.rs"), String) == RepliBuild.RustGlue.glue_text(RepliBuild.RustGlue.Row[])[1]
        @test !isfile(joinpath(root, "Cargo.lock"))            # the crate has none to start from
        stamp = mtime(joinpath(root, "Cargo.toml"))
        sleep(0.05)
        RB._write_root(r, c)
        @test mtime(joinpath(root, "Cargo.toml")) == stamp     # unchanged text is not rewritten
        # The crate's lock file pins its dependencies: the root starts from a
        # copy, and takes a new copy when the crate's changes, not otherwise.
        write(joinpath(up, "Cargo.lock"), "version = 3\n")
        RB._write_root(r, c)
        @test read(joinpath(root, "Cargo.lock"), String) == "version = 3\n"
        write(joinpath(root, "Cargo.lock"), "version = 3\n# cargo added the root package\n")
        RB._write_root(r, c)
        @test occursin("cargo added", read(joinpath(root, "Cargo.lock"), String))
        write(joinpath(up, "Cargo.lock"), "version = 4\n")
        RB._write_root(r, c)
        @test read(joinpath(root, "Cargo.lock"), String) == "version = 4\n"
        @test readdir(up) == ["Cargo.lock"]                    # nothing was written into the crate's tree
    end
end

@testset "cargo: the command line" begin
    rc(opt) = RCM.RustConfig("/p/crate/Cargo.toml", "", "", opt, String[], true, "/p/build/rust/target")
    crate = RB.Crate("rb-fix", "rb_fix", "/p/crate/Cargo.toml", "/p/crate")
    args = RB.cargo_args(rc(0), crate; windows = false)
    @test args[1] == "rustc"
    config(args) = [args[i+1] for i in eachindex(args) if args[i] == "--config"]
    # cargo builds the ROOT crate, into the build directory.
    for needed in (["--manifest-path", "/p/build/rust/root/Cargo.toml"], ["--target-dir", "/p/build/rust/target"])
        i = findfirst(==(needed[1]), args)
        @test i !== nothing && args[i+1] == needed[2]
    end
    # The compiler's messages come as JSON: an error names its file and line, which is how
    # a per-type line that does not compile is found.
    @test "--lib" in args && "--message-format=json" in args
    # What the root crate's own manifest says is not said again.
    @test !("--crate-type" in args) && !("--package" in args) && !("--features" in args) && !("--locked" in args)
    # The profile is stated whole: a debug build at 0, with one IR file per crate.
    @test "profile.dev.opt-level=0" in config(args) && "profile.dev.debug=2" in config(args)
    @test "profile.dev.codegen-units=1" in config(args) && "profile.dev.panic=\"unwind\"" in config(args)
    @test "profile.dev.incremental=false" in config(args)
    @test "profile.dev.debug-assertions=true" in config(args) && "profile.dev.overflow-checks=true" in config(args)
    # After `--`: the root crate's link, which is the library's.
    after = args[findfirst(==("--"), args)+1:end]
    @test after == ["-C", "link-arg=-Wl,-Bsymbolic"]              # ELF: the library binds itself

    # Above opt-level 0: a release build that keeps incremental on, which is
    # what keeps a small `pub fn` its symbol.
    args3 = RB.cargo_args(rc(3), crate; windows = false)
    @test "profile.dev.opt-level=3" in config(args3) && "profile.dev.incremental=true" in config(args3)
    # ...and it is stated a second time where no configuration file can undo
    # it: `build.incremental` overrides every profile's own setting.
    @test "build.incremental=true" in config(args3) && "build.incremental=false" in config(args)
    @test "profile.dev.debug-assertions=false" in config(args3) && "profile.dev.overflow-checks=false" in config(args3)
    @test "profile.dev.debug=2" in config(args3)

    # The compile that lists the crate's functions: the wrapped crate alone, with the build's own
    # profile and target directory (its dependencies are the ones already compiled), keeping dead code.
    disc = RB.discovery_args(rc(3), crate; )
    @test disc[1] == "rustc" && disc[end-4:end] == ["-p", "rb-fix", "--", "-C", "link-dead-code"]
    @test count(==("--"), disc) == 1 && !any(a -> startswith(a, "link-arg="), disc)
    @test config(disc) == config(args3) && "--lib" in disc
    for needed in (["--manifest-path", "/p/build/rust/root/Cargo.toml"], ["--target-dir", "/p/build/rust/target"])
        i = findfirst(==(needed[1]), disc)
        @test i !== nothing && disc[i+1] == needed[2]
    end

    # PE: no self-binding flag (a DLL never imports its own symbols), and no clock in the header.
    pe = RB.cargo_args(rc(0), crate; windows = true)
    @test pe[findfirst(==("--"), pe)+1:end] == ["-C", "link-arg=-Wl,--no-insert-timestamp"]
end

@testset "cargo: the flags every rustc of the build gets" begin
    plain = ["--emit=llvm-ir,link", "-Csymbol-mangling-version=v0", "--cap-lints=warn"]
    # A function the optimizer merged into another stays a function, with its own debug info.
    @test RB.MERGED_FUNCTION_FLAGS == ["-Cllvm-args=-mergefunc-use-aliases=false", "-Cllvm-args=-mergefunc-preserve-debug-info"]
    ours = vcat(plain, RB.MERGED_FUNCTION_FLAGS)
    @test RB.rustflags(; base = Dict{String,String}()) == ours
    # ...unless this rustc's LLVM does not know those options, which it says by refusing to start.
    @test RB.rustflags(; base = Dict{String,String}(), keep_merged = false) == plain
    @test RB._unknown_llvm_argument("rustc: Unknown command line argument '-mergefunc-use-aliases=false'.  Try: 'rustc --help'")
    @test !RB._unknown_llvm_argument("error[E0308]: mismatched types") && !RB._unknown_llvm_argument("Unknown command line argument '-foo'")
    # The user's own flags stay, in front; either spelling of them.
    @test RB.rustflags(; base = Dict("RUSTFLAGS" => "-C target-cpu=native  -g")) == vcat(["-C", "target-cpu=native", "-g"], ours)
    @test RB.rustflags(; base = Dict("CARGO_ENCODED_RUSTFLAGS" => "-Clink-arg=-fuse-ld=lld\x1f--cfg\x1ffoo=\"a b\"",
                                     "RUSTFLAGS" => "ignored")) ==
          vcat(["-Clink-arg=-fuse-ld=lld", "--cfg", "foo=\"a b\""], ours)
    # They reach cargo as the one variable it reads before any configuration file.
    tc = RB.Toolchain("cargo", "rustc", "1.0.0", "x86_64-unknown-linux-gnu", "20.0.0", "id")
    cmd = withenv("RUSTFLAGS" => "-g", "CARGO_ENCODED_RUSTFLAGS" => nothing) do
        RB._cargo(tc, ["metadata"], tempdir())
    end
    env = Dict(split(e, '='; limit = 2)[1] => split(e, '='; limit = 2)[2] for e in cmd.env)
    @test split(env["CARGO_ENCODED_RUSTFLAGS"], '\x1f') == vcat(["-g"], ours)
    # Whether the build is incremental is the profile's to say. The two
    # variables that override every profile do not reach cargo: a CI
    # environment sets CARGO_INCREMENTAL=0, and with it an optimized build
    # gave a third of the fixture's functions no symbol (2026-11-20).
    for var in ("CARGO_INCREMENTAL", "CARGO_BUILD_INCREMENTAL")
        inherited = withenv(() -> RB._cargo(tc, ["metadata"], tempdir()), var => "0")
        @test !any(e -> startswith(e, var * "="), inherited.env)
    end
    without = withenv(() -> RB._cargo(tc, ["metadata"], tempdir(); keep_merged = false), "RUSTFLAGS" => nothing, "CARGO_ENCODED_RUSTFLAGS" => nothing)
    @test any(e -> e == "CARGO_ENCODED_RUSTFLAGS=" * join(plain, '\x1f'), without.env)
    @test !haskey(env, "RUSTFLAGS") && env["CARGO_TERM_COLOR"] == "never"
end

@testset "cargo: which crate" begin
    metadata(pkgs) = Dict{String,Any}("workspace_root" => "/w", "packages" => pkgs)
    pkg(name, manifest, targets) = Dict{String,Any}("name" => name, "manifest_path" => manifest, "targets" => targets)
    lib(name) = Dict{String,Any}("name" => name, "kind" => ["lib"], "crate_types" => ["lib"])
    bin(name) = Dict{String,Any}("name" => name, "kind" => ["bin"], "crate_types" => ["bin"])
    rc(manifest; package = "") = RCM.RustConfig(manifest, "", package, 0, String[], true, "/w/build/rust/target")

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
    # A library another Rust crate cannot depend on: the root crate has nothing to link.
    c_only = metadata([pkg("capi", "/w/Cargo.toml", [Dict{String,Any}("name" => "capi", "kind" => ["cdylib"],
                                                                      "crate_types" => ["cdylib"])])])
    msg = rust_error(() -> RB.select_crate(c_only, rc("/w/Cargo.toml")))
    @test occursin("only as a C library", msg) && occursin("`lib` or `rlib`", msg)
    both = metadata([pkg("capi", "/w/Cargo.toml", [Dict{String,Any}("name" => "capi", "kind" => ["cdylib", "rlib"],
                                                                    "crate_types" => ["cdylib", "rlib"])])])
    @test RB.select_crate(both, rc("/w/Cargo.toml")).name == "capi"
end

@testset "cargo: what was built" begin
    mktempdir() do dir
        manifest = joinpath(dir, "crate", "Cargo.toml")
        mkpath(dirname(manifest)); write(manifest, "")
        other = joinpath(dir, "vendor", "rb_fix", "Cargo.toml")       # another crate of the same name
        mkpath(dirname(other)); write(other, "")
        crate = RB.Crate("rb-fix", "rb_fix", manifest, dirname(manifest))
        so = "/t/debug/librb_fix_rb." * Libdl.dlext
        artifact(name, types, files, mpath; fresh = false) =
            "{\"reason\":\"compiler-artifact\",\"manifest_path\":$(repr(mpath)),\"target\":{\"name\":\"$name\"," *
            "\"crate_types\":$(repr(types)),\"kind\":$(repr(types))},\"filenames\":$(repr(files)),\"fresh\":$fresh}"
        messages(; root_fresh = true, crate_fresh = true) = join([
            "warning: a line that is not JSON",
            "{\"reason\":\"compiler-message\",\"message\":{\"rendered\":\"x\"}}",
            artifact("rb_fix", ["lib"], ["/t/debug/deps/librb_fix-0000aaaa.rlib"], other; fresh = true),
            artifact("rb_fix", ["lib"], ["/t/debug/deps/librb_fix-3cb2bb3e.rlib", "/t/debug/deps/librb_fix-3cb2bb3e.rmeta"],
                     manifest; fresh = crate_fresh),
            artifact("rb_fix_rb", ["dylib"], [so], "/t/root/Cargo.toml"; fresh = root_fresh),
            "{\"reason\":\"build-finished\",\"success\":true}"], "\n")
        a = RB.find_artifacts(messages(), crate)
        @test a.library == so && a.fresh
        # The IR is beside the crate's rlib, under its name: THIS crate's, told from its namesake by manifest.
        @test replace(a.ir, '\\' => '/') == "/t/debug/deps/rb_fix-3cb2bb3e.ll"
        # Either one recompiled is "built".
        @test !RB.find_artifacts(messages(; root_fresh = false), crate).fresh
        @test !RB.find_artifacts(messages(; crate_fresh = false), crate).fresh
        # No library for the root, or none for the crate, is an error that says what cargo did report.
        no_root = join(filter(l -> !occursin("rb_fix_rb", l), split(messages(), '\n')), "\n")
        msg = rust_error(() -> RB.find_artifacts(no_root, crate))
        @test occursin("no dynamic library for the root crate `rb_fix_rb`", msg) && occursin("rb_fix (lib)", msg)
        no_crate = join(filter(l -> !occursin("3cb2bb3e", l), split(messages(), '\n')), "\n")
        @test occursin("no library for the crate `rb_fix`", rust_error(() -> RB.find_artifacts(no_crate, crate)))
        @test occursin("no artifact at all", rust_error(() -> RB.find_artifacts("", crate)))
    end
end

@testset "cargo: where rustc wrote a crate's files" begin
    # Which directory a crate's own files land in is cargo's business and has
    # changed: `deps/` for every crate through cargo 1.99, and
    # `build/<package>/<hash>/out/` per unit from 1.100. The build looked in
    # `deps/` and failed on the first build with a new cargo (2026-11-20). It
    # reads the place from the names cargo reports, under either layout.
    crate_named(dir) = RB.Crate("rb-fix", "rb_fix", joinpath(dir, "crate", "Cargo.toml"), joinpath(dir, "crate"))
    function tree(dir, files)
        for (path, bytes) in files
            mkpath(dirname(joinpath(dir, path)))
            write(joinpath(dir, path), bytes)
        end
    end
    artifact(name, types, files, mpath) =
        "{\"reason\":\"compiler-artifact\",\"manifest_path\":$(repr(mpath)),\"target\":{\"name\":\"$name\"," *
        "\"crate_types\":$(repr(types)),\"kind\":$(repr(types))},\"filenames\":$(repr(files)),\"fresh\":false}"
    so = "librb_fix_rb." * Libdl.dlext
    unix(p) = replace(p, '\\' => '/')

    for (layout, out_crate, out_root, stale_root) in (
            ("deps/", "target/debug/deps", "target/debug/deps", nothing),
            ("build/<package>/<hash>/out/", "target/debug/build/rb-fix/aaaa1111/out", "target/debug/build/rb_fix_rb/bbbb2222/out",
             "target/debug/build/rb_fix_rb/cccc3333/out"))
        mktempdir() do dir
            crate = crate_named(dir)
            tree(dir, ["crate/Cargo.toml" => "",
                       "$out_crate/librb_fix-aaaa1111.rlib" => "the build's", "$out_crate/rb_fix-aaaa1111.ll" => "ir",
                       "$out_root/$so" => "the library", "$out_root/rb_fix_rb.ll" => "root ir",
                       "target/debug/$so" => "the library"])
            profile = joinpath(dir, "target", "debug")
            # The directories that exist, and only those.
            @test (layout, unix.(RB._output_dirs(profile))) ==
                  (layout, unique!(unix.([joinpath(dir, out_crate), joinpath(dir, out_root)])) |> sort!)
            # The root crate's IR is beside the file the reported library is a copy of.
            @test (layout, unix(RB.root_ir(joinpath(profile, so), crate))) == (layout, unix(joinpath(dir, out_root, "rb_fix_rb.ll")))
            if stale_root !== nothing
                # A profile directory keeps one unit per set of flags it was
                # built with: another opt-level left a library and an IR file
                # of the same names. The one that is the reported library wins.
                tree(dir, ["$stale_root/$so" => "an older build", "$stale_root/rb_fix_rb.ll" => "older ir"])
                @test unix(RB.root_ir(joinpath(profile, so), crate)) == unix(joinpath(dir, out_root, "rb_fix_rb.ll"))
                # With nothing to tell them apart it is not guessed.
                write(joinpath(profile, so), "neither")
                @test RB.root_ir(joinpath(profile, so), crate) === nothing
            end

            # The build's own message names the crate's library with its hash: the IR is beside it.
            built = artifact("rb_fix", ["lib"], [joinpath(dir, out_crate, "librb_fix-aaaa1111.rlib"),
                                                 joinpath(dir, out_crate, "librb_fix-aaaa1111.rmeta")], crate.manifest) * "\n" *
                    artifact("rb_fix_rb", ["dylib"], [joinpath(profile, so)], joinpath(dir, "root", "Cargo.toml"))
            @test unix(RB.find_artifacts(built, crate).ir) == unix(joinpath(dir, out_crate, "rb_fix-aaaa1111.ll"))
            # The compile that lists the crate's functions is reported as the
            # copy cargo makes for the user (no hash) beside the hashed
            # `.rmeta`: the hash and the directory are read from that name.
            listing_dir = layout == "deps/" ? out_crate : "target/debug/build/rb-fix/dddd4444/out"
            tree(dir, ["target/debug/librb_fix.rlib" => "the listing's",
                       "$listing_dir/librb_fix-dddd4444.rlib" => "the listing's", "$listing_dir/rb_fix-dddd4444.ll" => "listing ir"])
            listed = artifact("rb_fix", ["lib"], [joinpath(profile, "librb_fix.rlib"),
                                                  joinpath(dir, listing_dir, "librb_fix-dddd4444.rmeta")], crate.manifest)
            @test (layout, unix(RB._crate_ir(listed, crate))) == (layout, unix(joinpath(dir, listing_dir, "rb_fix-dddd4444.ll")))
            # Only the copy named: the file it is a copy of is found, by content, among every unit's.
            only_copy = artifact("rb_fix", ["lib"], [joinpath(profile, "librb_fix.rlib")], crate.manifest)
            @test (layout, unix(RB._crate_ir(only_copy, crate))) == (layout, unix(joinpath(dir, listing_dir, "rb_fix-dddd4444.ll")))
            # A namesake from another manifest is not this crate.
            other = artifact("rb_fix", ["lib"], [joinpath(dir, listing_dir, "librb_fix-dddd4444.rmeta")], joinpath(dir, "vendor", "Cargo.toml"))
            @test RB._crate_ir(other, crate) === nothing
        end
    end
    # No profile directory at all is no directory, not an error.
    @test isempty(RB._output_dirs(joinpath(tempdir(), "no-such-target-$(getpid())")))
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
