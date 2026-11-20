# `RepliBuild.discover` on a Cargo project, without a toolchain: which crate a
# manifest leads to (from `cargo metadata`'s answer, given here as cargo
# prints it), the project file that names it, and that an existing `[rust]`
# project is never rescanned. The call that asks cargo is in
# test_rust_build_live.jl.

using Test
using RepliBuild

const _RD = RepliBuild.RustBuild

# `cargo metadata --no-deps` for a manifest, cut to what is read: each package
# is (name, manifest, [(target name, kinds)]).
function _rd_metadata(root, packages)
    Dict{String,Any}("workspace_root" => root,
        "packages" => Any[Dict{String,Any}("name" => n, "manifest_path" => m,
                              "targets" => Any[Dict{String,Any}("name" => t, "kind" => Any[k...]) for (t, k) in targets])
                          for (n, m, targets) in packages])
end
_rd_error(f) = try
    f()
    ""
catch e
    sprint(showerror, e)
end

@testset "discover: which crate a Cargo.toml leads to" begin
    root = "/somewhere/geo"
    manifest = root * "/Cargo.toml"

    # A crate: the manifest's own package, and `[rust] package` has nothing to say.
    one = _rd_metadata(root, [("geo-kit", manifest, [("geo_kit", ["lib"])])])
    (crate, package) = _RD.discovered_crate(one, manifest)
    @test (crate.package, crate.name, package) == ("geo-kit", "geo_kit", "")
    @test crate.manifest == manifest && crate.workspace_root == root

    # A crate with a binary beside its library is still that crate.
    both = _rd_metadata(root, [("geo", manifest, [("geo", ["bin"]), ("geo", ["lib"])])])
    @test _RD.discovered_crate(both, manifest)[2] == ""

    # A workspace root that is a package too wins over its members.
    rooted = _rd_metadata(root, [("geo", manifest, [("geo", ["rlib"])]),
                                 ("geo-extra", root * "/extra/Cargo.toml", [("geo_extra", ["lib"])])])
    @test _RD.discovered_crate(rooted, manifest)[1].name == "geo"

    # A workspace with no package of its own and one library crate: that one, by name.
    virtual = _rd_metadata(root, [("geo-cli", root * "/cli/Cargo.toml", [("geo-cli", ["bin"])]),
                                  ("geo-core", root * "/core/Cargo.toml", [("geo_core", ["lib"])])])
    (crate, package) = _RD.discovered_crate(virtual, manifest)
    @test (crate.name, package) == ("geo_core", "geo-core")
    @test crate.manifest == root * "/core/Cargo.toml"
    # The same when the root package is only a binary.
    binroot = _rd_metadata(root, [("geo", manifest, [("geo", ["bin"])]),
                                  ("geo-core", root * "/core/Cargo.toml", [("geo_core", ["lib"])])])
    @test _RD.discovered_crate(binroot, manifest)[2] == "geo-core"

    # cargo answers for the whole workspace from any member's manifest. A
    # member asked for by its own manifest is that member: a library is
    # wrapped, and a binary is refused as a binary, whatever its siblings are.
    cli = root * "/cli/Cargo.toml"
    @test _RD.discovered_crate(virtual, root * "/core/Cargo.toml") |> x -> (x[1].name, x[2]) == ("geo_core", "")
    msg = _rd_error(() -> _RD.discovered_crate(virtual, cli))
    @test occursin("\"geo-cli\" has no library target", msg) && !occursin("geo_core", msg)

    # Several library crates: which to wrap is not in the manifest. The error
    # lists them and shows the file to write.
    several = _rd_metadata(root, [("geo-core", root * "/core/Cargo.toml", [("geo_core", ["lib"])]),
                                  ("geo-io", root * "/io/Cargo.toml", [("geo_io", ["lib"])])])
    msg = _rd_error(() -> _RD.discovered_crate(several, manifest))
    @test occursin("2 library crates: geo-core, geo-io", msg)
    @test occursin("[rust]\nmanifest = \"Cargo.toml\"\npackage = \"geo-core\"", msg)

    # Nothing to wrap: a binary, a workspace of binaries, a C-only library.
    @test occursin("no library target", _rd_error(() -> _RD.discovered_crate(
        _rd_metadata(root, [("geo", manifest, [("geo", ["bin"])])]), manifest)))
    @test occursin("none of its packages has a library target", _rd_error(() -> _RD.discovered_crate(
        _rd_metadata(root, [("a", root * "/a/Cargo.toml", [("a", ["bin"])])]), manifest)))
    @test occursin("only as a C library", _rd_error(() -> _RD.discovered_crate(
        _rd_metadata(root, [("geo", manifest, [("geo", ["cdylib"])])]), manifest)))
end

@testset "discover: the project file it writes" begin
    crate = _RD.Crate("geo-kit", "geo_kit", "/somewhere/geo/Cargo.toml", "/somewhere/geo")
    text = _RD.project_text(crate)
    data = RepliBuild.TOML.parse(text)
    @test data["project"] == Dict("name" => "geo_kit", "root" => ".")
    @test data["rust"] == Dict("manifest" => "Cargo.toml", "opt_level" => 0)
    @test sort!(collect(keys(data))) == ["project", "rust"]          # nothing a build can find out is asked of the file
    @test RepliBuild.TOML.parse(_RD.project_text(crate, "geo-kit"))["rust"]["package"] == "geo-kit"

    # It is a project file the configuration loads, as a Rust project named after the crate.
    mktempdir() do dir
        write(joinpath(dir, "Cargo.toml"), "[package]\nname = \"geo-kit\"\nversion = \"0.1.0\"\n")
        path = joinpath(dir, "replibuild.toml")
        write(path, text)
        @test RepliBuild.ConfigurationManager.is_rust_toml(path)
        cfg = RepliBuild.ConfigurationManager.load_config(path)
        @test cfg.rust !== nothing && cfg.rust.opt_level == 0 && isempty(cfg.rust.package)
        @test replace(cfg.rust.manifest, '\\' => '/') == replace(joinpath(realpath(dir), "Cargo.toml"), '\\' => '/') ||
              replace(cfg.rust.manifest, '\\' => '/') == replace(joinpath(dir, "Cargo.toml"), '\\' => '/')
        @test RepliBuild.ConfigurationManager.get_module_name(cfg) == "GeoKit"
        @test !RepliBuild.ConfigurationManager.is_rust_toml(joinpath(dir, "Cargo.toml"))
        @test !RepliBuild.ConfigurationManager.is_rust_toml(joinpath(dir, "absent.toml"))
        write(joinpath(dir, "broken.toml"), "[rust\n")
        @test !RepliBuild.ConfigurationManager.is_rust_toml(joinpath(dir, "broken.toml"))
    end
end

@testset "discover: an existing [rust] project is never rescanned" begin
    # A scan looks for C and C++ sources; run on a Rust project it would write
    # a C++ project with none. With `force` too, the file is kept byte for
    # byte, whatever else its directory holds.
    mktempdir() do dir
        path = joinpath(dir, "replibuild.toml")
        text = "# mine\n[project]\nname = \"geo\"\n\n[rust]\nmanifest = \"crate/Cargo.toml\"\nfeatures = [\"simd\"]\n" *
               "instantiate = [\"geo::largest::<f64>\"]\n"
        write(path, text)
        write(joinpath(dir, "stray.c"), "int stray(void) { return 1; }\n")
        for force in (false, true)
            out = redirect_stdout(() -> RepliBuild.Discovery.discover(dir; force), devnull)
            @test out == path
            @test read(path, String) == text
        end
    end
    # A project file that is not a Rust one says what the project is: a
    # Cargo.toml beside it does not make it a crate.
    mktempdir() do dir
        write(joinpath(dir, "Cargo.toml"), "[package]\nname = \"geo\"\nversion = \"0.1.0\"\n")
        write(joinpath(dir, "replibuild.toml"), "[project]\nname = \"geo\"\n\n[wrap]\nlanguage = \"c\"\n")
        redirect_stdout(() -> RepliBuild.Discovery.discover(dir), devnull)
        @test !RepliBuild.ConfigurationManager.is_rust_toml(joinpath(dir, "replibuild.toml"))
    end
end

@testset "discover: what it does not run it does not compile" begin
    # `discover` ends by building, or building and registering, when asked to.
    # Called directly, those functions are known when `discover` is compiled,
    # and the first `discover` of a session compiled all of `build`, `wrap`
    # and `register` before it ran (87 s, 2026-11-19), to keep a file or to
    # write one. They are reached through `invokelatest`: timing is not
    # something a test can hold, so this holds the shape.
    src = read(joinpath(pkgdir(RepliBuild), "src", "Builder", "Discovery.jl"), String)
    m = match(r"\nfunction _build_and_wrap\(.*?\n(.*?)\nend\n"s, src)
    @test m !== nothing
    body = m.captures[1]
    @test occursin("Base.invokelatest(parent_mod.register, config_path)", body)
    @test occursin("Base.invokelatest(parent_mod.build, config_path)", body)
    @test !occursin(r"parent_mod\.\w+\(", body)
    # ...and the body of `discover` reaches them only through that function.
    d = match(r"\nfunction discover\(target_dir::String.*?\n(.*?)\nend\n"s, src)
    @test d !== nothing && !occursin(r"parent_mod\.(build|register|wrap)", d.captures[1])
end
