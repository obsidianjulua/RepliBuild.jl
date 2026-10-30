#!/usr/bin/env julia
# test/test_dep_submodules.jl — a git dependency fetches the submodules it lists
#
# `_clone_dep` ran a plain `git clone`, so a superproject such as boostorg/boost
# came back as an empty tree and every Boost module had to be its own dependency.
# `[dependencies.<n>] submodules = [...]` fetches exactly the listed paths, at the
# commits the superproject's tree records, and each one is one more source root
# of the same dependency. These tests drive the real load_config →
# resolve_dependencies path against local repos over file:// (needs `git` only:
# no network, no C/LLVM toolchain), plus one forced `discover`.

using Test
using TOML
using RepliBuild

const _SUB_CM   = RepliBuild.ConfigurationManager
const _SUB_DR   = RepliBuild.DependencyResolver
const _SUB_DISC = RepliBuild.Discovery

# Three slashes and forward separators: the only spelling that is a file URL on
# Windows as well (test_dep_cache.jl says why).
_sub_file_url(p) = "file://" * (Sys.iswindows() ? "/" * replace(abspath(p), '\\' => '/') : abspath(p))

# git refuses file:// for submodules unless asked (protocol.file.allow, CVE-2022-39253).
# The test allows it for the processes it starts; RepliBuild itself never does.
const _SUB_GIT_ENV = ("GIT_CONFIG_COUNT" => "1",
                      "GIT_CONFIG_KEY_0" => "protocol.file.allow",
                      "GIT_CONFIG_VALUE_0" => "always")

@testset "git dependency submodules" begin
    if Sys.which("git") === nothing
        @warn "git not found — skipping dependency submodule tests"
    else
        mktempdir() do sb
            # Identity, signing and line endings pinned: nothing may depend on the
            # developer's global git config (test_dep_cache.jl has the story).
            gitrun(repo, args...) = run(`git -C $repo -c user.email=t@example.com -c user.name=tester -c commit.gpgsign=false -c tag.gpgsign=false -c core.autocrlf=false -c protocol.file.allow=always $(collect(args))`)
            gitout(repo, args...) = String(strip(read(`git -C $repo $(collect(args))`, String)))
            function newrepo(dir, files)
                mkpath(dir)
                run(`git -C $dir init -q`)
                for (rel, text) in files
                    p = joinpath(dir, rel); mkpath(dirname(p)); write(p, text)
                end
                gitrun(dir, "add", "-A"); gitrun(dir, "commit", "-q", "-m", "init")
                return lowercase(gitout(dir, "rev-parse", "HEAD"))
            end

            # Two submodules the recipe lists, one it never does.
            up_a = joinpath(sb, "up_a")
            sha_a1 = newrepo(up_a, ["include/liba.h" => "int liba(void);\n",
                                    "src/liba.c"     => "int liba(void){return 1;}\n",
                                    "tests/a_test.c" => "int a_test(void){return 0;}\n"])
            up_b = joinpath(sb, "up_b")
            newrepo(up_b, ["include/libb.h" => "int libb(void);\n",
                           "libb.c"         => "int libb(void){return 2;}\n",
                           "skipme.c"       => "int skipme(void){return 0;}\n"])
            up_c = joinpath(sb, "up_c")
            newrepo(up_c, ["include/c_extra.h" => "int c_extra(void);\n",
                           "c_extra.c"         => "int c_extra(void){return 3;}\n"])

            # The superproject: v1 records a1 for libs/a.
            sup = joinpath(sb, "super")
            newrepo(sup, ["top.c" => "int top(void){return 0;}\n"])
            gitrun(sup, "submodule", "add", "--quiet", "--", _sub_file_url(up_a), "libs/a")
            gitrun(sup, "submodule", "add", "--quiet", "--", _sub_file_url(up_b), "libs/b")
            gitrun(sup, "submodule", "add", "--quiet", "--", _sub_file_url(up_c), "extras/c")
            # Upstream's .gitmodules may tell `git submodule update` to skip one;
            # the recipe's list decides, so libs/b must be fetched regardless.
            gitrun(sup, "config", "-f", ".gitmodules", "submodule.libs/b.update", "none")
            gitrun(sup, "add", ".gitmodules")
            gitrun(sup, "commit", "-q", "-m", "v1"); gitrun(sup, "tag", "v1")

            # up_a moves on after v1, so v1's recorded commit is not up_a's tip and
            # a depth-1 fetch has to ask for it by name. v2 records the new commit.
            write(joinpath(up_a, "src", "liba2.c"), "int liba2(void){return 12;}\n")
            gitrun(up_a, "add", "-A"); gitrun(up_a, "commit", "-q", "-m", "a2")
            sha_a2 = lowercase(gitout(up_a, "rev-parse", "HEAD"))
            gitrun(sup, "update-index", "--cacheinfo", "160000,$sha_a2,libs/a")
            gitrun(sup, "commit", "-q", "-m", "v2"); gitrun(sup, "tag", "v2")
            @test sha_a1 != sha_a2

            proj = joinpath(sb, "proj"); mkpath(proj)
            toml_path = joinpath(proj, "replibuild.toml")
            write_toml(tag, subs; exclude=String[]) = open(toml_path, "w") do io
                println(io, "[project]\nname = \"subtest\"\nroot = \"", escape_string(proj), "\"\n")
                println(io, "[dependencies.super]\ntype = \"git\"\nurl = \"", _sub_file_url(sup), "\"\ntag = \"", tag, "\"")
                println(io, "submodules = [", join(repr.(subs), ", "), "]")
                println(io, "exclude = [", join(repr.(exclude), ", "), "]")
            end

            depsdir = joinpath(proj, ".replibuild_cache", "deps")
            dep     = joinpath(depsdir, "super")
            marker  = joinpath(depsdir, "super.resolved")
            under(p, root) = startswith(p, joinpath(dep, splitpath(root)...))
            resolve() = withenv(_SUB_GIT_ENV...) do
                _SUB_DR.resolve_dependencies(_SUB_CM.load_config(toml_path))
            end
            function marker_subs()
                s = nothing
                isfile(marker) && for l in eachline(marker); startswith(l, "submodules=") && (s = l[12:end]); end
                s
            end
            # path => prefix char of `git submodule status` (' ' at the recorded
            # commit, '-' not initialized, '+' checked out elsewhere).
            function sub_status()
                st = Dict{String,Char}()
                for l in eachline(`git -C $dep submodule status`)
                    st[split(l[2:end])[2]] = l[1]
                end
                st
            end

            @testset "fetches exactly the listed submodules" begin
                write_toml("v1", ["libs/b", "libs/a"]; exclude=["libs/b/skipme"])
                @test _SUB_CM.load_config(toml_path).dependencies.items["super"].submodules ==
                      ["libs/a", "libs/b"]           # sorted at load
                cfg = resolve()
                st = sub_status()
                @test st == Dict("libs/a" => ' ', "libs/b" => ' ', "extras/c" => '-')
                # At the commit the superproject records, not the submodule's tip.
                @test lowercase(gitout(joinpath(dep, "libs", "a"), "rev-parse", "HEAD")) == sha_a1
                @test marker_subs() == "libs/a,libs/b"

                srcs = cfg.compile.source_files
                @test joinpath(dep, "top.c") in srcs
                @test joinpath(dep, "libs", "a", "src", "liba.c") in srcs
                @test joinpath(dep, "libs", "b", "libb.c") in srcs
                @test !(joinpath(dep, "libs", "a", "tests", "a_test.c") in srcs)  # skipped dir
                @test !(joinpath(dep, "libs", "b", "skipme.c") in srcs)           # exclude, by its path in the dependency
                @test !any(s -> under(s, "extras"), srcs)
                @test count(==(joinpath(dep, "libs", "a", "src", "liba.c")), srcs) == 1

                incs = cfg.compile.include_dirs
                for d in (dep, joinpath(dep, "libs", "a"), joinpath(dep, "libs", "a", "include"),
                          joinpath(dep, "libs", "a", "src"), joinpath(dep, "libs", "b"),
                          joinpath(dep, "libs", "b", "include"))
                    @test d in incs
                end
                @test !any(s -> under(s, "extras"), incs)

                # Unchanged recipe: a cache hit, same answer.
                cfg2 = resolve()
                @test Set(cfg2.compile.source_files) == Set(srcs)
            end

            @testset "a submodule removed from the list contributes nothing" begin
                # libs/b stays checked out in the cache; it must not leak back in
                # through the superproject's own source walk.
                write_toml("v1", ["libs/a"])
                cfg = resolve()
                @test marker_subs() == "libs/a"
                @test isfile(joinpath(dep, "libs", "b", "libb.c"))
                @test !any(s -> under(s, "libs/b"), cfg.compile.source_files)
                @test !any(s -> under(s, "libs/b"), cfg.compile.include_dirs)
                @test joinpath(dep, "libs", "a", "src", "liba.c") in cfg.compile.source_files
            end

            @testset "a submodule added to the list is fetched" begin
                write_toml("v1", ["extras/c", "libs/a"])
                cfg = resolve()
                @test sub_status()["extras/c"] == ' '
                @test marker_subs() == "extras/c,libs/a"
                @test joinpath(dep, "extras", "c", "c_extra.c") in cfg.compile.source_files
                @test joinpath(dep, "extras", "c", "include") in cfg.compile.include_dirs
            end

            @testset "a tag bump moves submodules to the new recorded commits" begin
                write_toml("v2", ["libs/a"])
                cfg = resolve()
                @test lowercase(gitout(joinpath(dep, "libs", "a"), "rev-parse", "HEAD")) == sha_a2
                @test joinpath(dep, "libs", "a", "src", "liba2.c") in cfg.compile.source_files
            end

            @testset "a submodule moved off its recorded commit is a hard error" begin
                gitrun(joinpath(dep, "libs", "a"), "commit", "-q", "--allow-empty", "-m", "drift")
                err = try resolve(); nothing catch e; e end
                @test err isa ErrorException
                err isa ErrorException && @test occursin("submodule 'libs/a'", err.msg)
                err isa ErrorException && @test occursin(sha_a2, err.msg)   # names the recorded commit
            end

            @testset "a listed path that is not a submodule is refused" begin
                for bad in ("libs/nope", "libs")   # absent; a directory, not a gitlink
                    rm(depsdir; recursive=true, force=true)
                    write_toml("v1", ["libs/a", bad])
                    err = try resolve(); nothing catch e; e end
                    @test err isa ErrorException
                    err isa ErrorException && @test occursin("does not record: $bad", err.msg)
                    @test !isdir(dep)   # no half-resolved clone left to read as a cache
                end
            end

            @testset "a marker from before submodules reads as none" begin
                legacy = joinpath(sb, "legacy.resolved")
                write(legacy, "url=u\ntag=v1\ncommit=$sha_a1\n")
                @test _SUB_DR._read_dep_marker(legacy) == ("u", "v1", sha_a1, String[])
            end
        end
    end
end

@testset "dependency submodules: entries refused at load" begin
    mktempdir() do dir
        toml = joinpath(dir, "replibuild.toml")
        write_dep(body) = write(toml, "[project]\nname = \"subload\"\n\n[dependencies.super]\n" * body)
        git = "type = \"git\"\nurl = \"https://example.invalid/super.git\"\n"

        write_dep(git * "submodules = [\"libs/math\", \"libs/config\", \"libs/math\"]\n")
        @test _SUB_CM.load_config(toml).dependencies.items["super"].submodules == ["libs/config", "libs/math"]
        write_dep(git)
        @test isempty(_SUB_CM.load_config(toml).dependencies.items["super"].submodules)

        bad_entries = ["../x", "libs/../x", "./libs/a", "-x", "--recursive", "/abs", "C:/x",
                       "libs\\\\a", "a b", "a;b", "a|b", "\$(x)", "`x`", "a*", "a//b", "libs/a/", ""]
        refused = 0
        for e in bad_entries
            write_dep(git * "submodules = [\"$e\"]\n")
            @test_throws ErrorException _SUB_CM.load_config(toml)
            refused += 1
        end
        @test refused == length(bad_entries)

        write_dep(git * "submodules = \"libs/a\"\n")             # not a list
        @test_throws ErrorException _SUB_CM.load_config(toml)
        write_dep(git * "submodules = [1]\n")                    # not a string
        @test_throws ErrorException _SUB_CM.load_config(toml)
        write_dep("type = \"local\"\npath = \"vendor\"\nsubmodules = [\"libs/a\"]\n")
        @test_throws ErrorException _SUB_CM.load_config(toml)    # inert off git: refused
    end
end

@testset "dependency submodules survive save_config and discover(force=true)" begin
    mktempdir() do proj
        write(joinpath(proj, "lib.c"), "int f(int x){return x+1;}\n")
        toml = joinpath(proj, "replibuild.toml")
        write(toml, """
        [project]
        name = "subdisc"
        root = "."

        [dependencies.super]
        type = "git"
        url = "https://example.invalid/super.git"
        tag = "v1"
        submodules = ["libs/math", "libs/config"]
        """)

        cfg = _SUB_CM.load_config(toml)
        _SUB_CM.save_config(cfg)
        @test TOML.parsefile(toml)["dependencies"]["super"]["submodules"] == ["libs/config", "libs/math"]

        kept = _SUB_DISC._collect_preserved_sections(toml)
        @test kept !== nothing && haskey(kept, ("dependencies", "super"))

        # The real path: re-discovery regenerates the TOML from the tree, which
        # names no dependencies at all.
        RepliBuild.discover(proj; force=true)
        doc = TOML.parsefile(toml)
        @test doc["dependencies"]["super"]["submodules"] == ["libs/config", "libs/math"]
        @test doc["dependencies"]["super"]["url"] == "https://example.invalid/super.git"
        @test doc["dependencies"]["super"]["tag"] == "v1"
        @test _SUB_CM.load_config(toml).dependencies.items["super"].submodules == ["libs/config", "libs/math"]
    end
end
