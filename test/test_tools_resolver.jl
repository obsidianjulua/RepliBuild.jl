# test_tools_resolver.jl — which program a non-LLVM tool is.
#
# readelf, objdump, nm, c++filt, git, cmake and ninja used to be whatever the
# name meant on PATH. `RepliBuild.Tools` now answers for each: an explicit
# override, the JLL's copy or PATH's in the order REPLIBUILD_TOOLS names, then
# the bare name. A JLL copy carries the environment it needs on the Cmd, never
# in ENV, and a command literal that names one of these tools must go through
# `Tools.tool_cmd` (the AST guard at the end).
#
# No toolchain: the JLL copies are dependencies of the package.

using Test
using RepliBuild

const _TR = RepliBuild.Tools

# Whether the JLL's copy of a tool is there and starts on this host, in the
# environment the question is asked in. It does everywhere this package's
# dependencies have a working build; where one does not, the resolver's
# answer is the other copy, and that is what these tests hold it to.
function _tr_jll_starts(name::String)::Bool
    p = _TR.PROVIDERS[name]
    path = _TR._jll_path(p)
    return path !== nothing && _TR._starts(name, p, path)
end

# What `resolve` must answer for `name` under `order`, from the two copies.
function _tr_expected(name::String, order::Symbol)::Symbol
    jll = _tr_jll_starts(name)
    sys = Sys.which(name) !== nothing
    order === :jll && return jll ? :jll : (sys ? :system : :missing)
    return sys ? :system : (jll ? :jll : :missing)
end

@testset "every tool resolves to the copy the order names, and it starts" begin
    checked = 0
    for r in _TR.report()
        @test r.source === _tr_expected(r.name, _TR.order())
        if r.source === :missing
            # Neither copy on this host (`ld.lld` where the JLL's does not
            # start and none is installed): the bare name, to fail by name.
            @test r.path == r.name
        else
            @test isfile(r.path)
            @test _TR._probe(_TR.tool_cmd(`$(r.name) --version`)) !== nothing
        end
        checked += 1
    end
    @test checked == length(_TR.PROVIDERS) == 8
end

@testset "the order is REPLIBUILD_TOOLS, and nothing else is accepted" begin
    @test withenv(() -> _TR.order(), "REPLIBUILD_TOOLS" => "jll") === :jll
    @test withenv(() -> _TR.order(), "REPLIBUILD_TOOLS" => " System ") === :system
    @test withenv(() -> _TR.order(), "REPLIBUILD_TOOLS" => nothing) === (Sys.iswindows() ? :system : :jll)
    @test_throws ErrorException withenv(() -> _TR.order(), "REPLIBUILD_TOOLS" => "both")
    @test_throws ErrorException _TR.resolve("no-such-tool")
end

@testset "a copy that is not there is passed over for the other" begin
    # No PATH: `system` first finds nothing and takes the JLL's, where it starts.
    withenv("REPLIBUILD_TOOLS" => "system", "PATH" => "") do
        checked = 0
        for name in ("nm", "git", "ninja")
            r = _TR.resolve(name)
            @test Sys.which(name) === nothing
            @test r.source === _tr_expected(name, :system)
            r.source === :missing ? (@test r.path == name) : (@test isfile(r.path))
            checked += 1
        end
        @test checked == 3
    end
    withenv("REPLIBUILD_TOOLS" => "jll") do
        for r in _TR.report()
            @test r.source === _tr_expected(r.name, :jll)
        end
    end
    jll_first = withenv(() -> [r.name for r in _TR.report() if r.source === :jll], "REPLIBUILD_TOOLS" => "jll")
    @info "tools whose JLL copy starts on this host" jll_first
end

@testset "an override is used as given" begin
    withenv("REPLIBUILD_TOOL_NM" => "/nowhere/my-nm", "REPLIBUILD_TOOL_CXXFILT" => "/nowhere/filt") do
        @test _TR.resolve("nm") == _TR.Resolved("nm", "/nowhere/my-nm", :override)
        @test _TR.tool_path("c++filt") == "/nowhere/filt"
        @test _TR.tool_cmd(`nm -D lib.so`).exec == ["/nowhere/my-nm", "-D", "lib.so"]
        # Another tool is not affected.
        @test _TR.resolve("readelf").source !== :override
    end
    # And the answer follows the variable back.
    @test _TR.resolve("nm").source !== :override
end

@testset "tool_cmd swaps the program and keeps the rest of the command" begin
    withenv("REPLIBUILD_TOOLS" => "jll") do
        dir = mktempdir()
        cmd = _TR.tool_cmd(Cmd(ignorestatus(`git -C $dir status --porcelain`); dir = dir))
        @test cmd.exec[1] == _TR.tool_path("git")
        @test cmd.exec[2:end] == ["-C", dir, "status", "--porcelain"]
        @test cmd.ignorestatus
        @test cmd.dir == dir
        # A path this module handed out is still its tool.
        again = _TR.tool_cmd(`$(_TR.tool_path("nm")) --version`)
        @test again.exec[1] == _TR.tool_path("nm")
        @test (again.env !== nothing) == (_TR.tool_source("nm") === :jll)
    end
    # Not one of this module's tools: the same object back.
    other = `clang++ --version`
    @test _TR.tool_cmd(other) === other
    @test _TR.tool_cmd(`/usr/bin/some-nm -D x`).env === nothing
end

@testset "a JLL copy's environment rides on the Cmd, never on ENV" begin
    before = copy(ENV)
    withenv("REPLIBUILD_TOOLS" => "jll") do
        envof(cmd) = Dict(split(e, '='; limit = 2)[1] => split(e, '='; limit = 2)[2] for e in cmd.env)
        if _TR.tool_source("nm") === :jll
            nm = envof(_TR.tool_cmd(`nm --version`))
            libdirs = String(RepliBuild.Tools.Binutils_jll.LIBPATH[])
            @test startswith(nm[_TR.LIBPATH_KEY], libdirs)
            # One entry for the loader path, whatever case the host spells it in.
            @test count(k -> uppercase(k) == uppercase(_TR.LIBPATH_KEY), collect(keys(nm))) == 1
            # What the command already had stays, behind the copy's own.
            kept = envof(_TR.tool_cmd(setenv(`nm --version`, _TR.LIBPATH_KEY => "/kept/dir")))
            @test endswith(kept[_TR.LIBPATH_KEY], "/kept/dir")
            @test startswith(kept[_TR.LIBPATH_KEY], libdirs)
        else
            # No JLL copy that starts here: the command is left as it is.
            @test _TR.tool_cmd(`nm --version`).env === nothing
        end
        if _TR.tool_source("git") === :jll
            git = envof(_TR.tool_cmd(`git --version`))
            # Where the artifact has a helper directory, git is told of it.
            haskey(git, "GIT_EXEC_PATH") && @test isdir(git["GIT_EXEC_PATH"])
            Sys.islinux() && @test haskey(git, "GIT_EXEC_PATH")
        end
        # cmake and ninja start the system compiler: no library path of Julia's
        # in front of the system's.
        for name in ("cmake", "ninja")
            cmd = _TR.tool_cmd(`$name --version`)
            cmd.env === nothing && continue
            got = _TR._env_get(envof(cmd), _TR.LIBPATH_KEY)
            base = _TR._env_get(ENV, _TR.LIBPATH_KEY)
            # Unchanged, or (cmake, where the loader path IS `PATH`) the
            # process's own with ninja's directory after it. Never a library
            # directory of the JLL's in front.
            appended = name == "cmake" && startswith(got, base) &&
                       isdir(String(last(split(got, _TR._PATHSEP))))
            @test got == base || appended
        end
    end
    @test ENV == before
end

@testset "a program that never answers is not a tool that starts" begin
    # A probe waits for the program, and a program can wait forever (a
    # loader's missing-library dialog on an unattended Windows machine).
    sleeper = `$(Base.julia_cmd()) --startup-file=no -e "sleep(600)"`
    t = @elapsed got = _TR._probe(sleeper; timeout = 2.0)
    @test got === nothing
    @test t < 30
    @test _TR._probe(`$(Base.julia_cmd()) --startup-file=no -e "print(41 + 1)"`) == "42"
    @test _TR._probe(`$(Base.julia_cmd()) --startup-file=no -e "exit(3)"`) === nothing
    @test _TR._probe(`$(joinpath(mktempdir(), "no-such-program")) --version`) === nothing
end

@testset "BuildBridge.execute runs the resolved copy" begin
    for name in ("nm", "readelf", "objdump")
        out, code = RepliBuild.BuildBridge.execute(name, ["--version"])
        @test code == 0
        @test RepliBuild.Compiler._is_gnu_binutils(out)
    end
    out, code = RepliBuild.BuildBridge.execute("cmake", ["--version"]; use_llvm_env = false)
    @test code == 0
    @test occursin("cmake version", out)
end

# A command literal whose program is one of these tools runs PATH's copy with
# no environment unless it goes through `Tools.tool_cmd`. Walked on the parsed
# source, since prose in docstrings spells the same commands.
@testset "no command literal names a resolved tool outside Tools.tool_cmd" begin
    src = joinpath(dirname(@__DIR__), "src")
    names = collect(keys(_TR.PROVIDERS))
    is_tool_cmd(ex) = ex isa Expr && ex.head === :macrocall && ex.args[1] isa GlobalRef &&
        ex.args[1].name === Symbol("@cmd") && ex.args[end] isa String &&
        any(n -> ex.args[end] == n || startswith(ex.args[end], n * " "), names)
    wrapped(ex) = ex isa Expr && ex.head === :call && ex.args[1] isa Expr &&
        ex.args[1].head === :. && ex.args[1].args[end] == QuoteNode(:tool_cmd)
    offenders = String[]
    literals = 0
    function walk(ex, inside::Bool, file)
        ex isa Expr || return
        if is_tool_cmd(ex)
            literals += 1
            inside || push!(offenders, "$file: `$(ex.args[end])`")
            return
        end
        now = inside || wrapped(ex)
        foreach(a -> walk(a, now, file), ex.args)
    end
    files = 0
    for (root, _, fs) in walkdir(src), f in fs
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        ast = Meta.parseall(read(path, String); filename = path)
        @test !(ast.head in (:incomplete, :error))
        walk(ast, false, relpath(path, src))
        files += 1
    end
    @test files > 20
    @test literals >= 10          # the git, nm and objdump sites exist and were seen
    @test isempty(offenders)
    isempty(offenders) || foreach(o -> println("  not through Tools.tool_cmd: ", o), offenders)
end
