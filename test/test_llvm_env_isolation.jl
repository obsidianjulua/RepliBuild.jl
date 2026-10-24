#!/usr/bin/env julia
# test/test_llvm_env_isolation.jl — the toolchain environment reaches the child
# process and never this process's ENV.
#
# `BuildBridge.run_command` wrapped every tool in `with_llvm_env`. That
# function saves the toolchain keys from `ENV` (PATH, LD_LIBRARY_PATH, CPATH,
# LLVM_DIR, ...), sets them, runs the command and restores them. It is called
# concurrently: the per-TU IR compile runs under `Threads.@threads` and the
# per-TU object compile under `asyncmap`. Overlapping cycles do not nest. A
# saves the user's LD_LIBRARY_PATH and sets "" (a system toolchain
# force-clears it), B saves "", A restores the user's value, and then B
# restores "" by deleting the variable. After a parallel C++ build the user's
# LD_LIBRARY_PATH and CPATH were gone, PATH kept the LLVM bin prefix, and
# LLVM_DIR/Clang_DIR stayed set. Every later subprocess inherited that. On
# 2026-10-02, two of three devtests runs lost the LD_LIBRARY_PATH that
# libJLCS needed, and the stress_test verify child died on libLLVM.so.22.1.
#
# The environment now rides on the command (`LLVMEnvironment.llvm_cmd`). The
# toolchain is fabricated here with the real `build_environment_vars`, so
# nothing needs LLVM. The children are this julia, printing what they received.
# A watcher samples ENV while the children run, because the old code fails
# that check whatever order the children finish in.

using Test
using RepliBuild

const _LE = RepliBuild.LLVMEnvironment
const _BB = RepliBuild.BuildBridge

# ── No caller of with_llvm_env in src/ ───────────────────────────────────────
# Its save/restore cycle is global, so a call on any path that can run
# concurrently brings the defect back. Checked on the parsed AST: comments, the
# docstrings that explain this, and the definition's own signature are not
# calls. First, because it is cheap and a failing testset ends the file.

@testset "nothing in src/ calls with_llvm_env" begin
    is_target(f) = f === :with_llvm_env ||
        (f isa Expr && f.head === :. && f.args[end] == QuoteNode(:with_llvm_env))
    is_short_def(ex) = ex.head === :(=) && ex.args[1] isa Expr &&
        (ex.args[1].head in (:call, :where) ||
         (ex.args[1].head === :(::) && ex.args[1].args[1] isa Expr &&
          ex.args[1].args[1].head === :call))

    function calls!(hits, ex)
        ex isa Expr || return hits
        # A definition's signature is a call expression too; only its body runs.
        if ex.head === :function || is_short_def(ex)
            length(ex.args) >= 2 && calls!(hits, ex.args[2])
            return hits
        end
        ex.head === :call && is_target(ex.args[1]) && push!(hits, ex)
        foreach(a -> calls!(hits, a), ex.args)
        return hits
    end
    calls_in(src::AbstractString) = calls!(Expr[], Meta.parseall(src))

    # The walker on shapes whose answer is known, so an empty result below
    # means no callers rather than a walker that sees nothing.
    @test length(calls_in("with_llvm_env(() -> run(c))")) == 1
    @test length(calls_in("g = LLVMEnvironment.with_llvm_env() do\n  run(c)\nend")) == 1
    @test length(calls_in("function f()\n  x::Int = with_llvm_env(h)\nend")) == 1
    @test isempty(calls_in("function with_llvm_env(f::Function)\n  f()\nend"))
    @test isempty(calls_in("with_llvm_env(f) = f()"))
    @test isempty(calls_in("# with_llvm_env(f)\nx = \"\"\"with_llvm_env() do\"\"\""))
    @test isempty(calls_in("export with_llvm_env"))

    src_dir = joinpath(dirname(@__DIR__), "src")
    offenders = String[]
    scanned = 0
    for (root, _, files) in walkdir(src_dir), f in files
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        ast = Meta.parseall(read(path, String); filename=path)
        bad = filter(a -> a isa Expr && a.head in (:incomplete, :error), ast.args)
        @test isempty(bad)
        scanned += 1
        isempty(calls!(Expr[], ast)) || push!(offenders, relpath(path, src_dir))
    end
    @test scanned > 30
    isempty(offenders) || @warn "with_llvm_env is called here; attach the environment with LLVMEnvironment.llvm_cmd instead" offenders
    @test isempty(offenders)
end

# A toolchain under a fake prefix, with the environment the real builder gives
# that source. "system" is what a box with an installed LLVM gets, and the
# mode that force-clears LD_LIBRARY_PATH and CPATH to "".
function _fake_toolchain(root::String, source::String; tools=Dict{String,String}())
    bin, lib, inc = joinpath(root, "bin"), joinpath(root, "lib"), joinpath(root, "include")
    env_vars = _LE.build_environment_vars(root, bin, lib, source)
    return _LE.LLVMToolchain(root, bin, lib, inc, joinpath(root, "libexec"),
                             joinpath(root, "share"), "0.0.0", 0, 0, 0, tools,
                             Dict{String,String}(), String[], String[], "",
                             env_vars, true, source)
end

# Prints what the child received for each key named in ARGS, "<unset>" for an
# absent key, after sleeping ARGS[1] seconds so the calls overlap.
const _CHILD = """
sleep(parse(Float64, ARGS[1]))
for k in ARGS[2:end]
    v = get(ENV, k, nothing)
    println("ENV ", k, "=", v === nothing ? "<unset>" : v)
end
"""

function _child_view(output::AbstractString)
    view = Dict{String,Union{String,Nothing}}()
    for m in eachmatch(r"^ENV ([^=\n]+)=(.*)$"m, output)
        view[m.captures[1]] = m.captures[2] == "<unset>" ? nothing : String(m.captures[2])
    end
    return view
end

# A key the toolchain maps to "" must be empty or absent for the child. Never
# inherited: that is the force-clear. Everything else must match exactly.
_received(expected::String, got) = isempty(expected) ? got in ("", nothing) : got == expected

@testset "LLVM tool environment: on the command, never in ENV" begin
    saved_toolchain = _LE.GLOBAL_LLVM_TOOLCHAIN[]
    jl = Base.julia_cmd()
    julia_exe = jl.exec[1]
    # Each testset below runs from this same ENV and gets it back afterwards,
    # so one that corrupts ENV cannot fail the next one.
    sentinels = ("PATH" => ENV["PATH"], "LD_LIBRARY_PATH" => "rb-sentinel-ld",
                 "CPATH" => "rb-sentinel-cpath", "RB_ENV_PROBE" => "inherited",
                 "LLVM_ROOT" => nothing, "LLVM_DIR" => nothing, "Clang_DIR" => nothing)
    isolated(f) = withenv(f, sentinels...)
    try
        tc = _fake_toolchain(mktempdir(), "system"; tools=Dict("julia" => julia_exe))
        _LE.GLOBAL_LLVM_TOOLCHAIN[] = tc
        keys_ = sort!(collect(keys(tc.env_vars)))
        @test tc.env_vars["LD_LIBRARY_PATH"] == "" && tc.env_vars["CPATH"] == ""
        snapshot() = Dict(k => get(ENV, k, nothing) for k in keys_)

        @testset "llvm_cmd attaches the environment and writes nothing" begin
            isolated() do
                before = snapshot()
                cmd = _LE.llvm_cmd(`tool --flag`)
                @test snapshot() == before
                @test cmd.exec == ["tool", "--flag"]
                env = _LE._env_dict(cmd.env)
                for (k, v) in tc.env_vars
                    @test get(env, k, nothing) == v      # "" stays set and empty
                end
                @test env["RB_ENV_PROBE"] == "inherited"

                # A command that already has an environment keeps it, with the
                # toolchain's over it. ENV is not consulted.
                own = _LE.llvm_cmd(ignorestatus(setenv(`tool`, "ONLY_MINE" => "1",
                                                       "LD_LIBRARY_PATH" => "mine")))
                e = _LE._env_dict(own.env)
                @test e["ONLY_MINE"] == "1"
                @test e["LD_LIBRARY_PATH"] == ""
                @test !haskey(e, "RB_ENV_PROBE")
                @test own.ignorestatus
            end
        end

        @testset "environment helpers, both name-case rules" begin
            @test _LE._env_dict(["A=1", "B=x=y", "EMPTY=", "=C:=C:\\work"]) ==
                  Dict("A" => "1", "B" => "x=y", "EMPTY" => "", "=C:" => "C:\\work")

            # Windows names are case-insensitive. ENV iteration there gives
            # `PATH`, but a Cmd's own env can say `Path`, and the toolchain key
            # is `Clang_DIR`. Both rules are run here, whatever the host.
            base = Dict("Path" => "C:\\old", "CLANG_DIR" => "stale", "KEEP" => "1")
            overlay = Dict("PATH" => "C:\\llvm;C:\\old", "Clang_DIR" => "C:\\llvm\\lib",
                           "CPATH" => "")
            @test _LE._overlay_env(base, overlay; casefold=true) ==
                  Dict("PATH" => "C:\\llvm;C:\\old", "Clang_DIR" => "C:\\llvm\\lib",
                       "CPATH" => "", "KEEP" => "1")
            @test _LE._overlay_env(base, overlay; casefold=false) ==
                  Dict("Path" => "C:\\old", "PATH" => "C:\\llvm;C:\\old",
                       "CLANG_DIR" => "stale", "Clang_DIR" => "C:\\llvm\\lib",
                       "CPATH" => "", "KEEP" => "1")
        end

        # Runs `body`, sampling ENV until it returns. Returns the result, the
        # number of samples, and the samples that differed from `before`.
        function watching_env(body, before)
            done = Threads.Atomic{Bool}(false)
            samples = Ref(0)
            changed = Dict{String,Any}[]
            watcher = @async while !done[]
                now = snapshot()
                samples[] += 1
                now == before || push!(changed, now)
                sleep(0.005)
            end
            result = try
                body()
            finally
                done[] = true
                wait(watcher)
            end
            return result, samples[], changed
        end

        n = 4
        child_args(i) = vcat(jl.exec[2:end],
                             ["--startup-file=no", "--history-file=no", "-e", _CHILD,
                              string(0.1 * i)], keys_)
        # The two concurrent shapes in Compiler.jl.
        shapes = [
            "asyncmap (_compile_ir_to_objects)" => () ->
                asyncmap(i -> _BB.execute(julia_exe, child_args(i)), 1:n; ntasks=n),
            "Threads.@threads (compile_to_ir)" => () -> begin
                out = Vector{Tuple{String,Int}}(undef, n)
                Threads.@threads for i in 1:n
                    out[i] = _BB.execute(julia_exe, child_args(i))
                end
                out
            end,
        ]
        Threads.nthreads() > 1 ||
            @info "one thread: the @threads shape runs its calls in sequence"

        for (label, run_shape) in shapes
            @testset "overlapping BuildBridge.execute: $label" begin
                isolated() do
                    before = snapshot()
                    results, samples, changed = watching_env(run_shape, before)

                    # ENV was the same throughout, not just at the end. The old
                    # code changed it for as long as any child ran.
                    @test samples >= 3
                    @test isempty(changed)
                    @test snapshot() == before

                    # Every child ran and received the toolchain environment.
                    @test length(results) == n
                    @test all(r -> r[2] == 0, results)
                    checked = 0
                    for (output, _) in results
                        view = _child_view(output)
                        for k in keys_
                            @test _received(tc.env_vars[k], get(view, k, "<no report>"))
                            checked += 1
                        end
                    end
                    @test checked == n * length(keys_)
                end
            end
        end

        @testset "run_tool attaches the environment too" begin
            isolated() do
                before = snapshot()
                (output, code) = _LE.run_tool("julia", child_args(0))
                @test code == 0
                view = _child_view(output)
                @test all(k -> _received(tc.env_vars[k], get(view, k, "<no report>")), keys_)
                @test snapshot() == before
            end
        end

        @testset "Discovery's include scan: prepare_cmd reaches the clang command" begin
            mktempdir() do dir
                src = joinpath(dir, "a.cpp")
                write(src, "int a() { return 0; }\n")
                seen = Cmd[]
                RepliBuild.ASTWalker.build_dependency_graph(
                    [src], String[]; use_clang=true, clang_path=julia_exe,
                    prepare_cmd = c -> (push!(seen, c); `$jl --version`))
                @test length(seen) == 1
                @test first(seen).exec[1] == julia_exe
                @test "-E" in first(seen).exec && "-H" in first(seen).exec
            end
        end

        @testset "with_llvm_env: overlapping calls queue instead of interleaving" begin
            # Nothing in src/ calls it. It is still exported, so it must not
            # corrupt ENV when it is used from several tasks. Task i holds the
            # toolchain environment for 0.05*i s. Interleaved, task 1 would
            # restore first and the last task would restore the values task 1
            # had set.
            isolated() do
                before = snapshot()
                inside = Vector{Dict{String,Any}}(undef, n)
                asyncmap(1:n; ntasks=n) do i
                    _LE.with_llvm_env() do
                        inside[i] = snapshot()
                        sleep(0.05 * i)
                    end
                end
                @test snapshot() == before
                @test all(s -> all(k -> _received(tc.env_vars[k], s[k]), keys_), inside)
            end

            # A variable set to "" is restored as "", not deleted.
            isolated() do
                withenv("CPATH" => "") do
                    pre = get(ENV, "CPATH", nothing)
                    _LE.with_llvm_env(() -> nothing)
                    @test get(ENV, "CPATH", nothing) == pre
                end
            end
        end
    finally
        _LE.GLOBAL_LLVM_TOOLCHAIN[] = saved_toolchain
    end
end
