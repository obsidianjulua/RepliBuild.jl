# test/test_debug_inspection.jl — RepliBuild.Debug: static inspection of emitted code
#
# devtests §14. Needs the toolchain (objdump) and a JIT'd library, and drives the
# object capture through a SUBPROCESS on purpose: MLIR's object cache has to
# exist when the engine is created, and engines are cached per library per
# process, so a test that ran after any other test touched vi_test could never
# obtain one in-process. Testing it in-process would either pass vacuously or
# depend on file order.

using Test
using RepliBuild

const D = RepliBuild.Debug
const VI = joinpath(@__DIR__, "vi_test")
const VI_SO = joinpath(VI, "julia", "libvi_test.so")

@testset "Debug: static inspection" begin

    @testset "artifact locations" begin
        want = joinpath(VI, ".debug")

        # Every spelling a caller plausibly has in hand must land in the same
        # place. The first version only matched `.debug` as an exact basename,
        # so `.debug/mlir` re-appended to `…/.debug/mlir/.debug/mlir` and the
        # failure blamed a missing directory for what was a path bug. Found at a
        # REPL within an hour of shipping, which is why the list is exhaustive
        # rather than representative.
        @test D.debug_root(VI) == want
        @test D.debug_root(want) == want
        @test D.debug_root(want * "/") == want
        @test D.debug_root(joinpath(want, "mlir")) == want
        @test D.debug_root(joinpath(want, "obj")) == want
        @test D.debug_root(joinpath(VI, "..", "vi_test")) == want
        isfile(VI_SO) && @test D.debug_root(VI_SO) == want
        for f in D.mlir_sources(VI)
            @test D.debug_root(f) == want
        end
        # Idempotent under repeated application.
        @test D.debug_root(D.debug_root(D.debug_root(VI))) == want

        # "" is the working directory, not an error: `D.thunks("")` while
        # standing in a package is the shortest thing to type.
        cd(VI) do
            @test D.debug_root("") == want
            @test D.debug_root(".") == want
        end

        # Same artifacts reached through unrelated spellings.
        @test D.thunks(VI) == D.thunks(joinpath(want, "mlir"))
    end

    @testset "a mistyped path says so" begin
        # A missing artifact and a bad path both present as an empty directory,
        # and at a REPL the second is far likelier. The error must lead with the
        # resolution instead of blaming the JIT for never running.
        err = try; D.thunks(joinpath(VI, "vi_test")); "" catch e; sprint(showerror, e) end
        @test occursin("resolved to", err)
        @test occursin("Accepted:", err)
        # ...and must NOT say that when the root is real and simply has no object.
        if !D.has_object(VI)
            e2 = try; D.disassemble(VI); "" catch e; sprint(showerror, e) end
            @test !occursin("resolved to", e2)
        end
    end

    @testset "MLIR sources and thunk enumeration" begin
        srcs = D.mlir_sources(VI)
        @test !isempty(srcs)
        @test all(endswith(".mlir"), srcs)
        @test all(isfile, srcs)

        t = D.thunks(VI)
        @test !isempty(t)
        @test all(endswith("_thunk"), t)
        @test issorted(t)
        @test length(unique(t)) == length(t)
        # The diamond's shapes: a plain method, a non-virtual `this` adjustment,
        # and a virtual override thunk. If the enumeration regex ever narrows,
        # these are the classes it would silently drop.
        @test "_ZNK5VBase3tagEv_thunk" in t
        @test "_ZThn16_N7DiamondD0Ev_thunk" in t
        @test "_ZTv0_n32_NK7Diamond3tagEv_thunk" in t
    end

    @testset "mlir_body extracts exactly one function" begin
        body = D.mlir_body(VI, "_ZNK5VBase3tagEv_thunk")
        @test startswith(body, "func.func @_ZNK5VBase3tagEv_thunk(")
        @test endswith(body, "}")
        @test occursin("jlcs.vcall", body)
        # One block, not a run-on to the next function.
        @test count(l -> startswith(l, "func.func @"), split(body, '\n')) == 1
        @test !occursin("_ZNK7Diamond3tagEv_thunk", body)
    end

    @testset "guards name the fix, not just the failure" begin
        # A missing object is the expected state — capture is opt-in — so the
        # error has to carry the recipe or every encounter costs a search.
        mi = joinpath(@__DIR__, "mi_test")
        if !D.has_object(mi)
            err = try; D.disassemble(mi); "" catch e; sprint(showerror, e) end
            @test occursin("REPLIBUILD_JIT_OBJDUMP", err)
            @test occursin("obj", err)
        end

        # A typo must surface the symbol that was meant.
        err = try; D.mlir_body(VI, "_ZNK5VBase3tgEv_thunk"); "" catch e; sprint(showerror, e) end
        @test occursin("_ZNK5VBase3tagEv_thunk", err)
    end

    @testset "object capture round trip" begin
        objdir = joinpath(VI, ".debug", "obj")
        rm(objdir; recursive=true, force=true)
        @test !D.has_object(VI)

        # Fresh process with capture on. `timeout -s KILL` because a wedged
        # julia ignores SIGTERM and this suite must not hang on it.
        script = joinpath(VI, "verify.jl")
        cmd = `timeout -s KILL 170 julia --project=$(dirname(@__DIR__)) $script`
        # addenv, NOT setenv: setenv REPLACES the environment, which strips PATH
        # and the depot and fails the child for reasons that look nothing like
        # the thing under test.
        ok = success(pipeline(addenv(cmd, "REPLIBUILD_JIT_OBJDUMP" => "1"),
                              stdout=devnull, stderr=devnull))
        @test ok

        if ok
            @test D.has_object(VI)
            obj = D.object_path(VI)
            @test endswith(obj, ".o")
            @test filesize(obj) > 0

            # The payoff: the emitted machine code carries DWARF pointing at the
            # generated MLIR, so objdump interleaves them with no live process.
            # This is the assertion that would fail if the sourceName parse or
            # the DIScope pass regressed — both are invisible to every other
            # test, because the thunks keep working without them.
            asm = D.disassemble(VI; symbol="_ZNK5VBase3tagEv_thunk")
            @test occursin("_ZNK5VBase3tagEv_thunk", asm)
            @test occursin("jlcs.vcall", asm)            # MLIR interleaved
            @test occursin(r"\bmov\b", asm)              # ...with machine code

            # Slot 2 of the vtable is 2 × 8 = 0x10, and that multiplication is
            # the thing the dialect is responsible for getting right.
            @test occursin("0x10(%rax)", asm)

            # source=false is the same code without the interleaving.
            plain = D.disassemble(VI; symbol="_ZNK5VBase3tagEv_thunk", source=false)
            @test occursin(r"\bmov\b", plain)
            @test !occursin("jlcs.vcall", plain)

            # DWARF is real, and it pins WHICH file the debugger will open.
            info = D.dwarf(VI; section="info")
            @test occursin("DW_AT_producer\t(\"MLIR\")", info)
            @test occursin(".mlir", info)
            @test occursin(joinpath(VI, ".debug", "mlir"), info)   # comp_dir is absolute

            # `.debug_info` holds the compile unit and NOTHING else — no
            # DW_TAG_subprogram, no variable DIEs. That is what LineTablesOnly
            # means here, and it is why `info locals` and `p %val_1` come back
            # empty in gdb while file/line work perfectly. Asserted so that a
            # future emissionKind=Full experiment shows up as this test failing
            # rather than as a surprise.
            @test !occursin("DW_TAG_subprogram", info)
            @test !occursin("DW_TAG_variable", info)

            # The function-level information lives in the line table instead.
            line = D.dwarf(VI; section="line")
            @test occursin("jlcs_", line)
            @test occursin(".mlir", line)

            # Unknown symbols must not read as an empty function.
            @test_throws ErrorException D.disassemble(VI; symbol="definitely_not_here")

            w = D.walk(VI, "_ZNK5VBase3tagEv_thunk")
            @test occursin("func.func @_ZNK5VBase3tagEv_thunk", w)
            @test occursin("0x10(%rax)", w)
        end
    end
end
