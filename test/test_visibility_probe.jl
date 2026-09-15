# test_visibility_probe.jl — `VisibilityProbe.visibility_probe`
#
# The probe answers one question per package: does this library annotate its
# exports, so that `-fvisibility=hidden` narrows the wrap surface to the public
# API instead of erasing it? It answers by compiling the package's real sources
# under its real flags with the flag added, and reading visibility off the IR —
# the compiler resolves the export macro, so nothing here parses a header.
#
# Two failure shapes this pins, both of which would produce a confident wrong
# answer rather than an error:
#
#   * Substring matching on the `define` line. A function named `hidden_helper`,
#     or a return type spelled `range(i32 -2147483647, -2147483648)`, must not be
#     read as visibility. Tokens before the `@`, never `occursin` on the line.
#   * Unmeasured read as unannotated. A probe whose sources all failed to compile
#     has found a broken invocation; calling that "annotates nothing" would rule
#     out a library that annotates everything (pcre2 run from the wrong directory
#     is the live instance — 29/29 fail without its `config/` include root).
#
# Fixture is written at runtime into a tempdir with an explicit
# `[compile] source_files`, so it needs no dependency resolution and nothing is
# tracked.

using Test
using RepliBuild
using JSON

const VP = RepliBuild.VisibilityProbe

# Annotated + bare-extern + static, one of each shape the flag distinguishes.
const ANNOTATED_C = """
#define PUB __attribute__((visibility("default")))
PUB int annotated_one(int x) { return x + 1; }
PUB int annotated_two(int x) { return x + 2; }
extern int bare_extern(int x);
int bare_extern(int x) { return x + 3; }
static int helper(int x) { return x * 2; }
int uses_helper(int x) { return helper(x); }
"""

# The lua/zlib shape: everything exported by plain `extern`.
const BARE_C = """
extern int plain_one(int x);
int plain_one(int x) { return x + 1; }
int plain_two(int x) { return x + 2; }
"""

function write_pkg(dir::String, name::String, body::String)
    src = joinpath(dir, "$name.c")
    write(src, body)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "$name"
    root = "."

    [compile]
    source_files = ["$name.c"]
    flags = ["-O2", "-fPIC"]

    [binary]
    type = "shared"

    [wrap]
    language = "c"
    """)
    return toml
end

@testset "VisibilityProbe" begin

    @testset "_classify_define tokenizes, never substring-matches" begin
        cd = VP._classify_define
        # Plain external definition.
        @test cd("define i32 @foo(i32 %0) {") == (:default, "foo")
        # Hidden and protected.
        @test cd("define hidden i32 @foo(i32 %0) {") == (:hidden, "foo")
        @test cd("define protected i32 @foo(i32 %0) {") == (:hidden, "foo")
        # Local linkage is not part of the surface at all.
        @test cd("define internal i32 @foo(i32 %0) {") === nothing
        @test cd("define private i32 @foo(i32 %0) {") === nothing
        # A NAME containing the keyword must not be read as visibility — this is
        # the whole reason the match is on tokens before the `@`.
        @test cd("define i32 @hidden_helper(i32 %0) {") == (:default, "hidden_helper")
        @test cd("define i32 @internal_thing(i32 %0) {") == (:default, "internal_thing")
        @test cd("define i32 @protected_x(i32 %0) {") == (:default, "protected_x")
        # A return-type attribute carrying punctuation and negative numbers —
        # clang emits exactly this at -O2.
        @test cd("define range(i32 -2147483647, -2147483648) i32 @rng(i32 %0) {") ==
              (:default, "rng")
        @test cd("define hidden range(i32 -1, 2) i32 @rngh(i32 %0) {") == (:hidden, "rngh")
        # C++ linkage keywords are external and keep their visibility answer.
        @test cd("define linkonce_odr hidden i32 @_ZN1A1fEv(ptr %0) {") ==
              (:hidden, "_ZN1A1fEv")
        @test cd("define weak_odr i32 @_ZN1B1gEv(ptr %0) {") == (:default, "_ZN1B1gEv")
        # Quoted symbol names.
        @test cd("define i32 @\"odd name\"(i32 %0) {") == (:default, "odd name")
        # Not a definition.
        @test cd("declare i32 @bar(i32)") === nothing
        @test cd("  define i32 @indented()") === nothing
        @test cd("; define i32 @commented()") === nothing
    end

    mktempdir() do root
        @testset "annotated library" begin
            dir = mkpath(joinpath(root, "annotated"))
            toml = write_pkg(dir, "annotated", ANNOTATED_C)
            r = VP.visibility_probe(toml)

            @test r.verdict === :annotated
            @test VP.annotates_exports(r)
            @test r.compiled == 1
            @test isempty(r.failed)
            # Three external definitions; `helper` is static and inlined away.
            @test r.external_defs >= 3
            @test "annotated_one" in r.exported
            @test "annotated_two" in r.exported
            # The bare `extern` is PUBLIC API and does NOT survive — the whole point.
            @test !("bare_extern" in r.exported)
            @test !("uses_helper" in r.exported)
        end

        @testset "unannotated library" begin
            dir = mkpath(joinpath(root, "bare"))
            toml = write_pkg(dir, "bare", BARE_C)
            r = VP.visibility_probe(toml)

            @test r.verdict === :unannotated
            @test !VP.annotates_exports(r)
            @test r.external_defs >= 2
            @test isempty(r.exported)
        end

        @testset "macro shims do not vote on the verdict" begin
            # The lua case, minimally. RB_SHIM_EXPORT annotates every generated
            # shim unconditionally, so a bare-`extern` library with [wrap.macros]
            # has annotated definitions in its IR that say nothing about the
            # library. Counting them inverted the answer on the first Hub sweep:
            # lua read `:annotated` 89/450 with all 89 shims and 0 library
            # functions, i.e. "safe to hide" for a package the flag erases.
            dir = mkpath(joinpath(root, "shimmed"))
            # The macro must live in a header the shim TU can include — a shim
            # cannot see a #define that only exists inside a .c file.
            write(joinpath(dir, "shimmed.h"), "#define MAGIC_NUMBER 42\n")
            write(joinpath(dir, "shimmed.c"), "#include \"shimmed.h\"\n" * BARE_C)
            write(joinpath(dir, "replibuild.toml"), """
            [project]
            name = "shimmed"
            root = "."

            [compile]
            source_files = ["shimmed.c"]
            include_dirs = ["."]
            flags = ["-O2", "-fPIC"]

            [binary]
            type = "shared"

            [wrap]
            language = "c"
            shim_headers = ["shimmed.h"]

            [wrap.macros.MAGIC_NUMBER]
            ret = "int"
            """)
            r = VP.visibility_probe(joinpath(dir, "replibuild.toml"))

            # The shim IS annotated and IS in `exported` — it really does survive.
            @test any(startswith(e, "replibuild_shim_") for e in r.exported)
            # ...and it is excluded from the library's own evidence.
            @test isempty(r.library_exported)
            @test r.verdict === :unannotated
            @test !VP.annotates_exports(r)
        end

        @testset "retained/dropped against a real wrapper surface" begin
            dir = mkpath(joinpath(root, "withmd"))
            toml = write_pkg(dir, "withmd", ANNOTATED_C)
            # Stand in for a built package: the wrapper carries all four today.
            mkpath(joinpath(dir, "julia"))
            open(joinpath(dir, "julia", "compilation_metadata.json"), "w") do io
                JSON.print(io, Dict("functions" => [
                    Dict("name" => n, "mangled" => n) for n in
                    ("annotated_one", "annotated_two", "bare_extern", "uses_helper")]))
            end
            r = VP.visibility_probe(toml)

            @test r.wrapped_today == 4
            @test sort(r.retained) == ["annotated_one", "annotated_two"]
            # The decision-grade list: what enabling the flag would cost.
            @test sort(r.dropped) == ["bare_extern", "uses_helper"]
        end

        @testset "unmeasured is an ERROR, never a verdict" begin
            dir = mkpath(joinpath(root, "broken"))
            write(joinpath(dir, "broken.c"), "#include \"nope_missing.h\"\nint f(void){return 0;}\n")
            toml = joinpath(dir, "replibuild.toml")
            write(toml, """
            [project]
            name = "broken"
            root = "."

            [compile]
            source_files = ["broken.c"]
            flags = ["-O2", "-fPIC"]

            [binary]
            type = "shared"
            """)
            err = try
                VP.visibility_probe(toml); nothing
            catch e; e end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            # Must say which mistake this is, or the reader draws the wrong verdict.
            @test occursin("NOT ONE", msg)
            @test occursin("unmeasured", msg)
        end

        @testset "no sources is an error too" begin
            dir = mkpath(joinpath(root, "empty"))
            toml = joinpath(dir, "replibuild.toml")
            write(toml, """
            [project]
            name = "emptypkg"
            root = "."

            [binary]
            type = "shared"
            """)
            @test_throws ErrorException VP.visibility_probe(toml)
        end

        @testset "missing config errors" begin
            @test_throws ErrorException VP.visibility_probe(joinpath(root, "absent.toml"))
        end
    end
end
