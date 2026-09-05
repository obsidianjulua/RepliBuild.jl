# =============================================================================
# Per-file IR cache invalidation on compile-config change
#
# Guards against a correctness bug: the per-file IR cache was mtime-only, so
# changing [compile].flags (or defines/includes/compiler) left every source
# mtime untouched and the stale IR — built with the OLD flags — was silently
# reused. The .so looked fine but was compiled wrong; the only workaround was
# `rm -rf .replibuild_cache build`. A compile fingerprint now gates the cache.
# =============================================================================

using Test

@testset "Per-file IR cache: compile-flag invalidation" begin
    dir = mktempdir()
    mkpath(joinpath(dir, "src"))

    # internal_helper has default visibility normally (exported); public_api is
    # explicitly visibility("default"). Under -fvisibility=hidden, internal_helper
    # drops from the dynamic symbol table — an observable, flag-caused difference.
    write(joinpath(dir, "src", "m.c"), """
    int internal_helper(int x) { return x + 1; }
    __attribute__((visibility("default"))) int public_api(int x) { return internal_helper(x) * 2; }
    """)

    toml = joinpath(dir, "replibuild.toml")
    base = """
    [project]
    name = "cachetest"
    # escape_string on every interpolated path: TOML basic strings take escapes,
    # so a Windows `C:\\Users\\...` would parse as `\\U` and fail.
    root = "$(escape_string(dir))"
    [compile]
    flags = ["-O1", "-fPIC"]
    source_files = ["$(escape_string(joinpath(dir, "src", "m.c")))"]
    [link]
    enable_lto = false
    [binary]
    type = "shared"
    [wrap]
    language = "c"
    [cache]
    enabled = true
    """
    write(toml, base)

    # TWO HASHES, AND THEY ANSWER DIFFERENT QUESTIONS. Per-file IR lands at
    # build/m.c-<8hex>.ll, where that hash identifies the SOURCE — it is stable
    # across compile-config changes. The COMPILE fingerprint lives in the .key
    # sidecar beside it, and that is what gates the cache.
    #
    # This testset watched `build/m.ll`, a path that has not existed since the
    # per-file cache was keyed. `mtime` on a missing file returns 0.0, so
    # `m3 > m2` compared 0.0 > 0.0 and failed standalone for reasons that had
    # nothing to do with invalidation — which works, and is asserted below on
    # the sidecar that actually carries the decision.
    build_dir = joinpath(dir, "build")
    ir_file() = only(filter(f -> startswith(f, "m.c-") && endswith(f, ".ll"),
                            readdir(build_dir)))
    ir_path() = joinpath(build_dir, ir_file())
    fingerprint() = strip(read(ir_path() * ".key", String))

    # Symbols by NAME, never by count — see build 3 for why a count cannot work.
    function exported(so)
        out = readchomp(`nm -g --defined-only $so`)
        Set(String(last(split(l))) for l in split(out, '\n') if occursin(" T ", l))
    end

    # Build 1 — cold
    so = RepliBuild.build(toml)
    @test isfile(ir_path() * ".key")   # fingerprint sidecar written
    f1, m1 = fingerprint(), mtime(ir_path())
    e1 = exported(so)
    @test "internal_helper" in e1 && "public_api" in e1     # both visible by default

    # Build 2 — no change → cache hit: same fingerprint, IR not recompiled
    RepliBuild.build(toml)
    @test fingerprint() == f1
    @test mtime(ir_path()) == m1

    # Build 3 — add a compile flag, NO manual cache clear → must recompile.
    write(toml, replace(base,
        "flags = [\"-O1\", \"-fPIC\"]" => "flags = [\"-O1\", \"-fPIC\", \"-fvisibility=hidden\"]"))
    so = RepliBuild.build(toml)
    f3 = fingerprint()
    @test f3 != f1                     # THE BUG THIS GUARDS: stale IR was reused
    @test mtime(ir_path()) > m1        # and the recompile actually happened

    # And the flag actually took effect. Asserted on the NAME, because the
    # export COUNT cannot see it: `internal_helper` is now external-linkage
    # but hidden — the LUAI_FUNC shape — so static promotion (2026-07-22,
    # added after this test was written) correctly re-exports it as
    # `__rb_cachetest_internal_helper` with default visibility. The old
    # `nsyms(so) == 1` was therefore unsatisfiable: two symbols go in, two
    # come out, and only their names record that anything happened.
    e3 = exported(so)
    @test "public_api" in e3
    @test !("internal_helper" in e3)                     # gone under its own name
    @test "__rb_cachetest_internal_helper" in e3         # …and back under promotion's

    # Build 4 — revert → the ORIGINAL fingerprint returns. Bidirectional, and a
    # stronger statement than "it recompiled": the same inputs must be keyed the
    # same way, so an invalidation scheme that merely churned (a timestamp, a
    # counter) would satisfy build 3 and fail here.
    write(toml, base)
    so = RepliBuild.build(toml)
    @test fingerprint() == f1
    @test "internal_helper" in exported(so)

    rm(dir; recursive=true, force=true)
end
