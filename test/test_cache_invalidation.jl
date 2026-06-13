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
    root = "$dir"
    [compile]
    flags = ["-O1", "-fPIC"]
    source_files = ["$(joinpath(dir, "src", "m.c"))"]
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

    ll = joinpath(dir, "build", "m.ll")
    nsyms(so) = parse(Int, readchomp(pipeline(`nm -g --defined-only $so`, `grep -c " T "`)))

    # Build 1 — cold
    so = RepliBuild.build(toml)
    m1 = mtime(ll)
    s1 = nsyms(so)
    @test isfile(ll * ".key")          # fingerprint sidecar written
    @test s1 == 2                      # both symbols exported by default

    # Build 2 — no change → cache hit, IR untouched
    RepliBuild.build(toml)
    m2 = mtime(ll)
    @test m2 == m1                     # per-file cache still works when nothing changed

    # Build 3 — add a compile flag, NO manual cache clear → must recompile
    write(toml, replace(base,
        "flags = [\"-O1\", \"-fPIC\"]" => "flags = [\"-O1\", \"-fPIC\", \"-fvisibility=hidden\"]"))
    so = RepliBuild.build(toml)
    m3 = mtime(ll)
    s3 = nsyms(so)
    @test m3 > m2                      # the bug: this stayed cached (stale IR)
    @test s3 == 1                      # and the new flag actually took effect

    # Build 4 — revert the flag → recompiles again (fingerprint is bidirectional)
    write(toml, base)
    so = RepliBuild.build(toml)
    @test nsyms(so) == 2

    rm(dir; recursive=true, force=true)
end
