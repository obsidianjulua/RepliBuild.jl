#!/usr/bin/env julia
# test/test_json_mmap_hygiene.jl — every JSON.parsefile in src/ must pass
# `use_mmap=false`.
#
# `JSON.parsefile` defaults to `use_mmap=true`:
#
#     s = use_mmap ? String(Mmap.mmap(io, Vector{UInt8}, sz)) : read(io, String)
#
# and Julia drops a mapping only when the GC finalizes it — which is to say, at
# no time you can name. POSIX does not care, because a mapped file still
# unlinks. Windows does: while the mapping is live the file cannot be deleted,
# and a directory holding it fails with ENOTEMPTY.
#
# That made `clean()` fail on a build tree RepliBuild had just produced itself,
# and the file left behind was `compilation_metadata.json` — not the .dll, which
# is what anyone would have gone looking for. It reproduced on every run and was
# invisible on Linux, where the same leaked mapping costs nothing.
#
# These files are metadata: kilobytes, read once. mmap buys nothing here and
# costs a platform-specific bug, so the rule is blanket rather than case-by-case.
#
# The guard is textual because the defect is textual: a new call site that omits
# the keyword is the failure mode, and it is invisible to any Linux test run.

using Test

@testset "JSON.parsefile never memory-maps" begin
    src_dir = joinpath(dirname(@__DIR__), "src")
    @test isdir(src_dir)

    offenders = String[]
    call_sites = 0

    for (root, _, files) in walkdir(src_dir), f in files
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        lines = readlines(path)
        for (i, line) in enumerate(lines)
            occursin("JSON.parsefile(", line) || continue
            # Skip prose: comment lines mention this call (this file's own
            # rationale is quoted in several src/ comments), and a comment is
            # not a call site.
            startswith(strip(line), "#") && continue
            call_sites += 1
            # The keyword may sit on the continuation line of a wrapped call.
            window = join(lines[i:min(i + 1, length(lines))], "\n")
            occursin("use_mmap=false", window) && continue
            push!(offenders, string(relpath(path, src_dir), ":", i, "  ", strip(line)))
        end
    end

    # Assert the sweep actually ran. A rename, a move, or a bad walkdir would
    # otherwise make this testset pass by finding nothing — the same "the catch
    # turned a whole testset into a no-op" failure CLAUDE.md records against the
    # symbol-hygiene sweep.
    @test call_sites > 5

    if !isempty(offenders)
        @info "JSON.parsefile call sites missing use_mmap=false:\n  " *
              join(offenders, "\n  ")
    end
    @test isempty(offenders)
end
