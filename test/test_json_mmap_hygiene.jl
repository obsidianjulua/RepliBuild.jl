#!/usr/bin/env julia
# test/test_json_mmap_hygiene.jl — every JSON read in src/ goes through
# `RepliBuild._read_json`.
#
# Two defects, both textual at the call site, both invisible to the run that
# would have to catch them:
#
# 1. JSON 0.21's `parsefile` defaults to `use_mmap=true`:
#
#        s = use_mmap ? String(Mmap.mmap(io, Vector{UInt8}, sz)) : read(io, String)
#
#    and Julia drops a mapping only when the GC finalizes it — which is to say,
#    at no time you can name. POSIX does not care, because a mapped file still
#    unlinks. Windows does: while the mapping is live the file cannot be
#    deleted, and a directory holding it fails with ENOTEMPTY. That made
#    `clean()` fail on a build tree RepliBuild had just produced itself, and the
#    file left behind was `compilation_metadata.json` — not the .dll, which is
#    what anyone would have gone looking for. No Linux run can observe it.
#
# 2. JSON 1.x rejects the fix for (1): `parsefile(path; use_mmap=false)` is a
#    `MethodError` in `JSON.LazyOptions`, and plain `parsefile` returns
#    `JSON.Object`, not `Dict`. Several reads sit inside a `try`, so on 1.x some
#    of them would not even fail loudly — they would fall back. No run on 0.21
#    can observe it.
#
# `_read_json` is `JSON.parse(read(path, String); dicttype=Dict{String,Any})`,
# which never maps and returns the same types on both majors. The rule is
# blanket: these files are metadata, read once, and a second read path is how
# either defect comes back.

using Test
using RepliBuild

const READ_ENTRY = r"JSON\.(parsefile|parse|lazyfile|lazy)\b"

# One parent, so both halves always run and report together: a top-level
# testset that fails throws at its `end`, and the second half would never run.
@testset "JSON reads (src/ → _read_json)" begin
    @testset "every read goes through _read_json" begin
        src_dir = joinpath(dirname(@__DIR__), "src")
        @test isdir(src_dir)

        offenders = String[]
        helper_defs = String[]
        call_sites = 0

        for (root, _, files) in walkdir(src_dir), f in files
            endswith(f, ".jl") || continue
            path = joinpath(root, f)
            for (i, line) in enumerate(readlines(path))
                # Skip prose: comments name these calls to explain the rule, and
                # a comment is not a call site.
                startswith(strip(line), "#") && continue
                where = string(relpath(path, src_dir), ":", i, "  ", strip(line))
                is_def = startswith(strip(line), "_read_json(path::AbstractString) =")
                if occursin(READ_ENTRY, line)
                    push!(is_def ? helper_defs : offenders, where)
                end
                occursin("_read_json(", line) && !is_def && (call_sites += 1)
            end
        end

        # The helper exists exactly once, and it is the shape the rule relies on.
        @test length(helper_defs) == 1
        @test all(d -> occursin("read(path, String)", d) &&
                       occursin("dicttype=Dict{String,Any}", d), helper_defs)

        # Assert the sweep actually ran. A rename, a move, or a bad walkdir would
        # otherwise make this testset pass by finding nothing — the same "the
        # catch turned a whole testset into a no-op" failure CLAUDE.md records
        # against the symbol-hygiene sweep.
        @test call_sites > 5

        if !isempty(offenders)
            @info "JSON read outside RepliBuild._read_json — use the helper:\n  " *
                  join(offenders, "\n  ")
        end
        @test isempty(offenders)
    end

    # Behavioural half: the types every consumer indexes into, under whichever
    # JSON major this environment resolved (the testset name records which).
    @testset "_read_json types under JSON $(pkgversion(RepliBuild.JSON))" begin
        mktempdir() do dir
            obj = joinpath(dir, "compilation_metadata.json")
            write(obj, """{"functions": [{"name": "f", "offset": null, "size": 8}],
                           "types": {"S": {"members": []}}}""")
            md = RepliBuild._read_json(obj)
            @test md isa Dict{String,Any}
            @test md["types"] isa Dict{String,Any}
            @test md["types"]["S"] isa Dict{String,Any}
            @test md["functions"] isa Vector{Any}
            @test md["functions"][1] isa Dict{String,Any}
            @test md["functions"][1]["offset"] === nothing
            @test md["functions"][1]["size"] === 8

            # compile_commands.json is an array at the root.
            arr = joinpath(dir, "compile_commands.json")
            write(arr, """[{"file": "a.c", "arguments": ["clang", "-c", "a.c"]}]""")
            cc = RepliBuild._read_json(arr)
            @test cc isa Vector{Any}
            @test cc[1] isa Dict{String,Any}

            # Deletable straight after the read: trivially true on POSIX, and the
            # check that matters on Windows, where a live mapping refuses it.
            rm(obj); rm(arr)
            @test !isfile(obj) && !isfile(arr)
        end
    end
end
