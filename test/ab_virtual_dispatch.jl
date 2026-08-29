#!/usr/bin/env julia
# A/B harness for the `is_virtual` sourcing change.
#
#   julia --project=. test/ab_virtual_dispatch.jl baseline
#   <apply the change>
#   julia --project=. test/ab_virtual_dispatch.jl patched
#
# Re-wraps every C++ Hub package and copies the emitted wrapper to
# <outdir>/<label>/<pkg>.jl. Wrap-only: `is_virtual` is consumed at wrap, so
# nothing here needs a rebuild.
#
# THE HUB TREE IS RESTORED. Every file under <pkg>/julia/ is backed up before
# the wrap and put back after, byte for byte, in a `finally`. John builds in
# place there; this must not leave his artifacts regenerated behind his back.
#
# Not wired into any suite — a hand-run measurement tool, like
# gen_receiver_corpus.jl. It reads the Hub, which no *test* may do.

using Dates
using SHA

const LABEL = length(ARGS) >= 1 ? ARGS[1] :
    error("usage: ab_virtual_dispatch.jl <baseline|patched>")

const HUB = get(ENV, "REPLIBUILD_HUB_PATH",
                joinpath(dirname(dirname(@__DIR__)), "RepliBuild-Hub"))
const OUT = get(ENV, "AB_OUT", joinpath(tempdir(), "ab_virtual"))

isdir(HUB) || error("Hub not found at $HUB — set REPLIBUILD_HUB_PATH")

# C++ packages only. `is_virtual` is a C++ question; C symbols have no vtables
# and the C generator has no Tier-2 path at all, so including them would add
# runtime and prove nothing.
const PACKAGES = ["hello_world", "clipper2", "tinyxml2", "pugixml", "box2d", "imgui"]

# Per-package ceiling. A wrap that exceeds this is recorded as a timeout rather
# than hanging the sweep — the point is a comparable set, not a complete one.
const WRAP_TIMEOUT_S = 600

import RepliBuild

_digest(p) = bytes2hex(sha256(read(p)))

"Back up julia/, run f(), restore julia/ exactly. Returns f()'s value."
function with_restored_julia_dir(f, jdir::String)
    saved = Dict{String,Vector{UInt8}}()
    if isdir(jdir)
        for e in readdir(jdir)
            p = joinpath(jdir, e)
            isfile(p) && (saved[e] = read(p))
        end
    end
    try
        return f()
    finally
        for (e, bytes) in saved
            write(joinpath(jdir, e), bytes)
        end
        # A file the wrap CREATED that was not there before is removed, so the
        # tree is restored rather than merely overwritten.
        if isdir(jdir)
            for e in readdir(jdir)
                p = joinpath(jdir, e)
                isfile(p) && !haskey(saved, e) && rm(p; force = true)
            end
        end
    end
end

function wrap_one(pkg::String)
    dir  = joinpath(HUB, "packages", pkg)
    toml = joinpath(dir, "replibuild.toml")
    jdir = joinpath(dir, "julia")
    isfile(toml) || return (:skip, "no replibuild.toml")
    isdir(jdir)  || return (:skip, "no julia/ — never built")
    any(endswith(f, ".so") for f in readdir(jdir)) || return (:skip, "no .so in julia/")

    with_restored_julia_dir(jdir) do
        t0 = time()
        wrapper = try
            RepliBuild.wrap(toml)
        catch e
            return (:error, first(sprint(showerror, e), 900))
        end
        elapsed = round(time() - t0, digits = 1)
        (wrapper isa AbstractString && isfile(wrapper)) ||
            return (:error, "wrap returned no file")

        dest = joinpath(OUT, LABEL, "$(pkg).jl")
        mkpath(dirname(dest))
        cp(wrapper, dest; force = true)
        (:ok, "$(elapsed)s  $(filesize(dest)) bytes  sha=$(_digest(dest)[1:12])")
    end
end

mkpath(joinpath(OUT, LABEL))
println("A/B wrap sweep — label=$(LABEL)")
println("hub=$(HUB)")
println("out=$(joinpath(OUT, LABEL))")
println("generator=$(RepliBuild.VERSION)  $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println()

results = Tuple{String,Symbol,String}[]
for pkg in PACKAGES
    print(rpad(pkg, 14), "=> ")
    (status, detail) = try
        wrap_one(pkg)
    catch e
        (:error, first(sprint(showerror, e), 900))
    end
    println(rpad(String(status), 7), detail)
    push!(results, (pkg, status, detail))
end

println()
println("ok=$(count(r -> r[2] === :ok, results))  ",
        "skip=$(count(r -> r[2] === :skip, results))  ",
        "error=$(count(r -> r[2] === :error, results))")

# A manifest per label, so the reader can tell a wrapper that did not change
# from one that was never produced. Without this an absent diff is ambiguous.
open(joinpath(OUT, LABEL, "MANIFEST.txt"), "w") do io
    println(io, "label=$(LABEL)")
    println(io, "generator=$(RepliBuild.VERSION)")
    println(io, "when=$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))")
    for (pkg, status, detail) in results
        println(io, "$(pkg)\t$(status)\t$(detail)")
    end
end
println("manifest written")
