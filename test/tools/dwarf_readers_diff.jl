# dwarf_readers_diff.jl — the readelf reader against the libdw reader, on
# libraries that already exist.
#
#     julia --project=. test/tools/dwarf_readers_diff.jl <lib.so>...
#     julia --project=. test/tools/dwarf_readers_diff.jl --hub <RepliBuild-Hub path>
#
# For each library it runs `Compiler.extract_dwarf_return_types` twice, once
# per `REPLIBUILD_DWARF` reader, and compares the five tables the parser
# returns, key by key. A parser input replacement is right only when every
# library gives identical tables; this is that check. It builds nothing and
# writes nothing. `--hub` takes every `packages/*/julia/lib*.so` that is not a
# thunks library (hand-run: a test may not read the Hub).
#
# Exit status 1 when any library differs or cannot be read by libdw.

using RepliBuild
const C = RepliBuild.Compiler

function differences!(out::Vector{String}, a, b, path::String; limit::Int = 12)
    length(out) >= limit && return out
    if a isa AbstractDict && b isa AbstractDict
        for k in sort!(collect(union(keys(a), keys(b))); by = string)
            if !haskey(a, k)
                push!(out, "$path/$k: only libdw has it")
            elseif !haskey(b, k)
                push!(out, "$path/$k: only readelf has it")
            else
                differences!(out, a[k], b[k], "$path/$k"; limit)
            end
        end
    elseif a isa AbstractVector && b isa AbstractVector
        if length(a) != length(b)
            push!(out, "$path: $(length(a)) entries from readelf, $(length(b)) from libdw")
        else
            foreach(i -> differences!(out, a[i], b[i], "$path[$i]"; limit), eachindex(a))
        end
    elseif !isequal(a, b)
        push!(out, "$path: readelf $(repr(a)), libdw $(repr(b))")
    end
    return out
end

function libraries(args)
    isempty(args) && error("usage: dwarf_readers_diff.jl <lib.so>... | --hub <RepliBuild-Hub path>")
    args[1] == "--hub" || return collect(String, args)
    length(args) == 2 || error("--hub takes the Hub's path")
    libs = String[]
    for pkg in sort!(readdir(joinpath(args[2], "packages"); join = true))
        dir = joinpath(pkg, "julia")
        isdir(dir) || continue
        for f in sort!(readdir(dir))
            (startswith(f, "lib") && endswith(f, ".so") && !occursin("thunks", f)) && push!(libs, joinpath(dir, f))
        end
    end
    isempty(libs) && error("no packages/*/julia/lib*.so under $(args[2])")
    return libs
end

function main(args)
    bad = 0
    libs = libraries(args)
    for lib in libs
        timed(reader) = withenv("REPLIBUILD_DWARF" => reader) do
            t = @elapsed tables = C.extract_dwarf_return_types(lib)
            (tables, t)
        end
        (r, rt) = timed("readelf")
        (l, lt) = try
            timed("libdw")
        catch e
            println("ERR  ", lib, ": ", first(sprint(showerror, e), 300))
            bad += 1
            continue
        end
        out = String[]
        for (i, name) in enumerate(("functions", "records", "globals", "typedefs", "record_abi"))
            differences!(out, r[i], l[i], name)
        end
        println(isempty(out) ? "same " : "DIFF ", basename(lib), "  ", length(r[1]), " functions, ",
                length(r[2]), " records; readelf ", round(rt; digits = 2), " s, libdw ", round(lt; digits = 2), " s")
        foreach(o -> println("     ", first(o, 260)), out)
        isempty(out) || (bad += 1)
    end
    println(bad == 0 ? "ALL SAME: $(length(libs)) libraries" : "DIFFERENT: $bad of $(length(libs))")
    return bad == 0 ? 0 : 1
end

exit(main(ARGS))
