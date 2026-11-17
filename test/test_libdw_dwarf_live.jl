# test_libdw_dwarf_live.jl — two DWARF readers, one answer.
#
# `REPLIBUILD_DWARF=libdw` reads a library's debug info in this process,
# through elfutils' libdw, and hands `parse_dwarf_dump` the text GNU readelf
# would have printed. The claim is that the parser cannot tell the difference,
# and this file holds it to that on libraries built here from
# test/libdw_test: a C++ unit (namespaces, an enum class, a virtual base
# pointer, bitfields, a function pointer member, a template, a union) and a C
# unit (an over-aligned record, `_Complex`, `__int128`, a SIMD vector,
# variadics), each at DWARF 5 and DWARF 4. The five tables the parser returns
# must be equal, key for key, and so must the per-unit file tables.
#
# It also reads one library with every GNU dumper overridden to a path that
# does not exist: the libdw reader needs none.

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const _LDL_C = RepliBuild.Compiler
const _LDL_SRC = joinpath(@__DIR__, "libdw_test")

# The first few places two tables differ, for the failure log.
function _ldl_differences!(out::Vector{String}, a, b, path::String)
    length(out) >= 10 && return out
    if a isa AbstractDict && b isa AbstractDict
        for k in sort!(collect(union(keys(a), keys(b))); by = string)
            if !haskey(a, k)
                push!(out, "$path/$k: only libdw has it")
            elseif !haskey(b, k)
                push!(out, "$path/$k: only readelf has it")
            else
                _ldl_differences!(out, a[k], b[k], "$path/$k")
            end
        end
    elseif a isa AbstractVector && b isa AbstractVector
        if length(a) != length(b)
            push!(out, "$path: $(length(a)) entries from readelf, $(length(b)) from libdw")
        else
            foreach(i -> _ldl_differences!(out, a[i], b[i], "$path[$i]"), eachindex(a))
        end
    elseif !isequal(a, b)
        push!(out, "$path: readelf $(repr(a)), libdw $(repr(b))")
    end
    return out
end

_ldl_read(lib, reader) =
    withenv(() -> _LDL_C.extract_dwarf_return_types(lib), "REPLIBUILD_DWARF" => reader)

if !Sys.islinux() || !RepliBuild.LibDW.available()
    @testset "libdw and readelf give one answer" begin
        @info "skipped: libdw reads ELF and needs Elfutils_jll"
        @test_skip false
    end
elseif requires("libdw and readelf give one answer", :clang, :clangxx)
    @testset "libdw and readelf give one answer" begin
        mktempdir() do dir
            builds = [
                ("shapes5", "clang++", ["-gdwarf-5"], "shapes.cpp"),
                ("shapes4", "clang++", ["-gdwarf-4"], "shapes.cpp"),
                ("shapes_sa", "clang++", ["-gdwarf-5", "-fstandalone-debug"], "shapes.cpp"),
                ("records5", "clang", ["-gdwarf-5"], "records.c"),
                ("records4", "clang", ["-gdwarf-4"], "records.c"),
            ]
            compared = 0
            for (name, compiler, flags, source) in builds
                lib = joinpath(dir, "lib$name.so")
                args = vcat(flags, ["-O1", "-fPIC", "-shared", joinpath(_LDL_SRC, source), "-o", lib])
                (out, code) = RepliBuild.BuildBridge.execute(compiler, args)
                @test code == 0
                code == 0 || (println(out); continue)

                @testset "$name" begin
                    r = _ldl_read(lib, "readelf")
                    l = _ldl_read(lib, "libdw")
                    # Not vacuous: the readelf side found the library's content.
                    @test !isempty(r[1])
                    @test !isempty(r[2])
                    diffs = String[]
                    for (i, table) in enumerate(("functions", "records", "globals", "typedefs", "record_abi"))
                        _ldl_differences!(diffs, r[i], l[i], table)
                    end
                    @test isempty(diffs)
                    foreach(d -> println("  ", first(d, 240)), diffs)

                    # The provenance filter's input, read two ways.
                    dumper = _LDL_C._dwarf_dumper()
                    @test dumper !== nothing
                    from_readelf = _LDL_C._dwarf_file_tables(dumper, lib)
                    from_libdw = RepliBuild.LibDW.file_tables(lib)
                    @test length(from_readelf) == length(from_libdw) >= 1
                    files = 0
                    for (r_table, l_table) in zip(from_readelf, from_libdw), (index, path) in r_table
                        # DWARF 4's table has no entry for the compilation
                        # directory, so readelf's name is relative to it;
                        # libdw's is resolved. The parser resolves the former.
                        got = get(l_table, index, "")
                        @test isabspath(path) ? got == path : (isabspath(got) && endswith(got, "/" * path))
                        @test !occursin('\t', path)      # a row read by the wrong column layout
                        files += 1
                    end
                    @test files >= 1
                end
                compared += 1
            end
            @test compared == length(builds)

            @testset "libdw needs no dumper" begin
                lib = joinpath(dir, "libshapes5.so")
                gone = joinpath(dir, "no-such-dumper")
                got = withenv("REPLIBUILD_DWARF" => "libdw", "REPLIBUILD_OBJDUMP" => gone,
                              "REPLIBUILD_TOOL_READELF" => gone, "REPLIBUILD_TOOL_OBJDUMP" => gone) do
                    @test _LDL_C._dwarf_dumper() === nothing     # the override really removed it
                    _LDL_C.extract_dwarf_return_types(lib)
                end
                @test haskey(got[1], "_ZNK3geo5Shape4areaEi")
                @test haskey(got[2], "Shape")
            end

            @testset "a file that is not ELF is an error that names it" begin
                junk = joinpath(dir, "junk.so")
                write(junk, "not an object file")
                err = try
                    RepliBuild.LibDW.info_dump(junk); nothing
                catch e
                    sprint(showerror, e)
                end
                @test err !== nothing && occursin("junk.so", err)
            end
        end
    end
end # requires
