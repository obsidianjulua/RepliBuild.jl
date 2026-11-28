# test_bundled_dwarfdump_live.jl — a C++ library's classes without a system LLVM.
#
# `DWARFParser.parse_vtables` runs `llvm-dwarfdump` when a C++ library is
# wrapped and again every time a JIT wrapper loads, so it was the one program a
# machine that only USES a finished wrapper had to have installed, under its
# bare name on PATH. It now comes from `Tools`: PATH's copy first (the dumper
# of the LLVM that compiled the library), else LLVM_unified_jll's, where that
# is LLVM 20 or newer.
#
# The bundled copy is at Julia's LLVM and the library is compiled by a newer
# one, so the claim to hold is that both read the same classes. This builds a
# library with single, multiple and virtual inheritance and compares what the
# two copies give, then reads it with nothing on PATH at all. On a Julia whose
# LLVM is older than 20 there is no bundled copy and those parts are skipped:
# LLVM 18's dumper prints a typedef without its scope, and the parser reads
# the names (test_tools_resolver.jl holds the floor).

using Test
using RepliBuild
using Libdl

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const _BDD_T = RepliBuild.Tools
const _BDD_D = RepliBuild.DWARFParser

const _BDD_SOURCE = """
namespace zoo {
struct Animal {
    int legs = 4;
    virtual ~Animal() {}
    virtual int speak() const { return 1; }
    virtual int weight(int scale) const { return scale; }
};
struct Swimmer {
    double depth = 0;
    virtual ~Swimmer() {}
    virtual int dive(int metres) { return metres; }
};
struct Duck : Animal, Swimmer {
    long feathers = 9000;
    int speak() const override { return 2; }
    int dive(int metres) override { return metres / 2; }
};
struct Base { int tag = 7; virtual ~Base() {} virtual int id() const { return tag; } };
struct Left : virtual Base { int l = 1; };
struct Right : virtual Base { int r = 2; };
struct Diamond : Left, Right { int d = 3; int id() const override { return 42; } };
struct Plain { int a; double b; };
}
__attribute__((visibility("default"))) int zoo_use(zoo::Duck *d, zoo::Diamond *m, zoo::Plain *p) {
    zoo::Animal *a = d; zoo::Swimmer *s = d;
    return a->speak() + a->weight(3) + s->dive(10) + m->id() + p->a;
}
__attribute__((visibility("default"))) zoo::Duck *zoo_duck() { return new zoo::Duck(); }
__attribute__((visibility("default"))) zoo::Diamond *zoo_diamond() { return new zoo::Diamond(); }
"""

# A class table as text, every field of every class in a fixed order.
function _bdd_canon(classes)::String
    io = IOBuffer()
    for name in sort!(collect(keys(classes)))
        c = classes[name]
        println(io, name, " size=", c.size, " vptr=", c.vtable_ptr_offset, " bases=", c.base_classes,
                " at=", c.base_offsets, " virtual=", c.virtual_bases, " vbase=", c.vbase_vtable_offsets)
        for m in c.virtual_methods
            println(io, "   v ", m.slot, " ", m.name, " ", m.mangled_name, " ", m.return_type, " ", m.parameters)
        end
        for m in c.members
            println(io, "   m ", m.offset, " ", m.name, " ", m.type_name)
        end
    end
    return String(take!(io))
end

if requires("a C++ library's classes without a system llvm-dwarfdump", :clangxx)
@testset "a C++ library's classes without a system llvm-dwarfdump" begin
    mktempdir() do dir
        src = joinpath(dir, "zoo.cpp")
        lib = joinpath(dir, "libzoo." * Libdl.dlext)
        write(src, _BDD_SOURCE)
        (out, code) = RepliBuild.BuildBridge.execute("clang++", ["-std=c++17", "-g", "-O1", "-fPIC", "-shared", src, "-o", lib])
        code == 0 || println(out)
        @test code == 0

        source_under(order) = withenv(() -> _BDD_T.tool_source("llvm-dwarfdump"), "REPLIBUILD_TOOLS" => order)
        read_with(order) = withenv(() -> _BDD_D.parse_vtables(lib), "REPLIBUILD_TOOLS" => order)

        have_jll = source_under("jll") === :jll
        have_system = source_under("system") === :system
        @info "llvm-dwarfdump copies on this host" bundled = have_jll system = have_system
        # One of them at least, or nothing below would run.
        @test have_jll || have_system

        by_copy = Dict{String,String}()
        for (order, there) in (("jll", have_jll), ("system", have_system))
            there || continue
            info = read_with(order)
            classes = info.classes
            # Not vacuous: the library's classes, their virtual methods and
            # all three shapes of inheritance were read.
            duck = first(c for (n, c) in classes if endswith(n, "Duck"))
            diamond = first(c for (n, c) in classes if endswith(n, "Diamond"))
            left = first(c for (n, c) in classes if endswith(n, "Left"))
            @test length(duck.base_classes) == 2
            @test any(m -> m.name == "speak", duck.virtual_methods)
            @test any(>(0), duck.base_offsets)                 # the second base is not at 0
            @test any(left.virtual_bases)                      # virtual inheritance is marked
            @test any(<(0), left.vbase_vtable_offsets)         # ...with its vtable slot
            @test length(diamond.base_classes) == 2
            @test any(endswith(n, "Plain") for n in keys(classes))
            by_copy[order] = _bdd_canon(classes)
        end
        # The bundled copy, at Julia's LLVM, and the system's, at the
        # compiler's, read the same classes.
        if have_jll && have_system
            @test by_copy["jll"] == by_copy["system"]
        else
            @test_skip by_copy["jll"] == by_copy["system"]
        end

        # `parse_vtables` also runs `nm`. Binutils_jll has no build for Windows,
        # so there is no machine without a PATH to stand in for there: a
        # wrapper's user has MSYS2's `nm`, and the two cases below need both
        # tools bundled.
        nm_bundled = withenv(() -> _BDD_T.tool_source("nm"), "REPLIBUILD_TOOLS" => "jll") === :jll
        Sys.islinux() && @test nm_bundled
        if have_jll && nm_bundled
            # A machine that only uses a wrapper: no LLVM, no binutils, no PATH.
            alone = withenv("REPLIBUILD_TOOLS" => "jll", "PATH" => "") do
                @test Sys.which("llvm-dwarfdump") === nothing
                _BDD_D.parse_vtables(lib)
            end
            @test _bdd_canon(alone.classes) == by_copy["jll"]
            # `system` first with no system copy falls to the bundled one.
            fallback = withenv("REPLIBUILD_TOOLS" => "system", "PATH" => "") do
                @test _BDD_T.tool_source("llvm-dwarfdump") === :jll
                _BDD_D.parse_vtables(lib)
            end
            @test _bdd_canon(fallback.classes) == by_copy["jll"]
        else
            @test_skip false
        end

        # The call goes through the resolver: a program named for it that does
        # not exist is used as given, and fails by its own name.
        err = try
            withenv(() -> _BDD_D.parse_vtables(lib), "REPLIBUILD_TOOL_LLVM_DWARFDUMP" => joinpath(dir, "no-such-dwarfdump"))
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("no-such-dwarfdump", sprint(showerror, err))
    end
end
end # requires
