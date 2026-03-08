# bench.jl — llvmcall vs bare ccall vs pure-C
# Run from project root: julia --project=. test/mydir/bench.jl

using Libdl

PROJ = @__DIR__
include(joinpath(PROJ, "julia", "Project.jl"))

const LIB   = joinpath(PROJ, "julia", "libproject.so")
const hdl   = Libdl.dlopen(LIB, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)

const sym_iadd  = Libdl.dlsym(hdl, :iadd)
const sym_fmadd = Libdl.dlsym(hdl, :fmadd)
const sym_hsum  = Libdl.dlsym(hdl, :hsum)

# ── timing helper: median of N timed loops, each doing K calls ───────────────
function bench_ns(f, K=200_000, rounds=9)
    times = Vector{Float64}(undef, rounds)
    for r in 1:rounds
        t0 = time_ns()
        for _ in 1:K; f(); end
        times[r] = (time_ns() - t0) / K
    end
    sort!(times)
    times[div(rounds,2)+1]   # median
end

# ── prevent DCE: accumulate into a global ───────────────────────────────────
sink = Ref{Float64}(0.0)

# ── test data for hsum ───────────────────────────────────────────────────────
const VEC = rand(Float64, 1024)
const C_BIN_FFAST = joinpath(PROJ, "bench_c_ffast")

# ─────────────────────────────────────────────────────────────────────────────
# iadd(int,int)->int
# ─────────────────────────────────────────────────────────────────────────────
t_llvm_iadd  = bench_ns(() -> (sink[] += Project.iadd(3, 7)))
t_ccall_iadd = bench_ns(() -> (sink[] += ccall(sym_iadd, Cint, (Cint,Cint), 3, 7)))

# ─────────────────────────────────────────────────────────────────────────────
# fmadd(double,double,double)->double
# ─────────────────────────────────────────────────────────────────────────────
t_llvm_fmadd  = bench_ns(() -> (sink[] += Project.fmadd(1.5, 2.5, 0.1)))
t_ccall_fmadd = bench_ns(() -> (sink[] += ccall(sym_fmadd, Cdouble, (Cdouble,Cdouble,Cdouble), 1.5, 2.5, 0.1)))

# ─────────────────────────────────────────────────────────────────────────────
# hsum(double*,int)->double  (256-element array)
# ─────────────────────────────────────────────────────────────────────────────
t_llvm_hsum  = bench_ns(() -> (sink[] += Project.hsum(pointer(VEC), length(VEC))))
t_ccall_hsum = bench_ns(() -> (sink[] += ccall(sym_hsum, Cdouble, (Ptr{Cdouble},Cint), pointer(VEC), length(VEC))))

using Printf

# ── pure-C baselines ──────────────────────────────────────────────────────────
function parse_c_results(bin)
    d = Dict{String,Float64}()
    isfile(bin) || return d
    for line in split(read(`$bin`, String), '\n')
        m = match(r"^(\w+)\s+([\d.]+)", line)
        m !== nothing && (d[m[1]] = parse(Float64, m[2]))
    end
    d
end

c_results       = parse_c_results(joinpath(PROJ, "bench_c"))
c_results_ffast = parse_c_results(joinpath(PROJ, "bench_c_ffast"))

# ─────────────────────────────────────────────────────────────────────────────
# Results table
# ─────────────────────────────────────────────────────────────────────────────
println("\n" * "="^80)
println("  Flags used to compile libproject.so: -O3 -march=native -ffast-math")
println("  llvmcall/ccall both call that vectorized library.")
println("  C baselines are standalone binaries measuring their own compiled hsum.")
println("="^80)
println(rpad("Function", 20), rpad("llvmcall", 14), rpad("ccall", 16),
        rpad("C (no ffast)", 16), "C (-ffast-math)")
println("-"^80)

function row(name, t_lc, t_cc)
    c1 = haskey(c_results,       name) ? @sprintf("%.2f", c_results[name])       : "—"
    c2 = haskey(c_results_ffast, name) ? @sprintf("%.2f", c_results_ffast[name]) : "—"
    println(rpad(name, 20), rpad(@sprintf("%.2f", t_lc), 14), rpad(@sprintf("%.2f", t_cc), 16),
            rpad(c1, 16), c2)
end

row("iadd",  t_llvm_iadd,  t_ccall_iadd)
row("fmadd", t_llvm_fmadd, t_ccall_fmadd)
row("hsum",  t_llvm_hsum,  t_ccall_hsum)
println("="^75)
println("(sink=", sink[], " — ensures calls aren't dead-code-eliminated)")
