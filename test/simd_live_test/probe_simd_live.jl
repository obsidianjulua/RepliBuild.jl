# Subprocess probe for test_simd_vectors_live.jl. Loads one wrapper (the C or
# the C++ build of simd_live) and calls every take_/make_ with the SSE
# registers poisoned first: a record built from its C-side image (a Tier 3
# call), its take_R checked against the value C computes, make_R's bytes
# against the image under the member mask; then the bare vectors. Prints
# `PROBE <name>/<take|make>: PASS | TRAP … | FAIL …` per call and PROBE_DONE.
# A call the wrapper refuses with an ABI Safety Trap prints `TRAP …`.
# Run: julia --project=<RepliBuild> probe_simd_live.jl <wrapper_path> [skip...]
# Every name after the wrapper (`v2f/take`) is skipped: a call that would
# crash this process runs in a probe of its own.

wrapper = ARGS[1]
const SKIP = Set(ARGS[2:end])
M = include(wrapper)

# (record, what take_R returns for the canonical value)
const RECORDS = [("r_v1f", 1.5), ("r_v2i", -37.0), ("r_v2f", 26.25), ("r_v1d", 6.75),
                 ("r_v2f_i", 715.5), ("r_v4f", 2160.5)]

tobytes(x) = (r = Ref(x); GC.@preserve r copy(unsafe_wrap(Array, Ptr{UInt8}(pointer_from_objref(r)), sizeof(x))))

function failure(e)
    msg = first(sprint(showerror, e), 300)
    return (e isa ErrorException && occursin("ABI Safety Trap", msg)) ? "TRAP " * msg : "FAIL threw " * msg
end

function probe(name, f)
    name in SKIP && return
    verdict = try
        M.poison_regs()
        f()
    catch e
        failure(e)
    end
    println("PROBE $name: ", verdict)
    flush(stdout)
end

for (rname, expected) in RECORDS
    T = getfield(M, Symbol(rname))
    nbytes = Int(getfield(M, Symbol("size_", rname))())
    buf = zeros(UInt8, nbytes)
    GC.@preserve buf getfield(M, Symbol("image_", rname))(Ptr{Cvoid}(pointer(buf)))
    mask = zeros(UInt8, nbytes)
    GC.@preserve mask getfield(M, Symbol("image_mask_", rname))(Ptr{Cvoid}(pointer(mask)))
    sizeof(T) == nbytes || println("PROBE $rname/size: FAIL julia sizeof=$(sizeof(T)) C=$nbytes")
    v = GC.@preserve buf unsafe_load(Ptr{T}(pointer(buf)))
    probe("$rname/take", () -> (got = getfield(M, Symbol("take_", rname))(v);
                                got == expected ? "PASS" : "FAIL got $got expected $expected"))
    probe("$rname/make", () -> (b = tobytes(getfield(M, Symbol("make_", rname))());
                                (b .& mask) == (buf .& mask) ? "PASS" :
                                "FAIL bytes $(bytes2hex(b)) expected $(bytes2hex(buf)) (padding masked)"))
end

# Bare vectors: the lanes as a tuple in, the lanes as a tuple out.
lanes(x) = Tuple(Float32(y) for y in (x isa Tuple ? x : getfield(x, 1)))
probe("v2f/take", () -> (got = M.take_v2f((1.25f0, 2.5f0)); got == 26.25 ? "PASS" : "FAIL got $got expected 26.25"))
probe("v2f/make", () -> (got = lanes(M.make_v2f(2.0f0)); got == (2f0, 4f0) ? "PASS" : "FAIL got $got"))
probe("v4f/take", () -> (got = M.take_v4f((0.5f0, 1f0, 1.5f0, 2f0)); got == 2160.5 ? "PASS" : "FAIL got $got expected 2160.5"))
probe("v4f/make", () -> (got = lanes(M.make_v4f(1.0f0)); got == (1f0, 2f0, 3f0, 4f0) ? "PASS" : "FAIL got $got"))
println("PROBE_DONE")
