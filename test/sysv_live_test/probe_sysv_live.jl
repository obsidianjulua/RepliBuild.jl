# Subprocess probe for test_sysv_live.jl. Loads the wrapper once and, for
# every record the wrapper has, builds the value from the C-side image (a Tier 3
# call), poisons the SSE registers, calls take_R through its Tier 2 thunk, and
# compares make_R's bytes with the image under the member mask (padding is
# unspecified after a trip through registers). Prints one line per direction,
# `PROBE <R>/take: …` and `PROBE <R>/make: …`, then PROBE_DONE. A call the
# wrapper refuses with an ABI Safety Trap prints `TRAP <message>`, which the
# test expects exactly where a trap is the right answer.
# Run: julia --project=<RepliBuild> probe_sysv_live.jl <wrapper_path>

wrapper = ARGS[1]
M = include(wrapper)

# (record, what take_R returns for the canonical value)
const CASES = [
    ("c_f_d", 24.0), ("c_d_f", 17.25), ("c_p_f", 4111.0), ("c_ubf_f", 15.0), ("c_f_pad", 24.0),
    ("c_wrap_df", 17.25), ("c_al16_f", 15.0), ("c_cf", 26.5), ("c_cd", 26.5), ("c_ubf_d", 25.0),
    ("c_ld", 1.5), ("c_u_ld_i", 1.5), ("c_s_f", 18.0), ("c_pk_sf", 187.0), ("c_pk_ci", 1007.0),
    ("c_u_pk", 1007.0), ("c_pk_alnest", 37.0), ("c_pp1_bf", 10007.0), ("c_u_fi", 15.0),
    ("c_bf2_f", 203.0), ("c_i128_l", 95.0), ("c_al16_big", 321.0),
    ("c_f_i", 71.5), ("c_d2", 26.5), ("c_c_d", 32.0),
]

tobytes(x) = ismutabletype(typeof(x)) ? collect(getfield(x, 1)) :
             (r = Ref(x); GC.@preserve r copy(unsafe_wrap(Array, Ptr{UInt8}(pointer_from_objref(r)), sizeof(x))))

# "TRAP …" for the generators' refusal stub, "FAIL threw …" for anything else.
function failure(e)
    msg = first(sprint(showerror, e), 300)
    return (e isa ErrorException && occursin("ABI Safety Trap", msg)) ? "TRAP " * msg : "FAIL threw " * msg
end

for (rname, expected) in CASES
    isdefined(M, Symbol("take_", rname)) || continue
    T = getfield(M, Symbol(rname))
    nbytes = Int(getfield(M, Symbol("size_", rname))())
    buf = zeros(UInt8, nbytes)
    GC.@preserve buf getfield(M, Symbol("image_", rname))(Ptr{Cvoid}(pointer(buf)))
    mask = zeros(UInt8, nbytes)
    GC.@preserve mask getfield(M, Symbol("image_mask_", rname))(Ptr{Cvoid}(pointer(mask)))
    masked(b) = b .& mask
    v = ismutabletype(T) ? T(Tuple(buf)) : GC.@preserve buf unsafe_load(Ptr{T}(pointer(buf)))
    sizeof(T) == nbytes || println("PROBE $rname/size: FAIL julia sizeof=$(sizeof(T)) C=$nbytes")
    take = getfield(M, Symbol("take_", rname))
    let verdict = try
            M.poison_regs()
            got = take(v)
            got == expected ? "PASS" : "FAIL got $got expected $expected"
        catch e
            failure(e)
        end
        println("PROBE $rname/take: ", verdict)
    end
    let verdict = try
            M.poison_regs()
            r = getfield(M, Symbol("make_", rname))()
            b = tobytes(r)
            masked(b) == masked(buf) ? "PASS" :
                "FAIL bytes $(bytes2hex(b)) expected $(bytes2hex(buf)) (padding masked)"
        catch e
            failure(e)
        end
        println("PROBE $rname/make: ", verdict)
    end
end
println("PROBE_DONE")
