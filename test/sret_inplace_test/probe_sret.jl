# Subprocess probe for test_sret_in_caller_buffer.jl. A wrong return convention
# leaves a dangling self-pointer or crashes, so this runs in a child. Prints one
# "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_sret.jl <wrapper_path>

using RepliBuild

wrapper, = ARGS
const M = include(wrapper)
const FG = RepliBuild.JLCSIRGenerator.FunctionGen
const MD = RepliBuild._read_json(joinpath(dirname(wrapper), "compilation_metadata.json"))
const SD = MD["struct_definitions"]
const RA = get(MD, "record_abi", Dict{String,Any}())

function probe(f, name)
    verdict = try
        p = String[]
        f(p)
        isempty(p) ? "PASS" : "FAIL " * join(p, "; ")
    catch e
        "FAIL threw " * first(sprint(showerror, e), 300)
    end
    println("PROBE $name: ", verdict)
end
check(p, what, got, want) = got == want || push!(p, "$what: got $got, want $want")
fn(name) = MD["functions"][findfirst(f -> get(f, "name", "") == name, MD["functions"])]

# make_self(7), called the way the thunk generator says its thunk takes it:
# in the caller's buffer when `returns_in_caller_buffer` holds, else the old
# shape (the thunk returns the bytes). Either way the object then lives in
# `buf`, and its `self` must be `buf`'s address.
probe("built_in_the_callers_buffer") do p
    f = fn("make_self")
    inplace = FG.returns_in_caller_buffer(f, SD, RA)
    check(p, "make_self returns in the caller's buffer", inplace, true)
    name = "_mlir_ciface_" * FG.thunk_symbol(f["mangled"], inplace)
    buf = Ref{M.SelfRef}()
    GC.@preserve buf begin
        addr = Base.unsafe_convert(Ptr{Cvoid}, buf)
        if inplace
            RepliBuild.JITManager.invoke(name, addr, Int32(7))
        else
            buf[] = RepliBuild.JITManager.invoke(name, M.SelfRef, Int32(7))
        end
        s = buf[]
        check(p, "self == the buffer", UInt(s.self), UInt(addr))
        check(p, "v, w", (s.v, s.w), (Int32(7), 3.5))
        # C++'s own view of the object where it lives.
        check(p, "self_ok", M.self_ok(buf), Int32(1))
        check(p, "self_value", M.self_value(buf), Int32(7))
    end
end

# A wrapper generated before the change calls the old name with the old shape.
# It must fail at the call, saying why, never reach the in-place thunk.
probe("stale_call_says_rewrap") do p
    old = "_mlir_ciface_" * fn("make_self")["mangled"] * "_thunk"
    msg = try
        RepliBuild.JITManager.invoke(old, M.SelfRef, Int32(7))
        "no error"
    catch e
        sprint(showerror, e)
    end
    occursin("_inplace` is: this wrapper was generated before", msg) ||
        push!(p, "old-name call: " * first(msg, 200))
end

# The control: a trivially copyable record takes the wrapper's own call, as
# it always did.
probe("trivial_record_unchanged") do p
    check(p, "make_plain predicate", FG.returns_in_caller_buffer(fn("make_plain"), SD, RA), false)
    r = M.make_plain(4)
    check(p, "make_plain", (r.a, r.b, r.c), (Int32(4), Int32(5), Int32(6)))
end

println("PROBE_DONE")
