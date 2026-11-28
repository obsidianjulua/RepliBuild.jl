# Calls the functions the ownership unit compiled into the library, on objects
# the library builds: each PROBE line is one fact about ending, copying or
# moving. Run in a child by test_ownership_unit_live.jl, which passes the
# library and the nine function names (end, copy, move for a class with a
# destructor, for std::vector<int>, and for a std::unique_ptr).
import Libdl
lib = Libdl.dlopen(ARGS[1])
sym(s) = Libdl.dlsym(lib, Symbol(s))
(t_end, t_copy, t_move, v_end, v_copy, v_move, u_end, u_copy, u_move) = sym.(ARGS[2:10])
live() = ccall(sym("_ZN3own10live_countEv"), Cint, ())
say(name, ok, detail = "") = println("PROBE ", name, ": ", ok ? "PASS" : "FAIL " * string(detail))
buf() = zeros(UInt8, 64)                       # room for any of them, zeroed

# ── A class with a destructor: every construction and ending is counted. ────
a = buf(); b = buf(); c = buf()
GC.@preserve a b c begin
    (pa, pb, pc) = pointer.((a, b, c))
    ccall(sym("own_make_tracked"), Cvoid, (Ptr{Cvoid}, Cint), pa, 5)
    say("made_is_one_alive", live() == 1, live())
    rc = ccall(t_copy, Cint, (Ptr{Cvoid}, Ptr{Cvoid}), pb, pa)
    say("copy_makes_a_second", rc == 0 && live() == 2, (rc, live()))
    say("copy_has_the_value", ccall(sym("own_tracked_value"), Cint, (Ptr{Cvoid},), pb) == 5)
    say("copy_leaves_the_original", ccall(sym("own_tracked_value"), Cint, (Ptr{Cvoid},), pa) == 5)
    rm = ccall(t_move, Cint, (Ptr{Cvoid}, Ptr{Cvoid}), pc, pa)
    say("move_takes_the_value", rm == 0 && ccall(sym("own_tracked_value"), Cint, (Ptr{Cvoid},), pc) == 5, rm)
    say("moved_from_is_empty_and_alive", ccall(sym("own_tracked_value"), Cint, (Ptr{Cvoid},), pa) == -1 && live() == 3, live())
    ccall(t_end, Cvoid, (Ptr{Cvoid},), pa)
    ccall(t_end, Cvoid, (Ptr{Cvoid},), pb)
    say("ending_two_leaves_one", live() == 1, live())
    ccall(t_end, Cvoid, (Ptr{Cvoid},), pc)
    say("ending_all_leaves_none", live() == 0, live())
end

# ── A container: the copy owns its own memory. ──────────────────────────────
a = buf(); b = buf(); c = buf()
GC.@preserve a b c begin
    (pa, pb, pc) = pointer.((a, b, c))
    total(p) = ccall(sym("own_points_sum"), Clong, (Ptr{Cvoid},), p)
    ccall(sym("own_make_points"), Cvoid, (Ptr{Cvoid}, Cint), pa, 100)
    rc = ccall(v_copy, Cint, (Ptr{Cvoid}, Ptr{Cvoid}), pb, pa)
    say("vector_copy_reads_the_same", rc == 0 && total(pb) == 4950 && total(pa) == 4950, (rc, total(pb)))
    ccall(v_end, Cvoid, (Ptr{Cvoid},), pa)
    say("vector_copy_outlives_the_original", total(pb) == 4950, total(pb))
    rm = ccall(v_move, Cint, (Ptr{Cvoid}, Ptr{Cvoid}), pc, pb)
    say("vector_move_takes_the_elements", rm == 0 && total(pc) == 4950 && total(pb) == 0, (rm, total(pc), total(pb)))
    ccall(v_end, Cvoid, (Ptr{Cvoid},), pb)
    ccall(v_end, Cvoid, (Ptr{Cvoid},), pc)
end

# ── What cannot be copied says so, and touches nothing. ─────────────────────
a = buf(); b = buf()
GC.@preserve a b begin
    (pa, pb) = pointer.((a, b))              # all zero is an empty unique_ptr
    b .= 0xab
    rc = ccall(u_copy, Cint, (Ptr{Cvoid}, Ptr{Cvoid}), pb, pa)
    say("unique_ptr_copy_is_refused", rc == 2 && all(==(0xab), b), rc)
    rm = ccall(u_move, Cint, (Ptr{Cvoid}, Ptr{Cvoid}), pb, pa)
    say("unique_ptr_moves", rm == 0 && all(==(0x00), b[1:sizeof(Ptr{Cvoid})]), rm)
    ccall(u_end, Cvoid, (Ptr{Cvoid},), pa)
    ccall(u_end, Cvoid, (Ptr{Cvoid},), pb)
end
say("nothing_left_alive", live() == 0, live())
println("PROBE_DONE")
