# Subprocess probe for test_pure_virtual_live.jl: calls pure virtual member
# functions through the wrapper, on objects of four dynamic types reached
# through a pointer to their abstract base, and asks for what the wrapper must
# refuse. A pure slot reached on an object of the abstract class itself ends the
# process (`__cxa_pure_virtual`), and so would a C++ exception no thunk caught,
# which is why this is a child. Prints one "PROBE <case>: PASS|FAIL …" line
# each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_pure_virtual.jl <wrapper_path> jit|aot

wrapper, mode = ARGS
using RepliBuild
const M = include(wrapper)
const JM = RepliBuild.JITManager

function probe(f, name)
    verdict = try
        p = String[]
        f(p)
        isempty(p) ? "PASS" : "FAIL " * join(unique(p)[1:min(end, 4)], "; ")
    catch e
        "FAIL threw " * first(sprint(showerror, e), 300)
    end
    println("PROBE $name: ", verdict)
    flush(stdout)
end

# Zeroed storage for one object of the class, as a raw pointer.
alloc(T::Symbol) = Libc.calloc(1, M.struct_size(T))
# kind: 0 Dog, 1 Puppy, 2 Bird, 3 Duck
with_animal(f, kind) = (a = M.make_animal(kind); try f(a) finally M.free_animal(a) end)
# The message of what `f` throws, or `nothing` when it returns.
function refusal(f)
    try
        f()
        return nothing
    catch e
        return e isa ErrorException ? e.msg : "threw $(typeof(e)): " * first(sprint(showerror, e), 200)
    end
end

# ── How the wrapper is bound ────────────────────────────────────────────────

if mode == "aot"
    # Every thunk is an address resolved at load, and no engine was built.
    probe("aot_thunks_resolved") do p
        slots = M._THUNK_SLOTS
        unresolved = [sym for (slot, sym) in slots if slot[] == C_NULL]
        isempty(unresolved) || push!(p, "unresolved: $(first(unresolved, 3))")
        any(s -> s[2] == "_mlir_ciface__ZNK6Animal4nameEv_thunk", slots) || push!(p, "no slot for Animal::name")
    end
else
    # The library's engine came up whole: a callee the module names and no
    # library defines would have taken every thunk with it.
    probe("jit_engine_initialized") do p
        (@atomic JM.GLOBAL_JIT.initialized) || push!(p, "no engine")
        errs = [sprint(showerror, e.init_error) for e in JM.GLOBAL_JIT.engines if e.init_error !== nothing]
        isempty(errs) || push!(p, first(errs))
    end
end

# ── The interface's own methods, through the interface ──────────────────────

probe("name_through_the_base") do p
    for (kind, want) in ((0, "dog"), (1, "puppy"), (2, "bird"), (3, "duck"))
        with_animal(kind) do a
            a isa Ptr{M.Animal} || push!(p, "make_animal returns $(typeof(a))")
            (got = M.Animal_name(a)) == want || push!(p, "Animal_name(kind $kind) = $(repr(got))")
            (got = unsafe_string(M.Animal_name_ptr(a))) == want || push!(p, "Animal_name_ptr(kind $kind) = $(repr(got))")
        end
    end
end

probe("scalar_in_scalar_out") do p
    for (kind, want) in ((0, 20.0), (1, 6.0), (2, 61.5), (3, 60.5))
        with_animal(kind) do a
            (got = M.Animal_speed(a, 2.0)) == want || push!(p, "Animal_speed(kind $kind, 2) = $got")
        end
    end
end

# `unsigned char` and `bool` are extended by the caller: 200 must not arrive
# sign-extended, and `false` must arrive as zero.
probe("integers_narrower_than_int") do p
    with_animal(0) do a
        (got = M.Animal_rank(a, 5, UInt8(200), true)) == 1205 || push!(p, "dog rank(5, 200, true) = $got")
        (got = M.Animal_rank(a, 5, UInt8(200), false)) == 205 || push!(p, "dog rank(5, 200, false) = $got")
    end
    with_animal(3) do a
        (got = M.Animal_rank(a, 5, UInt8(255), true)) == 2265 || push!(p, "duck rank(5, 255, true) = $got")
    end
end

probe("void_return_and_pointer_argument") do p
    with_animal(2) do a
        r = M.Animal_rename(a, "tweety")
        r === nothing || push!(p, "rename returned $(repr(r))")
        (got = M.Animal_name(a)) == "tweety" || push!(p, "name after rename = $(repr(got))")
    end
end

# The override throws: the thunk's landing pad (libJLCS's guard on Windows)
# turns it into a Julia exception, and the next call is clean.
probe("exception_from_the_override") do p
    for (kind, who, twice) in ((0, "dog", 8), (2, "bird", 12))
        with_animal(kind) do a
            try
                got = M.Animal_risky(a, -1)
                push!(p, "risky(-1) returned $got")
            catch e
                e isa JM.CxxException || push!(p, "threw $(typeof(e))")
                occursin("$who: negative", sprint(showerror, e)) || push!(p, "message: $(first(sprint(showerror, e), 120))")
            end
            (got = M.Animal_risky(a, 4)) == twice || push!(p, "risky(4) after the throw = $got")
        end
    end
end

# A member that is not virtual and calls the pure one, and a virtual that has
# a symbol: both as before.
probe("defined_members_unchanged") do p
    with_animal(1) do a
        (got = M.Animal_describe(a)) == "puppy with 4 legs" || push!(p, "describe = $(repr(got))")
        (got = M.Animal_legs(a)) == 4 || push!(p, "legs = $got")
        (got = M.Dog_name(a)) == "puppy" || push!(p, "Dog_name(puppy) = $(repr(got))")
    end
end

# ── The factory struct ──────────────────────────────────────────────────────

probe("factory_struct_proxies") do p
    n = M.live_animals()
    let obj = M.Animal(Cint(1))
        obj isa M.Animal || push!(p, "Animal(1) is a $(typeof(obj))")
        (got = M.name(obj)) == "puppy" || push!(p, "name(obj) = $(repr(got))")
        (got = M.speed(obj, 2.0)) == 6.0 || push!(p, "speed(obj, 2) = $got")
        (got = M.describe(obj)) == "puppy with 4 legs" || push!(p, "describe(obj) = $(repr(got))")
        finalize(obj)
    end
    M.live_animals() == n || push!(p, "live_animals $(M.live_animals()), was $n")
end

# ── What is passed over ─────────────────────────────────────────────────────

probe("private_protected_variadic_and_internal_have_no_wrapper") do p
    for name in (:Animal_secret, :Animal_hidden, :Logger_log, :secret, :hidden)
        isdefined(M, name) && push!(p, "$name is defined")
    end
    # They are still what the library calls.
    with_animal(0) do a
        (got = M.Animal_secret_plus(a, 1)) == 43 || push!(p, "secret_plus = $got")
    end
    lg = M.make_logger()
    (got = M.Logger_lines(lg)) == 0 || push!(p, "Logger_lines = $got")
    M.drop_logger(lg)
    # A class of one translation unit: its pure virtual is the library's own.
    bound = [n for n in names(M; all = true) if occursin("Step_step", string(n))]
    isempty(bound) || push!(p, "bound: $bound")
    (got = M.run_step(21)) == 42 || push!(p, "run_step = $got")
end

probe("pure_virtual_destructor_is_bound_by_symbol_only") do p
    t = M.make_coin()
    (got = M.Token_value(t)) == 25 || push!(p, "Token_value(coin) = $got")
    isdefined(M, :Token_destroy_Token) || push!(p, "no Token_destroy_Token")
    isdefined(M, :Token_delete_Token) || push!(p, "no Token_delete_Token")
    # `Coin`'s destructor is implicit; where its symbol is not in the wrapper,
    # `Token`'s deleting destructor frees the same block.
    (isdefined(M, :Coin_delete_Coin) ? M.Coin_delete_Coin : M.Token_delete_Token)(t)
end

# ── What is refused ─────────────────────────────────────────────────────────

probe("record_by_value_is_refused") do p
    with_animal(0) do a
        for (what, call) in (("returns a record by value", () -> M.Animal_span(a)),
                             ("takes a record by value", () -> M.Animal_weigh(a, M.Span(1.0, 2.0))),
                             ("returns a record by value", () -> M.Animal_blob(a)))
            msg = refusal(call)
            msg === nothing && (push!(p, "a call that $what returned"); continue)
            (occursin("ABI Safety Trap", msg) && occursin("pure virtual", msg) && occursin(what, msg)) ||
                push!(p, "message: $(first(msg, 160))")
        end
        # The override is a function of its own class, with a symbol.
        s = M.Dog_span(a)
        (s.lo, s.hi) == (1.0, 4.0) || push!(p, "Dog_span = $s")
        (got = M.Dog_weigh(a, M.Span(1.0, 2.0))) == 3 || push!(p, "Dog_weigh = $got")
    end
end

# No constructor of an abstract class runs: the storage is untouched, so no
# object exists whose pure slots a later call could reach.
probe("abstract_class_is_not_constructed") do p
    for (T, build) in ((:Animal, q -> M.Animal_Animal(q, 4)), (:Shape, q -> M.Shape_Shape(q)),
                       (:Poly, q -> M.Poly_Poly(q, 3)), (:Token, q -> M.Token_Token(q)))
        q = alloc(T)
        msg = refusal(() -> build(q))
        if msg === nothing
            push!(p, "$(T)'s constructor returned")
        else
            (occursin("ABI Safety Trap", msg) && occursin("is an abstract class", msg)) ||
                push!(p, "$T: $(first(msg, 160))")
        end
        all(iszero, unsafe_wrap(Array, Ptr{UInt8}(q), M.struct_size(T))) || push!(p, "$(T)'s storage was written")
        Libc.free(q)
    end
end

# ── Classes that define what they inherit ───────────────────────────────────

probe("concrete_subclass_built_in_place") do p
    s = alloc(:Square)
    M.Square_Square(s, 3)
    (got = M.Shape_area(s)) == 9 || push!(p, "Shape_area(square) = $got")
    (got = M.Shape_sides(s)) == 4 || push!(p, "Shape_sides(square) = $got")
    (got = M.Poly_sides(s)) == 4 || push!(p, "Poly_sides(square) = $got")
    (got = M.Square_area(s)) == 9 || push!(p, "Square_area = $got")
    M.Square_destroy_Square(s)
    Libc.free(s)
end

probe("interface_with_no_symbol_at_all") do p
    l = M.make_counter(7)
    M.Listener_on_event(l, 5)
    M.Listener_on_event(l, 6)
    (got = M.counter_seen(l)) == 11 || push!(p, "counter_seen = $got")
    (got = M.Listener_priority(l)) == 7 || push!(p, "Listener_priority = $got")
    (got = M.dispatch(l, 1)) == 7 || push!(p, "dispatch = $got")
    (got = M.counter_seen(l)) == 12 || push!(p, "counter_seen after dispatch = $got")
    M.drop_listener(l)
end

probe("pure_virtual_with_a_definition") do p
    b = M.make_derived()
    (got = M.Base_tag(b)) == 8 || push!(p, "Base_tag(derived) = $got")
    M.drop_base(b)
end

probe("namespace_reference_bool_and_typed_pointer") do p
    with_animal(0) do a
        with_animal(2) do other
            k = M.zoo_hire(a)
            (got = M.zoo_Keeper_feeds(k, a)) === true || push!(p, "feeds(favourite) = $(repr(got))")
            (got = M.zoo_Keeper_feeds(k, other)) === false || push!(p, "feeds(other) = $(repr(got))")
            fav = M.zoo_Keeper_favourite(k)
            fav isa Ptr{M.Animal} || push!(p, "favourite returns $(typeof(fav))")
            Ptr{Cvoid}(fav) == Ptr{Cvoid}(a) || push!(p, "favourite is another object")
            M.zoo_fire(k)
        end
    end
end

probe("nothing_leaked") do p
    M.live_animals() == 0 || push!(p, "live_animals = $(M.live_animals())")
end

println("PROBE_DONE")
