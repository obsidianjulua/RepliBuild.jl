# Subprocess probe for test_name_screen_live.jl: loads the two wrappers of
# test/name_screen_test (a C library and a C++ one whose identifiers are names
# Julia and the wrapper already use), calls each library function under the
# name it is bound as, and checks that Base is still Base's. Prints one
# "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_names.jl <c wrapper> <c++ wrapper>

c_wrapper, cpp_wrapper = ARGS

@enum Fruit apple = 3 pear = 4            # a Julia enum of the caller's own

function probe(f, name)
    verdict = try
        p = String[]
        f(p)
        isempty(p) ? "PASS" : "FAIL " * join(unique(p)[1:min(end, 6)], "; ")
    catch e
        "FAIL threw " * first(replace(sprint(showerror, e), '\n' => ' '), 300)
    end
    println("PROBE $name: ", verdict)
end

# A module that fails to load ends the probe here, with no PROBE_DONE.
const C = include(c_wrapper)
const X = include(cpp_wrapper)

# The library function reached through `name` in `M`, called on 1: its value,
# or `nothing` when `name` is not a function of `M`'s that takes an integer.
function via(M, name)
    s = Symbol(name)
    isdefined(M, s) || return nothing
    f = getfield(M, s)
    (f isa Function && parentmodule(f) === M) || return nothing
    return try f(1) catch; nothing end
end

probe("c_ordinary_functions") do p
    C.plain(1) == 2 || push!(p, "plain(1) = $(C.plain(1))")
    C.label() == "lbl" || push!(p, "label() = $(repr(C.label()))")
end

# A name Julia can never give a library function, a name the wrapper binds,
# and a name its own text uses: each C function is reachable as `<name>_`, and
# the name itself is not the C function.
probe("c_moved_functions") do p
    for (name, k) in (("String", 101), ("Ptr", 102), ("Ref", 103), ("Cint", 104), ("eval", 105),
                      ("include", 106), ("ccall", 107), ("Base", 108), ("Libdl", 109), ("RepliBuild", 110),
                      ("nothing", 111), ("unsafe_convert", 112), ("LIBRARY_PATH", 114),
                      ("METADATA", 115), ("Namescreenc", 116))
        (got = via(C, name * "_")) == 1 + k || push!(p, "$(name)_(1) = $(repr(got)), want $(1 + k)")
        via(C, name) === nothing || push!(p, "`$name` is the C function too")
    end
end

# Base functions the wrapper's text does not use stay the library's, under
# their own names, as functions of the module and not methods of Base's.
# `isfile` is one the wrapper's header calls, as `Base.isfile`.
probe("c_kept_functions") do p
    for (name, k) in (("isfile", 113), ("length", 117), ("print", 118), ("show", 119))
        (got = via(C, name)) == 1 + k || push!(p, "$name(1) = $(repr(got)), want $(1 + k)")
        isdefined(C, Symbol(name * "_")) && push!(p, "`$(name)_` is bound")
        getfield(C, Symbol(name)) === getfield(Base, Symbol(name)) && push!(p, "`$name` is Base's")
    end
    (length("abc") == 3 && !isfile("/no/such/file")) || push!(p, "Base.length or Base.isfile changed")
    C._check_build_identity() === nothing || push!(p, "_check_build_identity")
end

probe("c_wrapper_names_are_its_own") do p
    C.nothing === nothing || push!(p, "nothing is $(C.nothing)")
    C.Base === Base || push!(p, "Base is $(C.Base)")
    C.String === String || push!(p, "String is $(C.String)")
    C.Integer === Integer || push!(p, "Integer is $(C.Integer)")
    C.Bool === Bool || push!(p, "Bool is $(C.Bool)")
    C.Symbol === Symbol || push!(p, "Symbol is $(C.Symbol)")
    (C.LIBRARY_PATH isa String && isfile(C.LIBRARY_PATH)) || push!(p, "LIBRARY_PATH is $(repr(C.LIBRARY_PATH))")
    C.METADATA isa Dict || push!(p, "METADATA is a $(typeof(C.METADATA))")
    C.Namescreenc === C || push!(p, "the module's name is $(C.Namescreenc)")
end

probe("c_records_enumerators_globals") do p
    i = C.make_integer(4)
    (typeof(i) === C.c_Integer && i.v == 4 && i.w == 2.0) || push!(p, "make_integer(4) = $i")
    C.take_integer(Ref(i)) == 5 || push!(p, "take_integer = $(C.take_integer(Ref(i)))")
    b = C.make_bool(6)
    (typeof(b) === C.c_Bool && b.v == 6) || push!(p, "make_bool(6) = $b")
    C.take_bool(Ref(b)) == 8 || push!(p, "take_bool = $(C.take_bool(Ref(b)))")
    # A record named after a Base type the wrapper's text does not use.
    t = C.make_task(8)
    (typeof(t) === C.Task && C.Task !== Base.Task && t.v == 8) || push!(p, "make_task(8) = $t")
    C.take_task(Ref(t)) == 11 || push!(p, "take_task = $(C.take_task(Ref(t)))")
    # C's bool beside the record named Bool.
    (C.truth(1) === true && C.truth(-1) === false) || push!(p, "truth = $(C.truth(1)), $(C.truth(-1))")
    (C.Kind_Symbol == C.Kind(1) && C.stat == C.Kind(2) && C.KindPlain == C.Kind(3)) || push!(p, "enumerators")
    C.kind_of(1) == C.Kind_Symbol || push!(p, "kind_of(1) = $(C.kind_of(1))")
    # An integer parameter takes an integer and an enum, as before.
    C.plain(apple) == 4 || push!(p, "plain(apple) = $(C.plain(apple))")
    (C.Dict_() == 7 && C.counter() == 9) || push!(p, "globals: $(C.Dict_()), $(C.counter())")
    unsafe_load(C.Dict__ptr()) == 7 || push!(p, "Dict__ptr")
end

probe("cpp_ordinary_functions") do p
    X.plain(1) == 2 || push!(p, "plain(1) = $(X.plain(1))")
    X.label() == "lbl" || push!(p, "label() = $(repr(X.label()))")
    X.eval_(1) == 106 || push!(p, "eval_(1) = $(X.eval_(1))")
    (X.Dict_() == 11 && X.Dict === Dict) || push!(p, "the global `Dict`: $(X.Dict_())")
end

# Each record: made by value through a thunk, read as fields, passed by
# reference, and its method called. `jl` is the name the wrapper reports.
probe("cpp_records") do p
    for (cname, jl, k) in (("Bool", "c_Bool", 10), ("Cvoid", "c_Cvoid", 20), ("Int", "c_Int", 30),
                           ("Integer", "c_Integer", 40), ("Cstring", "c_Cstring", 50), ("UInt8", "c_UInt8", 60),
                           ("String", "c_String", 70), ("Task", "Task", 80), ("Widget", "Widget", 90))
        T = getfield(X, Symbol(jl))
        (T isa DataType && parentmodule(T) === X && sizeof(T) == 16) || push!(p, "$jl is $T")
        x = getfield(X, Symbol("make_" * cname))(4)
        (typeof(x) === T && x.v == 4 && x.w == 2.0) || push!(p, "make_$cname(4) = $x")
        (got = getfield(X, Symbol("take_" * cname))(Ref(x))) == 6 + k || push!(p, "take_$cname = $got")
        get = unique!([n for n in (jl * "_get", cname * "_get") if isdefined(X, Symbol(n))])
        length(get) == 1 || push!(p, "$cname::get is bound as $get")
        isempty(get) || (got = getfield(X, Symbol(get[1]))(Ref(x))) == 4 + k || push!(p, "$(get[1]) = $got")
    end
end

probe("cpp_own_types_beside_the_records") do p
    (X.truth(Ref(X.make_Bool(1))) === true && X.truth(Ref(X.make_Bool(-1))) === false) || push!(p, "truth")
    X.byte_of(Ref(X.make_UInt8(65))) === 0x41 || push!(p, "byte_of = $(X.byte_of(Ref(X.make_UInt8(65))))")
    out = Ref{Cint}(0)
    X.nop(Ref(X.make_Cvoid(5)), out)
    out[] == 12 || push!(p, "nop wrote $(out[])")
    (X.Bool === Bool && X.Cvoid === Cvoid && X.Int === Int && X.Integer === Integer &&
     X.Cstring === Cstring && X.UInt8 === UInt8 && X.String === String) || push!(p, "a wrapper name is the library's")
end

# A record holding the record named Bool, by value both ways.
probe("cpp_record_holding_one_by_value") do p
    h = X.hold(5)
    (h.inner isa X.c_Bool && h.inner.v == 5 && h.inner.w == 10.0 && h.tag == 15) || push!(p, "hold(5) = $h")
    X.held(h) == 30 || push!(p, "held = $(X.held(h))")
end

# Method proxies are top-level names too.
probe("cpp_method_proxies") do p
    f = X.File(Cint(3))                       # the owning type: its factory, and its deleter as finalizer
    for (name, want, moved) in (("open", 4, false), ("isempty", 0, false), ("String", 5, true),
                                ("get", 6, false), ("size", 7, false))
        bound = moved ? name * "_" : name
        g = isdefined(X, Symbol(bound)) ? getfield(X, Symbol(bound)) : nothing
        (g isa Function && parentmodule(g) === X) || (push!(p, "`$bound` is not the module's function"); continue)
        (got = g(f)) == want || push!(p, "$bound(f) = $got, want $want")
    end
    # The wrapper's header calls `open` and `isempty` too, as Base's: with the
    # proxies in the module it still loaded, and this still runs.
    X._check_build_identity() === nothing || push!(p, "_check_build_identity")
end

# Neither wrapper gave a type of Base a method (a constructor is one), and
# what the audit saw from a caller's side is gone.
probe("base_is_unchanged") do p
    for T in (String, Ptr, Ref, Cint, Integer, Bool, Dict, Symbol, Task), M in (C, X)
        n = count(m -> parentmodule(m) === M, methods(T))
        n == 0 || push!(p, "$(nameof(M)) defines $n method(s) of $T")
    end
    applicable(String, 1) && push!(p, "String(1) = $(String(1))")
    applicable(Ptr, 1) && push!(p, "Ptr(1) = $(Ptr(1))")
    Cint(apple) == 3 || push!(p, "Cint(apple) = $(Cint(apple))")
    Ref(1)[] == 1 || push!(p, "Ref(1)")
    (isempty("") && !isfile("/no/such/file") && nothing === Core.nothing) || push!(p, "isempty, isfile or nothing")
    hasmethod(Base.unsafe_convert, Tuple{Int}) && push!(p, "Base.unsafe_convert(::Int) exists")
end

println("PROBE_DONE")
