# Subprocess probe for test_factory_ctor_collisions_live.jl: loads the wrapper
# of factory_ctor_test, makes an owning object through each factory, and
# checks the shared signature names its factories. Prints one
# "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_factory_ctors.jl <wrapper_path>

wrapper, = ARGS

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

M = nothing
probe("wrapper_loads") do p
    global M = include(wrapper)
    M isa Module || push!(p, "include returned $(typeof(M))")
end

# Before the fix the second definition of `Animal(::Cint)` replaced the first:
# every Animal had the doubled legs.
probe("each_factory_makes_its_own_animal") do p
    a = M.Animal(Val(:make_animal), Cint(4))
    b = M.Animal(Val(:make_animal_doubled), Cint(4))
    a isa M.Animal || push!(p, "Animal(Val(:make_animal), …) gave $(typeof(a))")
    b isa M.Animal || push!(p, "Animal(Val(:make_animal_doubled), …) gave $(typeof(b))")
    (n = M.legs(a)) == 4 || push!(p, "make_animal: $n legs, not 4")
    (n = M.legs(b)) == 8 || push!(p, "make_animal_doubled: $n legs, not 8")
end

probe("a_signature_one_factory_has_stays_plain") do p
    c = M.Animal(Cint(2), Cint(3))
    (n = M.legs(c)) == 5 || push!(p, "make_animal_pair: $n legs, not 5")
end

probe("the_bare_shared_signature_is_a_method_error") do p
    e = try
        M.Animal(Cint(4)); nothing
    catch err
        err
    end
    e isa MethodError || push!(p, "Animal(Cint(4)) gave $(e === nothing ? "an object" : typeof(e))")
    if e isa MethodError
        msg = sprint(showerror, e)
        for f in ("make_animal", "make_animal_doubled")
            occursin("Val{:$f}", msg) || push!(p, "the MethodError does not list the $f constructor")
        end
    end
    e = try
        M.Zebra(Cint(4)); nothing
    catch err
        err
    end
    e isa MethodError || push!(p, "Zebra(Cint(4)) gave $(e === nothing ? "an object" : typeof(e))")
end

probe("the_early_form_makes_each_zebra") do p
    a = M.Zebra(Val(:make_zebra), Cint(3))
    b = M.Zebra(Val(:make_zebra_striped), Cint(3))
    (n = M.stripes(a)) == 3 || push!(p, "make_zebra: $n stripes, not 3")
    (n = M.stripes(b)) == 9 || push!(p, "make_zebra_striped: $n stripes, not 9")
    # The record that names the class holds one.
    r = Ref(M.setproperties(M.Pen(); z = Ptr{M.Zebra}(b.handle), n = Cint(2)))
    (n = GC.@preserve b r M.pen_stripes(r)) == 18 || push!(p, "pen_stripes $n, not 18")
end

@noinline function churn(n)
    for i in 1:n
        M.Animal(Val(:make_animal), Cint(i)); M.Animal(Val(:make_animal_doubled), Cint(i))
        M.Animal(Cint(i), Cint(1))
        M.Zebra(Val(:make_zebra), Cint(i)); M.Zebra(Val(:make_zebra_striped), Cint(i))
    end
    return nothing
end
probe("finalizers_free_what_the_factories_made") do p
    for _ in 1:4; GC.gc(true); end
    a0, z0 = M.animals_alive(), M.zebras_alive()
    churn(25)
    for _ in 1:4; GC.gc(true); end
    yield()
    # 75 Animals and 50 Zebras were made; most of them are freed.
    (a = M.animals_alive()) < a0 + 75 || push!(p, "no Animal freed ($a0 -> $a)")
    (z = M.zebras_alive()) < z0 + 50 || push!(p, "no Zebra freed ($z0 -> $z)")
end

println("PROBE_DONE")
