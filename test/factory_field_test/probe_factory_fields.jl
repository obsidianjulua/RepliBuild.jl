# Subprocess probe for test_factory_field_order_live.jl: loads the wrapper of
# factory_field_test, builds records that hold pointers to its factory
# classes, and calls through them. Prints one "PROBE <case>: PASS|FAIL …" line
# each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_factory_fields.jl <wrapper_path>

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

# What failed: the module did not load (`UndefVarError: Animal`).
M = nothing
probe("wrapper_loads") do p
    global M = include(wrapper)
    M isa Module || push!(p, "include returned $(typeof(M))")
end
# Without the module, every probe below fails with its reason.

# A raw pointer to the object a factory struct owns.
aptr(x) = Ptr{M.Animal}(x.handle)

probe("fields_keep_the_class_type") do p
    for (rec, f, want) in ((M.Cage, :pet, Ptr{M.Animal}), (M.Den, :pack, Ptr{Ptr{M.Animal}}),
                           (M.Ark, :pets, NTuple{3, Ptr{M.Animal}}), (M.Aviary, :bird, Ptr{M.Bird}),
                           (M.Aviary, :prey, Ptr{M.Animal}))
        (t = fieldtype(rec, f)) === want || push!(p, "$rec.$f is $t, not $want")
    end
end

probe("cage_holds_an_animal") do p
    a = M.Animal(Cint(4))
    c = M.setproperties(M.Cage(); pet = aptr(a), n = Cint(3))
    r = Ref(c)
    (n = GC.@preserve a r M.cage_legs(r)) == 12 || push!(p, "cage_legs $n, not 12")
    # The pointer read back is the object, and a method takes it.
    c.pet == aptr(a) || push!(p, "cage.pet $(c.pet) is not the animal $(aptr(a))")
    (n = GC.@preserve a M.Animal_legs(c.pet)) == 4 || push!(p, "Animal_legs(cage.pet) $n, not 4")
    (n = M.legs(a)) == 4 || push!(p, "legs(a) $n, not 4")
end

probe("den_holds_pointers_to_animals") do p
    a, b = M.Animal(Cint(4)), M.Animal(Cint(2))
    pack = [aptr(a), aptr(b)]
    GC.@preserve a b pack begin
        d = M.setproperties(M.Den(); pack = pointer(pack), count = Cint(2))
        r = Ref(d)
        (n = GC.@preserve r M.den_legs(r)) == 6 || push!(p, "den_legs $n, not 6")
        (n = M.Animal_legs(unsafe_load(d.pack, 2))) == 2 || push!(p, "second of the pack $n, not 2")
    end
end

probe("ark_holds_an_array_of_animals") do p
    a, b = M.Animal(Cint(4)), M.Animal(Cint(8))
    ark = M.Ark((aptr(a), Ptr{M.Animal}(C_NULL), aptr(b)))
    r = Ref(ark)
    (n = GC.@preserve a b r M.ark_legs(r)) == 12 || push!(p, "ark_legs $n, not 12")
    (n = GC.@preserve a b M.Animal_legs(ark.pets[3])) == 8 || push!(p, "pets[3] $n, not 8")
end

probe("a_factory_takes_the_record_that_holds_its_class") do p
    a = M.Animal(Cint(4))
    c = M.setproperties(M.Cage(); pet = aptr(a), n = Cint(3))
    x = GC.@preserve a M.Animal(c)
    x isa M.Animal || push!(p, "Animal(::Cage) gave $(typeof(x))")
    (n = M.legs(x)) == 7 || push!(p, "legs $n, not 7")
end

probe("a_second_factory_class_and_its_record") do p
    prey = M.Animal(Cint(4))
    home = Ref(M.setproperties(M.Aviary(); prey = aptr(prey)))
    bird = GC.@preserve prey home M.Bird(Base.unsafe_convert(Ptr{M.Aviary}, home), Cint(2))
    bird isa M.Bird || push!(p, "Bird(...) gave $(typeof(bird))")
    v = Ref(M.setproperties(M.Aviary(); bird = Ptr{M.Bird}(bird.handle), prey = aptr(prey)))
    # 2 wings plus the prey's 4 legs, and the prey's 4 again.
    (n = GC.@preserve prey bird v M.aviary_sum(v)) == 10 || push!(p, "aviary_sum $n, not 10")
end

# The owning types still free what they made.
@noinline churn(n) = (for i in 1:n; M.Animal(Cint(i)); M.Bird(Ptr{M.Aviary}(C_NULL), Cint(i)); end; nothing)
probe("finalizers_free_what_the_factories_made") do p
    for _ in 1:4; GC.gc(true); end
    a0, b0 = M.animals_alive(), M.birds_alive()
    churn(25)
    for _ in 1:4; GC.gc(true); end
    yield()
    (a = M.animals_alive()) < a0 + 25 || push!(p, "no Animal freed ($a0 -> $a)")
    (b = M.birds_alive()) < b0 + 25 || push!(p, "no Bird freed ($b0 -> $b)")
end

println("PROBE_DONE")
