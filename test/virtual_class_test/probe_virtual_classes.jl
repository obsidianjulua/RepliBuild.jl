# Subprocess probe for test_virtual_classes_live.jl: builds objects of classes
# with a virtual base in place through the wrapper's constructors, ends them
# through its destructors, and calls virtual methods of classes that add no
# data member through pointers to their subclasses. Bound to the base-object
# constructor, the first group segfaults, which is why this is a child. Prints
# one "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_virtual_classes.jl <wrapper_path>

wrapper, = ARGS
const M = include(wrapper)

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
arities(f) = sort!([m.nargs - 1 for m in methods(f)])

# ── Constructors and destructors of classes with a virtual base ─────────────

# `Both b;` then `b.~Both()`: the complete-object pair builds and ends the
# virtual base, which the live counter sees.
probe("both_constructed_in_place") do p
    n = M.live_roots()
    b = alloc(:Both)
    M.Both_Both(b)
    M.live_roots() == n + 1 || push!(p, "live_roots $(M.live_roots()) after construction, was $n")
    (got = M.Both_sum(b)) == 9 + 1 + 2 + 3 || push!(p, "sum = $got")
    (got = M.Both_who(b)) == 3 || push!(p, "who = $got")
    # The virtual base is where the object's own vtable says it is.
    (got = M.Root_who(M.Both_as_Root(b))) == 3 || push!(p, "who through Root = $got")
    M.Both_destroy_Both(b)
    M.live_roots() == n || push!(p, "live_roots $(M.live_roots()) after destruction, was $n")
    Libc.free(b)
end

# A constructor with parameters: each lands in its own argument.
probe("both_constructed_with_arguments") do p
    n = M.live_roots()
    b = alloc(:Both)
    M.Both_Both(b, 40, 50)
    (got = M.Both_sum(b)) == 9 + 40 + 2 + 50 || push!(p, "sum = $got")
    M.Both_destroy_Both(b)
    M.live_roots() == n || push!(p, "live_roots $(M.live_roots()), was $n")
    Libc.free(b)
end

probe("left_and_right_constructed_in_place") do p
    n = M.live_roots()
    l = alloc(:Left)
    M.Left_Left(l)
    (got = M.Left_who(l)) == 1 || push!(p, "Left who = $got")
    (got = M.Root_who(M.Left_as_Root(l))) == 1 || push!(p, "Left who through Root = $got")
    l7 = alloc(:Left)
    M.Left_Left(l7, 7)
    (got = unsafe_load(Ptr{Cint}(l7 + M.member_offset(:Left, :l)))) == 7 || push!(p, "Left(7).l = $got")
    r = alloc(:Right)
    M.Right_Right(r)
    (got = M.Root_who(M.Right_as_Root(r))) == 0 || push!(p, "Right who through Root = $got")
    M.live_roots() == n + 3 || push!(p, "live_roots $(M.live_roots()) with three objects, was $n")
    M.Left_destroy_Left(l); M.Left_destroy_Left(l7); M.Right_destroy_Right(r)
    M.live_roots() == n || push!(p, "live_roots $(M.live_roots()) after destruction, was $n")
    foreach(Libc.free, (l, l7, r))
end

# The deleting destructor takes the object and nothing else, and frees it.
probe("delete_takes_the_object_alone") do p
    n = M.live_roots()
    b = M.spawn_both()
    (got = M.Both_sum(b)) == 9 + 4 + 2 + 5 || push!(p, "spawned sum = $got")
    M.Both_delete_Both(b)
    M.live_roots() == n || push!(p, "live_roots $(M.live_roots()) after delete, was $n")
end

probe("no_wrapper_takes_a_vtt") do p
    for f in (M.Both_Both, M.Left_Left, M.Right_Right, M.Solid_Solid)
        all(m -> !(:vtt in Base.method_argnames(m)), methods(f)) || push!(p, "$f has a vtt parameter")
    end
    arities(M.Both_Both) == [1, 3] || push!(p, "Both_Both arities $(arities(M.Both_Both))")
    arities(M.Left_Left) == [1, 2] || push!(p, "Left_Left arities $(arities(M.Left_Left))")
    for f in (M.Both_destroy_Both, M.Both_delete_Both, M.Left_destroy_Left, M.Left_delete_Left,
              M.Right_destroy_Right, M.Right_delete_Right, M.Root_destroy_Root)
        arities(f) == [1] || push!(p, "$f arities $(arities(f))")
    end
end

# A subclass of an abstract class with a virtual base is built whole.
probe("solid_is_built_through_its_abstract_base") do p
    n = M.live_roots()
    s = alloc(:Solid)
    M.Solid_Solid(s)
    (got = M.Solid_area(s)) == 42 || push!(p, "area = $got")
    M.Solid_destroy_Solid(s)
    M.live_roots() == n || push!(p, "live_roots $(M.live_roots()), was $n")
    Libc.free(s)
end

# The abstract class itself has only a base-object constructor: refused.
probe("abstract_with_a_virtual_base_is_refused") do p
    s = alloc(:Sketch)
    try
        M.Sketch_Sketch(s)
        push!(p, "Sketch_Sketch returned")
    catch e
        msg = sprint(showerror, e)
        e isa ErrorException || push!(p, "threw $(typeof(e))")
        occursin("only the base-object constructor of `Sketch`", msg) || push!(p, "message: $(first(msg, 160))")
    end
    Libc.free(s)
end

# No virtual base: the base-object constructor is the constructor, and it is
# refused all the same. The class is abstract, and an object built by it alone
# would have `__cxa_pure_virtual` where `get` is (test_pure_virtual_live.jl).
probe("abstract_without_a_virtual_base_is_refused") do p
    q = alloc(:Plain)
    try
        M.Plain_Plain(q, 31)
        push!(p, "Plain_Plain returned")
    catch e
        msg = sprint(showerror, e)
        e isa ErrorException || push!(p, "threw $(typeof(e))")
        occursin("`Plain` is an abstract class", msg) || push!(p, "message: $(first(msg, 160))")
    end
    (got = unsafe_load(Ptr{Cint}(q + M.member_offset(:Plain, :v)))) == 0 || push!(p, "v = $got: the storage was written")
    Libc.free(q)
end

# ── Virtual dispatch ────────────────────────────────────────────────────────

# kind: 0 Dog, 1 Puppy, 2 Bird, 3 Duck, 4 Mallard
with_animal(f, kind) = (a = M.spawn_animal(kind); try f(a) finally M.drop_animal(a) end)

# Dog adds no data member; a Puppy reached through Dog's wrappers is a Puppy.
probe("memberless_class_dispatches") do p
    with_animal(1) do a
        (got = M.Dog_name(a)) == "puppy" || push!(p, "Dog_name(puppy) = $got")
        (got = M.Dog_speed(a, 2.0)) == 6.0 || push!(p, "Dog_speed(puppy, 2) = $got")
        (got = M.Animal_name(a)) == "puppy" || push!(p, "Animal_name(puppy) = $got")
        (got = M.Animal_speed(a, 2.0)) == 6.0 || push!(p, "Animal_speed(puppy, 2) = $got")
        (got = M.Puppy_name(a)) == "puppy" || push!(p, "Puppy_name(puppy) = $got")
    end
    with_animal(0) do a
        (got = M.Dog_name(a)) == "dog" || push!(p, "Dog_name(dog) = $got")
        (got = M.Dog_speed(a, 2.0)) == 20.0 || push!(p, "Dog_speed(dog, 2) = $got")
    end
end

# Duck adds none either and has two bases; a Mallard comes back as an Animal*.
probe("memberless_class_with_two_bases_dispatches") do p
    with_animal(4) do a
        (got = M.Duck_name(a)) == "mallard" || push!(p, "Duck_name(mallard) = $got")
        (got = M.Bird_name(a)) == "mallard" || push!(p, "Bird_name(mallard) = $got")
        # Duck::depth overrides its SECOND base's method, in Duck's own vtable.
        (got = M.Duck_depth(a)) == 50 || push!(p, "Duck_depth(mallard) = $got")
        (got = M.Swimmer_depth(M.Duck_as_Swimmer(a))) == 50 || push!(p, "Swimmer_depth(mallard) = $got")
    end
    with_animal(3) do a
        (got = M.Duck_name(a)) == "duck" || push!(p, "Duck_name(duck) = $got")
        (got = M.Duck_depth(a)) == 5 || push!(p, "Duck_depth(duck) = $got")
    end
end

# Bird has a data member; the unit that defines `Bird::speed` only declares
# the class. Both of Bird's virtual methods still go through the vtable.
probe("class_declared_in_a_later_unit_dispatches") do p
    with_animal(4) do a
        (got = M.Bird_speed(a, 2.0)) == 14.0 || push!(p, "Bird_speed(mallard, 2) = $got")
        (got = M.Bird_name(a)) == "mallard" || push!(p, "Bird_name(mallard) = $got")
        (got = M.Animal_speed(a, 2.0)) == 14.0 || push!(p, "Animal_speed(mallard, 2) = $got")
    end
    with_animal(2) do a
        (got = M.Bird_speed(a, 2.0)) == 61.5 || push!(p, "Bird_speed(bird, 2) = $got")
        (got = M.Bird_name(a)) == "bird" || push!(p, "Bird_name(bird) = $got")
    end
end

# `noexcept`, so a plain ccall could carry it: it must not.
probe("noexcept_virtual_dispatches") do p
    with_animal(1) do a
        (got = M.Animal_rank(a)) == 3 || push!(p, "Animal_rank(puppy) = $got")
        (got = M.Dog_rank(a)) == 3 || push!(p, "Dog_rank(puppy) = $got")
    end
    with_animal(0) do a
        (got = M.Animal_rank(a)) == 2 || push!(p, "Animal_rank(dog) = $got")
    end
    with_animal(2) do a
        (got = M.Animal_rank(a)) == 1 || push!(p, "Animal_rank(bird) = $got")
    end
end

println("PROBE_DONE")
