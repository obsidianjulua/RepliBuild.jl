include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const MI = "std::map<int, int>"
case(MI, "make") do
    m = M.create_std_map_int_int()
    m[1] = 10; m[7] = 70
    works((length(m), m[7], haskey(m, 1), haskey(m, 2)), (2, Cint(70), true, false), "create_std_map_int_int(), setindex!, getindex, haskey")
end
case(MI, "takes a Julia Dict") do
    takes(10, "a Dict{Cint,Cint}") do
        M.map_get(Dict{Cint,Cint}(1 => 10), 1)
    end
end
case(MI, "pass by reference") do
    m = M.create_std_map_int_int(); m[1] = 10; m[7] = 70
    works((M.map_get(m, 7), M.map_get(m, 3), M.map_last_key(m)), (70, -1, 7), "a CppMap")
end
case(MI, "pass by value") do
    m = M.create_std_map_int_int(); m[1] = 10; m[7] = 70
    works((M.map_size_val(m), M.map_size_val(m)), (2, 2), "a CppMap, passed twice")
end
case(MI, "return by value") do
    r = M.map_make(4)
    opaque((M.map_get(Ref(r), 3), M.map_get(Ref(r), 9), M.map_size_val(r)), (9, -1, 4), "comes back as its bytes, a $(describe(r)); lookups through Ref(r) are right")
end
case(MI, "intact after the return") do
    # The map's end marker is inside the map object, and the tree's root points
    # back at it. The library reads its last key through rbegin().
    r = M.map_make(4)
    intact(M.map_last_key_self(), M.map_last_key(Ref(r)), 3)
end

const MS = "std::map<std::string, int>"
case(MS, "make") do
    m = M.create_std_map_std_string_int()
    m["one"] = 1
    works((length(m), m["one"]), (1, Cint(1)), "create_std_map_std_string_int() with String keys")
end
case(MS, "return by value") do
    r = fn(:smap_make)()
    opaque((M.smap_get(Ref(r), "two"), M.smap_get(Ref(r), "six")), (2, -1), "its bytes; the key is a Julia String")
end

const UM = "std::unordered_map<int, int>"
case(UM, "make") do
    m = M.create_std_unordered_map_int_int()
    m[1] = 10; m[7] = 70
    works((length(m), m[7], M.umap_get(m, 7), M.umap_get(m, 2)), (2, Cint(70), 70, -1), "create_std_unordered_map_int_int()")
end
case(UM, "return by value") do
    # Every key: the bucket of the element the map's own list starts with is
    # reached through a pointer into the map object.
    r = M.umap_make(5)
    opaque(ntuple(k -> M.umap_get(Ref(r), k - 1), 5), (10, 11, 12, 13, 14), "its bytes, a $(describe(r)); all 5 keys were found through Ref(r) on this run")
end

case(UM, "intact after the return") do
    # The bucket of the map's first element points at the list head inside the
    # map object. Lookups above went through that pointer.
    r = M.umap_make(5)
    intact(M.umap_intact_self(), M.umap_intact(Ref(r)), 1)
end

const SI = "std::set<int>"
case(SI, "make") do
    nomake(r"set", "the entry is in [types] templates and gets no constructor")
end
case(SI, "return by value") do
    r = M.set_make(4)
    opaque((M.set_has(Ref(r), 6), M.set_has(Ref(r), 5)), (1, 0), "its bytes")
end
case(SI, "intact after the return") do
    r = M.set_make(4)
    intact(M.set_last_self(), M.set_last(Ref(r)), 6)
end
done()
