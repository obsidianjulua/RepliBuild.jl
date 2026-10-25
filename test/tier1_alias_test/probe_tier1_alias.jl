# Subprocess probe for test_tier1_alias_binding.jl: loads a decoy library that
# defines every public name of the fixture RTLD_GLOBAL, then the wrapper, and
# calls each Tier 1 function. Prints one "PROBE <name>: PASS|FAIL …" line per
# case, then PROBE_DONE. The failure being guarded is a slice calling another
# library's copy of a function: with real zlib that is heap corruption and a
# process stuck in its own crash reporter, so this runs in a child.
# Run: julia --project=<RepliBuild> probe_tier1_alias.jl <wrapper> <decoy library>

using Libdl
using InteractiveUtils: code_llvm

const WRAPPER, DECOY = ARGS[1], ARGS[2]

# The decoy first, RTLD_GLOBAL: every name the fixture exports is now defined
# earlier in the process, the way Julia's own libz.so.1 defines zlib's.
const DECOY_HANDLE = Libdl.dlopen(DECOY, Libdl.RTLD_NOW | Libdl.RTLD_GLOBAL)

include(WRAPPER)
using .Zcollide
const Z = Zcollide

probe(name, ok::Bool, detail) = println("PROBE $name: ", ok ? "PASS" : "FAIL $detail")

# Premise: the process already holds another definition of each public name.
# On ELF that is what `dlsym(RTLD_DEFAULT, …)` answers, and the fixture's copy
# is not it. Windows has no process-wide namespace, and nothing to collide in;
# the decoy's copy stands in for "another definition exists".
const LIB_HANDLE = Z.LIB_HANDLE[]
own(sym) = Libdl.dlsym(LIB_HANDLE, sym; throw_error = false)
elsewhere(sym) = Sys.iswindows() ?
    Libdl.dlsym(DECOY_HANDLE, sym; throw_error = false) :
    ccall(:dlsym, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), C_NULL, String(sym))
for sym in (:gzclose_w, :gzclose, :adler32_z, :z_verbose, :z_errmsg)
    o, e = own(sym), elsewhere(sym)
    probe("collision_$sym", o !== nothing && e != C_NULL && e != o,
          "library=$o process=$e")
end

# Every function of the fixture takes Tier 1 here.
const TIER1 = ("gzclose_w", "gzclose", "gzclose_r", "adler32_z", "adler32",
               "zc_verbosity", "zc_set_verbosity", "zc_table_at", "zc_set_seed")
probe("tier1_surface", Set(TIER1) == Z.TIER1_FUNCTIONS,
      "TIER1_FUNCTIONS = $(sort!(collect(Z.TIER1_FUNCTIONS)))")
tiers = Dict(f => Z.dispatch_tier(Symbol(f)) for f in TIER1)
probe("dispatch_tier", all(t -> t === :tier1, values(tiers)), repr(tiers))

# Each slice binds the library's own symbols by alias, and nothing by a public
# name. These are the declares of the slices that ship with the wrapper.
want = Dict("gzclose_w"        => ["__rb_zcollide_zc_seed"],
            "adler32_z"        => ["__rb_zcollide_zc_seed"],
            "zc_set_seed"      => ["__rb_zcollide_zc_seed"],
            "gzclose"          => ["__rb_zcollide_gzclose_w"],
            "adler32"          => ["__rb_zcollide_adler32_z"],
            "zc_verbosity"     => ["__rb_zcollide_z_verbose"],
            "zc_set_verbosity" => ["__rb_zcollide_z_verbose"],
            "zc_table_at"      => ["__rb_zcollide_z_errmsg"])
for (f, syms) in want
    probe("declares_$f", get(Z.TIER1_DECLARES, f, nothing) == syms,
          "TIER1_DECLARES[$f] = $(repr(get(Z.TIER1_DECLARES, f, nothing)))")
end
# gzclose_r reaches gzclose, or gzclose_w once gzclose is inlined into it;
# either way only the library's alias names.
r_decl = get(Z.TIER1_DECLARES, "gzclose_r", String[])
probe("declares_gzclose_r",
      !isempty(r_decl) && all(s -> s in ("__rb_zcollide_gzclose", "__rb_zcollide_gzclose_w"), r_decl),
      repr(r_decl))
stray = [(f, s) for (f, syms) in Z.TIER1_DECLARES for s in syms if !startswith(s, "__rb_zcollide_")]
probe("no_public_declare", isempty(stray), repr(stray))

# The values: the fixture's sentinels, not the decoy's and not zlib's.
# zlib's gzclose_w(NULL) is Z_STREAM_ERROR (-2), the decoy's -1000.
v = Z.gzclose(C_NULL);     probe("gzclose", v == 4243, "got $v")
v = Z.gzclose_r(C_NULL);   probe("gzclose_r", v == 4343, "got $v")
v = Z.adler32(1, C_NULL, 2); probe("adler32", v == 1 + 2 + 4242 + 1, "got $v")
t = [Z.zc_table_at(i) for i in 0:3]
probe("table", t == [11, 13, 17, 19], "got $t")
Z.zc_set_verbosity(5)
seen = Z.zc_verbosity()
own_state = unsafe_load(Ptr{Cint}(own(:z_verbose)))
decoy_state = unsafe_load(Ptr{Cint}(Libdl.dlsym(DECOY_HANDLE, :z_verbose)))
probe("state", seen == 5 && own_state == 5 && decoy_state == -70,
      "read $seen, library's z_verbose $own_state, decoy's $decoy_state")

# The calls above were the slices, not the ccall bodies: the kernel's native IR
# calls the alias. (`gzclose` is spliced, so its callee is visible here.)
llvm = sprint(io -> code_llvm(io, Z._TIER1_gzclose, (Ptr{Cvoid},); debuginfo = :none))
probe("spliced", occursin("__rb_zcollide_gzclose_w", llvm) && !occursin("@gzclose_w", llvm),
      first(llvm, 600))

println("PROBE_DONE")
