include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const MS = "std::chrono::milliseconds"
const Ms = M.duration_long_std_ratio_1L_1000L
case(MS, "make") do
    works(M.ms_count(Ms(250)), 250, "the struct's one field, the count")
end
case(MS, "pass by value") do
    works(M.ms_count(M.ms_make(1500)), 1500)
end
case(MS, "pass by reference") do
    works(M.ms_count_ref(Ref(Ms(7))), 7)
end
case(MS, "return by value") do
    works(M.ms_make(1500)._r, 1500, "read from the field `_r`")
end

const DS = "std::chrono::duration<double>"
case(DS, "make") do
    works(M.secs_of(M.duration_double_std_ratio_1L_1L(1.5)), 1.5)
end

const TP = "std::chrono::system_clock::time_point"
case(TP, "return by value") do
    works(M.tp_epoch_plus(10)._d._r, 10_000_000_000, "nanoseconds, two fields deep")
end
case(TP, "pass by value") do
    works(M.tp_seconds(M.tp_epoch_plus(10)), 10)
end
case(TP, "make") do
    T = typeof(M.tp_epoch_plus(0))
    works(M.tp_seconds(T(M.duration_long_std_ratio_1L_1000000000L(3_000_000_000))), 3, "struct of a struct of the count")
end
done()
