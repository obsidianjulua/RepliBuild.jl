using BenchmarkTools
data = rand(1000)

function bare_loop(d, n)
    acc = 0.0
    for i in 1:n
        acc += d[i]
    end
    acc
end

println("Passed as arg:")
@btime bare_loop($data, 1000)

println("Captured non-const global:")
@btime (() -> begin
    acc = 0.0
    for i in 1:1000
        acc += data[i]
    end
    acc
end)()

const const_data = data
println("Captured const global:")
@btime (() -> begin
    acc = 0.0
    for i in 1:1000
        acc += const_data[i]
    end
    acc
end)()

