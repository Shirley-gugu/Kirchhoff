using Laplacians
using SparseArrays
using LinearAlgebra
using Random

function load_csc(path)
    open(path, "r") do io
        n = Int(read(io, Int64))
        nz = Int(read(io, Int64))
        colptr = Vector{Int64}(undef, n + 1)
        read!(io, colptr)
        colptr .+= 1
        row32 = Vector{Int32}(undef, nz)
        read!(io, row32)
        rowval = Int64.(row32) .+ 1
        A = SparseMatrixCSC(n, n, colptr, rowval, ones(Float64, nz))
        return A
    end
end

function probe(rng, n)
    q = rand(rng, [-1.0, 1.0], n)
    q .-= sum(q) / n
    return q
end

function hutchlap(A, p, seed)
    Random.seed!(seed)
    rng = MersenneTwister(seed)
    n = size(A, 1)
    solve = approxchol_lap(A)
    total = 0.0
    for _ in 1:p
        q = probe(rng, n)
        total += dot(q, solve(q))
    end
    return n * total / p
end

function hutchmv(A, d, p, ell, seed)
    rng = MersenneTwister(seed)
    n = size(A, 1)
    total = 0.0
    tmp = zeros(n)
    for _ in 1:p
        q = probe(rng, n)
        y = q ./ d
        acc = 0.0
        for k in 1:ell
            acc += dot(q, y)
            if k < ell
                mul!(tmp, A, y)
                @inbounds @simd for i in eachindex(y)
                    y[i] = 0.5 * (y[i] + tmp[i] / d[i])
                end
            end
        end
        total += 0.5 * acc
    end
    return n * total / p
end

function hutchwalk(A, d, p, ell, r, seed)
    probe_rng = MersenneTwister(seed)
    walk_rng = MersenneTwister(seed + 10_000_000)
    n = size(A, 1)
    colptr, rowval = A.colptr, A.rowval
    deg = diff(colptr)
    total = 0.0
    for _ in 1:p
        q = probe(probe_rng, n)
        f = q ./ d
        acc = 0.0
        for _ in 1:r
            s0 = rand(walk_rng, 1:n)
            s = s0
            pathsum = f[s]
            for _ in 2:ell
                if rand(walk_rng, Bool)
                    s = rowval[colptr[s] + rand(walk_rng, 0:(deg[s]-1))]
                end
                pathsum += f[s]
            end
            acc += q[s0] * pathsum
        end
        total += acc
    end
    return n^2 * total / (2 * p * r)
end

function measure(f, name)
    start = time_ns()
    value = f()
    elapsed = (time_ns() - start) / 1e9
    println(name, '\t', value, '\t', elapsed)
end

function main()
    path = ARGS[1]
    A = load_csc(path)
    n = size(A, 1)
    d = Float64.(diff(A.colptr))
    @assert nnz(A) == 2 * div(nnz(A), 2)
    println("GRAPH\t", n, '\t', nnz(A) ÷ 2)
    toy = sparse([1, 2, 2, 3], [2, 1, 3, 2], ones(4), 3, 3)
    td = Float64.(diff(toy.colptr))
    hutchlap(toy, 1, 1)
    hutchmv(toy, td, 1, 2, 1)
    hutchwalk(toy, td, 1, 2, 10, 1)
    measure("HutchLap") do
        hutchlap(A, 1, 101)
    end
    measure("HutchMV") do
        hutchmv(A, d, 1, 100, 101)
    end
    measure("HutchWalk") do
        hutchwalk(A, d, 1, 100, 10000, 101)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
