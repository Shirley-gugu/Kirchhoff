using Laplacians
using SparseArrays
include("Utils.jl")

"""Return a HutchLap estimate of the Kirchhoff index `R(G) = N * tr(L^dagger)`."""
function hutch(A::SparseMatrixCSC{Float64}, t::Int64, ob::Bool = false)
    ans = 0
    f = approxchol_lap(A)
    n = size(A)[1]
    for i = 1:t
        local x = rand([1,-1],n)
        ans += x' * f(x)
        if ob && (i % 5 == 0)
            println("ite:", i, "  Kirchhoff estimate:", n * ans / i)
        end
    end
    return n * ans / t
end

"""Return a HutchMV estimate of the Kirchhoff index, including the factor `N`."""
function hutchIter(g, l, p)
    n = nv(g)
    alpha = adaptive_walk_alpha(g)
    Adj = adjacency_matrix(g, Float64)
    d = degree(g)
    Dinv = spdiagm(1 ./ d)
    P =  Dinv * Adj
    result = 0.0
    for i = 1 : p
        x = rand([1,-1],n)
        q = x .- sum(x) / n
        f = Dinv * q
        result += cuIter(P, q, f, l, alpha)
    end
    return n * result / p
end


"""A computable lower bound on `tr(L^dagger)` used by Theorem 5.9."""
function walk_trace_lower_bound(g)
    n = nv(g)
    n >= 3 || throw(ArgumentError("HutchWalk requires at least three vertices"))
    is_connected(g) || throw(ArgumentError("HutchWalk requires a connected graph"))
    dmax = maximum(degree(g))
    denominator = 2 * ne(g) - 1 - dmax
    denominator > 0 || throw(ArgumentError("invalid trace lower-bound denominator"))
    return 1 / (1 + dmax) + (n - 2)^2 / denominator
end

"""
Return the smallest multiple of `N` satisfying the walk-sampling condition in
Theorem 5.9. This is the number of walks per Hutchinson probe.
"""
function required_walks(g, l, p, epsilon_w, delta; tau0=walk_trace_lower_bound(g))
    n = nv(g)
    l >= 0 || throw(ArgumentError("l must be nonnegative"))
    p >= 1 || throw(ArgumentError("p must be positive"))
    0 < epsilon_w < 1 || throw(ArgumentError("epsilon_w must lie in (0,1)"))
    0 < delta < 1 || throw(ArgumentError("delta must lie in (0,1)"))
    tau0 > 0 || throw(ArgumentError("tau0 must be positive"))
    dmin = minimum(degree(g))
    dmin > 0 || throw(ArgumentError("HutchWalk requires a connected graph"))
    alpha = adaptive_walk_alpha(g)

    raw = (BigFloat(32) * BigFloat(alpha)^2 * BigFloat(n)^2 *
           BigFloat(l + 1)^2 * log(BigFloat(4) / BigFloat(delta))) /
          (BigFloat(p) * BigFloat(epsilon_w)^2 * BigFloat(tau0)^2 *
           BigFloat(dmin)^2)
    raw <= typemax(Int) - n || throw(ArgumentError("the certified walk budget exceeds Int capacity"))
    lower = max(n, ceil(Int, raw))
    return cld(lower, n) * n
end

function _hutch_walk_trace(g, l, p, r; balanced_starts)
    n = nv(g)
    l >= 0 || throw(ArgumentError("l must be nonnegative"))
    p >= 1 || throw(ArgumentError("p must be positive"))
    r >= 1 || throw(ArgumentError("r must be positive"))
    balanced_starts && r % n != 0 &&
        throw(ArgumentError("a balanced start multiset requires r to be a multiple of N"))

    alpha = adaptive_walk_alpha(g)
    start_nodes_d = if balanced_starts
        cu(collect(repeat(1:n, inner=div(r, n))))
    else
        cu(rand(collect(vertices(g)), r))
    end
    d = degree(g)
    volume = 2 * ne(g)
    indices = Array{Int32}(zeros(volume))
    indptr = Array{Int32}(zeros(n + 1))
    curr = 1
    for i in 1:n
        indptr[i] = curr
        neighbor = neighbors(g, i)
        for j in neighbor
            indices[curr] = j
            curr+=1
        end
    end
    indptr[n+1] = curr
    result = 0.0
    indices_d = cu(indices)
    indptr_d = cu(indptr)
    numblocks = ceil(Int, r/768) > 256 ? 256 : ceil(Int, r/768)
    
    hitting_d = cu(Array{Float32}(zeros(r)))
    for i = 1 : p
        sum_hitting_d = cu(Float32[0]) 
        x = rand([-1,1],n)
        q = x .- sum(x) / n
        f = 1 ./ d .* q
        f_d = cu(f)
        q_d = cu(q)
        @cuda threads = 768 blocks=numblocks walk_kernel(start_nodes_d, indptr_d, indices_d, hitting_d, sum_hitting_d, f_d, q_d, r, l, Float32(alpha))
        result += alpha * n / r * Array(sum_hitting_d)[1]
    end
    return result / p
end

"""
The theorem-aligned HutchWalk implementation. `epsilon_w` is the random-walk
error budget and `delta` is the total failure probability used in the paper's
split, so the walk tail is bounded by `delta/2`. The return value is the
Kirchhoff-index estimate, including the final factor `N`.
"""
function hutchWalk(g, l, p; epsilon_w, delta, tau0=walk_trace_lower_bound(g))
    r = required_walks(g, l, p, epsilon_w, delta; tau0=tau0)
    return nv(g) * _hutch_walk_trace(g, l, p, r; balanced_starts=true)
end

"""
Empirical fixed-budget variant used only for the timing tables. It carries no
Theorem 5.9 confidence guarantee. The return value is the Kirchhoff estimate.
"""
function hutchWalkEmpirical(g, l, p)
    n = nv(g)
    r0 = 1000000
    r1 = 100000
    if n > r0
        return n * _hutch_walk_trace(g, l, p, r0; balanced_starts=false)
    elseif n > r1
        return n * _hutch_walk_trace(g, l, p, 3 * n; balanced_starts=true)
    else
        return n * _hutch_walk_trace(g, l, p, 5 * n; balanced_starts=true)
    end
end
