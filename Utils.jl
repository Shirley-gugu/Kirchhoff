using Graphs
#using LinearAlgebra
using SparseArrays, Krylov, LinearOperators
using CUDA.CUSPARSE
using CUDA

include("NetworkModel.jl")
include("graph.jl")

# Use the simple walk on non-bipartite graphs and the 1/2-lazy walk on
# bipartite graphs. The explicit two-coloring keeps this choice independent of
# version-specific graph-library helpers and costs O(N + M).
function adaptive_walk_alpha(g)
    n = nv(g)
    color = fill(Int8(0), n)
    queue = Vector{Int}(undef, n)
    for root in vertices(g)
        color[root] != 0 && continue
        head = 1
        tail = 1
        queue[tail] = root
        color[root] = 1
        while head <= tail
            u = queue[head]
            head += 1
            for v in neighbors(g, u)
                if color[v] == 0
                    color[v] = -color[u]
                    tail += 1
                    queue[tail] = v
                elseif color[v] == color[u]
                    return 1.0
                end
            end
        end
    end
    return 0.5
end

function solverVecsWrap(solver::Function, X)
    shape = size(X)
    res = solver(X[:, 1])
    for i = 2 : shape[2]
        res = [res solver(X[:, i])]
    end
    return res
end

function network2MatrixWrap(t::Int64)
    G = identityDegree(t, 0)
    n = nv(G)
    A = adjacency_matrix(G, Float64)
    G = 0
    return A, n
end

function network2LgzWrap(t::Int64, name::String)
    G = identityDegree(t, 0)
    savegraph(name * ".lgz", G)
end

function simplNetLgzWrap(t::Int64, q::Int64, name::String)
    G = simplicialNet(t, q)
    savegraph(name * ".lgz", G)
end

function realDataLgzWrap(name)
    G = realDataHelperJL(name)
    savegraph(name * ".lgz", G)
end

function basicPropagation(P::SparseMatrixCSC, Di::SparseMatrixCSC, x, n::Int, m::Int, l::Int, alpha::Float64 = 1.0)
    xs = sum(x)
    cur = Di * x
    res = Di * x
    for i = 1 : l
        cur = (1 - alpha) .* cur .+ alpha .* (P * cur)
        res += cur
    end
    return alpha * (x' * res - (l + 1) * xs^2 / (2 * m))
end


function cuIter(P::SparseMatrixCSC, x, f, l, alpha::Float64)
    P_d = CuSparseMatrixCSR{Float32}(P)
    f_d = CuArray{Float32}(f)
    res_d = CuArray{Float32}(f)
    alpha_d = Float32(alpha)
    for i = 1 : l
        next_d = P_d * f_d
        CUDA.@sync f_d .= (1f0 - alpha_d) .* f_d .+ alpha_d .* next_d
        CUDA.@sync res_d .+= f_d
    end
    return alpha * (x' * Array(res_d))
end

function exactResult(P::SparseMatrixCSC, x, l::Int) 
    res = x 
    for i = 1 : l + 2
        x = P * x 
        res += x
    end
    return res
end

function cuResult(P::SparseMatrixCSC, x, l::Int)
    P_d = CuSparseMatrixCSC(P)
    x_d = CuArray{Float32}(x)
    res_d = CuArray{Float32}(x)
    for i = 1 : l + 2
        CUDA.@sync x_d = P_d * x_d
        CUDA.@sync res_d .+= x_d
    end
    return Array(res_d)
end

function realDataHelper(name::String) 
    data, _ = readdlm(name, ',', Int64, header=true)
    shape = size(data)
    m = shape[1]
    n = 0
    Label = Dict{Int32, Int32}()
    Origin = Dict{Int32, Int32}()

    getID(x :: Int) = haskey(Label, x) ? Label[x] : Label[x] = n += 1
    
    u = Int[]
    v = Int[]

    tot = 0
    for i = 1 : m
        x = data[i, 1]
        y = data[i, 2]
        if x!=y
            u1 = getID(x)
            v1 = getID(y)
            Origin[u1] = x
            Origin[v1] = y
            push!(u, u1)
            push!(v, v1)
            tot += 1
        end
    end
    nbr=[ [ ] for i in 1:n ]
    for i=1:tot
        u1=u[i];
        v1=v[i];
        push!(nbr[u1],v1);
        push!(nbr[v1],u1);
    end
    return findconnect(Graph(n, tot, u, v,nbr))
end

function realDataHelperJL(fileName)
    Label = Dict{Int32, Int32}()
    Origin = Dict{Int32, Int32}()
    n = 0
    getID(x :: Int) = haskey(Label, x) ? Label[x] : Label[x] = n += 1
    g = SimpleGraph(0)
    open(fileName) do io
        str = readline(io)
       str = readline(io)
        while !eof(io)
            str = readline(io)
            str = split(str, ' ')
            #print(str)
            x = parse(Int, str[1])
            y = parse(Int, str[2])
            if x!=y
                u = getID(x)
                v = getID(y)
                count = max(u, v)
                if (count > nv(g))
                    add_vertices!(g, count - nv(g))
                end
                if (!add_edge!(g, u, v) && !has_edge(g, u, v))
                    println("Error: add edge (", u, ",", v, ") failed")
                end
            end
        end
    end
    components = connected_components(g)
    lcc_v = components[argmax([length(c) for c in components])]
    lcc, _ = induced_subgraph(g, lcc_v)
    return lcc
end


function walk_kernel(start_nodes, indptr, indices, hitting, sum_hitting, f, x, r, l, alpha)
    thread_id = (blockIdx().x - UInt32(1)) * blockDim().x + threadIdx().x 
    stride = blockDim().x * gridDim().x
    hitting_tmp::Float32 = 0
    walk_id = thread_id
    while walk_id <= r
        curr_node = start_nodes[walk_id]
        hitting[walk_id] = f[curr_node]

        for j in 1 : l
            if rand() < alpha
                start_idx = indptr[curr_node]
                end_idx = indptr[curr_node + 1]
                rand_float = rand()
                choice::Int32 = floor(Int32, rand_float * (end_idx - start_idx))
                curr_node = indices[start_idx + choice]
            end
            hitting[walk_id] += f[curr_node]
        end

        hitting[walk_id] *= x[start_nodes[walk_id]]
        hitting_tmp += hitting[walk_id]
        walk_id += stride
    end
    if hitting_tmp != 0
        CUDA.atomic_add!(pointer(sum_hitting), hitting_tmp)
    end
    return 
end

    
     



