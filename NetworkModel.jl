using Graphs
using LinearAlgebra
using SparseArrays

function identityDegree(t::Int, q::Int)
    n :: Int64 = (4^t + 2) * 2 / 3
    g = SimpleGraph(n)
    add_edge!(g, 1, 2)
    cur_n :: Int64 = 2
    for i = 1:t
        cur_edges = deepcopy(edges(g))
        for e in cur_edges
            #println(t, " ", e)
            if q == 0
                add_edge!(g, e.src, cur_n+1)
                add_edge!(g, e.src, cur_n+2)
                add_edge!(g, e.dst, cur_n+1)
                add_edge!(g, e.dst, cur_n+2)
                rem_edge!(g, e.src, e.dst)
            elseif q == 1
                add_edge!(g, e.src, cur_n+1)
                add_edge!(g, cur_n+1, cur_n+2)
                add_edge!(g, e.dst, cur_n+2)
            end
            cur_n += 2
        end
    end
    return g    
end

function simplicialNet(t::Int, q::Int)
    n :: Int64 = 2 * (((q+1)*(q+2)/2)^(t+1) + q + 2) / (q + 3)
    g = SimpleGraph(n)
    cur_n = q + 2
    for i = 1 : q + 1 
        for j = i + 1 : q + 2 
            add_edge!(g, i, j)
        end
    end
    for i = 1:t
        cur_edges = deepcopy(edges(g))
        for e in cur_edges
            for j = cur_n + 1 : cur_n + q - 1 
                for k = j + 1 : cur_n + q 
                    add_edge!(g, j, k)
                end
            end
            for j = cur_n + 1 : cur_n + q 
                add_edge!(g, e.src, j)
                add_edge!(g, e.dst, j)
            end
            cur_n += q
        end
    end
    return g    
end




