include("three_cpu.jl")

function read_configs(path)
    configs = Tuple{String,Int,Int,Int}[]
    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue
        startswith(line, "#") && continue
        fields = split(line, ',')
        length(fields) == 4 || error("expected method,p,ell,r")
        push!(configs, (String(fields[1]), parse(Int, fields[2]),
                        parse(Int, fields[3]), parse(Int, fields[4])))
    end
    return configs
end

function run_graph(path, configs_path, seed_start, trials)
    A = load_csc(path)
    n = size(A, 1)
    d = Float64.(diff(A.colptr))
    println("# graph\t", path, '\t', n, '\t', nnz(A) ÷ 2)
    toy = sparse([1, 2, 2, 3], [2, 1, 3, 2], ones(4), 3, 3)
    td = Float64.(diff(toy.colptr))
    hutchlap(toy, 1, 1)
    hutchmv(toy, td, 1, 2, 1)
    hutchwalk(toy, td, 1, 2, 10, 1)
    configs = read_configs(configs_path)
    println("method\tp\tell\tr\ttrial\testimate\tseconds")
    flush(stdout)
    for trial in 1:trials
        seed = seed_start + trial - 1
        for index in randperm(MersenneTwister(seed + 999), length(configs))
            method, p, ell, r = configs[index]
            start = time_ns()
            value = try
                if method == "HutchLap"
                    hutchlap(A, p, seed)
                elseif method == "HutchMV"
                    hutchmv(A, d, p, ell, seed)
                elseif method == "HutchWalk"
                    hutchwalk(A, d, p, ell, r, seed)
                else
                    error("unknown method: $method")
                end
            catch exception
                println(stderr, "ERROR\t", method, '\t', p, '\t', ell, '\t', r,
                        '\t', trial, '\t', sprint(showerror, exception))
                NaN
            end
            seconds = (time_ns() - start) / 1e9
            println(method, '\t', p, '\t', ell, '\t', r, '\t', trial,
                    '\t', value, '\t', seconds)
            flush(stdout)
        end
    end
end

run_graph(ARGS[1], ARGS[2], parse(Int, ARGS[3]), parse(Int, ARGS[4]))
