include("FastTrace.jl")
include("Utils.jl")
using BenchmarkTools
#=
G = loadgraph("q0t5.lgz")
A = adjacency_matrix(G, Float64)
=#
#data = readdlm("./P1a.tsv", Int64, header=false) 
#g = realDataHelper("skitter.csv")
#A = adjsp(g)
expList = "expList.txt"
output = "realExpResult.txt"

function experimentRealNetHutchLap(expList, output)
    expListFile = open(expList)
    outputFile = open(output, "a")
    redirect_stdout(outputFile) do
        while !eof(expListFile)
            str = "./datalgz/" * readline(expListFile)
            g = loadgraph(str)
            Adj = adjacency_matrix(g, Float64)
            println(str)
            println("========HutchLap 1=======")
            res = @btime hutch($Adj, 1)
            println("result: ", res)
            println("========HutchLap 5=======")
            res = @btime hutch($Adj, 5)
            println("result: ", res)
            println("========HutchLap 10=======")
            res = @btime hutch($Adj, 10)
            println("result: ", res)
            println("\n")
        end
    end
    close(expListFile)
    close(outputFile)
    return
end

function experimentRealNetHutchIter(expList, output)
    expListFile = open(expList)
    outputFile = open(output, "a")
    redirect_stdout(outputFile) do
        println("==============================")
        while !eof(expListFile)
            str = "./datalgz/" * readline(expListFile)
            g = loadgraph(str)
            println(str)
            println("========HutchIter 1 l 10000=======")
            res = @btime hutchIter($g, 10000, 1)
            println("result: ", res)
            println("========HutchIter 5 l 10000=======")
            res = @btime hutchIter($g, 10000, 5)
            println("result: ", res)
            println("========HutchIter 10 l 10000=======")
            res = @btime hutchIter($g, 10000, 10)
            println("result: ", res)
            println("\n")
        end
    end
    close(expListFile)
    close(outputFile)
    return
end


function experimentRealNetHutchWalk(expList, output)
    expListFile = open(expList)
    outputFile = open(output, "a")
    redirect_stdout(outputFile) do
        println("==============================")
        while !eof(expListFile)
            str = "./datalgz/" * readline(expListFile)
            g = loadgraph(str)
            println(str)
            println("========HutchWalk 1 l 10000=======")
            res = @btime hutchWalkEmpirical($g, 10000, 1)
            println("result: ", res)
            println("========HutchWalk 5 l 10000=======")
            res = @btime hutchWalkEmpirical($g, 10000, 5)
            println("result: ", res)
            println("========HutchWalk 10 l 10000=======")
            res = @btime hutchWalkEmpirical($g, 10000, 10)
            println("result: ", res)
            println("\n")
        end
    end
    close(expListFile)
    close(outputFile)
    return
end

function experimentExact(exactList, output)
    expListFile = open(expList)
    outputFile = open(output, "a")
    redirect_stdout(outputFile) do
        println("==============================")
        while !eof(expListFile)
            str = "./datalgz/" * readline(expListFile)
            g = loadgraph(str)
            println(str)
            println("========HutchWalk 1 l 10000=======")
            res = @btime hutchWalkEmpirical($g, 10000, 1)
            println("result: ", res)
            println("========HutchWalk 5 l 10000=======")
            res = @btime hutchWalkEmpirical($g, 10000, 5)
            println("result: ", res)
            println("========HutchWalk 10 l 10000=======")
            res = @btime hutchWalkEmpirical($g, 10000, 10)
            println("result: ", res)
            println("\n")
        end
    end
    close(expListFile)
    close(outputFile)
    return
end


