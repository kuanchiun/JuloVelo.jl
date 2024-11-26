using JuloVelo
using Test
using BSON
using Flux
using CUDA

tests = [
    "network",
    "velocity"
]

@testset "JuloVelo.jl" begin
    for t in tests
        include("$(t).jl")
    end
end
