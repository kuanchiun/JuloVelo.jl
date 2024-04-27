using JuloVelo
using Test
using BSON
using Flux

tests = [
    "network"
]

@testset "JuloVelo.jl" begin
    for t in tests
        include("$(t).jl")
    end
end
