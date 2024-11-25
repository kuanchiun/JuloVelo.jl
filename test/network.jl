@testset "network" begin
    ngenes, ncells = (50, 10)

    batched_c = rand(Float32, ngenes, ncells)
    batched_u = rand(Float32, ngenes, ncells)
    batched_s = rand(Float32, ngenes, ncells)

    @testset "SpatialDense" begin
        layer_cpu = SpatialDense(ncells * 2 => 10, ngenes)
        layer_gpu = SpatialDense(ncells * 2 => 10, ngenes) |> gpu
        x_cpu = reshape(vcat(batched_u, batched_s), ncells * 2, 1, ngenes)
        x_gpu = x_cpu |> gpu
        y_cpu = layer_cpu(x_cpu)
        y_gpu = layer_gpu(x_gpu)
        @test size(y_cpu) == (10, 1, ngenes)
        @test size(y_gpu) == (10, 1, ngenes)

        layer_cpu = SpatialDense(2 => 10, ngenes)
        layer_gpu = SpatialDense(2 => 10, ngenes) |> gpu
        x_cpu = vcat(permutedims(reshape(batched_u', ncells, 1, ngenes), (2, 1, 3)), permutedims(reshape(batched_s', ncells, 1, ngenes), (2, 1, 3)))
        x_gpu = x_cpu |> gpu
        y_cpu = layer_cpu(x_cpu)
        y_gpu = layer_gpu(x_gpu)
        @test size(y_cpu) == (10, ncells, ngenes)
        @test size(y_gpu) == (10, ncells, ngenes)
    end

    @testset "Load_network" begin
        BSON.@load "../models/Circle.bson" circle 
        BSON.@load "../models/Induction.bson" Induction
        BSON.@load "../models/Repression.bson" Repression
        @test circle isa Flux.Chain
        @test Induction isa Flux.Chain
        @test Repression isa Flux.Chain
    end
end