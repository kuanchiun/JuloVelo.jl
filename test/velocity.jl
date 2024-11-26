@testset "velocity" begin
    ngenes, ncells = (1000, 2000)

    X = rand(Float32, 2, ncells, ngenes)
    embedding = rand(Float32, ncells, 2)
    velocity = rand(Float32, 2, ncells, ngenes)

    spliced_matrix = X[2, :, :]'
    velocity_spliced_matrix = velocity[2, :, :]'
    velocity_spliced_matrix = sqrt.(abs.(velocity_spliced_matrix) .+ 1) ./ sign.(velocity_spliced_matrix)

    neighbor_graph = get_neighbor_graph(embedding, 100)

    @testset "gpu" begin
        if CUDA.functional()
            CUDA.allowscalar(false)
            velocity_embedding = velocity_projection(spliced_matrix, velocity_spliced_matrix, neighbor_graph, embedding; use_gpu = true)
            @test size(velocity_embedding) == (ncells, 2)
        end
    end

    @testset "cpu" begin
        velocity_embedding = velocity_projection(spliced_matrix, velocity_spliced_matrix, neighbor_graph, embedding; use_gpu = false)
        @test size(velocity_embedding) == (ncells, 2)
    end
end

