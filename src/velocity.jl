function velocity_estimation(data::JuloVeloObject; dt::AbstractFloat = 0.5f0)
    ncells = data.ncells
    ngenes = data.train_genes_number
    u = data.train_u'
    s = data.train_s'
    X = data.X
    Kinetic = data.param["velocity_model"]
    
    kinetics = Kinetic(X)
    
    du, ds = kinetic_equation(u, s, kinetics)
    
    # Multiply by dt
    du = du .* dt
    ds = ds .* dt
    
    # Reshape du, ds and create dt_vector
    û = permutedims(reshape(du, ncells, 1, ngenes), (2, 1, 3))
    ŝ = permutedims(reshape(ds, ncells, 1, ngenes), (2, 1, 3))
    X̂ = vcat(û, ŝ) .+ eps(Float32)
    
    data.param["velocity"] = X̂
    data.param["kinetics"] = kinetics
    
    @info "Velocity saved in param.velocity"
    return data
end 

function kinetic_equation(u::AbstractArray, s::AbstractArray, kinetic::AbstractArray)
    α = kinetic[1, :, :]
    β = kinetic[2, :, :]
    γ = kinetic[3, :, :]

    du = α .* 2.0f0 .- β .* u
    ds = β .* u .- γ .* s

    return du, ds
end

function compute_cell_velocity(data::JuloVeloObject)
    X = data.X
    embedding = data.embedding
    velocity = data.param["velocity"]
    
    spliced_matrix = X[2, :, :]'
    velocity_spliced_matrix = velocity[2, :, :]'
    velocity_spliced_matrix = sqrt.(abs.(velocity_spliced_matrix) .+ 1) ./ sign.(velocity_spliced_matrix)
    
    neighbor_graph = get_neighbor_graph(embedding)
    velocity_embedding = velocity_projection(spliced_matrix, velocity_spliced_matrix, neighbor_graph, embedding)
    
    data.param["velocity_embedding"] = velocity_embedding
    
    return data
end

function velocity_correlation(spliced_matrix, velocity_spliced_matrix)
    correlation_matrix = zeros32(size(spliced_matrix)[2], size(velocity_spliced_matrix)[2])
    @inbounds for i in axes(spliced_matrix, 2)
        correlation_matrix[i, :] = correlation_coefficient(spliced_matrix, velocity_spliced_matrix, i)
    end
    correlation_matrix[diagind(correlation_matrix)] .= 0
    
    return correlation_matrix
end

function correlation_coefficient(spliced_matrix, velocity_spliced_matrix, i)
    spliced_matrix = spliced_matrix'
    velocity_spliced_matrix = velocity_spliced_matrix'
    spliced_matrix = spliced_matrix .- spliced_matrix[i, :]'
    velocity_spliced_matrix = velocity_spliced_matrix[i, :]
    
    mean_spliced_matrix = spliced_matrix .- mean(spliced_matrix, dims = 2)
    mean_velocity_spliced_matrix = velocity_spliced_matrix .- mean(velocity_spliced_matrix)
    
    ss_mean_spliced_matrix = sum(mean_spliced_matrix .^ 2, dims = 2)
    ss_mean_velocity_spliced_matrix = sum(mean_velocity_spliced_matrix .^ 2)
    
    correlation = mean_spliced_matrix * mean_velocity_spliced_matrix
    length = sqrt.(ss_mean_spliced_matrix .* ss_mean_velocity_spliced_matrix) .+ eps(Float32)
    
    correlation = correlation ./ length
    
    return correlation'
end

function get_neighbor_graph(embedding; n_neighbors = 200)
    graph = zeros32(size(embedding)[1], size(embedding)[1])
    
    kdTree = KDTree(embedding')
    idxs, distss = knn(kdTree, embedding', n_neighbors + 1, true);
    
    idxs = reduce(vcat, transpose.(idxs))[:, 2:end]
    
    for i in axes(idxs, 1)
        graph[i, idxs[i, :]] .= 1
    end
    
    return graph
end

function velocity_projection(spliced_matrix, velocity_spliced_matrix, neighbor_graph, embedding)
    function replace_nan(v)
        return map(x -> isnan(x) ? zero(x) : x, v)
    end
    
    σ = 0.05
    ncells = size(embedding)[1]
    correlation_coefficient = velocity_correlation(spliced_matrix, velocity_spliced_matrix)
    
    probablity_matrix = exp.(correlation_coefficient ./ σ) .* neighbor_graph
    probablity_matrix = probablity_matrix ./ sum(probablity_matrix, dims = 2)
    
    velocity_field = zeros32(ncells, ncells, 2)
    for i in axes(embedding', 2)
        temp = embedding' .- embedding'[:, i]
        velocity_field[i, :, 1] = temp[1, :]'
        velocity_field[i, :, 2] = temp[2, :]'
    end
    
    uni_scale = (velocity_field[:, :, 1] .^ 2 + velocity_field[:, :, 2] .^ 2) .^ 0.5
    velocity_field = velocity_field ./ uni_scale
    velocity_field = replace_nan(velocity_field)
    
    velocity_embedding = sum(velocity_field .* probablity_matrix, dims = 2)
    velocity_embedding = velocity_embedding .- sum(neighbor_graph .* velocity_field, dims = 2) ./ sum(neighbor_graph, dims = 2)
    
    velocity_embedding = reshape(velocity_embedding, ncells, 2)
    
    return velocity_embedding
end
