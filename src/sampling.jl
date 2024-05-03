function gaussian_kernel(X::AbstractArray; mu = 0.0f0, sigma = 1.0f0)
    return exp.(-(X .- mu) .^ 2.0f0 ./ (2.0f0 * sigma ^ 2.0f0)) ./ (2.0f0 * pi * sigma ^ 2.0f0) ^ 0.5f0
end

function meshgrid(x::AbstractVector, y::AbstractVector)
    function meshgrid!(x, y)
        return [x, y]
    end
    return vcat(meshgrid!.(x, y')'...)
end

function density_sampling!(X::AbstractMatrix, ncells::Int; sample_number::Int = 2400, step = (30, 30))
    grs = []
    for dim in axes(X, 1)
        min, max = minimum(X[dim, :]), maximum(X[dim, :])
        min = min - 0.025f0 * abs(max - min)
        max = max + 0.025f0 * abs(max - min)
        gr = range(min, max, 30)
        push!(grs, gr)
    end
    gridpoints = meshgrid(grs[1], grs[2]) + rand(Normal(), step[1] * step[2], 2) * 0.15f0
    
    kdTree = KDTree(X)
    grid_idxs, grid_distss = knn(kdTree, gridpoints', 10)
    idx_choice = sort(unique(vcat(grid_idxs...)))
    
    density_idxs, density_dists = knn(kdTree, X[:, idx_choice], 10)
    density_kernel = gaussian_kernel.(density_dists; mu = 0, sigma = 0.5)
    density_kernel = sum.(density_kernel)
    
    # Save sparse point region
    sparse_point = density_kernel .< percentile(density_kernel, 15)
    sparse_idx = idx_choice[sparse_point]
    
    # Add points in dense region
    dense_number = sample_number - length(sparse_idx)
    dense_idx = rand(reshape(1:ncells, :)[1:ncells .∉ Ref(sparse_idx)], dense_number)
    
    # Merge point
    idx_choice = vcat(sparse_idx, dense_idx)
    
    return idx_choice
end

function density_sampling(data::JuloVeloObject; sample_number::Int = 2400, step = (30, 30))
    X = data.X
    train_X = Array{Float32}(undef, 2, sample_number, 0)
    for i in axes(X, 3)
        temp_X = X[:, X[1, :, i] .!= 0, i]
        selected_point = density_sampling!(temp_X, size(temp_X)[2]; sample_number)
        train_X = cat(train_X, temp_X[:, selected_point], dims = 3)
    end
    
    data.train_X = train_X
    
    return data
end
