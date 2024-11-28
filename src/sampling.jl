function gaussian_kernel(X::AbstractArray; mu::AbstractFloat = 0.0f0, sigma::AbstractFloat = 1.0f0)
    return exp.(-(X .- mu) .^ 2.0f0 ./ (2.0f0 * sigma ^ 2.0f0)) ./ (2.0f0 * pi * sigma ^ 2.0f0) ^ 0.5f0
end

function meshgrid(x::AbstractVector, y::AbstractVector)
    function meshgrid!(x, y)
        return [x, y]
    end
    return vcat(meshgrid!.(x, y')'...)
end

function density_sampling(adata::Muon.AnnData; sample_number::Int = 2400, step = (30, 30), overwrite::Bool = false)
    # Check if data is reshaped
    if ~haskey(adata.uns, "X")
        @info "Could not find \"X\" in adata.uns"
        @info "Please use reshape_data() first"
        return adata
    end
    
    # Check if data is already sampled
    if haskey(adata.uns, "train_X")
        if overwrite
            @info "Warning! Overwrite train_X in adata.uns"
        else
            @info "Already sample cells"
            @info "Use overwrite = true to sample cells again if you want"
            return adata
        end        
    end
    
    # Extract data
    X = adata.uns["X"]
    
    # Density sampling
    train_X = Array{Float32}(undef, 2, sample_number, 0)
    for i in axes(X, 3)
        temp_X = X[:, X[1, :, i] .!= 0, i]
        selected_point = density_sampling!(temp_X, size(temp_X)[2]; sample_number = sample_number, step = step)
        train_X = cat(train_X, temp_X[:, selected_point], dims = 3)
    end
    
    # Write to anndata
    adata.uns["train_X"] = train_X
    
    @info "Sample $(sample_number) cells"
    
    return adata
end

function density_sampling!(X::AbstractMatrix, ncells::Int; sample_number::Int = 2400, step::Tuple{Int, Int} = (30, 30))
    # initial point on 2D space
    grs = []
    for dim in axes(X, 1)
        min, max = minimum(X[dim, :]), maximum(X[dim, :])
        min = min - 0.025f0 * abs(max - min)
        max = max + 0.025f0 * abs(max - min)
        gr = range(min, max, 30)
        push!(grs, gr)
    end
    # Add gaussian noise
    gridpoints = meshgrid(grs[1], grs[2]) + rand(Normal(), step[1] * step[2], 2) * 0.15f0
    
    # Find point far away from the gaussian point
    kdTree = KDTree(X)
    grid_idxs, _ = knn(kdTree, gridpoints', 10)
    idx_choice = sort(unique(vcat(grid_idxs...)))
    
    # Find the sparse point
    _, density_dists = knn(kdTree, X[:, idx_choice], 10)
    density_kernel = gaussian_kernel.(density_dists; mu = 0.0f0, sigma = 0.5f0)
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
