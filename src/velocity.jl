function kinetics_equation(u::AbstractArray, s::AbstractArray, kinetics::AbstractArray)
    α = kinetics[1, :, :]
    β = kinetics[2, :, :]
    γ = kinetics[3, :, :]

    du = α .* 2.0f0 .- β .* u
    ds = β .* u .- γ .* s

    return du, ds
end

function velocity_estimation(adata::Muon.AnnData, Kinetics::Chain; dt::AbstractFloat = 0.5f0)
    # Extract data
    X = adata.uns["X"]
    u = X[1, :, :]
    s = X[2, :, :]
    
    # Get ncells and ngenes
    _, ncells, ngenes = size(X)
    
    # Calculate kinetics
    kinetics = Kinetics(X)
    
    # Calculate du and ds
    du, ds = kinetics_equation(u, s, kinetics)
    
    # Multiply du, ds by dt
    du = du .* dt
    ds = ds .* dt
    
    # Reshape du, ds and create dt_vector
    û = permutedims(reshape(du, ncells, 1, ngenes), (2, 1, 3))
    ŝ = permutedims(reshape(ds, ncells, 1, ngenes), (2, 1, 3))
    X̂ = vcat(û, ŝ) .+ eps(Float32)
    
    # Write to anndata
    adata.uns["velocity"] = X̂
    adata.uns["kinetics"] = kinetics
    
    # Info for velocity and kinetics
    @info "Velocity saved in adata.uns[\"velocicy\"]"
    @info "kinetics saved in adata.uns[\"kinetics\"]"
end

function velocity_correlation(spliced_matrix::AbstractMatrix, velocity_spliced_matrix::AbstractMatrix)
    correlation_matrix = zeros32(size(spliced_matrix)[2], size(velocity_spliced_matrix)[2])
    @inbounds for i in axes(spliced_matrix, 2)
        correlation_matrix[i, :] = correlation_coefficient(spliced_matrix, velocity_spliced_matrix, i)
    end
    correlation_matrix[diagind(correlation_matrix)] .= 0
    
    return correlation_matrix
end

function correlation_coefficient(spliced_matrix::AbstractMatrix, velocity_spliced_matrix::AbstractMatrix, i::Int)
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

function get_neighbor_graph(embedding::AbstractMatrix, n_neighbors::Int)
    graph = zeros32(size(embedding)[1], size(embedding)[1])
    
    kdTree = KDTree(embedding')
    idxs, distss = knn(kdTree, embedding', n_neighbors + 1, true);
    
    idxs = reduce(vcat, transpose.(idxs))[:, 2:end]
    
    for i in axes(idxs, 1)
        graph[i, idxs[i, :]] .= 1
    end
    
    return graph
end

function velocity_projection(spliced_matrix::AbstractMatrix, velocity_spliced_matrix::AbstractMatrix, neighbor_graph::AbstractMatrix, embedding::AbstractMatrix)
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

function compute_cell_velocity(adata::Muon.AnnData; 
        pipeline_type::AbstractString = "JuloVelo", n_neighbors::Int = 200, datapath::AbstractString = "", celltype = "clusters", basis = "umap")
    
    if pipeline_type == "JuloVelo"
        X = adata.uns["X"]
        embedding = adata.obsm["X_$basis"]
        velocity = adata.uns["velocity"]
        
        spliced_matrix = X[2, :, :]'
        velocity_spliced_matrix = velocity[2, :, :]'
        velocity_spliced_matrix = sqrt.(abs.(velocity_spliced_matrix) .+ 1) ./ sign.(velocity_spliced_matrix)
    
        neighbor_graph = get_neighbor_graph(embedding, n_neighbors)
        velocity_embedding = velocity_projection(spliced_matrix, velocity_spliced_matrix, neighbor_graph, embedding)
        
        adata.obsm["velocity_$basis"] = velocity_embedding
        
    elseif pipeline_type == "cellDancer"
        to_cellDancer(adata; datapath = datapath, celltype = celltype, basis = basis)
        
        py"""
        import pandas as pd
        import celldancer as cd

        JuloVelo_df = pd.read_csv("JuloVelo_result.csv")
        JuloVelo_df = cd.compute_cell_velocity(cellDancer_df=JuloVelo_df, projection_neighbor_choice='gene', expression_scale='power10', projection_neighbor_size=200, speed_up=(100,100))
        
        JuloVelo_df.to_csv("JuloVelo_result.csv", index = None)
        """
        
        ncells = size(adata.uns["X"])[2]
        JuloVelo_df = CSV.read("JuloVelo_result.csv", DataFrame)
        velocity_embedding = Matrix(JuloVelo_df[1:ncells, ["velocity1", "velocity2"]])
        adata.obsm["velocity_$basis"] = velocity_embedding
    end
    
    return adata
end

function estimate_pseudotime(adata::Muon.AnnData, n_path::Union{Int, Nothing} = nothing; n_repeat::Int = 10, n_jobs::Int = 8, datapath::AbstractString = "", celltype = "clusters", basis = "umap")
    if isnothing(n_path)
        throw(ArgumentError("empty n_path, please give the estimation number of differentiation flow."))
    end
    
    to_cellDancer(adata; datapath = datapath, celltype = celltype, basis = basis)
    
    @info "Start estimate pseudotime, it may take a long time."
    
    py"""
    import pandas as pd
    import celldancer as cd
    import celldancer.utilities as cdutil
    import random

    JuloVelo_df = pd.read_csv("JuloVelo_result.csv")

    # set parameters
    dt = 0.05
    t_total = {dt:int(10/dt)}
    n_repeats = $n_repeat

    # estimate pseudotime
    JuloVelo_df = cd.pseudo_time(cellDancer_df=JuloVelo_df,
                               grid=(30,30),
                               dt=dt,
                               t_total=t_total[dt],
                               n_repeats=n_repeats,
                               speed_up=(100,100),
                               n_paths = $n_path,
                               psrng_seeds_diffusion=[i for i in range(n_repeats)],
                               n_jobs=$n_jobs)
    
    JuloVelo_df.to_csv("JuloVelo_result.csv", index = None)
    """
    
    ncells = size(adata.uns["X"])[2]
    JuloVelo_df = CSV.read("JuloVelo_result.csv", DataFrame)
    pseudotime = JuloVelo_df[1:ncells, "pseudotime"]
    adata.obs[!, "pseudotime"] = pseudotime
    
    return adata
end

function kinetics_embedding(adata::Muon.AnnData; basis::AbstractString = "pca", min_dist::AbstractFloat = 0.5, n_neighbors::Int = 50)
    if ~haskey(adata.uns, "kinetics")
        @info "Could not find \"kinetics\" in adata.uns"
        @info "Please run velocity_estimation() first"
        return adata
    else
        kinetics = adata.uns["kinetics"]
    end

    α = kinetics[1, :, :]'
    β = kinetics[2, :, :]'
    γ = kinetics[3, :, :]'
    embedding = vcat(α, β, γ)
    
    if basis == "pca"
        M = MultivariateStats.fit(PCA, embedding; maxoutdim = 2)
        kinetics_embedding = predict(M, embedding)'
    elseif basis == "umap"
        kinetics_embedding = umap(embedding; min_dist = min_dist, n_neighbors = n_neighbors)'
    end
    
    adata.obsm["kinetics_$basis"] = kinetics_embedding
    
    return adata
end