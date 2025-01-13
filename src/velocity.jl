"""
TODO: Add docsstrings
"""
function kinetics_equation_training(u::AbstractArray, s::AbstractArray, kinetics::AbstractArray)
    α = kinetics[1, :]
    β = kinetics[2, :]
    γ = kinetics[3, :]

    du = α .* 2.0f0 .- β .* u
    ds = β .* u .- γ .* s
    
    return du, ds
end

"""
TODO: Add docsstrings
"""
function kinetics_equation_inference(u::AbstractArray, s::AbstractArray, kinetics::AbstractArray)
    α = kinetics[1, :, :]
    β = kinetics[2, :, :]
    γ = kinetics[3, :, :]
    
    du = α .* 2.0f0 .- β .* u
    ds = β .* u .- γ .* s

    return du, ds
end

"""
TODO: Add docsstrings
"""
function velocity_estimation(adata::Muon.AnnData, Kinetics::Chain; dt::AbstractFloat = 0.5f0)
    # Extract data
    X = adata.uns["X"]
    u = X[1, :, :]
    s = X[2, :, :]
    
    # Get ncells and ngenes
    _, ncells, ngenes = size(X)
    
    # Calculate kinetics
    kinetics = Kinetics(X)
    α = kinetics[1, :, :]
    β = kinetics[2, :, :]
    γ = kinetics[3, :, :]
    
    # Calculate du and ds
    du, ds = kinetics_equation_inference(u, s, kinetics)
    
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
    adata.uns["alpha"] = α
    adata.uns["beta"] = β
    adata.uns["gamma"] = γ
    adata.uns["train_u"] = u
    adata.uns["train_s"] = s
    
    # Info for velocity and kinetics
    @info "Velocity saved in adata.uns[\"velocity\"]"
    @info "kinetics saved in adata.uns[\"kinetics\"]"
end

"""
TODO: Add docsstrings
"""
function velocity_correlation(spliced_matrix::AbstractMatrix, velocity_spliced_matrix::AbstractMatrix; use_gpu::Bool = true)
    correlation_matrix = zeros32(size(spliced_matrix)[2], size(velocity_spliced_matrix)[2])
    if use_gpu
        spliced_matrix = spliced_matrix |> gpu
        velocity_spliced_matrix = velocity_spliced_matrix |> gpu
    end 
    @inbounds for i in axes(spliced_matrix, 2)
        correlation_matrix[i, :] = cpu(correlation_coefficient(spliced_matrix, velocity_spliced_matrix, i))
    end
    correlation_matrix[diagind(correlation_matrix)] .= 0
    
    return correlation_matrix
end

"""
TODO: Add docsstrings
"""
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

"""
TODO: Add docsstrings
"""
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

"""
TODO: Add docsstrings
"""
function velocity_projection(spliced_matrix::AbstractMatrix, velocity_spliced_matrix::AbstractMatrix, neighbor_graph::AbstractMatrix, embedding::AbstractMatrix; use_gpu::Bool = true, dimension::AbstractString = "2d")
    function replace_nan(v)
        return map(x -> isnan(x) ? zero(x) : x, v)
    end
    
    σ = 0.05
    ncells = size(embedding)[1]
    correlation_coefficient = velocity_correlation(spliced_matrix, velocity_spliced_matrix; use_gpu = use_gpu)
    
    probablity_matrix = exp.(correlation_coefficient ./ σ) .* neighbor_graph
    probablity_matrix = probablity_matrix ./ sum(probablity_matrix, dims = 2)
    
    if dimension == "2d"
        velocity_field = zeros32(ncells, ncells, 2)
        for i in axes(embedding', 2)
            temp = embedding' .- embedding'[:, i]
            velocity_field[i, :, 1] = temp[1, :]'
            velocity_field[i, :, 2] = temp[2, :]'
        end
        uni_scale = (velocity_field[:, :, 1] .^ 2 + velocity_field[:, :, 2] .^ 2) .^ 0.5
    elseif dimension == "3d"
        velocity_field = zeros32(ncells, ncells, 3)
        for i in axes(embedding', 2)
            temp = embedding' .- embedding'[:, i]
            velocity_field[i, :, 1] = temp[1, :]'
            velocity_field[i, :, 2] = temp[2, :]'
            velocity_field[i, :, 3] = temp[3, :]'
        end
        uni_scale = (velocity_field[:, :, 1] .^ 2 + velocity_field[:, :, 2] .^ 2 + velocity_field[:, :, 3] .^ 2) .^ 0.5
    end
    
    velocity_field = velocity_field ./ uni_scale
    velocity_field = replace_nan(velocity_field)
    
    velocity_embedding = sum(velocity_field .* probablity_matrix, dims = 2)
    velocity_embedding = velocity_embedding .- sum(neighbor_graph .* velocity_field, dims = 2) ./ sum(neighbor_graph, dims = 2)
    
    if dimension == "2d"
        velocity_embedding = reshape(velocity_embedding, ncells, 2)
    elseif dimension == "3d"
        velocity_embedding = reshape(velocity_embedding, ncells, 3)
    end
    
    return velocity_embedding
end

"""
TODO: Add docsstrings
"""
function compute_cell_velocity(adata::Muon.AnnData;
    n_neighbors::Int = 200, datapath::AbstractString = "", celltype::AbstractString = "clusters", basis::AbstractString = "umap", use_gpu::Bool = true, dimension::AbstractString = "2d")

    X = adata.uns["X"]
    embedding = adata.obsm["X_$basis"]
    velocity = adata.uns["velocity"]

    spliced_matrix = X[2, :, :]'
    velocity_spliced_matrix = velocity[2, :, :]'
    velocity_spliced_matrix = sqrt.(abs.(velocity_spliced_matrix) .+ 1) ./ sign.(velocity_spliced_matrix)

    neighbor_graph = get_neighbor_graph(embedding, n_neighbors)
    velocity_embedding = velocity_projection(spliced_matrix, velocity_spliced_matrix, neighbor_graph, embedding; use_gpu = use_gpu, dimension = dimension)

    adata.obsm["velocity_$basis"] = velocity_embedding

    if use_gpu
        CUDA.reclaim()
    end

    return adata
end

"""
TODO: Add docsstrings
"""
function estimate_pseudotime(adata::Muon.AnnData;
    n_neighbors::Int = 15, 
    n_jobs::Int = 8, 
    basis::AbstractString = "umap", 
    use_velocity_graph::Bool = true)

    if ~haskey(adata.obsm, "X_$basis") | ~haskey(adata.obsm, "velocity_$basis")
        throw(ArgumentError("basis is not found in adata.obsm."))
    end

    train_gene_index = adata.var[!, "train_genes"]

    idx = 1
    velocity = Array{Float32}(undef, size(adata.X)[1], 0)
    for i in range(1, size(adata.X)[2])
        if adata.var[!, "train_genes"][i]
            velocity = hcat(velocity, adata.uns["velocity"][2, :, idx])
            idx += 1
        else
            velocity = hcat(velocity, zeros32(size(adata.X)[1]))
        end
    end

    root_key = adata.uns["iroot"]
    X = adata.X
    Ms = Matrix(adata.layers["Ms"])
    Mu = Matrix(adata.layers["Mu"])
    embedding = Matrix(adata.obsm["X_$basis"])
    neighbors = adata.uns["neighbors"]
    distances = Matrix(adata.obsp["distances"])
    connectivities = Matrix(adata.obsp["connectivities"])

    py"""
    import scvelo as scv
    import anndata as ad
    import numpy as np
    import scipy

    def estimate_pseudotime(X, root_key, velocity, Mu, Ms, embedding, neighbors, distances, connectivities, basis):
        temp_adata = ad.AnnData(np.array(X))
        temp_adata.uns["iroot"] = root_key
        temp_adata.layers["velocity"] = np.array(velocity)
        temp_adata.layers["Mu"] = np.array(Mu)
        temp_adata.layers["Ms"] = np.array(Ms)
        temp_adata.obsm[f"X_{str(basis)}"] = np.array(embedding)
        temp_adata.uns["neighbors"] = neighbors
        temp_adata.obsp["distances"] = scipy.sparse.csr_matrix(distances)
        temp_adata.obsp["connectivities"] = scipy.sparse.csr_matrix(connectivities)
        
        scv.tl.velocity_graph(temp_adata, n_jobs = $n_jobs, n_neighbors = $n_neighbors)
        temp_adata.uns["velocity_params"]["embeddings"] = [str(basis)]
        scv.tl.velocity_pseudotime(temp_adata, use_velocity_graph = $use_velocity_graph, root_key = temp_adata.uns["iroot"])

        velocity_pseudotime = list(temp_adata.obs["velocity_pseudotime"])
        velocity_self_transition = list(temp_adata.obs["velocity_self_transition"])
        velocity_params = temp_adata.uns["velocity_params"]
        velocity_graph = temp_adata.uns["velocity_graph"]
        velocity_graph_neg = temp_adata.uns["velocity_graph_neg"]

        velocity_dictionary = {
            "velocity_pseudotime":velocity_pseudotime,
            "velocity_self_transition":velocity_self_transition,
            "velocity_params":velocity_params,
            "velocity_graph":velocity_graph,
            "velocity_graph_neg":velocity_graph_neg
        }

        return velocity_dictionary
    """

    velocity_dictionary = py"estimate_pseudotime"(X, root_key, velocity, Mu, Ms, embedding, neighbors, distances, connectivities, basis)

    adata.layers["velocity"] = velocity
    adata.obs[!, "velocity_pseudotime"] = velocity_dictionary["velocity_pseudotime"]
    adata.obs[!, "velocity_self_transition"] = velocity_dictionary["velocity_self_transition"]
    adata.uns["velocity_params"] = Dict{String, Union{AbstractString, Number, DataFrame, Dict, CategoricalArray{<:AbstractString}, CategoricalArray{<:Number}, AbstractArray{<:Union{Missing, Integer}}, AbstractArray{<:AbstractString}, AbstractArray{<:Number}, StructArray}}(py"$velocity_dictionary['velocity_params']")
    adata.uns["velocity_graph"] = sparse(velocity_dictionary["velocity_graph"].A)
    adata.uns["velocity_graph_neg"] = sparse(velocity_dictionary["velocity_graph_neg"].A)

    return adata
end

"""
TODO: Add docsstrings
"""
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

"""
TODO: Deprecated
"""
#=
function estimate_pseudotime(adata::Muon.AnnData, n_path::Union{Int, Nothing} = nothing;
    n_repeat::Int = 10, 
    n_jobs::Int = 8, 
    datapath::AbstractString = "", 
    celltype::AbstractString = "clusters", 
    basis::AbstractString = "umap", 
    pipeline_type::AbstractString = "scvelo",
    use_velocity_graph::Bool = true)

    if ~haskey(adata.obsm, "X_$basis") | ~haskey(adata.obsm, "velocity_$basis")
        throw(ArgumentError("basis is not found in adata.obsm."))
    end

    if pipeline_type == "celldancer"
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
    
    elseif pipeline_type == "scvelo"
        train_gene_index = adata.var[!, "train_genes"]

        idx = 1
        velocity = Array{Float32}(undef, size(adata.X)[1], 0)
        for i in range(1, size(adata.X)[2])
            if adata.var[!, "train_genes"][i]
                velocity = hcat(velocity, adata.uns["velocity"][2, :, idx])
                idx += 1
            else
                velocity = hcat(velocity, zeros32(size(adata.X)[1]))
            end
        end

        root_key = adata.uns["iroot"]
        X = adata.X
        Ms = Matrix(adata.layers["Ms"])
        Mu = Matrix(adata.layers["Mu"])
        basis = Matrix(adata.obsm["X_umap"])
        neighbors = adata.uns["neighbors"]
        distances = Matrix(adata.obsp["distances"])
        connectivities = Matrix(adata.obsp["connectivities"])

        py"""
        import scvelo as scv
        import anndata as ad
        import numpy as np
        import scipy

        def estimate_pseudotime(X, root_key, velocity, Mu, Ms, basis, neighbors, distances, connectivities):
            temp_adata = ad.AnnData(np.array(X))
            temp_adata.uns["iroot"] = root_key
            temp_adata.layers["velocity"] = np.array(velocity)
            temp_adata.layers["Mu"] = np.array(Mu)
            temp_adata.layers["Ms"] = np.array(Ms)
            temp_adata.obsm["X_umap"] = np.array(basis)
            temp_adata.uns["neighbors"] = neighbors
            temp_adata.obsp["distances"] = scipy.sparse.csr_matrix(distances)
            temp_adata.obsp["connectivities"] = scipy.sparse.csr_matrix(connectivities)
            
            scv.tl.velocity_graph(temp_adata, n_jobs = $n_jobs, n_neighbors = 15)
            temp_adata.uns["velocity_params"]["embeddings"] = ["umap"]
            scv.tl.velocity_pseudotime(temp_adata, use_velocity_graph = $use_velocity_graph, root_key = temp_adata.uns["iroot"])

            velocity_pseudotime = list(temp_adata.obs["velocity_pseudotime"])
            velocity_self_transition = list(temp_adata.obs["velocity_self_transition"])
            velocity_params = temp_adata.uns["velocity_params"]
            velocity_graph = temp_adata.uns["velocity_graph"]
            velocity_graph_neg = temp_adata.uns["velocity_graph_neg"]

            velocity_dictionary = {
                "velocity_pseudotime":velocity_pseudotime,
                "velocity_self_transition":velocity_self_transition,
                "velocity_params":velocity_params,
                "velocity_graph":velocity_graph,
                "velocity_graph_neg":velocity_graph_neg
            }

            return velocity_dictionary
        """

        velocity_dictionary = py"estimate_pseudotime"(X, root_key, velocity, Mu, Ms, basis, neighbors, distances, connectivities)

        adata.layers["velocity"] = velocity
        adata.obs[!, "velocity_pseudotime"] = velocity_dictionary["velocity_pseudotime"]
        adata.obs[!, "velocity_self_transition"] = velocity_dictionary["velocity_self_transition"]
        adata.uns["velocity_params"] = Dict{String, Union{AbstractString, Number, DataFrame, Dict, CategoricalArray{<:AbstractString}, CategoricalArray{<:Number}, AbstractArray{<:Union{Missing, Integer}}, AbstractArray{<:AbstractString}, AbstractArray{<:Number}, StructArray}}(py"$velocity_dictionary['velocity_params']")
        adata.uns["velocity_graph"] = sparse(velocity_dictionary["velocity_graph"].A)
        adata.uns["velocity_graph_neg"] = sparse(velocity_dictionary["velocity_graph_neg"].A)
    end

    return adata
end
=#