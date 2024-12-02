"""
TODO: Add docsstrings
"""
function normalize(adata::Muon.AnnData; use_raw::Bool = false)
    # Check if already normalize
    if haskey(adata.layers, "norm_u") || haskey(adata.layers, "norm_s")
        @info "Already normalized unspliced and spliced count"
        return adata
    end
    
    # If use raw count
    if use_raw
        # Check unspliced and spliced in adata.layers
        if ~haskey(adata.layers, "spliced") || ~haskey(adata.layers, "unspliced")
            throw(error("Could not find 'unspliced' or 'spliced' in adata.layer"))
        else
            # Extract matrix
            u = permutedims(Matrix{Float32}(adata.layers["unspliced"]), (2, 1))
            s = permutedims(Matrix{Float32}(adata.layers["spliced"]), (2, 1))
        end
    else
        # Check Mu and Ms in adata.layers
        if ~haskey(adata.layers, "Mu") || ~haskey(adata.layers, "Ms")
            throw(error("Could not find 'Mu' or 'Ms' in adata.layer"))
        else
            # Extract matrix
            u = permutedims(adata.layers["Mu"], (2, 1))
            s = permutedims(adata.layers["Ms"], (2, 1))
        end
    end
    
    # Store max value
    u_max = vec(maximum(u, dims = 2))
    s_max = vec(maximum(s, dims = 2))
    
    # Normalization
    u = mapslices(x -> x ./ (maximum(x) + eps(Float32)), u, dims = 2)
    s = mapslices(x -> x ./ (maximum(x) + eps(Float32)), s, dims = 2)
    
    # Write to anndata
    adata.layers["norm_u"] = permutedims(u, (2, 1))
    adata.layers["norm_s"] = permutedims(s, (2, 1))
    adata.var[!, "u_max"] = u_max
    adata.var[!, "s_max"] = s_max
    
    return adata
end

"""
TODO: Add docsstrings
"""
function normalize(adata_rna::Muon.AnnData, adata_atac::Muon.AnnData; use_raw::Bool = false)
    # RNA normalize
    normalize(adata_rna; use_raw = use_raw)
    
    # Check if already normalize
    if haskey(adata_atac.layers, "norm_c")
        @info "Already normalized ATAC count"
        return adata
    end
    
    # Extract matrix
    c = permutedims(Matrix{Float32}(adata_atac.X), (2, 1))
    
    # ATAC normalization
    c = mapslices(x -> x ./ (maximum(x) + eps(Float32)), c, dims = 2)
    
    # Write to anndata
    adata_atac.layers["norm_c"] = permutedims(c, (2, 1))
    
    return adata_rna, adata_atac
end

"""
TODO: Add docsstrings
"""
function filter_and_gene_kinetics_predetermination(adata::Muon.AnnData; 
    filter_criteria::AbstractFloat = 0.5f0,
    pseudotime::AbstractString = "dpt_pseudotime", 
    clusters::AbstractString = "leiden", 
    root::AbstractString = "iroot", 
    cluster_correlation_criteria::AbstractFloat = 0.3f0, 
    pseudotime_correlation_criteria::AbstractFloat = 0.2f0,
    overwrite::Bool = false)

    filter_genes(adata; 
        filter_criteria = filter_criteria, 
        overwrite = overwrite)

    gene_kinetics_predetermination(adata; 
        pseudotime = pseudotime, 
        clusters = clusters, 
        root = root, 
        cluster_correlation_criteria = cluster_correlation_criteria, 
        pseudotime_correlation_criteria = pseudotime_correlation_criteria, 
        overwrite = overwrite)

    return adata
end

"""
TODO: Add docsstrings
"""
function filter_and_gene_kinetics_predetermination(adata_rna::Muon.AnnData, adata_atac::Muon.AnnData; 
    filter_criteria::AbstractFloat = 0.5f0,
    pseudotime::AbstractString = "dpt_pseudotime", 
    clusters::AbstractString = "leiden", 
    root::AbstractString = "iroot", 
    cluster_correlation_criteria::AbstractFloat = 0.3f0, 
    pseudotime_correlation_criteria::AbstractFloat = 0.2f0,
    overwrite::Bool = false)

    filter_genes(adata_rna; 
        filter_criteria = filter_criteria, 
        overwrite = overwrite)

    gene_kinetics_predetermination(adata_rna, adata_atac; 
        pseudotime = pseudotime, 
        clusters = clusters, 
        root = root, 
        cluster_correlation_criteria = cluster_correlation_criteria, 
        pseudotime_correlation_criteria = pseudotime_correlation_criteria, 
        overwrite = overwrite)

    return adata_rna, adata_atac
end

"""
TODO: Add docsstrings
"""
function filter_genes(adata::Muon.AnnData; 
    filter_criteria::AbstractFloat = 0.5f0, overwrite::Bool = false)
    # Check if genes are already filtered
    if "bad_correlation_genes" in names(adata.var)
        if overwrite
            @info "Warning! Overwrite bad_correlation_genes in adata.var"
        else
            @info "Already filtered genes"
            @info "Use overwrite = true to filter genes again if you want"
            return adata
        end
    end

    # Check if norm_u and norm_s in adata.layer
    if ~haskey(adata.layers, "norm_u") || ~haskey(adata.layers, "norm_s")
        @info "Could not find \"norm_u\" or \"norm_s\" in adata.layer"
        @info "Please use normalize() first"
        return adata
    end

    # Extract data
    u = permutedims(adata.layers["norm_u"], (2, 1))
    s = permutedims(adata.layers["norm_s"], (2, 1))
    genes = Array{AbstractString}(adata.var_names)

    # Filter genes using Spearman correlation between norm_u and norm_s
    gene_correlation = corspearman.(eachrow(u), eachrow(s))

    # Find genes that pass or fail on criteria
    pass_gene_index = gene_correlation .>= filter_criteria
    fail_gene_index = gene_correlation .< filter_criteria

    # Write to anndata
    adata.var[!, "bad_correlation_genes"] = fail_gene_index

    # info for bad correlation genes
    @info "$(sum(fail_gene_index)) genes are filtered due to low correlation between norm_u and norm_s"
    @info "See adata.var[!, \"bad_correlation_genes\"] for details"

    return adata
end

"""
TODO: Add docsstrings
"""
function find_neighbor(X, neighbor_number)
    kdTree = KDTree(X)
    idxs, _ = knn(kdTree, X, neighbor_number + 1, true)
    NN = reduce(vcat, transpose.(idxs))[:, 2:end]
    return NN
end

"""
TODO: Add docsstrings
"""
function calculate_neighbor_vector(X, sample_number, neighbor_number)
    NN = find_neighbor(X, neighbor_number)
    repeat_idx = transpose(reduce(hcat, [repeat([i], neighbor_number) for i in 1:sample_number]))

    neighbor_vector = mapreduce(i -> X[:, NN[i, :]] - X[:, repeat_idx[i, :]], hcat, axes(NN, 1))
    neighbor_vector = reshape(neighbor_vector, 2, neighbor_number, sample_number)
    neighbor_vector = neighbor_vector .+ eps(Float32)

    return neighbor_vector
end

"""
TODO: Add docsstrings
"""
function to_device(X, Kinetics, neighbor_vector, device)
    X = X |> device
    Kinetics = Kinetics |> device
    neighbor_vector = neighbor_vector |> device

    return X, Kinetics, neighbor_vector
end

"""
TODO: Add docsstrings
"""
function gpu_functional(use_gpu)
    if use_gpu
        # Check CUDA is functional
        if CUDA.functional()
            device = gpu
            @info "Training on gpu"
        else
            device = cpu
            @info "CUDA is not functional, back to cpu"
            @info "Training on cpu"
        end
    else
        device = cpu
        @info "Training on cpu"
    end

    return device
end

"""
TODO: Add docsstrings
"""
round4(x::AbstractFloat)::AbstractFloat = round(x, digits = 4)

"""
TODO: Add docsstrings
"""
function to_cellDancer(adata::Muon.AnnData; datapath::AbstractString = "", celltype::AbstractString = "clusters", basis::AbstractString = "umap")
    # Extract data
    X = adata.uns["X"]
    kinetics = adata.uns["kinetics"]
    velocity = adata.uns["velocity"]
    embedding = adata.obsm["X_$basis"]
    celltype = Array{AbstractString}(adata.obs[!, celltype])
    genes = Array{AbstractString}(adata.var_names)
    train_genes = genes[adata.var[!, "train_genes"]]
    
    # Get ncells and ngenes
    _, ncells, ngenes = size(X)
    
    # Create cell index
    cellindex = collect(range(0, ncells - 1))
    cellindex = string.(cellindex);
    cellindex = repeat(cellindex, ngenes)
    
    # Create basic information for cellDancer
    use_genes = [train_genes[i] for i in 1:ngenes for j in 1:ncells]
    u = reduce(vcat, X[1, :, :])
    s = reduce(vcat, X[2, :, :])
    û = reduce(vcat, X[1, :, :] + velocity[1, :, :])
    ŝ = reduce(vcat, X[2, :, :] + velocity[2, :, :])
    α = reduce(vcat, kinetics[1, :, :])
    β = reduce(vcat, kinetics[2, :, :])
    γ = reduce(vcat, kinetics[3, :, :])
    loss = repeat([0.05], ncells * ngenes)
    cell_name = ["cell_$j" for i in 1:ngenes for j in 1:ncells]
    celltypes = repeat(celltype, ngenes)
    embedding1 = repeat(embedding[:, 1], ngenes)
    embedding2 = repeat(embedding[:, 2], ngenes)
    
    # Create basic table for celldancer
    table = hcat(
            cellindex,
            use_genes,
            u,
            s,
            û,
            ŝ,
            α,
            β,
            γ,
            loss,
            cell_name,
            celltypes,
            embedding1,
            embedding2
        )
    
    index = ["cellIndex", 
        "gene_name", 
        "unsplice", 
        "splice", 
        "unsplice_predict", 
        "splice_predict", 
        "alpha", 
        "beta", 
        "gamma", 
        "loss", 
        "cellID", 
        "clusters", 
        "embedding1", 
        "embedding2"]
    
    # Add velocity embedding information if existed
    if haskey(adata.obsm, "velocity_$basis")
        velocity_embedding = adata.obsm["velocity_$basis"]
        velocity1 = repeat(velocity_embedding[:, 1], ngenes)
        velocity2 = repeat(velocity_embedding[:, 2], ngenes)
        table = hcat(table, velocity1, velocity2)
        push!(index, "velocity1")
        push!(index, "velocity2")
    end
    
    # Add pseudotime information if existed
    if "pseudotime" in names(adata.obs)
        pseudotime = adata.obs[!, "pseudotime"]
        pseudotime = repeat(pseudotime, ngenes)
        table = hcat(table, pseudotime)
        push!(index, "pseudotime")
    end
    
    dataframe = DataFrame(table, index)
    CSV.write(joinpath(datapath, "JuloVelo_result.csv"), dataframe)
    
    return nothing
end

"""
TODO: Add docsstrings
"""
function save_model(Kinetics; filename = "models.bson")
    if endswith(filename, "bson")
        BSON.@save(filename, Kinetics)
    else
        BSON.@save("$filename.bson", Kinetics)
    end

    return nothing
end

"""
TODO: Add docsstrings
"""
function load_model(filename)
    BSON.@load filename Kinetics
    return Kinetics
end
