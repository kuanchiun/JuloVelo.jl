function filter_gene(data::JuloVeloObject; criteria::AbstractFloat = 0.5)
    # Check if genes are filter first
    if ~isnothing(data.bad_correlation_genes)
        @info "Already filtered genes, do nothing"
        return data
    end
    
    c = data.c
    u = data.u
    s = data.s
    genes = data.genes
    datatype = data.datatype
    
    #Filter genes using correlation between unspliced and spliced pattern
    gene_correlation = corspearman.(eachrow(u), eachrow(s)) # Get correlation between u/s
    
    pass = gene_correlation .>= criteria # Find genes that have good correlation
    fail = gene_correlation .< criteria # Find genes that habe bad correlation
    
    pass_genes = genes[pass]
    bad_correlation_genes = genes[fail]
    u = u[pass, :]
    s = s[pass, :]
    if datatype == "multi"
        c = c[pass, :]
    end
    
    data.temp_u = u
    data.temp_s = s
    data.bad_correlation_genes = bad_correlation_genes
    data.temp_genes = pass_genes
    if datatype == "multi"
        data.temp_c = c
    else
        data.temp_c = nothing
    end
        
    @info "$(length(data.bad_correlation_genes)) genes are filtered due to low correlation between unspliced and spliced RNA"
    @info "See .bad_correlation_genes for details"
    
    return data
end

function find_neighbor(X::AbstractArray, neighbor_number::Int, ngenes::Int)
    # Initialize Array for Nearest Neighbor
    NN = Array{Int32}(undef, size(X, 2), neighbor_number, ngenes)
    # Calculate Nearest Neighbor for each cell in each gene
    for i in 1:ngenes
        kdTree = KDTree(X[:, :, i])
        idxs, dists = knn(kdTree, X[:, :, i], neighbor_number + 1, true)
        NN[:, :, i] = reduce(vcat, transpose.(idxs))[:, 2:end]
    end

    return NN
end

function calculate_neighbor_vector(X, ngenes::Int, sample_number::Int, neighbor_number::Int)
    NN = find_neighbor(X, neighbor_number, ngenes)
    repeat_idx = transpose(reduce(hcat, [repeat([i], neighbor_number) for i in 1:sample_number]))
    
    neighbor_vector = mapreduce(j -> mapreduce(i -> X[:, NN[i, :, j], j] - X[:, repeat_idx[i, :], j], hcat, axes(NN, 1)), hcat, axes(X, 3))
    neighbor_vector = reshape(neighbor_vector, 2, neighbor_number, :)
    neighbor_vector = neighbor_vector .+ eps(Float32)
    
    return neighbor_vector
end

function to_device(train_X::AbstractArray, Kinetic::Chain, train_neighbor_vector::AbstractArray; use_gpu = true)
    if use_gpu
        use_cuda = CUDA.functional()
        if use_cuda
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
    
    train_X = train_X |> device
    Kinetic = Kinetic |> device
    train_neighbor_vector = train_neighbor_vector |> device
    
    return train_X, Kinetic, train_neighbor_vector
end

function to_celldancer(data::JuloVeloObject; datapath = "")
    # Get information
    ncells = data.ncells
    ngenes = data.train_genes_number
    train_genes = data.train_genes
    train_genes_number = data.train_genes_number
    X = data.X
    velocity = data.param["velocity"]
    embedding = data.embedding
    Kinetic = data.param["velocity_model"]
    kinetic = Kinetic(X)
    celltype = isnothing(data.celltype) ? data.clusters : data.celltype
    # Create cell index
    cellindex = collect(range(0, ncells - 1))
    cellindex = string.(cellindex);
    cellindex = repeat(cellindex, ngenes)
    # Create basic information for celldancer
    use_genes = [train_genes[i] for i in 1:train_genes_number for j in 1:ncells]
    u = reduce(vcat, X[1, :, :])
    s = reduce(vcat, X[2, :, :])
    û = reduce(vcat, X[1, :, :] + velocity[1, :, :])
    ŝ = reduce(vcat, X[2, :, :] + velocity[2, :, :])
    α = reduce(vcat, kinetic[1, :, :])
    β = reduce(vcat, kinetic[2, :, :])
    γ = reduce(vcat, kinetic[3, :, :])
    loss = repeat([0.05], ncells * train_genes_number)
    cell_name = ["cell_$j" for i in 1:train_genes_number for j in 1:ncells]
    celltypes = repeat(celltype, train_genes_number)
    embedding1 = repeat(embedding[:, 1], train_genes_number)
    embedding2 = repeat(embedding[:, 2], train_genes_number)
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
    
    index = ["cellIndex", "gene_name", "unsplice", "splice", "unsplice_predict", "splice_predict", "alpha", "beta", "gamma", "loss", "cellID", "clusters", "embedding1", "embedding2"]
    # Add velocity embedding information if existed
    if haskey(data.param, "velocity_embedding")
        velocity_embedding = data.param["velocity_embedding"]
        velocity1 = repeat(velocity_embedding[:, 1], train_genes_number)
        velocity2 = repeat(velocity_embedding[:, 2], train_genes_number)
        table = hcat(table, velocity1, velocity2)
        push!(index, "velocity1")
        push!(index, "velocity2")
    end
    # Add pseudotime information if existed
    if haskey(data.param, "pseudotime")
        pseudotime = data.param["pseudotime"]
        pseudotime = repeat(pseudotime, train_genes_number)
        table = hcat(table, pseudotime)
        push!(index, "pseudotime")
    end
    
    dataframe = DataFrame(table, index)
    CSV.write(joinpath(datapath, "JuloVelo_result.csv"), dataframe)
    
    return nothing
end

function to_anndata(data::JuloVeloObject)
    # Extract information
    X = data.X
    genes = data.train_genes
    embedding = data.embedding
    velocity_embedding = data.param["velocity_embedding"]
    model = data.param["velocity_model"]
    velocity = data.param["velocity"]
    celltype = data.celltype
    leiden = data.clusters
    if haskey(data.param, "JuloVelo_pseudotime")
        pseudotime = data.param["JuloVelo_pseudotime"]
    end
    # Get spliced and unspliced RNA
    u = X[1, :, :]
    s = X[2, :, :]
    cellID = ["cell_$i" for i in 1:data.ncells]
    # Get kinetics parameters
    kinetics = model(X)
    α = kinetics[1, :, :]
    β = kinetics[2, :, :]
    γ = kinetics[3, :, :]
    
    # Get velocity
    û = velocity[1, :, :]
    ŝ = velocity[2, :, :]
    # Create Anndata
    adata = AnnData(X = s, obs_names = cellID, var_names = genes)
    # Add dynamo required element to adata
    adata.obs[!, "clusters"] = celltype
    adata.obs[!, "leiden"] = leiden
    adata.layers["M_s"] = s
    adata.layers["X_spliced"] = s
    adata.layers["M_u"] = u
    adata.layers["X_unspliced"] = u
    adata.layers["alpha"] = α
    adata.layers["beta"] = β
    adata.layers["gamma"] = γ
    adata.layers["velocity_U"] = û
    adata.layers["velocity_S"] = ŝ
    adata.obsm["X_ebd"] = embedding
    adata.obsm["velocity_ebd"] = velocity_embedding
    if haskey(data.param, "JuloVelo_pseudotime")
        adata.obs[!, "pseudotime"] = pseudotime
    end

    if ~isnothing(data.train_c)
        c = data.train_c'
        adata.layers["M_c"] = c
    end
    
    if any(ismissing.(adata.obsm["velocity_ebd"]))
        adata.obsm["velocity_ebd"] = replace(adata.obsm["velocity_ebd"], missing => NaN)
    end

    return adata
end

function write_adata(adata::Muon.AnnData; filename::AbstractString = "JuloVelo")
    writeh5ad("$filename.h5ad", adata)

    return nothing
end
