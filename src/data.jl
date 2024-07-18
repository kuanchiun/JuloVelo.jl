abstract type AbstractJuloVeloObject end

mutable struct JuloVeloObject <: AbstractJuloVeloObject
    # Data
    X::Union{AbstractArray{<:Float32}, Nothing}
    c::Union{AbstractMatrix{<:Float32}, Nothing}
    u::AbstractMatrix{<:Float32}
    s::AbstractMatrix{<:Float32}
    train_c::Union{AbstractMatrix{<:Float32}, Nothing}
    train_u::Union{AbstractMatrix{<:Float32}, Nothing}
    train_s::Union{AbstractMatrix{<:Float32}, Nothing}
    temp_c::Union{AbstractMatrix{<:Float32}, Nothing}
    temp_u::Union{AbstractMatrix{<:Float32}, Nothing}
    temp_s::Union{AbstractMatrix{<:Float32}, Nothing}
    
    # Training data
    train_X::Union{AbstractArray{<:Float32}, Nothing}
    gene_kinetics::Union{AbstractVector{<:AbstractString}, Nothing}
    
    # Gene
    ngenes::Int
    genes::AbstractVector{<:AbstractString}
    temp_genes::Union{AbstractVector{<:AbstractString}, Nothing}
    train_genes::Union{AbstractVector{<:AbstractString}, Nothing}
    train_genes_number::Union{Int, Nothing}
    bad_correlation_genes::Union{AbstractVector{<:AbstractString}, Nothing}
    bad_kinetics_genes::Union{AbstractVector{<:AbstractString}, Nothing}
    
    # Cell characteristic
    ncells::Int
    pseudotime::AbstractVector{<:AbstractFloat}
    clusters::AbstractVector{<:AbstractString}
    root::AbstractString
    embedding::AbstractArray{<:AbstractFloat}
    celltype::Union{AbstractVector{<:AbstractString}, Nothing}
    datatype::AbstractString
    
    # parameter
    param::Dict{<:AbstractString, <:Any}
    
    function JuloVeloObject(datapath::AbstractString, root::Union{AbstractString, Int}; datatype::AbstractString = "gex", normalized::Bool = true)
        data = new()
        # Check datatype
        if ~(lowercase(datatype) in ["gex", "multi"])
            throw(error("Datatype argument must be 'gex' or 'multi'."))
        end
        
        if lowercase(datatype) == "gex"
            u, s, genes, pseudotime, clusters, embedding, celltype, M = load_data(datapath; datatype, normalized)
        else
            c, u, s, genes, pseudotime, clusters, embedding, celltype, M = load_data(datapath; datatype, normalized)
        end
        
        root = typeof(root) == String ? root : string(root)
        
        # Data
        data.X = nothing
        data.u = u
        data.s = s
        data.train_c = nothing
        data.train_u = nothing
        data.train_s = nothing
        data.temp_c = nothing
        data.temp_u = nothing
        data.temp_s = nothing
        if lowercase(datatype) == "multi"
            data.c = c
        else
            data.c = nothing
        end
        # Training data
        data.train_X = nothing
        data.gene_kinetics = nothing
        # Gene
        data.ngenes = size(u, 1)
        data.genes = genes
        data.temp_genes = nothing
        data.train_genes = nothing
        data.train_genes_number = nothing
        data.bad_correlation_genes = nothing
        data.bad_kinetics_genes = nothing
        # Cell characteristic
        data.ncells = size(u, 2)
        data.pseudotime = pseudotime
        data.clusters = clusters
        data.root = root
        data.embedding = embedding
        data.celltype = celltype
        data.datatype = lowercase(datatype)
        
        data.param = Dict{String, Any}()
        data.param["PCA_model"] = M
        
        return data
    end
    
    function JuloVeloObject(adata::Muon.AnnData, root::Union{AbstractString, Int}; clusters = "leiden", celltype = "clusters", basis = "umap", normalized::Bool = true)
        data = new()
        u, s, genes, pseudotime, clusters, embedding, celltype = load_data(adata; clusters, celltype, basis, normalized)
        root = typeof(root) == String ? root : string(root)
    
        # Data
        data.X = nothing
        data.c = nothing
        data.u = u
        data.s = s
        data.train_c = nothing
        data.train_u = nothing
        data.train_s = nothing
        data.temp_c = nothing
        data.temp_u = nothing
        data.temp_s = nothing
        # Training data
        data.train_X = nothing
        data.gene_kinetics = nothing
        # Gene
        data.ngenes = size(u, 1)
        data.genes = genes
        data.temp_genes = nothing
        data.train_genes = nothing
        data.train_genes_number = nothing
        data.bad_correlation_genes = nothing
        data.bad_kinetics_genes = nothing
        # Cell characteristic
        data.ncells = size(u, 2)
        data.pseudotime = pseudotime
        data.clusters = clusters
        data.root = root
        data.embedding = embedding
        data.celltype = celltype
        data.datatype = "gex"
        
        data.param = Dict{String, Any}()
    
        return data
    end
    
    function JuloVeloObject(adata_rna::Muon.AnnData, adata_atac::Muon.AnnData, root::Union{AbstractString, Int}; clusters = "leiden", celltype = "clusters", embedding = "umap", normalized::Bool = true)
        data = new()
        c, u, s, genes, pseudotime, clusters, embedding, celltype = load_data(adata_rna, adata_atac; clusters, celltype, basis, normalized)
        root = typeof(root) == String ? root : string(root)
        
         # Data
        data.X = nothing
        data.c = c
        data.u = u
        data.s = s
        data.train_c = nothing
        data.train_u = nothing
        data.train_s = nothing
        data.temp_c = nothing
        data.temp_u = nothing
        data.temp_s = nothing
        # Training data
        data.train_X = nothing
        data.gene_kinetics = nothing
        # Gene
        data.ngenes = size(u, 1)
        data.genes = genes
        data.temp_genes = nothing
        data.train_genes = nothing
        data.train_genes_number = nothing
        data.bad_correlation_genes = nothing
        data.bad_kinetics_genes = nothing
        # Cell characteristic
        data.ncells = size(u, 2)
        data.pseudotime = pseudotime
        data.clusters = clusters
        data.root = root
        data.embedding = embedding
        data.celltype = celltype
        data.datatype = "multi"
        
        data.param = Dict{String, Any}()
    
        return data
    end
end

function load_data(datapath::AbstractString; datatype::AbstractString = "gex", normalized::Bool = true)
    # Load RNA from CSV
    mu = CSV.read(joinpath(datapath, "Mu.csv"), DataFrame)
    ms = CSV.read(joinpath(datapath, "Ms.csv"), DataFrame)
    # Load pseudotime from pickle
    pseudotime = Float32.(Pickle.load(joinpath(datapath, "pseudotime.pkl")))
    # Load leiden from pickle
    clusters = String.(Pickle.load(joinpath(datapath, "clusters.pkl")))
    
    # Extract gene list
    genes = String.(mu[!, 1])
    
    # Convert dataframe ot matrix
    u = Float32.(Matrix(mu[!, 2:end]))
    s = Float32.(Matrix(ms[!, 2:end]))
    
    # Matrix normalized
    if normalized
        u = mapslices(x -> x ./ (maximum(x) + eps(Float32)), u, dims = 2)
        s = mapslices(x -> x ./ (maximum(x) + eps(Float32)), s, dims = 2)
    end
    
    # chromatin for multi datatype
    if lowercase(datatype) == "multi"
        mc = CSV.read(joinpath(datapath, "Mc.csv"), DataFrame)
        c = Float32.(Matrix(mc[!, 2:end]))
        if normalized
            c = mapslices(x -> x ./ (maximum(x) + eps(Float32)), c, dims = 2)
        end
    end
    
    # Load embedding from pickle
    if isfile(joinpath(datapath, "embedding.pkl"))
        embedding = Pickle.npyload(joinpath(datapath, "embedding.pkl"))
        M = nothing
    else
        @info "Not providing embedding information, construct PCA embbeding."
        gene_space = vcat(u, s)
        M = MultivariateStats.fit(PCA, gene_space, maxoutdim = 2)
        embedding = permutedims(MultivariateStats.predict(M, gene_space), (2, 1))
    end
    
    # Load cellt type from pickle
    if isfile(joinpath(datapath, "celltype.pkl"))
        celltype = String.(Pickle.npyload(joinpath(datapath, "celltype.pkl")))
    else
        @info "Not providing celltype information."
        celltype = nothing
    end
    
    # Return
    if lowercase(datatype) == "gex"
        return u, s, genes, pseudotime, clusters, embedding, celltype, M
    else
        return c, u, s, genes, pseudotime, clusters, embedding, celltype, M
    end
end

function load_data(adata::Muon.AnnData; clusters = "leiden", celltype = "clusters", basis = "umap", normalized::Bool = true)
    u = adata.layers["Mu"]'
    s = adata.layers["Ms"]'
    genes = Array{AbstractString}(adata.var_names)
    pseudotime = adata.obs[!, "dpt_pseudotime"]
    cls = Array{AbstractString}(adata.obs[!, clusters])
    embedding = adata.obsm["X_$basis"]
    cts = Array{AbstractString}(adata.obs[!, celltype])
    
    if normalized
        u = mapslices(x -> x ./ (maximum(x) + eps(Float32)), u, dims = 2)
        s = mapslices(x -> x ./ (maximum(x) + eps(Float32)), s, dims = 2)
    end
    
    return u, s, genes, pseudotime, cls, embedding, cts
end

function load_data(adata_rna::Muon.AnnData, adata_atac::Muon.AnnData; clusters = "leiden", celltype = "clusters", basis = "umap", normalized::Bool = true)
    u = adata_rna.layers["Mu"]'
    s = adata_rna.layers["Ms"]'
    genes = Array{AbstractString}(adata_rna.var_names)
    pseudotime = adata_rna.obs[!, "dpt_pseudotime"]
    cls = Array{AbstractString}(adata_rna.obs[!, clusters])
    embedding = adata_rna.obsm["X_$basis"]
    cts = Array{AbstractString}(adata_rna.obs[!, celltype])
    
    if normalized
        u = mapslices(x -> x ./ (maximum(x) + eps(Float32)), u, dims = 2)
        s = mapslices(x -> x ./ (maximum(x) + eps(Float32)), s, dims = 2)
    end
    
    c = adata_atac.X'
    
    if normalized
        c = mapslices(x -> x ./ (maximum(x) + eps(Float32)), c, dims = 2)
    end
    
    return c, u, s, genes, pseudotime, cls, embedding, cts
end

function reshape_data(data::JuloVeloObject)
    if isnothing(data.train_u)
        @info "train_u is empty, define gene kinetics first"
        define_gene_kinetic(data)
    end
    
    u = data.train_u
    s = data.train_s
    train_genes_number = data.train_genes_number
    ncells = data.ncells
    
    X = vcat(
        permutedims(reshape(u', ncells, 1, train_genes_number), (2, 1, 3)),
        permutedims(reshape(s', ncells, 1, train_genes_number), (2, 1, 3))
    )
    
    data.X = X
    
    @info "data is reshaped to $(size(X))"
    @info "$(train_genes_number) genes are used in velocity estimation"
    
    return data
end
