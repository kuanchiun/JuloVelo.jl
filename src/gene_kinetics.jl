function define_gene_kinetic(data::JuloVeloObject)
    # Check if genes are filter first
    if isnothing(data.bad_correlation_genes)
        @info "uns.bad_correlation_genes is empty, filter gene first"
        filter_gene(data)
    end
    
    if ~isnothing(data.bad_kinetics_genes)
        @info "Already define kinetics for genes, do nothing"
        return data
    end
    
    c = data.temp_c
    u = data.temp_u
    s = data.temp_s
    genes = data.temp_genes
    pseudotime = data.pseudotime
    clusters = data.clusters
    root = data.root
    datatype = data.datatype
    
    gene_kinetics = Array{String}(undef, 0)
    pass = Array{Bool}(undef, 0)
    fail = Array{Bool}(undef, 0)
    for i in axes(genes, 1)
        if datatype == "gex"
            kinetics = define_gene_kinetic!(u[i, :], s[i, :], pseudotime, clusters, root)
        else
            kinetics = define_gene_kinetic!(c[i, :], u[i, :], s[i, :], pseudotime, clusters, root)
        end
        
        if kinetics == "fail"
            push!(pass, false)
            push!(fail, true)
        else
            push!(pass, true)
            push!(fail, false)
            push!(gene_kinetics, kinetics)
        end
    end
    
    u = u[pass, :]
    s = s[pass, :]
    pass_genes = genes[pass]
    bad_kinetics_genes = genes[fail]
    train_genes_number = length(pass_genes)
    
    data.train_u = u
    data.train_s = s
    data.train_genes = pass_genes
    data.bad_kinetics_genes = bad_kinetics_genes
    data.train_genes_number = train_genes_number
    data.gene_kinetics = gene_kinetics
    
    @info "$(length(data.bad_kinetics_genes)) genes are filtered due to bad kinetics"
    @info "See .bad_kinetics_genes for details"
    
    return data
end

function define_gene_kinetic!(u::AbstractVector, s::AbstractVector, pseudotime::AbstractVector, clusters::AbstractVector, root::String)
    # Cluster condition initialization
    cluster_kinetics = Array{Int}(undef, 0)
    
    for cluster in Set(clusters)
        # Get information for each cluster
        cell_idx = findall(x -> x == cluster, clusters)
        cluster_u = u[cell_idx]
        cluster_s = s[cell_idx]
        cluster_pseudotime = pseudotime[cell_idx]
        
        # Calculate correlation and linear equation
        us = corspearman(cluster_u, cluster_s)
        tu = linear_fit(cluster_pseudotime, cluster_u)
        ts = linear_fit(cluster_pseudotime, cluster_s)
        
        # Filter if unspliced and spliced have non-sigificant correlation
        if abs(us) < 0.3f0
            continue
        end
        
        # Filter if pseudotime has non-sigificant correlation with unspliced and spliced
        if abs(tu[2]) < 0.2f0 || abs(ts[2]) < 0.2f0 || tu[2]/ts[2] < 0
            continue
        end
        
        # Filter if is root cluster
        if cluster == root
            continue
        end
        
        # Determine cluster kinetics
        if tu[2] > 0 # Induction 
            push!(cluster_kinetics, 1)
        elseif tu[2] < 0 # Repression
            push!(cluster_kinetics, -1)
        end
    end
    
    # Check the number of different kinetics in total clusters
    cluster_kinetics = Set(cluster_kinetics)
    
    if length(cluster_kinetics) == 0
        return "fail"
    elseif length(cluster_kinetics) == 2
        return "circle"
    elseif 1 in cluster_kinetics
        return "induction"
    elseif -1 in cluster_kinetics
        return "repression"
    end
end

function define_gene_kinetic!(c::AbstractVector, u::AbstractVector, s::AbstractVector, pseudotime::AbstractVector, clusters::AbstractVector, root::String)
    # Cluster condition initialization
    cluster_kinetics = Array{Int}(undef, 0)
    
    for cluster in Set(clusters)
        # Get information for each cluster
        cell_idx = findall(x -> x == cluster, clusters)
        cluster_c = c[cell_idx]
        cluster_u = u[cell_idx]
        cluster_s = s[cell_idx]
        cluster_pseudotime = pseudotime[cell_idx]
        
        # Calculate correlation and linear equation
        us = corspearman(cluster_u, cluster_s)
        tu = linear_fit(cluster_pseudotime, cluster_u)
        ts = linear_fit(cluster_pseudotime, cluster_s)
        cu = linear_fit(cluster_c, cluster_u)
        
        # Filter if unspliced and spliced have non-sigificant correlation
        if abs(us) < 0.3f0
            continue
        end
        
        if cu[1] < 0.2f0 && cu[2] < 0.2f0 # slope or intercept > 0.2 in linear regression between chromatin and unspliced RNA 
            continue
        end
        
        # Filter if pseudotime has non-sigificant correlation with unspliced and spliced
        if abs(tu[2]) < 0.2f0 || abs(ts[2]) < 0.2f0 || tu[2]/ts[2] < 0
            continue
        end
        
        # Filter if is root cluster
        if cluster == root
            continue
        end
        
        # Determine cluster kinetics
        if tu[2] > 0 # Induction 
            push!(cluster_kinetics, 1)
        elseif tu[2] < 0 # Repression
            push!(cluster_kinetics, -1)
        end
    end
    
    # Check the number of different kinetics in total clusters
    cluster_kinetics = Set(cluster_kinetics)
    
    if length(cluster_kinetics) == 0
        return "fail"
    elseif length(cluster_kinetics) == 2
        return "circle"
    elseif 1 in cluster_kinetics
        return "induction"
    elseif -1 in cluster_kinetics
        return "repression"
    end
end

function kinetics_embedding(data::JuloVeloObject; basis::AbstractString = "pca", min_dist::AbstractFloat = 0.5, n_neighbors::Int = 50)
    if ~haskey(data.param, "kinetics")
        X = data.X
        Kinetics = data.param["velocity_model"]
        kinetics = Kinetics(X)
        data.param["kinetics"] = kinetics
    else
        kinetics = data.param["kinetics"]
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
    data.param["kinetics_embedding"] = kinetics_embedding

    return data
end
