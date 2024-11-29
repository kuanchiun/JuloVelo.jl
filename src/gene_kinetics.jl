function gene_kinetics_predetermination(adata::Muon.AnnData; 
    pseudotime::AbstractString = "dpt_pseudotime", 
    clusters::AbstractString = "leiden", 
    root::AbstractString = "root", 
    cluster_correlation_criteria::AbstractFloat = 0.3f0, 
    pseudotime_correlation_criteria::AbstractFloat = 0.2f0,
    overwrite::Bool = false)

    # Check if genes are filtered
    if ~("bad_correlation_genes" in names(adata.var))
        @info "Could not find \"bad_correlation_genes\" in adata.var"
        @info "Please use filter_genes() first"
        return adata
    end
    # Check if genes are already pre-determine kinetics
    if "bad_kinetics_genes" in names(adata.var)
        if overwrite
            @info "Warning! Overwrite bad_kinetics_genes in adata.var"
        else
            @info "Already pre-determine gene kinetics"
            @info "Use overwrite = true to pre-determine gene kinetics again if you want"
            return adata
        end
    end

    # Extract data
    u = permutedims(adata.layers["norm_u"], (2, 1))
    s = permutedims(adata.layers["norm_s"], (2, 1))
    pseudotime = adata.obs[!, pseudotime]
    clusters = adata.obs[!, clusters]
    root = string(adata.uns[root])

    # Extract gene information
    all_genes = Array{AbstractString}(adata.var_names)
    bad_correlation_index = adata.var[!, "bad_correlation_genes"]

    # Pre-determine gene kinetics
    gene_kinetics = Array{String}(undef, 0)
    bad_kinetics_index = Array{Bool}(undef, 0)
    train_genes_index = Array{Bool}(undef, 0)

    for i in axes(all_genes, 1)
        if bad_correlation_index[i]
            push!(bad_kinetics_index, false)
            push!(train_genes_index, false)
            push!(gene_kinetics, "/")
        else
            kinetics = gene_kinetics_predetermination!(u[i, :], s[i, :], pseudotime, clusters, root; 
                cluster_correlation_criteria = cluster_correlation_criteria, 
                pseudotime_correlation_criteria = pseudotime_correlation_criteria)
            
            if kinetics != "fail"
                push!(bad_kinetics_index, false)
                push!(train_genes_index, true)
                push!(gene_kinetics, kinetics)
            else
                push!(bad_kinetics_index, true)
                push!(train_genes_index, false)
                push!(gene_kinetics, "/")
            end
        end
    end

    # Write to anndata
    adata.var[!, "bad_kinetics_genes"] = bad_kinetics_index
    adata.var[!, "train_genes"] = train_genes_index
    adata.var[!, "gene_kinetics"] = gene_kinetics

    # info for bad kinetics genes and training genes
    @info "$(sum(bad_kinetics_index)) genes are filtered due to bad kinetics"
    @info "See adata.var[!, \"bad_kinetics_genes\"] for details"
    @info "$(sum(train_genes_index)) genes are used to infer velocity"
    @info "See adata.var[!, \"train_genes\"] for details"

    return adata
end

function gene_kinetics_predetermination(adata_rna::Muon.AnnData, adata_atac::Muon.AnnData; 
    pseudotime::AbstractString = "dpt_pseudotime", 
    clusters::AbstractString = "leiden", 
    root::AbstractString = "root", 
    cluster_correlation_criteria::AbstractFloat = 0.3f0, 
    pseudotime_correlation_criteria::AbstractFloat = 0.2f0,
    chromatin_slope_criteria::AbstractFloat = 0.2f0, 
    chromatin_intercept_criteria::AbstractFloat = 0.2f0,
    overwrite::Bool = false)

    # Check if genes are filtered
    if ~("bad_correlation_genes" in names(adata_rna.var))
        @info "Could not find \"bad_correlation_genes\" in adata_rna.var"
        @info "Please use filter_genes() first"
        return adata_rna
    end
    # Check if genes are already pre-determine kinetics
    if "bad_kinetics_genes" in names(adata_rna.var)
        if overwrite
            @info "Warning! Overwrite bad_kinetics_genes in adata_rna.var"
        else
            @info "Already pre-determine gene kinetics"
            @info "Use overwrite = true to pre-determine gene kinetics again if you want"
            return adata_rna
        end
    end

    # Extract data
    c = permutedims(adata_atac.layers["norm_c"], (2, 1))
    u = permutedims(adata_rna.layers["norm_u"], (2, 1))
    s = permutedims(adata_rna.layers["norm_s"], (2, 1))
    pseudotime = adata_rna.obs[!, pseudotime]
    clusters = adata_rna.obs[!, clusters]
    root = string(adata_rna.uns[root])

    # Extract gene information
    all_genes = Array{AbstractString}(adata_rna.var_names)
    bad_correlation_index = adata_rna.var[!, "bad_correlation_genes"]

    # Pre-determine gene kinetics
    gene_kinetics = Array{String}(undef, 0)
    bad_kinetics_index = Array{Bool}(undef, 0)
    train_genes_index = Array{Bool}(undef, 0)

    for i in axes(all_genes, 1)
        if bad_correlation_index[i]
            push!(bad_kinetics_index, false)
            push!(train_genes_index, false)
            push!(gene_kinetics, "/")
        else
            kinetics = gene_kinetics_predetermination!(c[i, :], u[i, :], s[i, :], pseudotime, clusters, root; 
                cluster_correlation_criteria = cluster_correlation_criteria, 
                pseudotime_correlation_criteria = pseudotime_correlation_criteria,
                chromatin_slope_criteria = chromatin_slope_criteria, 
                chromatin_intercept_criteria = chromatin_intercept_criteria)
            
            if kinetics != "fail"
                push!(bad_kinetics_index, false)
                push!(train_genes_index, true)
                push!(gene_kinetics, kinetics)
            else
                push!(bad_kinetics_index, true)
                push!(train_genes_index, false)
                push!(gene_kinetics, "/")
            end
        end
    end

    # Write to anndata
    adata_rna.var[!, "bad_kinetics_genes"] = bad_kinetics_index
    adata_rna.var[!, "train_genes"] = train_genes_index
    adata_rna.var[!, "gene_kinetics"] = gene_kinetics

    # info for bad kinetics genes and training genes
    @info "$(sum(bad_kinetics_index)) genes are filtered due to bad kinetics"
    @info "See adata_rna.var[!, \"bad_kinetics_genes\"] for details"
    @info "$(sum(train_genes_index)) genes are used to infer velocity"
    @info "See adata_rna.var[!, \"train_genes\"] for details"

    return adata_rna
end

function gene_kinetics_predetermination!(u::AbstractVector, s::AbstractVector, pseudotime::AbstractVector, clusters::AbstractVector, root::AbstractString; 
    cluster_correlation_criteria::AbstractFloat = 0.3f0, 
    pseudotime_correlation_criteria::AbstractFloat = 0.2f0)
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
        if abs(us) < cluster_correlation_criteria
            continue
        end
        
        # Filter if pseudotime has non-sigificant correlation with unspliced and spliced
        if abs(tu[2]) < pseudotime_correlation_criteria || abs(ts[2]) < pseudotime_correlation_criteria || tu[2]/ts[2] < 0
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

function gene_kinetics_predetermination!(c::AbstractVector, u::AbstractVector, s::AbstractVector, pseudotime::AbstractVector, clusters::AbstractVector, root::AbstractString; 
    cluster_correlation_criteria::AbstractFloat = 0.3f0, 
    pseudotime_correlation_criteria::AbstractFloat = 0.2f0, 
    chromatin_slope_criteria::AbstractFloat = 0.2f0, 
    chromatin_intercept_criteria::AbstractFloat = 0.2f0)
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
        if abs(us) < cluster_correlation_criteria
            continue
        end
        
        # Filter slope or intercept > 0.2 in linear regression between chromatin and unspliced RNA 
        if cu[1] < chromatin_slope_criteria && cu[2] < chromatin_intercept_criteria
            continue
        end
        
        # Filter if pseudotime has non-sigificant correlation with unspliced and spliced
        if abs(tu[2]) < pseudotime_correlation_criteria || abs(ts[2]) < pseudotime_correlation_criteria || tu[2]/ts[2] < 0
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
