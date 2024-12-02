"""
TODO: Add docsstrings
"""
function write_adata(adata::Muon.AnnData; filename::AbstractString = "JuloVelo", basis::AbstractString = "umap")
    if haskey(adata.obsm, "velocity_$basis")
        if any(ismissing.(adata.obsm["velocity_$basis"]))
            adata.obsm["velocity_$basis"] = replace(adata.obsm["velocity_$basis"], missing => NaN)
        end
    end
    
    if endswith(filename, "h5ad") | endswith(filename, "h5")
        writeh5ad("$filename", adata)
    else
        writeh5ad("$filename.h5ad", adata)
    end

    return nothing
end

"""
TODO: Add docsstrings
"""
function read_adata(adatapath::AbstractString)
    return readh5ad(adatapath)
end

"""
TODO: Add docsstrings
"""
function merge_multiome_adata(adata_rna::Muon.AnnData, adata_atac::Muon.AnnData)
    c = adata_atac.layers["norm_c"]
    adata_rna.layers["norm_c"] = c
    
    return adata_rna, adata_atac
end

"""
TODO: Add docsstrings
"""
function reshape_data(adata::Muon.AnnData)
    # Check if gene are pre-determined kinetics
    if ~("train_genes" in names(adata.var))
        @info "Could not find \"train_genes\" in adata.var"
        @info "Please use gene_kinetics_predetermination() first"
        return adata
    end
    
    # Extract data
    u = permutedims(adata.layers["norm_u"], (2, 1))
    s = permutedims(adata.layers["norm_s"], (2, 1))
    train_gene_index = adata.var[!, "train_genes"]
    
    # Get training matrix
    train_u = u[train_gene_index, :]
    train_s = s[train_gene_index, :]
    
    # Get number of cells and genes
    ngenes, ncells = size(train_u)
    
    # Reshape data
    X = vcat(
        permutedims(reshape(train_u', ncells, 1, ngenes), (2, 1, 3)),
        permutedims(reshape(train_s', ncells, 1, ngenes), (2, 1, 3))
    )
    
    # Write to anndata
    adata.uns["X"] = X
    
    # Info for reshaped data
    @info "Data is reshaped to $(size(X))"
    
    return adata
end
