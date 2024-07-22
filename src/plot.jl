function plot_kinetics_embedding(adata::Muon.AnnData;
        label = "clusters",
        basis::AbstractString = "pca", 
        min_dist::AbstractFloat = 0.5, 
        n_neighbors::Int = 50, 
        figsize::Tuple{Int, Int} = (1800, 1000),
        left_margins = 1.0Plots.cm,
        bottom_margins = 1.0Plots.cm,
        xlabel::AbstractString = "PC1",
        ylabel::AbstractString = "PC2",
        legendfontsize::Int = 16,
        xguidefontsize::Int = 18, 
        yguidefontsize::Int = 18,
        dpi::Int = 300)

    theme(:vibrant, framestyle = :axes, grid = true, markersize = 3, linewidth = 1.4, palette = :tab20) 

    if ~haskey(adata.obsm, "kinetics_$basis")
        kinetics_embedding(adata; basis = basis, min_dist = min_dist, n_neighbors = n_neighbors)
    else
        kinetics_embedding = adata.obsm["kinetics_$basis"]
    end

    celltype = label in names(adata.obs) ? adata.obs[!, label] : adata.obs[!, "leiden"]
    
    labels = sort(unique(celltype))
    p = scatter(size = figsize, fg_legend = :transparent, dpi = dpi) #
    for label in labels
        scatter!(kinetics_embedding[celltype .== label, 1], 
        kinetics_embedding[celltype .== label, 2], 
        label = label, 
        legend=:outerright, 
        legendfontsize = legendfontsize, 
        xlabel = xlabel, 
        ylabel = ylabel,
        xticks = nothing,
        yticks = nothing,
        left_margins = left_margins,
        bottom_margins = bottom_margins,
        xguidefontsize = xguidefontsize, 
        yguidefontsize = yguidefontsize,
        )
    end
    
    return p
end
