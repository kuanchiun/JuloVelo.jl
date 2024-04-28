function kinetics_embedding(data::JuloVeloObject; figuresize::Tuple{<:Int, <:Int} = (1800, 1000))
    if ~haskey(data.param, "kinetics")
        X = data.X
        Kinetics = data.param["velocity_model"]
        kinetics = Kinetics(X)
        data.param["kinetics"] = kinetics
    else
        kinetics = data.param["kinetics"]
    end
    
    if ~haskey(data.param, "kinetics_embedding")
        α = kinetics[1, :, :]'
        β = kinetics[2, :, :]'
        γ = kinetics[3, :, :]'
        embedding = vcat(α, β, γ)
        M = MultivariateStats.fit(PCA, embedding; maxoutdim = 2)
        kinetics_embedding = predict(M, embedding)'
        data.param["kinetics_embedding"] = kinetics_embedding
    else
        kinetics_embedding = data.param["kinetics_embedding"]
    end

    celltype = isnothing(data.celltype) ? data.clusters : data.celltype
    
    labels = Set(celltype)
    p = scatter(size = figuresize, fg_legend = :transparent, axis=nothing) #
    for label in labels
        scatter!(kinetics_embedding[celltype .== label, 1], kinetics_embedding[celltype .== label, 2], label = label, legend=:outerright, legendfontsize = 16, xlabel = "PC1", ylabel = "PC2", margin = 2Plots.cm)
    end
    
    return p
end