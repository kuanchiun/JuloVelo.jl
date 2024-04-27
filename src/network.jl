"""
    SpatialDense(in => out, depth, σ=identity; bias=true, init=glorot_uniform)

Create a 3D fully connected layer, whose forward pass is given by:

    y = σ.(W * x .+ bias)
"""
function create_bias_3D(weights::AbstractArray, bias::Bool, dims::Tuple)
    bias ? fill!(similar(weights, dims), 0) : false
end

function create_bias_3D(weights::AbstractArray, bias::AbstractArray, dims::Tuple)
    size(bias) == dims || throw(DimensionMismatch("expect bias of size $(dims), got size $(size(bias))"))
    convert(AbstractArray{eltype(weights)}, bias)
end

struct SpatialDense{F, M<:AbstractArray, B}
    weight::M
    bias::B
    σ::F
    function SpatialDense(W::M, bias = true, σ::F = identity) where {M<:AbstractArray, F}
            b = create_bias_3D(W, bias, (size(W, 1), 1, size(W, 3)))
            new{F, M, typeof(b)}(W, b, σ)
    end
end

function SpatialDense((in, out)::Pair{<:Integer, <:Integer}, depth::Integer, σ = identity; init = Flux.glorot_uniform, bias = true)
    SpatialDense(init(out, in, depth), bias, σ)
end

Flux.@functor SpatialDense

function (l::SpatialDense)(x)
    σ = NNlib.fast_act(l.σ, x)
    xT = Flux._match_eltype(l, x)
    output = NNlib.batched_mul(l.weight, xT)
    output = σ.(output .+ l.bias)
    
    return output
end

function Base.show(io::IO, l::SpatialDense)
    print(io, "SpatialDense(", size(l.weight, 2), " => ", size(l.weight, 1))
    print(io, ", depth = ", size(l.weight, 3))
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == false && print(io, "; bias=false")
    print(io, ")")
end

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
        
        if cu[1] > 0.2f0 || cu[2] > 0.2f0 # slope or intercept > 0.2 in linear regression between chromatin and unspliced RNA 
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

function build_velocity_model(data::JuloVeloObject)
    modelpath = "../models/"
    gene_kinetics = data.gene_kinetics
    
    BSON.@load joinpath(modelpath, "Circle.bson") circle
    BSON.@load joinpath(modelpath, "Induction.bson") Induction
    BSON.@load joinpath(modelpath, "Repression.bson") Repression
    
    l1_weight = Array{Float32}(undef, 100, 2, 0)
    l2_weight = Array{Float32}(undef, 100, 100, 0)
    l3_weight = Array{Float32}(undef, 3, 100, 0)
    l1_bias = Array{Float32}(undef, 100, 1, 0)
    l2_bias = Array{Float32}(undef, 100, 1, 0)
    l3_bias = Array{Float32}(undef, 3, 1, 0)
    
    for kinetics in gene_kinetics
        if kinetics == "circle"
            l1_weight = cat(l1_weight, circle.layers.l1.weight, dims = 3)
            l2_weight = cat(l2_weight, circle.layers.l2.weight, dims = 3)
            l3_weight = cat(l3_weight, circle.layers.l3.weight, dims = 3)
            l1_bias = cat(l1_bias, circle.layers.l1.bias, dims = 3)
            l2_bias = cat(l2_bias, circle.layers.l2.bias, dims = 3)
            l3_bias = cat(l3_bias, circle.layers.l3.bias, dims = 3)
        elseif kinetics == "induction"
            l1_weight = cat(l1_weight, Induction.layers.l1.weight, dims = 3)
            l2_weight = cat(l2_weight, Induction.layers.l2.weight, dims = 3)
            l3_weight = cat(l3_weight, Induction.layers.l3.weight, dims = 3)
            l1_bias = cat(l1_bias, Induction.layers.l1.bias, dims = 3)
            l2_bias = cat(l2_bias, Induction.layers.l2.bias, dims = 3)
            l3_bias = cat(l3_bias, Induction.layers.l3.bias, dims = 3)
        elseif kinetics == "repression"
            l1_weight = cat(l1_weight, Repression.layers.l1.weight, dims = 3)
            l2_weight = cat(l2_weight, Repression.layers.l2.weight, dims = 3)
            l3_weight = cat(l3_weight, Repression.layers.l3.weight, dims = 3)
            l1_bias = cat(l1_bias, Repression.layers.l1.bias, dims = 3)
            l2_bias = cat(l2_bias, Repression.layers.l2.bias, dims = 3)
            l3_bias = cat(l3_bias, Repression.layers.l3.bias, dims = 3)
        end
    end
    
    Kinetic = Chain(
        l1 = SpatialDense(l1_weight, l1_bias, leakyrelu),
        l2 = SpatialDense(l2_weight, l2_bias, leakyrelu),
        l3 = SpatialDense(l3_weight, l3_bias, sigmoid)
    )
    
    data.param["velocity_model"] = Kinetic
    
    @info "Successfully initialize velocity model, see param.velocity_model for details"
    
    return data
end

function optimizer_setting(Kinetic::Chain, learning_rate::AbstractFloat, optimizer::AbstractString)
    if lowercase(optimizer) == "adam"
        rule = Optimisers.Adam(learning_rate)
        @info "Training with Adam Optimizer"
    elseif lowercase(optimizer) == "radam"
        rule = Optimisers.RAdam(learning_rate)
        @info "Training with RAdam Optimizer"
    else
        throw(ArgumentError("Optimizer only support adam and radam"))
    end
    
    opt_state = Optimisers.setup(rule, Kinetic)
    
    return opt_state
end

function train(data::JuloVeloObject; epochs::Int = 100, sample_number::Int = 2400, neighbor_number::Int = 30, learning_rate::AbstractFloat = 0.001, optimizer = "adam", λ::AbstractFloat = 0.004, dt::AbstractFloat = 0.5f0, logger = true, checktime::Int = 5)
    train_X = data.train_X
    ncells = data.ncells
    train_genes_number = data.train_genes_number
    Kinetic = data.param["velocity_model"]
    
    train_neighbor_vector = calculate_neighbor_vector(train_X, train_genes_number, sample_number, neighbor_number)
    train_X, Kinetic, train_neighbor_vector = to_device(train_X, Kinetic, train_neighbor_vector)
    opt_state = optimizer_setting(Kinetic, learning_rate, optimizer)
    
    if logger
        savepath = joinpath(@__DIR__, "saves/")
        tblogger = TBLogger(savepath, tb_overwrite)
        set_step_increment!(tblogger, 0)
        @info "TensorBoard logging at \"$(savepath)\""
    end
    
    @info "Start training"
    for epoch in 0:epochs
        if epoch == 0
            report(epoch, train_X, Kinetic, train_neighbor_vector, train_genes_number, neighbor_number, sample_number, λ, dt)
            continue
        end
        
        gs = Zygote.gradient(m -> eval_loss(train_X, m, train_neighbor_vector, neighbor_number, sample_number, λ, dt), Kinetic)
        opt_state, Kinetic = Optimisers.update(opt_state, Kinetic, gs[1])
        
        train_loss = report(epoch, train_X, Kinetic, train_neighbor_vector, train_genes_number, neighbor_number, sample_number, λ, dt)
        
        if logger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" cos_loss=train_loss.cos_loss penalty=train_loss.penalty AvgMeanCos=train_loss.AvgMeanCos
            end
        end
        
        if epoch % checktime == 0
            let Kinetic = cpu(Kinetic)
                data.param["velocity_model"] = Kinetic
            end
            @info "Model is updated in .param.velocity_model"
        end
        
        train_loss = nothing
        test_loss = nothing
        gs = nothing
        GC.gc(true)
        CUDA.reclaim()
    end
end

function velocity_estimation(data::JuloVeloObject; dt::AbstractFloat = 0.5f0)
    ncells = data.ncells
    ngenes = data.train_genes_number
    u = data.train_u'
    s = data.train_s'
    X = data.X
    Kinetic = data.param["velocity_model"]
    
    kinetic = Kinetic(X)
    
    du, ds = kinetic_equation(u, s, kinetic)
    
    # Multiply by dt
    du = du .* dt
    ds = ds .* dt
    
    # Reshape du, ds and create dt_vector
    û = permutedims(reshape(du, ncells, 1, ngenes), (2, 1, 3))
    ŝ = permutedims(reshape(ds, ncells, 1, ngenes), (2, 1, 3))
    X̂ = vcat(û, ŝ) .+ eps(Float32)
    
    data.param["velocity"] = X̂
    
    @info "Velocity saved in param.velocity"
    return data
end  