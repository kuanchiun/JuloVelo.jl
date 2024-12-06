"""
TODO: Add docsstrings
"""
function assign_velocity_model(gene_kinetics::AbstractString)
    #modelpath = joinpath(pkgdir(JuloVelo), "models")

    BSON.@load joinpath(joinpath(PROJECT_PATH, "models"), "Circle.bson") circle
    BSON.@load joinpath(joinpath(PROJECT_PATH, "models"), "Induction.bson") Induction
    BSON.@load joinpath(joinpath(PROJECT_PATH, "models"), "Repression.bson") Repression

    if gene_kinetics == "circle"
        return circle
    elseif gene_kinetics == "induction"
        return Induction
    elseif gene_kinetics == "repression"
        return Repression
    else
        throw(ArgumentError("Error gene kinetics detected."))
    end
end

"""
TODO: Add docsstrings
"""
function optimizer_setting(Kinetics::Chain, learning_rate::AbstractFloat, optimizer::AbstractString)
    if lowercase(optimizer) == "adam"
        rule = Optimisers.Adam(learning_rate)
    elseif lowercase(optimizer) == "radam"
        rule = Optimisers.RAdam(learning_rate)
    else
        throw(ArgumentError("Optimizer only supports adam and radam"))
    end
    
    opt_state = Optimisers.setup(rule, Kinetics)
    
    return opt_state
end

"""
TODO: Add docsstrings
"""
function train(adata;
    epochs::Int = 100,
    neighbor_number::Int = 30,
    learning_rate::AbstractFloat = 0.0001f0,
    optimizer::AbstractString = "adam",
    λ::AbstractFloat = 0.004f0,
    dt::AbstractFloat = 0.5f0,
    use_gpu::Bool = true)

    # Extract data
    train_X = adata.uns["train_X"]
    _, sample_number, train_gene_number = size(train_X)
    gene_kinetics = deepcopy(adata.var[!, "gene_kinetics"])
    gene_kinetics = filter!(x -> x != "/", gene_kinetics)

    l1_weight = Array{Float32}(undef, 100, 2, 0)
    l2_weight = Array{Float32}(undef, 100, 100, 0)
    l3_weight = Array{Float32}(undef, 3, 100, 0)
    l1_bias = Array{Float32}(undef, 100, 1, 0)
    l2_bias = Array{Float32}(undef, 100, 1, 0)
    l3_bias = Array{Float32}(undef, 3, 1, 0)

    @info "Start training"

    device = gpu_functional(use_gpu)

    if lowercase(optimizer) == "adam"
        @info "Training with Adam Optimizer"
    elseif lowercase(optimizer) == "radam"
        @info "Training with RAdam Optimizer"
    else
        throw(ArgumentError("Optimizer only supports adam and radam"))
    end

    # loop
    @showprogress for i in 1:train_gene_number
        train_X_single = train_X[:, :, i]
        gene_kinetics_single = gene_kinetics[i]
        neighbor_vector_single = calculate_neighbor_vector(train_X_single, sample_number, neighbor_number)
        Kinetics = assign_velocity_model(gene_kinetics_single)

        train_X_single, Kinetics, neighbor_vector_single = to_device(train_X_single, Kinetics, neighbor_vector_single, device)
        opt_state = optimizer_setting(Kinetics, learning_rate, optimizer)

        for epoch in 0:epochs
            gs = Zygote.gradient(m -> eval_loss(train_X_single, m, neighbor_vector_single, sample_number, neighbor_number, λ; dt = dt), Kinetics)
            opt_state, Kinetics = Optimisers.update(opt_state, Kinetics, gs[1])
        end

        Kinetics = Kinetics |> cpu

        l1_weight = cat(l1_weight, Kinetics.layers.l1.weight, dims = 3)
        l2_weight = cat(l2_weight, Kinetics.layers.l2.weight, dims = 3)
        l3_weight = cat(l3_weight, Kinetics.layers.l3.weight, dims = 3)
        l1_bias = cat(l1_bias, Kinetics.layers.l1.bias, dims = 3)
        l2_bias = cat(l2_bias, Kinetics.layers.l2.bias, dims = 3)
        l3_bias = cat(l3_bias, Kinetics.layers.l3.bias, dims = 3)
    end

    Kinetics = Chain(
        l1 = SpatialDense(l1_weight, l1_bias, leakyrelu),
        l2 = SpatialDense(l2_weight, l2_bias, leakyrelu),
        l3 = SpatialDense(l3_weight, l3_bias, sigmoid)
    )

    gs = nothing
        GC.gc(true)
    if use_gpu
        CUDA.reclaim()
    end

    return Kinetics
end

"""
TODO: Add docsstrings
"""
function forward(X::AbstractArray, Kinetics::Chain; dt::AbstractFloat = 0.5f0)
    u = X[1, :]
    s = X[2, :]

    kinetics = Kinetics(X)

    du, ds = kinetics_equation_training(u, s, kinetics)

    du = du .* dt
    ds = ds .* dt

    du = reshape(du, 1, 1, :)
    ds = reshape(ds, 1, 1, :)
    dt_vector = cat(du, ds, dims = 1) .+ eps(Float32)
    
    return dt_vector
end

"""
TODO: Add docsstrings
"""
function eval_loss(X::AbstractArray, Kinetics::Chain, neighbor_vector::AbstractArray, sample_number::Int, neighbor_number::Int, λ::AbstractFloat; dt::AbstractFloat = 0.5f0)
    dt_vector = forward(X, Kinetics; dt = dt)

    cos_loss = cosine_loss(neighbor_vector, dt_vector, sample_number, neighbor_number)

    penalty = (l2_penalty(Kinetics.layers.l1.weight) + 
    l2_penalty(Kinetics.layers.l2.weight) + 
    l2_penalty(Kinetics.layers.l3.weight)) * λ

    loss = cos_loss + penalty

    return loss
end
