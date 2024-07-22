function build_velocity_model(adata::Muon.AnnData)
    # Check if gene are pre-determined kinetics
    if ~("train_genes" in names(adata.var))
        @info "Could not find \"train_genes\" in adata.var"
        @info "Please use gene_kinetics_predetermination() first"
        return adata
    end
    
    # Set pre-trained model path
    modelpath = joinpath(pkgdir(JuloVelo), "models")
    
    # Load gene kinetics
    gene_kinetics = deepcopy(adata.var[!, "gene_kinetics"])
    gene_kinetics = filter!(x -> x != "/", gene_kinetics)
    
    
    # Load pre-trained model
    BSON.@load joinpath(modelpath, "Circle.bson") circle
    BSON.@load joinpath(modelpath, "Induction.bson") Induction
    BSON.@load joinpath(modelpath, "Repression.bson") Repression
    
    # Initialize model structure
    l1_weight = Array{Float32}(undef, 100, 2, 0)
    l2_weight = Array{Float32}(undef, 100, 100, 0)
    l3_weight = Array{Float32}(undef, 3, 100, 0)
    l1_bias = Array{Float32}(undef, 100, 1, 0)
    l2_bias = Array{Float32}(undef, 100, 1, 0)
    l3_bias = Array{Float32}(undef, 3, 1, 0)
    
    # Assign pre-trained model weight and bias for each gene
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
    
    # Create model
    Kinetics = Chain(
        l1 = SpatialDense(l1_weight, l1_bias, leakyrelu),
        l2 = SpatialDense(l2_weight, l2_bias, leakyrelu),
        l3 = SpatialDense(l3_weight, l3_bias, sigmoid)
    )
    
    # Info for model
    @info "Successfully initialize velocity model"
    
    return Kinetics
end

function optimizer_setting(Kinetics::Chain, learning_rate::AbstractFloat, optimizer::AbstractString)
    if lowercase(optimizer) == "adam"
        rule = Optimisers.Adam(learning_rate)
        @info "Training with Adam Optimizer"
    elseif lowercase(optimizer) == "radam"
        rule = Optimisers.RAdam(learning_rate)
        @info "Training with RAdam Optimizer"
    else
        throw(ArgumentError("Optimizer only supports adam and radam"))
    end
    
    opt_state = Optimisers.setup(rule, Kinetics)
    
    return opt_state
end

function train(adata::Muon.AnnData, Kinetics::Chain; 
    epochs::Int = 100, 
    neighbor_number::Int = 30, 
    learning_rate::AbstractFloat = 0.0001f0,
    optimizer::AbstractString = "adam", 
    λ::AbstractFloat = 0.004f0, 
    dt::AbstractFloat = 0.5f0, 
    logger::Bool = true, 
    checktime::Int = 5, 
    use_gpu::Bool = true)

    # Extract data
    train_X = adata.uns["train_X"]
    _, sample_number, train_gene_number = size(train_X)

    # Create neighbor vector
    train_neighbor_vector = calculate_neighbor_vector(train_X, train_gene_number, sample_number; neighbor_number = neighbor_number)
        
    # Send data to device
    train_X, Kinetics, train_neighbor_vector = to_device(train_X, Kinetics, train_neighbor_vector; use_gpu = use_gpu)

    # Set optimizer state
    opt_state = optimizer_setting(Kinetics, learning_rate, optimizer)

    # Set logger
    if logger
        savepath = joinpath(pwd(), "saves/")
        tblogger = TBLogger(savepath, tb_overwrite)
        set_step_increment!(tblogger, 0)
        @info "TensorBoard logging at \"$(savepath)\""
    end

    # Train
    @info "Start training"
    for epoch in 0:epochs
        if epoch == 0
            report(epoch, train_X, Kinetics, train_neighbor_vector, train_gene_number, sample_number; 
                neighbor_number = neighbor_number, λ = λ, dt = dt)
            continue
        end
        
        gs = Zygote.gradient(m -> eval_loss(train_X, m, train_neighbor_vector, sample_number;
                neighbor_number = neighbor_number, λ = λ, dt = dt), Kinetics)
        opt_state, Kinetics = Optimisers.update(opt_state, Kinetics, gs[1])
        
        train_loss = report(epoch, train_X, Kinetics, train_neighbor_vector, train_gene_number, sample_number; 
            neighbor_number = neighbor_number, λ = λ, dt = dt)
        
        if logger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" cos_loss=train_loss.cos_loss penalty=train_loss.penalty AvgMeanCos=train_loss.AvgMeanCos
            end
        end
        
        if epoch % checktime == 0
            let Kinetics = cpu(Kinetics)
                BSON.@save joinpath(pwd(), "saves/Kinetics.bson") Kinetics
            end
            @info "Model is saved in saves/Kinetics.bson"
        end
        
        train_loss = nothing
        gs = nothing
        GC.gc(true)
        if use_gpu
            CUDA.reclaim()
        end
    end

    Kinetics = Kinetics |> cpu

    return Kinetics
end

function forward(X::AbstractArray, Kinetics::Chain; dt::AbstractFloat = 0.05f0)
    # Split data to unsplice and splice
    u = X[1, :, :]
    s = X[2, :, :]

    # Predict kinetic
    kinetics = Kinetics(X)

    # Estimate du, ds
    du, ds = kinetics_equation(u, s, kinetics)

    # Multiply by dt
    du = du .* dt
    ds = ds .* dt
    
    # Reshape du, ds and create dt_vector
    du = reshape(du, 1, 1, :)
    ds = reshape(ds, 1, 1, :)
    dt_vector = cat(du, ds, dims = 1) .+ eps(Float32)
    
    return dt_vector
end

function eval_loss(X::AbstractArray, Kinetics::Chain, neighbor_vector::AbstractArray, sample_number::Int;
        neighbor_number::Int = 30, λ::AbstractFloat = 0.004f0, dt::AbstractFloat = 0.5f0)
    
    # Predict du, ds
    dt_vector = forward(X, Kinetics)
    
    # Calculate cosine similarity loss
    cos_loss = cosine_loss(neighbor_vector, dt_vector, sample_number; neighbor_number = neighbor_number)
    
    # L2 penalty
    penalty = (l2_penalty(Kinetics.layers.l1.weight) + 
    l2_penalty(Kinetics.layers.l2.weight) + 
    l2_penalty(Kinetics.layers.l3.weight)) * λ
    
    
    # loss
    loss = cos_loss + penalty
    #loss = cos_loss
    
    return loss
end

function eval_loss_report(X::AbstractArray, Kinetics::Chain, neighbor_vector::AbstractArray, train_gene_number::Int, sample_number::Int;
        neighbor_number::Int = 30, λ::AbstractFloat = 0.004f0, dt::AbstractFloat = 0.5f0)
    
    # Predict du, ds
    dt_vector = forward(X, Kinetics)
    
    # Calculate cosine similarity loss
    cos_loss = cosine_loss(neighbor_vector, dt_vector, sample_number; neighbor_number = neighbor_number)
    
    # Calculate average cosine similarity
    AvgMeanCos = 1 - (cos_loss / (train_gene_number * sample_number))
    
    # L2 penalty
    penalty = (l2_penalty(Kinetics.layers.l1.weight) + 
    l2_penalty(Kinetics.layers.l2.weight) + 
    l2_penalty(Kinetics.layers.l3.weight)) * λ
    
    return (cos_loss = cos_loss |> round4, penalty = penalty |> round4, AvgMeanCos = AvgMeanCos |> round4)
end

function report(epoch::Int, train_X::AbstractArray, Kinetics::Chain, train_neighbor_vector::AbstractArray, train_gene_number::Int, sample_number::Int;
        neighbor_number::Int = 30, λ::AbstractFloat = 0.004f0, dt::AbstractFloat = 0.5f0)
    
    train = eval_loss_report(train_X, Kinetics, train_neighbor_vector, train_gene_number, sample_number;
        neighbor_number = neighbor_number, λ = λ, dt = dt)
    
    println("Epoch: $epoch  Train: $(train)")
    return train
end
