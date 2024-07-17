function build_velocity_model(data::JuloVeloObject)
    modelpath = joinpath(pkgdir(JuloVelo), "models")
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
    use_cuda = CUDA.functional()
    
    train_neighbor_vector = calculate_neighbor_vector(train_X, train_genes_number, sample_number, neighbor_number)
    train_X, Kinetic, train_neighbor_vector = to_device(train_X, Kinetic, train_neighbor_vector)
    opt_state = optimizer_setting(Kinetic, learning_rate, optimizer)
    
    if logger
        savepath = joinpath(pwd(), "saves/")
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
        if use_cuda
            CUDA.reclaim()
        end
    end
end

function forward(X::AbstractArray, Kinetic::Chain; dt = 0.05f0)
    # Split data to unsplice and splice
    u = X[1, :, :]
    s = X[2, :, :]

    # Predict kinetic
    kinetic = Kinetic(X)

    # Estimate du, ds
    du, ds = kinetic_equation(u, s, kinetic)

    # Multiply by dt
    du = du .* dt
    ds = ds .* dt
    
    # Reshape du, ds and create dt_vector
    du = reshape(du, 1, 1, :)
    ds = reshape(ds, 1, 1, :)
    dt_vector = cat(du, ds, dims = 1) .+ eps(Float32)
    
    return dt_vector
end

function eval_loss(X::AbstractArray, Kinetic::Chain, neighbor_vector::AbstractArray, neighbor_number::Int, sample_number::Int, λ::AbstractFloat, dt::AbstractFloat)
    # Predict du, ds
    dt_vector = forward(X, Kinetic)
    
    # Calculate cosine similarity loss
    cos_loss = cosine_loss(neighbor_vector, dt_vector, neighbor_number, sample_number)
    
    # L2 penalty
    penalty = (l2_penalty(Kinetic.layers.l1.weight) + 
    l2_penalty(Kinetic.layers.l2.weight) + 
    l2_penalty(Kinetic.layers.l3.weight)) * λ
    
    
    # loss
    loss = cos_loss + penalty
    #loss = cos_loss
    
    return loss
end

function eval_loss_report(X::AbstractArray, Kinetic::Chain, neighbor_vector::AbstractArray, train_genes_number::Int, neighbor_number::Int, sample_number::Int, λ::AbstractFloat, dt::AbstractFloat)
    # Predict du, ds
    dt_vector = forward(X, Kinetic)
    
    # Calculate cosine similarity loss
    cos_loss = cosine_loss(neighbor_vector, dt_vector, neighbor_number, sample_number)
    
    # Calculate average cosine similarity
    AvgMeanCos = 1 - (cos_loss / (train_genes_number * sample_number))
    
    # L2 penalty
    penalty = (l2_penalty(Kinetic.layers.l1.weight) + 
    l2_penalty(Kinetic.layers.l2.weight) + 
    l2_penalty(Kinetic.layers.l3.weight)) * λ
    
    return (cos_loss = cos_loss |> round4, penalty = penalty |> round4, AvgMeanCos = AvgMeanCos |> round4)
end

round4(x::AbstractFloat)::AbstractFloat = round(x, digits = 4)

function report(epoch::Int, train_X::AbstractArray, Kinetic::Chain, train_neighbor_vector::AbstractArray, train_genes_number::Int, neighbor_number::Int, sample_number::Int, λ::AbstractFloat, dt::AbstractFloat) 
    train = eval_loss_report(train_X, Kinetic, train_neighbor_vector, train_genes_number, neighbor_number, sample_number, λ, dt)
    println("Epoch: $epoch  Train: $(train)")
    return train
end
