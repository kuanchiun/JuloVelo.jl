function cosine_similarity(x::AbstractArray, y::AbstractArray)
    dot = NNlib.batched_mul(permutedims(x, (2, 1, 3)), y)
    xnorm = sum(x .* x, dims = 1) .^ 0.5f0
    ynorm = sum(y .* y, dims = 1) .^ 0.5f0
    
    cos_sim = dot ./ permutedims(xnorm .* ynorm, (2, 1, 3))
    
    return cos_sim
end

function cosine_loss(neighbor_vector::AbstractArray, dt_vector::AbstractArray, neighbor_number::Int, sample_number::Int)
    cos_sim = reshape(cosine_similarity(neighbor_vector, dt_vector), neighbor_number, sample_number, :)
    cos_loss = sum(1 .- maximum(cos_sim, dims = 1))
    
    return cos_loss
end

function l2_penalty(weight::AbstractArray)
    return sum(abs2, weight)
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