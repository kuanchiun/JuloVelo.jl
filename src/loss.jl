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
