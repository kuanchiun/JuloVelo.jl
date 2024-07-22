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

function (l::SpatialDense)(x::AbstractArray)
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
