module JuloVelo

using LinearAlgebra
using StatsBase
using Statistics
using Random

using BSON
using CSV
using CUDA
using CurveFit
using DataFrames
using Distributions
using Flux
using MLUtils
using MultivariateStats
using Muon
using NearestNeighbors
using NNlib
using Optimisers
using Pickle
using PyCall
using Plots
using Zygote

using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!


export
    # Data
    AbstractJuloVeloObject,
    JuloVeloObject,
    load_data,
    reshape_data,

    # gene kinetics
    define_gene_kinetic,
    define_gene_kinetic!,

    # layer
    SpatialDense,

    # loss
    cosine_similarity,
    cosine_loss,
    l2_penalty,

    # network
    build_velocity_model,
    optimizer_setting,
    train,
    forward,
    eval_loss,
    eval_loss_report,
    round4,
    report,

    # plot
    kinetics_embedding,

    # sampling
    gaussian_kernel,
    meshgrid,
    density_sampling,
    density_sampling!,

    # utils
    filter_gene,
    find_neighbor,
    calculate_neighbor_vector,
    to_device,
    to_celldancer,
    to_dynamo,
    write_adata,

    # velocity
    velocity_estimation,
    kinetic_equation,
    compute_cell_velocity


include("data.jl")
include("gene_kinetics.jl")
include("layer.jl")
include("loss.jl")
include("network.jl")
include("plot.jl")
include("sampling.jl")
include("utils.jl")
include("velocity.jl")

end
