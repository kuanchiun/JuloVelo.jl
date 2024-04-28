module JuloVelo

using LinearAlgebra, StatsBase, Random, Statistics
using CUDA

using Plots
using Flux, NNlib, MLUtils, Optimisers
using Zygote, Optimisers
using NearestNeighbors
using Pickle, BSON
using CSV, DataFrames
using CurveFit
using Distributions
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!

export
    # Data
    AbstractJuloVeloObject,
    JuloVeloObject,
    load_data,
    reshape_data,

    # loss
    cosine_similarity,
    cosine_loss,
    l2_penalty,
    eval_loss,
    eval_loss_report,
    round4,
    report,

    # network
    SpatialDense,
    define_gene_kinetic,
    define_gene_kinetic!,
    build_velocity_model,
    optimizer_setting,
    train,
    velocity_estimation,

    # sampling
    gaussian_kernel,
    meshgrid,
    density_sampling,
    density_sampling!,
    kinetic_equation,
    forward,

    # utils
    filter_gene,
    find_neighbor,
    calculate_neighbor_vector,
    to_device,
    to_celldancer,

    # plot
    kinetics_embedding

include("data.jl")
include("loss.jl")
include("network.jl")
include("sampling.jl")
include("utils.jl")
include("plot.jl")


end
