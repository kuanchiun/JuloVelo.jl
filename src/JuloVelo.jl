module JuloVelo

using Base
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
using ProgressBars
using PyCall
using Plots
using UMAP
using Zygote

using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!


export
    # Data
    write_adata,
    read_adata,
    merge_multiome_adata,
    reshape_data,

    # gene kinetics
    gene_kinetics_predetermination,
    gene_kinetics_predetermination!,

    # layer
    SpatialDense,

    # loss
    cosine_similarity,
    cosine_loss,
    l2_penalty,

    # network
    assign_velocity_model,
    optimizer_setting,
    train,
    forward,
    eval_loss,

    # plot
    plot_kinetics_embedding,

    # sampling
    gaussian_kernel,
    meshgrid,
    density_sampling,
    density_sampling!,

    # utils
    normalize,
    filter_and_gene_kinetics_predetermination,
    filter_genes,
    find_neighbor,
    calculate_neighbor_vector,
    to_device,
    gpu_functional,
    round4,
    to_cellDancer,
    save_model,
    load_model,

    # velocity
    kinetics_equation_training,
    kinetics_equation_inference,
    velocity_estimation,
    velocity_correlation,
    correlation_coefficient,
    get_neighbor_graph,
    velocity_projection,
    compute_cell_velocity,
    estimate_pseudotime,
    kinetics_embedding


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
