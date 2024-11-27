# JuloVelo

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kuanchiun.github.io/JuloVelo.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kuanchiun.github.io/JuloVelo.jl/dev/)
[![Build Status](https://github.com/kuanchiun/JuloVelo.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/kuanchiun/JuloVelo.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/kuanchiun/JuloVelo.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kuanchiun/JuloVelo.jl)


A local RNA velocity model for cell differentiation inference based on Julia programming language.
This project is currently WIP and highly inspired by [cellDancer](https://guangyuwanglab2021.github.io/cellDancer_website/).

# Requirement
- Julia >= 1.10

- Python >= 3.7
- cellDancer == 1.1.7

# The minimum example of using JuloVelo

Preprocessing
```
import scvelo as scv
import scanpy as sc
import numpy as np

# Load anndata from scvelo
adata = scv.datasets.gastrulation_erythroid()

# Preprocessing
scv.pp.filter_and_normalize(adata, min_shared_counts = 20, n_top_genes = 2000)
scv.pp.moments(adata, n_pcs = 50, n_neighbors = 50)

# Clustering
sc.tl.leiden(adata)
sc.pl.umap(adata, color = "leiden")

# Calculate diffusion pseudotime
adata.uns["iroot"] = np.flatnonzero(adata.obs["leiden"] == '6')[0]
sc.tl.dpt(adata)

# Avoid PooledArray bug in Julia
adata.var["highly_variable_genes"] = ["true" if item == "True" else "false" for item in adata.var["highly_variable_genes"]]

# Write anndata for JuloVelo
adata.write_h5ad("JuloVelo_pre.h5ad")
```

Training model
```
using JuloVelo
using Plots

# Load anndata and change iroot to root cluster
adata = read_adata("JuloVelo_pre.h5ad")
adata.uns["iroot"] = 6

# Normalize read count
normalize(adata)

# Filter genes and pre-determine gene kinetics
filter_and_gene_kinetics_predetermination(adata; filter_criteria = 0.3)

# Reshape data for model compatible
reshape_data(adata)

# Initialize velocity model
Kinetics = build_velocity_model(adata)

# Reduce sample size
density_sampling(adata)

# Training
train(adata, Kinetics)

# Calculate velocity
velocity_estimation(adata, Kinetics);

# Compute cell velocity on embedding space
compute_cell_velocity(adata);

# Restore pseudotime
estimate_pseudotime(adata, 3, n_repeat = 3, celltype = "celltype");

# Write anndata
write_adata(adata)
```