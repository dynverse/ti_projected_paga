#!/usr/local/bin/python

import dynclipy
task = dynclipy.main()

# avoid errors due to no $DISPLAY environment variable available when running sc.pl.paga
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import h5py
import json

import scanpy.api as sc
import anndata
import numba
import warnings

import time
checkpoints = {}

#   ____________________________________________________________________________
#   Load data                                                               ####

counts = task["counts"]

parameters = task["parameters"]

if "groups_id" in task["priors"]:
  groups_id = task["priors"]['groups_id']
else:
  groups_id = None

# create dataset
if groups_id is not None:
  obs = pd.DataFrame(groups_id)
  obs["louvain"] = obs["group_id"].astype("category")
  adata = anndata.AnnData(counts, obs)
else:
  adata = anndata.AnnData(counts)
  
#   ____________________________________________________________________________
#   Basic preprocessing                                                     ####

# normalisation & filtering
if counts.shape[1] < 100 and parameters["filter_features"]:
  print("You have less than 100 features, but the filter_features parameter is true. This will likely result in an error. Disable filter_features to avoid this")

if parameters["filter_features"]:
  n_top_genes = min(2000, counts.shape[1])
  sc.pp.recipe_zheng17(adata, n_top_genes=n_top_genes)

# precalculating some dimensionality reductions
sc.tl.pca(adata, n_comps=parameters["n_comps"])
with warnings.catch_warnings():
  warnings.simplefilter('ignore', numba.errors.NumbaDeprecationWarning)
  sc.pp.neighbors(adata, n_neighbors=parameters["n_neighbors"])

# denoise the graph by recomputing it in the first few diffusion components
if parameters["n_dcs"] != 0:
  sc.tl.diffmap(adata, n_comps=parameters["n_dcs"])

#   ____________________________________________________________________________
#   Cluster, infer trajectory, infer pseudotime, compute dimension reduction ###

# add grouping if not provided
if groups_id is None:
  sc.tl.louvain(adata, resolution=parameters["resolution"])

# run paga
sc.tl.paga(adata)

# compute a layout for the paga graph
# - this simply uses a Fruchterman-Reingold layout, a tree layout or any other
#   popular graph layout is also possible
# - to obtain a clean visual representation, one can discard low-confidence edges
#   using the parameter threshold
sc.pl.paga(adata, threshold=0.01, layout='fr', show=False)

# run umap for a dimension-reduced embedding, use the positions of the paga
# graph to initialize this embedding
if parameters["embedding_type"] == 'umap':
  sc.tl.umap(adata, init_pos='paga')
  dimred_name = 'X_umap'
else:
  sc.tl.draw_graph(adata, init_pos='paga')
  dimred_name = "X_draw_graph_" + parameters["embedding_type"]

checkpoints["method_aftermethod"] = time.time()

#   ____________________________________________________________________________
#   Process & save output                                                   ####

# grouping
grouping = pd.DataFrame({"cell_id": counts.index, "group_id": adata.obs.louvain})

# milestone network
if parameters["tree"]:
  milestone_network = pd.DataFrame(
    adata.uns["paga"]["connectivities_tree"].todense(),
    index=adata.obs.louvain.cat.categories,
    columns=adata.obs.louvain.cat.categories
  ).stack().reset_index()
  milestone_network.columns = ["from", "to", "length"]
  milestone_network = milestone_network.query("length > 0").reset_index(drop=True)
  milestone_network["directed"] = False
else:
  milestone_network = pd.DataFrame(
    np.triu(adata.uns["paga"]["connectivities"].todense(), k = 0),
    index=adata.obs.louvain.cat.categories,
    columns=adata.obs.louvain.cat.categories
  ).stack().reset_index()
  milestone_network.columns = ["from", "to", "length"]
  milestone_network = milestone_network.query("length >= " + str(parameters["connectivity_cutoff"])).reset_index(drop=True)
  milestone_network["directed"] = False

# dimred
dimred = pd.DataFrame([x for x in adata.obsm[dimred_name].T]).T
dimred.columns = ["comp_" + str(i+1) for i in range(dimred.shape[1])]
dimred["cell_id"] = adata.obs.index

# dimred milestones
dimred_milestones = dimred.copy()
dimred_milestones["milestone_id"] = adata.obs.louvain.tolist()
dimred_milestones = dimred_milestones.groupby("milestone_id").mean().reset_index()

# timings
timings = pd.Series(checkpoints)
timings.index.name = "name"
timings.name = "timings"

# save
dataset = dynclipy.wrap_data(cell_ids = adata.obs.index)
dataset.add_dimred_projection(
  grouping = grouping, 
  milestone_network = milestone_network,
  dimred = dimred,
  dimred_milestones = dimred_milestones
)
dataset.write_output(task["output"])
