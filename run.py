#!/usr/local/bin/python

# avoid errors due to no $DISPLAY environment variable available when running sc.pl.paga
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import h5py
import json

import scanpy.api as sc
import anndata

import time
checkpoints = {}

import dynclipy

#   ____________________________________________________________________________
#   Load data                                                               ####
task = dynclipy.main()
task = dynclipy.main(
  ["--dataset", "/code/example.h5", "--output", "/mnt/output"],
  "/code/definition.yml"
)

counts = task["counts"]

params = task["params"]

if "groups_id" in task["priors"]:
  groups_id = task["priors"]['groups_id']
else:
  groups_id = None

# create dataset
if groups_id is not None:
  obs = pd.DataFrame(groups_id)
  obs["louvain"] = obs["group_id"].astype("category")
  adata = anndata.AnnData(counts.values, obs)
else:
  adata = anndata.AnnData(counts.values)
  
#   ____________________________________________________________________________
#   Basic preprocessing                                                     ####

n_top_genes = min(2000, counts.shape[1])

# normalisation & filtering
# the recipe_zheng17 only works when > 150 cells because of `np.arange(10, 105, 5)` in filter_genes_dispersion. This should be fixed in the next scanpy release (> 1.2.2) as it is already fixed on github
if counts.shape[1] >= 150:
  sc.pp.recipe_zheng17(adata, n_top_genes=n_top_genes)
else:
  sc.pp.normalize_per_cell(adata)
  sc.pp.scale(adata)

# precalculating some dimensionality reductions
sc.tl.pca(adata, n_comps=params["n_comps"])
sc.pp.neighbors(adata, n_neighbors=params["n_neighbors"])

# denoise the graph by recomputing it in the first few diffusion components
if params["n_dcs"] != 0:
  sc.tl.diffmap(adata, n_comps=params["n_dcs"])

#   ____________________________________________________________________________
#   Cluster, infer trajectory, infer pseudotime, compute dimension reduction ###

# add grouping if not provided
if groups_id is None:
  sc.tl.louvain(adata, resolution=params["resolution"])

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
if params["embedding_type"] != 'fa':
  sc.tl.draw_graph(adata, init_pos='paga')
else:
  sc.tl.umap(adata, init_pos='paga')

checkpoints["method_aftermethod"] = time.time()

#   ____________________________________________________________________________
#   Process & save output                                                   ####

# grouping
grouping = pd.DataFrame({"cell_id": counts.index, "group_id": adata.obs.louvain})

# milestone network
milestone_network = pd.DataFrame(
  np.triu(adata.uns["paga"]["connectivities"].todense(), k = 0),
  index=adata.obs.louvain.cat.categories,
  columns=adata.obs.louvain.cat.categories
).stack().reset_index()
milestone_network.columns = ["from", "to", "length"]
milestone_network = milestone_network.query("length >= " + str(params["connectivity_cutoff"])).reset_index(drop=True)
milestone_network["directed"] = False

# dimred
dimred = pd.DataFrame([x for x in adata.obsm['X_umap'].T]).T
dimred.columns = ["comp_" + str(i) for i in range(dimred.shape[1])]
dimred["cell_id"] = counts.index

# dimred milestones
dimred_milestones = dimred.copy()
dimred_milestones["milestone_id"] = adata.obs.louvain.tolist()
dimred_milestones = dimred_milestones.groupby("milestone_id").mean().reset_index()

# timings
timings = pd.Series(checkpoints)
timings.index.name = "name"
timings.name = "timings"

# save
dataset = dynclipy.wrap_data(cell_ids = counts.index)
dataset.add_dimred_projection(
  grouping = grouping, 
  milestone_network = milestone_network,
  dimred = dimred,
  dimred_milestones = dimred_milestones
)
dataset.write_output(task["output"])
