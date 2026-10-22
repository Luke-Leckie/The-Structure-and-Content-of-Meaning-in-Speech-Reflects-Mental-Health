#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import math
import gc
import pickle
from scipy.linalg import norm
from sklearn.preprocessing import normalize
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import os
from itertools import combinations
import numpy as np
from random import seed, sample
import random
import numpy as np
from collections import Counter
import networkx as nx
import scipy.stats as stats
from scipy.stats import sem
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.decomposition import PCA
import gc
import itertools
from scipy.linalg import eigvalsh
import networkx as nx
import numpy as np
from networkx.algorithms import approximation
from scipy import stats
import collections
import itertools
import gudhi as gd
import gudhi.representations


SPARSE=False
sparse_param=0
dims_simplex=3

def compute_persistence_diagram(data, ML=2):
    # Compute the Rips complex
    rips_complex = gd.RipsComplex(points=data, max_edge_length=ML)
    # Construct a simplex tree
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    # Compute persistent homology
    persistence = simplex_tree.persistence()
    # Plot persistence diagram`
    gd.plot_persistence_diagram(persistence)
    gd.plot_persistence_barcode(persistence)

    plt.show()
    return persistence

def add_geometric_centroid(data):
    """
    Given an (N, D) array 'data' of N points in D dimensions,
    computes the mean (geometric centroid) and appends it
    as an extra row at the end.

    Returns:
      data_with_centroid: an (N+1, D) array,
        where the last row is the centroid.
      centroid_index: the integer index of the new centroid row.
    """
    centroid = np.mean(data, axis=0)             # shape (D,)
    data_with_centroid = np.vstack([data, centroid])
    centroid_index = data_with_centroid.shape[0] - 1
    return data_with_centroid, centroid_index

def build_centroid_distance_matrix(data_with_centroid, centroid_index, large_val=1e6):
    """
    Creates a distance matrix where only edges from the 'centroid_index'
    to other points have the real Euclidean distance.
    All other pairwise distances are set to 'large_val'.
    """
    N = data_with_centroid.shape[0]
    dist_matrix = np.full((N, N), large_val, dtype=float)

    # Diagonal = 0
    np.fill_diagonal(dist_matrix, 0.0)

    # Compute distances between centroid and each other point
    for i in range(N):
        if i == centroid_index:
            continue
        # Real distance from centroid -> i
        dist = cosine_similarity(data_with_centroid[centroid_index].reshape(1,-1),\
                                 data_with_centroid[i].reshape(1,-1))
        dist_matrix[centroid_index, i] = dist
        dist_matrix[i, centroid_index] = dist

    return dist_matrix

def compute_persistence_centroid(data, max_edge_length=4.0, plotting=True):
    """
    1) Compute the geometric centroid of 'data' and append it as an extra row.
    2) Build a distance matrix such that only the centroid can connect to other points.
    3) Construct the Rips complex using the custom distance matrix.
    4) Compute and plot the persistence diagram and barcode.

    Returns:
      persistence: The list of (dim, (birth, death)) intervals from GUDHI
    """
    # 1) Add the centroid
    data_with_centroid, centroid_index = add_geometric_centroid(data)

    # 2) Build the custom distance matrix
    dist_matrix = build_centroid_distance_matrix(data_with_centroid, centroid_index, large_val=1e6)

    # 3) Create RipsComplex from the distance matrix
    rips_complex = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_edge_length)
    
    if SPARSE:
        rips_complex = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_edge_length, sparse=sparse_param)
    else:
        rips_complex = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dims_simplex)
    del dist_matrix  # Delete objects
    gc.collect()  # Force garbage collection to free memory
    # 4) Compute persistent homology
    persistence = simplex_tree.persistence()
    if not plotting:
        return rips_complex, simplex_tree, persistence
    # 5) Plot the persistence diagram & barcode
    gd.plot_persistence_diagram(persistence)
    gd.plot_persistence_barcode(persistence)
    plt.show()

    return rips_complex, simplex_tree, persistence


def get_alive_components_over_scales(births, deaths, step=0.025):
    """
    Given lists of (birth, death) intervals for a particular homology dimension,
    compute how many such features (e.g., connected components if D=0, loops if D=1, etc.)
    are 'alive' at increments of 'step' from 0 up to max(deaths).

    Returns:
    - scales: list of scale values (0, 0.025, 0.05, ...)
    - alive_counts: corresponding list of how many features are alive at each scale
    """
    if len(births) == 0:
        # No intervals => no features
        return [], []
    
    max_death = max(deaths)
    scales = np.arange(0, max_death + 1e-9, step)
    
    alive_counts = []
    for s in scales:
        # Count intervals that are alive: birth <= s < death
        count_alive = sum(1 for (b, d) in zip(births, deaths) if b <= s < d)
        alive_counts.append(count_alive)
    
    return list(scales), alive_counts

def get_rips_time_centroid(df, embeddings='sentence_embeddings_pca', step=0.025, D=0):
    """
    For each row in df, build a Rips complex, extract dimension-D intervals
    (e.g., D=0 => connected components, D=1 => loops, etc.),
    then compute how many such features are 'alive' at increments of 'step'.

    Creates two new columns in df:
    - f"scales_dim{D}": The scale values
    - f"alive_dim{D}": The counts of alive features at each scale
    """
    # Copy the DataFrame to avoid SettingWithCopy warnings
    df = df.copy()
    
    # Prepare two new columns (lists)
    df[f'centroid_scales_dim{D}'] = None
    df[f'centroid_alive_dim{D}'] = None
    df[f'rt_centroid'] = None

    for idx, row in df.iterrows():
#         if idx in [26, 27, 28]:
#             print('warning: skipping a computationally intensive sample')
#             continue
        # Get the embeddings for this row
        embed = row[embeddings]
        if not isinstance(embed, (list, np.ndarray)) or len(embed) == 0:
            continue
        
        # Build the Rips Complex
        rips_complex, simplex_tree,persistence= compute_persistence_centroid(embed,max_edge_length=MLC,plotting=False)
        births_dimD = []
        deaths_dimD = []
        
        for dim, (b, d) in persistence:
            if dim == D and d != float('inf'):  # ignoring infinite intervals
                births_dimD.append(b)
                deaths_dimD.append(d)

        # Compute how many features are alive at each scale
        scales, alive_components = get_alive_components_over_scales(births_dimD, deaths_dimD, step=step)
        df.at[idx, f"rt_centroid"] = max(deaths_dimD)
        # Store these lists in the new columns
        df.at[idx, f"centroid_scales_dim{D}"] = scales
        df.at[idx, f"centroid_alive_dim{D}"] = alive_components
        del rips_complex, simplex_tree, persistence, births_dimD, deaths_dimD, scales, alive_components  # Delete objects
        gc.collect()  # Force garbage collection to free memory
        
        #time.sleep(0.2)  # Pause for 0.5 seconds
    return df

def ff3(x):
    return x*(x-1)*(x-2)

def avg_tetr_cc(g):
    tetrahedra = itertools.islice(itertools.groupby(
        nx.enumerate_all_cliques(g), len), 3, 4)
    try:
        tetrahedra = next(tetrahedra)[1]
    except StopIteration:
        return 0
    cnts = collections.Counter(itertools.chain(*tetrahedra))
    return 6 * sum(cnt / ff3(g.degree[v]) for v, cnt in cnts.items()) / len(g)



def compute_graph_metrics(G):
    """
    Computes various network metrics for a given graph G, including Laplacian eigenvalues.
    
    Metrics:
    - Shortest Path (Weighted & Unweighted)
    - Number of Triangles
    - Number of Tetrahedra (4-cliques)
    - Modularity using Louvain (Weighted)
    - Clustering Coefficient
    - Max & Mean Degree
    - Max & Mean Betweenness Centrality
    - Max & Mean Strength (Weighted Degree)
    - Second Smallest Laplacian Eigenvalue (Fiedler Value)
    - Largest Laplacian Eigenvalue

    Parameters:
    - G (networkx.Graph): A 3-skeleton graph with weighted edges.

    Returns:
    - Dictionary with computed graph metrics.
    """
    metrics = {
        "shortest_path_unweighted": np.nan,
        "nodes":np.nan,
        "shortest_path_weighted": np.nan,
        "num_triangles": np.nan,
        "num_tetrahedra": np.nan, #maybe also area of these
        "modularity_louvain": np.nan,
        "clustering_coefficient": np.nan,
        "max_degree": np.nan,
        "mean_degree": np.nan,
        "max_betweenness": np.nan,
        "mean_betweenness": np.nan,
        "max_strength": np.nan,

        "mean_strength": np.nan,
        "fiedler_value": np.nan,
        "largest_laplacian_eigenvalue": np.nan,
        'spectral_gap':np.nan,
        'modularity_unw':np.nan,
        'num_comms_unw':np.nan,
        'num_comms':np.nan,

        "diameter":np.nan,
        "weighted_diameter":np.nan,
        "std_degree":np.nan,
        "std_strength":np.nan,
        "efficiency":np.nan,

        "fiedler_value_unw": np.nan,
        "largest_laplacian_eigenvalue_unw": np.nan,
        'spectral_gap_unw':np.nan
    }

    if not G or G.number_of_nodes() < 2:
        return metrics

    # Sorted nodes
    sorted_nodes = sorted(G.nodes())

    # Shortest Path (Unweighted & Weighted)
    first_node, last_node = sorted_nodes[0], sorted_nodes[-1]
    if nx.has_path(G, first_node, last_node):
        metrics["shortest_path_unweighted"] = nx.shortest_path_length(G, source=first_node, target=last_node)
        metrics["shortest_path_weighted"] = nx.shortest_path_length(G, source=first_node, target=last_node, weight='weight')

    try:
        metrics["diameter"] = nx.diameter(G, weight=None)
        metrics["weighted_diameter"] = nx.diameter(G, weight='weight')
    except Exception as e:
        metrics["diameter"] = np.nan
        metrics["weighted_diameter"] = np.nan
        
    try:
        metrics["efficiency"] = nx.global_efficiency(G)

    except Exception as e:
        metrics["efficiency"] = np.nan
    
    # Number of triangles (3-cliques)
    metrics["num_triangles"] = sum(nx.triangles(G).values()) // 3  # Each triangle counted 3 times

    # Number of tetrahedra (4-cliques)
    metrics["num_tetrahedra"] = avg_tetr_cc(G)  # Ensure avg_tetr_cc is defined


    # Clustering Coefficient
    metrics["clustering_coefficient"] = nx.average_clustering(G, weight='weight')

    # Degree (Max & Mean)
    degrees = dict(G.degree())
    metrics["max_degree"] = max(degrees.values())
    metrics["mean_degree"] = np.mean(list(degrees.values()))
    metrics["std_degree"] = np.std(list(degrees.values()))
    # Betweenness Centrality (Max & Mean)
    betweenness = nx.betweenness_centrality(G, weight='weight')
    metrics["max_betweenness"] = max(betweenness.values())
    metrics["mean_betweenness"] = np.mean(list(betweenness.values()))
    for u, v, data in G.edges(data=True):
        original_weight = data.get('weight', 1)  # default to 1 if no weight provided
        # Avoid division by zero:
        if original_weight != 0:
            data['inv_weight'] = 1 / original_weight
        else:
            data['inv_weight'] = 0  # or some default value that makes sense for your case

    # Louvain Modularity (Weighted)
    # comms = nx.community.louvain_communities(G, weight='inv_weight')
    # metrics["modularity_louvain"] = nx.community.modularity(G, comms, weight='inv_weight')
    # metrics["num_comms"]=len(comms)
    
    # comms_u = nx.community.louvain_communities(G, weight=None)
    # metrics["modularity_unw"] = nx.community.modularity(G, comms_u, weight=None)
    # metrics["num_comms_unw"]=len(comms_u)
    #comms_leiden = leiden_communities(G, weight='inv_weight')
   # metrics["modularity_leiden"] = nx.community.modularity(G, comms_leiden, weight='inv_weight')
    #metrics["num_comms_leiden"] = len(comms_leiden)
    # Strength (Weighted Degree) (Max & Mean)
    strength = {node: sum(G[node][nbr].get('inv_weight', 1) for nbr in G[node]) for node in G.nodes()}
    metrics["max_strength"] = max(strength.values())
    metrics["mean_strength"] = np.mean(list(strength.values()))
    metrics["std_strength"] = np.std(list(strength.values()))
    metrics['nodes']=len(G.nodes())
    # Compute Laplacian Eigenvalues
    L = nx.laplacian_matrix(G, weight='inv_weight').toarray()  # Convert sparse matrix to dense NumPy array
    eigenvalues = eigvalsh(L)  # Compute eigenvalues

    if len(eigenvalues) > 1:  # Ensure there are at least two eigenvalues
        metrics["fiedler_value"] = eigenvalues[1]  # Second smallest eigenvalue (λ₂)
        metrics["largest_laplacian_eigenvalue"] = eigenvalues[-1]  # Largest eigenvalue (λ_max)
        metrics["spectral_gap"] = eigenvalues[-1] - eigenvalues[1]  # λ_max - λ₂
        
    L_unw = nx.laplacian_matrix(G, weight=None).toarray()  # Convert sparse matrix to dense NumPy array
    eigenvalues_unw = eigvalsh(L_unw)  # Compute eigenvalues

    if len(eigenvalues_unw) > 1:  # Ensure there are at least two eigenvalues
        metrics["fiedler_value_unw"] = eigenvalues_unw[1]  # Second smallest eigenvalue (λ₂)
        metrics["largest_laplacian_eigenvalue_unw"] = eigenvalues_unw[-1]  # Largest eigenvalue (λ_max)
        metrics["spectral_gap_unw"] = eigenvalues_unw[-1] - eigenvalues_unw[1]  # λ_max - λ₂
        
    return metrics

def gini_coefficient(x):
    """Compute the Gini coefficient of a 1D array x."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or x.mean() == 0:
        return np.nan
    # pairwise differences
    diffs = np.abs(x[:, None] - x[None, :])
    return diffs.sum() / (2 * x.size**2 * x.mean())

# ---------- helper -----------------------------------------------------------
def _metric_block(b, d, p):
    if len(p) == 0:
        # empty – return NaNs so downstream math won’t crash
        keys = ['births','deaths']
        return dict.fromkeys(keys, np.nan)
    return {
        'births':b.tolist(),
        'deaths':d.tolist()
    }

def compute_distribution_stats(births, deaths, persistences):
    """
    Return a flat dict with the metrics for the full diagram and the same
    metrics prefixed with 'sig_' for bars whose lifetime exceeds
        τ = Q3 + 3·IQR   (IQR = Q3 - Q1 of lifetimes).
    """
    births = np.asarray(births,  dtype=float)
    deaths = np.asarray(deaths,  dtype=float)
    pers   = np.asarray(persistences, dtype=float)

    # ---------- metrics for ALL bars ---------------------------------------
    out = _metric_block(births, deaths, pers)
   # lifetimes = deaths - births        
    # -------- more sensitive whisker (scale = 1.5) -------------------------
 #   out=_add_sig_block(out,
   #                label="sig2",
     #              lifetimes=lifetimes,
       #            births=births, deaths=deaths, pers=pers,
         #          scale=1.5)
    return out


def visualize_rips_simplicial_complex(embed, dataset_name, entry, max_edge_length=3):
    """
    1) Builds a Rips complex (via GUDHI) from a set of high-dimensional points.
    2) Extracts simplices (up to dimension 2) from the simplex tree.
       - Edges (1-simplices) and triangles (2-simplices).
    3) Uses PCA to reduce the points to 3D.
    4) Plots a 3D visualization:
       - Nodes are shown as a scatter plot.
       - Edges are drawn as lines.
       - Triangles are drawn as filled polygons (using Poly3DCollection).
    
    Parameters:
    -----------
    embed : np.ndarray of shape (N, D)
        The high-dimensional point cloud.
    max_edge_length : float
        The maximum edge length used in the Rips complex.
    """
    # 1) Build the Rips complex and create the simplex tree
    rips_complex = gd.RipsComplex(points=embed, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dims_simplex)
    
    # 2) Extract simplices:
    edges = []
    triangles = []
    
    # get_skeleton(2) returns all simplices up to dimension 2
    for simplex, fvalue in simplex_tree.get_skeleton(4):
        if len(simplex) == 2:
            # 1-simplices: edges
            edges.append(simplex)
        elif len(simplex) == 3:
            # 2-simplices: triangles
            triangles.append(simplex)
    # 3) Use PCA to reduce the point cloud to 3D
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(embed)  # shape (N, 3)
    n_points = coords_3d.shape[0]
    
    # Prepare colormap for nodes (using 'magma_r')
    norm = plt.Normalize(vmin=0, vmax=n_points - 1)
    cmap = plt.get_cmap('plasma_r')
    node_colors = cmap(norm(np.arange(n_points)))
    
    # 4) Create the 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot nodes
    sc = ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                    c=node_colors, s=30, alpha=0.9)
    
    # Plot edges as lines
    for edge in edges:
        i, j = edge
        x_vals = [coords_3d[i, 0], coords_3d[j, 0]]
        y_vals = [coords_3d[i, 1], coords_3d[j, 1]]
        z_vals = [coords_3d[i, 2], coords_3d[j, 2]]
        # Optionally, color edge based on one endpoint's index or the average.
        avg_idx = int(np.mean(edge))
        edge_color = cmap(norm(avg_idx))
        ax.plot(x_vals, y_vals, z_vals, color=edge_color, alpha=0.8, linewidth=1.5)
    
    # Plot triangles as filled faces
    face_polys = []
    face_colors = []
    for tri in triangles:
        # Get the 3 vertices for this triangle
        pts = [coords_3d[idx] for idx in tri]
        face_polys.append(pts)
        # Color can be computed from the average index of the triangle's vertices
        avg_idx = int(np.mean(tri))
        face_colors.append(cmap(norm(avg_idx)))
    
    # Create a Poly3DCollection for the triangles with a set transparency (alpha)
    poly_collection = Poly3DCollection(face_polys, alpha=0.3, edgecolor='k')
    poly_collection.set_facecolor(face_colors)
    ax.add_collection3d(poly_collection)
    
    # Set title and labels
    ax.set_title(f"", pad=20)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    
    # Add colorbar for node indices
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("Node Index")
        # Define three different viewing angles
    angles = [(15, 180), (30, 90), (45, 0)]  # (elevation, azimuth) in degrees
    dir_fig_save=working_dir+f'rips_skeletons/{dataset_name}_{window}_{step}/'
    os.makedirs(dir_fig_save, exist_ok=True)

    # Save figures from different angles
    for i, (elev, azim) in enumerate(angles):
        ax.view_init(elev=elev, azim=azim)  # Set camera angle
        filename = dir_fig_save+f"{entry}_{i}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save figure
       # print(f"Saved: {filename}")

    #plt.show()



def compute_euler_characteristic(simplex_tree, max_dim=4):
    """
    Compute the Euler characteristic of a simplicial complex represented by a GUDHI simplex tree.
    
    Parameters:
      simplex_tree: A GUDHI simplex tree containing simplices up to dimension max_dim.
      max_dim: Maximum dimension to consider (e.g., 3 for tetrahedra).
    
    Returns:
      euler: The Euler characteristic computed as 
             f0 - f1 + f2 - f3 + ... (up to max_dim).
    """
    if not simplex_tree:
        return None
    # Dictionary to store counts for each dimension
    simplex_counts = {}
    for d in range(max_dim + 1):
        # In GUDHI, a d-simplex is a simplex with d+1 vertices.
        simplices_d = [simplex for simplex, filt in simplex_tree.get_skeleton(d) if len(simplex) == d + 1]
        simplex_counts[d] = len(simplices_d)
    #    print(f"Number of {d}-simplices (f_{d}): {simplex_counts[d]}")
    
    # Euler characteristic: sum_{d=0}^{max_dim} (-1)^d * f_d
    euler = sum(((-1) ** d) * simplex_counts[d] for d in range(max_dim + 1))
    return euler

def get_rips_complex_G(df, embedding=str('embeddings')):
    df['graph']=None
    df['density']=None
    df['edges']=None
    df['tris']=None
    df['tetra']=None
    df['penta']=None

    for idx, row in df.iterrows():
        G=nx.Graph()
        
        H = {}  # Hypergraph as a dictionary: {hyperedge_id: [vertices]}
        embed = row[embedding] 
        rips_complex = row['rt_rips']
        if not rips_complex:
            continue
        simplex_tree =  rips_complex.create_simplex_tree(max_dimension=dims_simplex)

        edges2 = []
        triangles = []
        tetrahedrons = []
        fives=[]
        # get_skeleton(2) returns all simplices up to dimension 2
        for simplex, fvalue in simplex_tree.get_skeleton(4):
            if len(simplex) >= 2:
                for (i, j) in itertools.combinations(simplex, 2):
                    G.add_edge(i, j, weight=fvalue)
            if len(simplex) == 2:
                edges2.append(simplex)
            elif len(simplex) == 3:
                triangles.append(simplex)
            elif len(simplex) == 4:
                tetrahedrons.append(simplex)
            elif len(simplex) == 5:
                fives.append(simplex)

                    
   #     print(len(G.nodes()), 'nodes')
#    print(len(embed), 'embed len')
        df['edges'].loc[idx]=len(edges2)
        df['tris'].loc[idx]=len(triangles)
        df['tetra'].loc[idx]=len(tetrahedrons)
        df['penta'].loc[idx]=len(tetrahedrons)
        df['graph'].loc[idx]=G
        df['density'].loc[idx]=nx.density(G)

        
    return df#df

    ####
import psutil
from scipy.spatial.distance import cdist
# Create a psutil Process object for memory checks
process = psutil.Process()

def get_rips_time(df, embeddings='sentence_embeddings', step=0.025, ML=2):
    """
    For each row in df, build a Rips complex, extract dimension-D intervals
    (e.g., D=0 => connected components, D=1 => loops, etc.),
    then compute how many such features are 'alive' at increments of 'step'.

    Creates two new columns in df:
    - f"scales_dim{D}": The scale values
    - f"alive_dim{D}": The counts of alive features at each scale
    """
    # Copy the DataFrame to avoid SettingWithCopy warnings
    df = df.copy()
    
    # Prepare two new columns (lists)
  #  df[f'scales_dim0'] = None
   # df[f'alive_dim0'] = None
    #df[f'scales_dim1'] = None
    #df[f'alive_dim1'] = None
    #df[f'scales_dim2'] = None
    #df[f'alive_dim2'] = None
    df['rt'] = None
    df["rt_rips"]=None
  #  df["rt_euler"]=None
    df["rips"]=None
    for idx, row in df.iterrows():
        # Get the embeddings for this row

        embed = row[embeddings]
        if not isinstance(embed, (list, np.ndarray)) or len(embed) == 0:
            continue
        
        # # Build the Rips Complex
        if SPARSE:
            rips_complex = gd.RipsComplex(points=embed, max_edge_length=ML, sparse=sparse_param)
        else:
            rips_complex = gd.RipsComplex(points=embed, max_edge_length=ML)
            #distance_matrix = cdist(embed, embed)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=dims_simplex)
        # Extract dimension-D intervals from persistence
        persistence = simplex_tree.persistence()
        

        for D in [0,1,2]:
            births_dimD = []
            deaths_dimD = []

            for dim, (b, d) in persistence:
                if dim == D and d != float('inf'):  # ignoring infinite intervals
                    births_dimD.append(b)
                    deaths_dimD.append(d)

            # Compute how many features are alive at each scale
        #    scales, alive_components = get_alive_components_over_scales(births_dimD, deaths_dimD, step=step)
            if len(deaths_dimD)>0 and D==0:
                df.at[idx, f"rt"] = max(deaths_dimD)+0.001
            # Store these lists in the new columns
          #  df.at[idx, f"scales_dim{D}"] = scales
            #df.at[idx, f"alive_dim{D}"] = alive_components
        if SPARSE:
            rips_complex_max = gd.RipsComplex(points=embed, max_edge_length=df["rt"].loc[idx], \
                                              sparse=sparse_param)
        else:
            try:
                rips_complex_max = gd.RipsComplex(points=embed, max_edge_length=df["rt"].loc[idx])
            except Exception as e:
                print(f"Error creating RipsComplex: {e}")
                rips_complex_max = gd.RipsComplex(points=embed, max_edge_length=ML)

        #simplex_tree_max = rips_complex.create_simplex_tree(max_dimension=dims_simplex)
       # st = rips_complex_max.create_simplex_tree()
        df.at[idx, f"rt_rips"] = rips_complex_max
        df.at[idx, f"rips"] = rips_complex

    return df

import math

def count_new_simplices_by_dimension_in_bins(simplex_tree, dimensions=(2, 3, 4, 5), bin_size=0.025):
    # Initialize a dictionary for each dimension of interest.
    counts = {dim: {} for dim in dimensions}
    
    # Get the (simplex, filtration) pairs from the simplex tree.
    simplex_filtration = simplex_tree.get_filtration()
    
    for simplex, filt in simplex_filtration:
        # Determine the dimension of the simplex.
        dim = len(simplex) - 1
        # Only process simplices with a dimension in the provided set.
        if dim in dimensions:
            bin_start = math.floor(filt / bin_size) * bin_size
            counts[dim][bin_start] = counts[dim].get(bin_start, 0) + 1
            
    return counts
def get_simplices_over_time(df, max_dimension=4,simplex_tree_type='rt_simplex_tree'):
    """
    For each row in the dataframe (which contains a simplex tree in the 'simplex_tree' column),
    compute the simplex counts at different filtration values for dimensions 2, 3, and 4.
    
    The filtration values and the corresponding counts are stored in separate columns.
    
    New columns added:
      - simplex_time_dim{D}_filtration: the list of filtration values.
      - simplex_time_dim{D}_count: the list of simplex counts for that dimension.
      
    Parameters:
        df: pandas DataFrame that contains a column 'simplex_tree'
        max_dimension: maximum dimension to be passed to count_simplices (not used in this snippet,
                       but can be used if you want to generalize further).
    
    Returns:
        The dataframe with additional columns.
    """
    
    # For each dimension, create two new columns for the filtration values and counts.
    for D in [2, 3, 4]:
        df[f'simplex_time_dim{D}_filtration'] = None
        df[f'simplex_time_dim{D}_count'] = None

    # Process each row in the dataframe.
    for idx, row in df.iterrows():
        # Calculate the counts for the current simplex tree.
        simplex_counts = count_new_simplices_by_dimension_in_bins(row[simplex_tree_type])
        for D in [2,3,4]:
            # Get the dictionary for the current dimension D.
            # (If there are no simplices of this dimension, we set empty lists.)
            if D in simplex_counts and simplex_counts[D]:
                # Sort the bins so that the lists are ordered by increasing filtration value.
                bins = sorted(simplex_counts[D].keys())
                counts_list = [simplex_counts[D][b] for b in bins]
            else:
                bins, counts_list = [], []
            df.at[idx, f'simplex_time_dim{D}_filtration'] = bins
            df.at[idx, f'simplex_time_dim{D}_count'] = counts_list

    return df



# In[4]:




import sys
import gc
import shutil
from tqdm import tqdm
from scipy.spatial import cKDTree
tqdm.pandas()




def run_TDA(TDA_data, overlap, df_name,data_save_dir,span, dim_reduction=10, ML=2,  reduce_dims=True):

    embeddings='embeddings'
    step_sparsity=int(span*overlap)
    final_data_save_dir=data_save_dir /'final_dfs/'
    os.makedirs(final_data_save_dir, exist_ok=True)
    print(f'Skipping every: {step_sparsity}')
    completed_files=os.listdir(data_save_dir)
    chunk_size=2
    step=int(span*overlap)#4
    def increase_sparsity(arr, skip=step_sparsity):
        length = len(arr)
        if length <= 1:
            return arr.copy()
        indices = np.arange(0, length - 1, skip + 1)
        indices = np.append(indices, length - 1)

        return arr[indices]

    data_save_dir_name=data_save_dir / f'{df_name}_{span}_{step}_{dim_reduction}_tuned/'
    if os.path.exists(data_save_dir_name):
        shutil.rmtree(data_save_dir_name)  # Deletes the entire directory and its contents
    os.makedirs(data_save_dir_name, exist_ok=True)

    TDA_data = TDA_data[
        TDA_data["embeddings"].apply(
            lambda x: (
                not isinstance(x, float)               # exclude floats
                and isinstance(x, (list, tuple, np.ndarray))  # must be list/tuple/np.ndarray
            )
        )]

    TDA_data['length'] = TDA_data['embeddings'].apply(len)

    TDA_data = TDA_data[
        TDA_data["embeddings"].apply(
            lambda x: (

                len(x) >= 3                        # length >= 3
            )
        )]

    if reduce_dims:
        print('reducing dims')
        all_vecs = []
        for row in TDA_data['embeddings']:
            arr = np.array(row)  
            all_vecs.append(arr)
        big_matrix = np.concatenate(all_vecs, axis=0)
        pca = PCA(n_components=dim_reduction, svd_solver='full')
        pca.fit(big_matrix)
        def transform_embeddings(emb_list):
            emb_array = np.array(emb_list)   # shape (k_i, 384)
            emb_pca = pca.transform(emb_array)  # shape (k_i, 50)
            return emb_pca
        TDA_data['embeddings'] = TDA_data['embeddings'].apply(transform_embeddings)
    
    TDA_data['embeddings'] =TDA_data['embeddings'].apply(increase_sparsity)
    
    print('performing TDA on ',df_name, ' span: ', span)#, 'step: ', step)
    cn=0


    for fi in tqdm(range(0, len(TDA_data), chunk_size), desc="Processing Chunks"):
        cn+=1
        if fi+chunk_size>=len(TDA_data):
            df_subset=TDA_data[fi:].reset_index(drop=True)
        else:
            df_subset=TDA_data[fi:fi+chunk_size].reset_index(drop=True)
        df_subset=get_rips_time(df_subset,embeddings=embeddings, ML=ML)
        df_with_graph=get_rips_complex_G(df_subset)
        graph_metrics = df_with_graph['graph'].apply(compute_graph_metrics)
        graph_metrics_df = pd.DataFrame(graph_metrics.tolist())
        df_with_graph = pd.concat([df_with_graph, graph_metrics_df], axis=1)
        # We'll accumulate new rows in a list of dictsdf_sorted
        new_rows = []
        dimensions = [0, 1, 2]
        df_with_graph['rt_euler']=None
        for idx, row in df_with_graph.iterrows():
            embed = row[embeddings]  # Adjust as needed
            dim_dict = {
                dim: {'births': [], 'deaths': [], 'pers': []}
                for dim in dimensions
            }
            rips_complex =row['rips']
            try:
                simplex_tree = \
                rips_complex.create_simplex_tree(max_dimension=dims_simplex)
            except Exception as e:
                continue

            persistence = simplex_tree.persistence()
            # euler_characteristic = sum((-1) ** (len(simplex) - 1) for simplex, _ in simplex_tree.get_simplices())
            # row["rt_euler"] = euler_characteristic
            for dim, (b, d) in persistence:
                if d == float('inf'):
                    continue
                if dim in dimensions:
                    dim_dict[dim]['births'].append(b)
                    dim_dict[dim]['deaths'].append(d)
                    dim_dict[dim]['pers'].append(d - b)
                # if dim==1:
                #     print(b,d)
            row_dict = row.to_dict()  # Start with original row's columns
            for dim in dimensions:
                bdp = dim_dict[dim]
                stats_dict = compute_distribution_stats(bdp['births'], bdp['deaths'], bdp['pers'])
                # prefix each stat key with dim
                for stat_key, stat_val in stats_dict.items():
                    row_dict[f"{stat_key}_dim{dim}"] = stat_val

            # Add row_dict to new_rows
            new_rows.append(row_dict)

        # Create a new DataFrame
        df_with_tda = pd.DataFrame(new_rows)
        df_with_tda.to_csv(data_save_dir_name / f'{df_name}_{fi}_{span}_TDA_results.csv')
        del df_subset
        del df_with_graph
        gc.collect()
    print(f'completed! {df_name} span: {span}')
    list_files=sorted(os.listdir(data_save_dir_name))
    ddfs=[]
    for f in list_files:
        ddfs.append(pd.read_csv(data_save_dir_name /f))
    data=pd.concat(ddfs)
    return data



def run_TDA_sequential(TDA_data, step, df_name,data_save_dir,span, TDA_sliding_window_size=50,dim_reduction=100, ML=2):

    embeddings='embeddings'
    step_sparsity=int(span*overlap)
    final_data_save_dir=data_save_dir /'final_dfs/'
    os.makedirs(final_data_save_dir, exist_ok=True)
    reduce_dims=True
    print(f'Skipping every: {step_sparsity}')
    completed_files=os.listdir(data_save_dir)
    chunk_size=2
    def increase_sparsity(arr, skip=step_sparsity):
        length = len(arr)
        if length <= 1:
            return arr.copy()
        indices = np.arange(0, length - 1, skip + 1)
        indices = np.append(indices, length - 1)

        return arr[indices]

    data_save_dir_name=data_save_dir / f'{df_name}_{span}_{step}_{dim_reduction}_tuned_sequential/'
    if os.path.exists(data_save_dir_name):
        shutil.rmtree(data_save_dir_name)  # Deletes the entire directory and its contents
    os.makedirs(data_save_dir_name, exist_ok=True)

    TDA_data = TDA_data[
        TDA_data["embeddings"].apply(
            lambda x: (
                not isinstance(x, float)               # exclude floats
                and isinstance(x, (list, tuple, np.ndarray))  # must be list/tuple/np.ndarray
            )
        )]

    TDA_data['length'] = TDA_data['embeddings'].apply(len)

    TDA_data = TDA_data[
        TDA_data["embeddings"].apply(
            lambda x: (

                len(x) >= 3                        # length >= 3
            )
        )]

    if reduce_dims:
        print('reducing dims')
        all_vecs = []
        for row in TDA_data['embeddings']:
            arr = np.array(row)  
            all_vecs.append(arr)
        big_matrix = np.concatenate(all_vecs, axis=0)
        pca = PCA(n_components=dim_reduction, svd_solver='full')
        pca.fit(big_matrix)
        def transform_embeddings(emb_list):
            emb_array = np.array(emb_list)   # shape (k_i, 384)
            emb_pca = pca.transform(emb_array)  # shape (k_i, 50)
            return emb_pca
        TDA_data['embeddings'] = TDA_data['embeddings'].apply(transform_embeddings)
    
    TDA_data['embeddings'] =TDA_data['embeddings'].apply(increase_sparsity)
    def extract_sliding_emb_winbdow(emb):
        wind_embs=[]
        for i in range(0, len(emb)-TDA_sliding_window_size, int(TDA_sliding_window_size*0.1)):
            wind_embs.append(emb[i:i+TDA_sliding_window_size])
        return wind_embs
    TDA_data['embeddings']=TDA_data['embeddings'].apply(extract_sliding_emb_winbdow)
    
    
    print('performing TDA on ',df_name, ' span: ', span)#, 'step: ', step)
    cn=0

    for fi in tqdm(range(0, len(TDA_data), chunk_size), desc="Processing Chunks"):
        cn+=1
        if fi+chunk_size>=len(TDA_data):
            df_subset=TDA_data[fi:].reset_index(drop=True)
        else:
            df_subset=TDA_data[fi:fi+chunk_size].reset_index(drop=True)
        new_rows = []
        dimensions = [0, 1, 2]
        for idx, row in df_subset.iterrows():
            embeds_all = row[embeddings]  # Adjust as needed
            for embed in embeds_all:

                dim_dict = {+
                    dim: {'births': [], 'deaths': [], 'pers': []}
                    for dim in dimensions
                }
                rips_complex = gd.RipsComplex(points=embed, max_edge_length=ML)

                try:
                    simplex_tree = \
                    rips_complex.create_simplex_tree(max_dimension=dims_simplex)
                except Exception as e:
                    continue

                persistence = simplex_tree.persistence()
                # euler_characteristic = sum((-1) ** (len(simplex) - 1) for simplex, _ in simplex_tree.get_simplices())
                # row["rt_euler"] = euler_characteristic
                for dim, (b, d) in persistence:
                    if d == float('inf'):
                        continue
                    if dim in dimensions:
                        dim_dict[dim]['births'].append(b)
                        dim_dict[dim]['deaths'].append(d)
                        dim_dict[dim]['pers'].append(d - b)

                row_dict = row.to_dict()  # Start with original row's columns
                for dim in dimensions:
                    bdp = dim_dict[dim]
                    stats_dict = compute_distribution_stats(bdp['births'], bdp['deaths'], bdp['pers'])
                    # prefix each stat key with dim
                    for stat_key, stat_val in stats_dict.items():
                        row_dict[f"{stat_key}_dim{dim}"] = stat_val

                # Add row_dict to new_rows
                new_rows.append(row_dict)

        # Create a new DataFrame
        df_with_tda = pd.DataFrame(new_rows)
        df_with_tda.to_csv(data_save_dir_name / f'{df_name}_{fi}_{span}_TDA_results.csv')
        del df_subset
        gc.collect()
    print(f'completed! {df_name} span: {span}')
    list_files=sorted(os.listdir(data_save_dir_name))
    ddfs=[]
    for f in list_files:
        ddfs.append(pd.read_csv(data_save_dir_name /f))
    data=pd.concat(ddfs)
    data['embeddings']=None
    return data


def run_TDA_growth(TDA_data, overlap, df_name,data_save_dir,span, TDA_step_size=20,dim_reduction=100, ML=2, reduce_dims=True):

    embeddings='embeddings'
    step_sparsity=int(span*overlap)
    final_data_save_dir=data_save_dir /'final_dfs/'
    os.makedirs(final_data_save_dir, exist_ok=True)
    print(f'Skipping every: {step_sparsity}')
    completed_files=os.listdir(data_save_dir)
    chunk_size=2
    step=int(span*overlap)#4
    def increase_sparsity(arr, skip=step_sparsity):
        length = len(arr)
        if length <= 1:
            return arr.copy()
        indices = np.arange(0, length - 1, skip + 1)
        indices = np.append(indices, length - 1)

        return arr[indices]

    data_save_dir_name=data_save_dir / f'{df_name}_{span}_{step}_{dim_reduction}_tuned_growth/'
    if os.path.exists(data_save_dir_name):
        shutil.rmtree(data_save_dir_name)  # Deletes the entire directory and its contents
    os.makedirs(data_save_dir_name, exist_ok=True)

    TDA_data = TDA_data[
        TDA_data["embeddings"].apply(
            lambda x: (
                not isinstance(x, float)               # exclude floats
                and isinstance(x, (list, tuple, np.ndarray))  # must be list/tuple/np.ndarray
            )
        )]

    TDA_data['length'] = TDA_data['embeddings'].apply(len)

    TDA_data = TDA_data[
        TDA_data["embeddings"].apply(
            lambda x: (

                len(x) >= 3                        # length >= 3
            )
        )]

    if reduce_dims:
        print('reducing dims')
        all_vecs = []
        for row in TDA_data['embeddings']:
            arr = np.array(row)  
            all_vecs.append(arr)
        big_matrix = np.concatenate(all_vecs, axis=0)
        pca = PCA(n_components=dim_reduction, svd_solver='full')
        pca.fit(big_matrix)
        def transform_embeddings(emb_list):
            emb_array = np.array(emb_list)   # shape (k_i, 384)
            emb_pca = pca.transform(emb_array)  # shape (k_i, 50)
            return emb_pca
        TDA_data['embeddings'] = TDA_data['embeddings'].apply(transform_embeddings)
    
    TDA_data['embeddings'] =TDA_data['embeddings'].apply(increase_sparsity)
    def extract_growing_emb_window(emb):
        wind_embs = []
        for i in range(0, len(emb)-TDA_step_size, int(TDA_step_size)):
            wind_embs.append(emb[0:i + int(TDA_step_size)])
        return wind_embs
    TDA_data['embeddings']=TDA_data['embeddings'].apply(extract_growing_emb_window)
    
    
    print('performing TDA on ',df_name, ' span: ', span)#, 'step: ', step)
    cn=0

    for fi in tqdm(range(0, len(TDA_data), chunk_size), desc="Processing Chunks"):
        cn+=1
        if fi+chunk_size>=len(TDA_data):
            df_subset=TDA_data[fi:].reset_index(drop=True)
        else:
            df_subset=TDA_data[fi:fi+chunk_size].reset_index(drop=True)
        new_rows = []
        dimensions = [0, 1, 2]
        for idx, row in df_subset.iterrows():
            embeds_all = row[embeddings]  # Adjust as needed
            for idd, embed in enumerate(embeds_all):

                dim_dict = {+
                    dim: {'births': [], 'deaths': [], 'pers': []}
                    for dim in dimensions
                }
                rips_complex = gd.RipsComplex(points=embed, max_edge_length=ML)

                try:
                    simplex_tree = \
                    rips_complex.create_simplex_tree(max_dimension=dims_simplex)
                except Exception as e:
                    continue

                persistence = simplex_tree.persistence()
                # euler_characteristic = sum((-1) ** (len(simplex) - 1) for simplex, _ in simplex_tree.get_simplices())
                # row["rt_euler"] = euler_characteristic
                for dim, (b, d) in persistence:
                    if d == float('inf'):
                        continue
                    if dim in dimensions:
                        dim_dict[dim]['births'].append(b)
                        dim_dict[dim]['deaths'].append(d)
                        dim_dict[dim]['pers'].append(d - b)

                row_dict = row.to_dict()  # Start with original row's columns
                row_dict['sample_id'] = idd
                row_dict['last'] = 'T' if idd == len(embeds_all) - 1 else 'F'
                for dim in dimensions:
                    bdp = dim_dict[dim]
                    stats_dict = compute_distribution_stats(bdp['births'], bdp['deaths'], bdp['pers'])
                    # prefix each stat key with dim
                    for stat_key, stat_val in stats_dict.items():
                        row_dict[f"{stat_key}_dim{dim}"] = stat_val

                # Add row_dict to new_rows
                new_rows.append(row_dict)

        # Create a new DataFrame
        df_with_tda = pd.DataFrame(new_rows)
        df_with_tda.to_csv(data_save_dir_name / f'{df_name}_{fi}_{span}_TDA_results.csv')
        del df_subset
        gc.collect()
    print(f'completed! {df_name} span: {span}')
    list_files=sorted(os.listdir(data_save_dir_name))
    ddfs=[]
    for f in list_files:
        ddfs.append(pd.read_csv(data_save_dir_name /f))
    data=pd.concat(ddfs)
    data['embeddings']=None
    return data



