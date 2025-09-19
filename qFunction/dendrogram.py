from matplotlib import pyplot as plt
import heapq
import numpy as np
from itertools import combinations
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import optimal_leaf_ordering, leaves_list



def manual_linkage(dist_matrix):
    """ manual calculation of single-linkage matrix using a min-heap.
    """
    n = len(dist_matrix)
    Z = np.zeros((n - 1, 4))
    
    # Store the current cluster ID for each original data point
    current_cluster_id = list(range(n))
    
    # Priority queue stores tuples: (distance, original_idx1, original_idx2)
    pq = []
    
    # Initialize the heap with all pairwise distances between original indices
    for i, j in combinations(range(n), 2):
        heapq.heappush(pq, (dist_matrix[i, j], i, j))
    
    next_cid = n
    
    for k in range(n - 1):
        while True:
            d, i, j = heapq.heappop(pq)
            
            # Find the representative cluster for each original index
            rep_i = current_cluster_id[i]
            rep_j = current_cluster_id[j]

            # If the clusters have already been merged, skip
            if rep_i == rep_j:
                continue
            
            # The clusters are a valid pair to merge
            break
        
        # Record the merge in the linkage matrix
        Z[k, 0] = min(rep_i, rep_j)
        Z[k, 1] = max(rep_i, rep_j)
        Z[k, 2] = d
        Z[k, 3] = k + 2 

        # Update the cluster IDs of the merged original data points
        for l in range(n):
            if current_cluster_id[l] == rep_i or current_cluster_id[l] == rep_j:
                current_cluster_id[l] = next_cid

        # Prepare for the next merge
        next_cid += 1

    return Z



def plot_manual_dendrogram(q_matrix, labels, line_color='#4285F4'):
    """
    Plots a dendrogram using a manually created linkage matrix.
    """
    # Calculate the distance matrix as the average of Q_matrix and its transpose
    dist_matrix = (q_matrix + q_matrix.T) / 2
    n = len(labels)
    Z = manual_linkage(dist_matrix)
    fig, ax = plt.subplots(figsize=(max(3.5, n * 0.25), 4))
    
    Z_opt = optimal_leaf_ordering(Z, squareform(dist_matrix, checks=False))
    order = leaves_list(Z_opt)
    
    coords = {i: (p, 0) for p, i in enumerate(order)}
    
    for k, (c1, c2, d, _) in enumerate(Z_opt, start=n):
        x1, h1 = coords[c1]
        x2, h2 = coords[c2]
        
        ax.plot([x1, x1], [h1, d], c=line_color)
        ax.plot([x2, x2], [h2, d], c=line_color)
        ax.plot([x1, x2], [d, d], c=line_color)
        
        coords[k] = ((x1 + x2) / 2, d)
    
    ax.set_xticks(range(n))
    ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=10)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, max(Z[:, 2]) * 1.05)
    ax.set_ylabel("Distance")
    plt.tight_layout()
    return plt
