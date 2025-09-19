import pandas as pd
import numpy as np
import umap
from matplotlib import pyplot as plt

def visualize_umap_embeddings(
    feature_names: list,
    umap_params: dict,
    q_matrix: np.ndarray,
    workflow_type: str = 'single',
    umap_fusion_params: dict = None,
    plot_title: str = "UMAP embedding of features",
    save_path: str = None
) -> np.ndarray:
    """
    Performs UMAP dimensionality reduction and visualizes the results.

    This function supports two distinct workflows based on a single input matrix (Q_matrix):
    1. 'single': Calculates a avearge distance matrix from Q_matrix and applies UMAP directly.
    2. 'concat': Calculates two separate matrices (forward and backward) from Q_matrix,
       concatenates the embeddings, and then applies a second UMAP on the concatenated data.

    Args:
        data_df (pd.DataFrame): The original DataFrame used to retrieve feature names
                                 for plotting labels (e.g., `data.T.index`).
        umap_params (dict): A dictionary of parameters for the UMAP model.
                            e.g., {'n_neighbors': 3, 'min_dist': 0.1, 'metric': 'precomputed'}.
        q_matrix (np.ndarray): The base matrix from which other matrices will be derived.
        workflow_type (str, optional): The type of UMAP workflow to run.
                                       Must be 'single' or 'concat'. Defaults to 'single'.
        umap_fusion_params (dict, optional): Parameters for the second UMAP model in the 'concat' workflow.
                                             If not provided, a default will be used.
        plot_title (str, optional): The title of the plot. Defaults to "UMAP embedding of features".
        save_path (str, optional): The file path to save the plot. If None, the plot is shown.

    Returns:
        np.ndarray: The final 2D UMAP embedding.
    """
    final_embedding = None

    # --- Workflow 1: Single UMAP on one matrix ---
    if workflow_type == 'single':
        print("Running UMAP on a single matrix derived from Q...")
        # Calculate the distance matrix as the average of Q and its transpose
        distance_matrix = (q_matrix + q_matrix.T) / 2
        
        reducer = umap.UMAP(**umap_params)
        final_embedding = reducer.fit_transform(distance_matrix)

    # --- Workflow 2: UMAP on two matrices, then on concatenated embeddings ---
    elif workflow_type == 'concat':
        print("Running UMAP on two matrices, then on concatenated embeddings...")
        
        # Calculate forward and backward matrices based on the formulas
        q_symmetric = np.triu(q_matrix) + np.triu(q_matrix, 1).T
        q_trans_symmetric = np.triu(q_matrix.T) + np.triu(q_matrix.T, 1).T
        
        # Default parameters for the second UMAP model
        if umap_fusion_params is None:
            umap_fusion_params = {'n_neighbors': 4, 'min_dist': 0.1, 'metric': 'euclidean', 'random_state': 42}

        # Apply UMAP to the two separate matrices
        umap_model = umap.UMAP(**umap_params)
        e_fwd = umap_model.fit_transform(q_symmetric)
        e_bwd = umap_model.fit_transform(q_trans_symmetric)

        # Concatenate the embeddings
        e_concat = pd.concat([pd.DataFrame(e_fwd), pd.DataFrame(e_bwd)], axis=1)

        # Apply UMAP on the concatenated embeddings
        umap_fusion = umap.UMAP(**umap_fusion_params)
        final_embedding = umap_fusion.fit_transform(e_concat)

    # --- Invalid workflow type ---
    else:
        raise ValueError("Invalid `workflow_type`. Choose 'single' or 'concat'.")

    # --- Visualization ---
    plt.figure(figsize=(6, 6))
    plt.scatter(final_embedding[:, 0], final_embedding[:, 1])

    # Label points with feature names
    for i, feat in enumerate(feature_names):
        plt.text(final_embedding[i, 0] + 0.01, final_embedding[i, 1] + 0.01, feat)

    plt.title(plot_title)
    plt.xlabel("UMAP_0")
    plt.ylabel("UMAP_1")

    if save_path:
        plt.savefig(save_path, dpi=700, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return final_embedding
