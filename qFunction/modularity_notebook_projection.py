
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler 

def gradient_ascent_modularity_unsupervised(G, k=2, eta=1.0, iterations=100, seed=42):
    np.random.seed(seed)

    A = np.asarray(G)
    n = A.shape[0]
    
    degree = A.sum(axis=1)
    total_weight = degree.sum()
    
    if total_weight == 0:
        print("Warning: Graph is empty. Returning random embedding.")
        return np.random.randn(n, k)

    S = np.random.randn(n, k)
    S, _ = np.linalg.qr(S)

    for i in tqdm(range(iterations), desc="Gradient Ascent for Modularity"):
        neighbor_agg = A @ S
        degree_S_sum = np.outer(degree, S.sum(axis=0))
        grad_modularity = neighbor_agg - degree_S_sum / total_weight
        
        S += eta * grad_modularity
        S, _ = np.linalg.qr(S)

    return S

def visualize_modularity_embedding_notebook(
    feature_names: list,
    q_matrix: np.ndarray,
    plot_title: str = "Modularity-based Feature Embedding",
    save_path: str = None
):
    
    print("Running Modularity Maximization")
    
    # 1. Transform Q-matrix to a symmetrized similarity matrix.
    similarity_matrix = 1.0 - q_matrix
    symmetric_matrix = (similarity_matrix + similarity_matrix.T) / 2
    
    # 2. Run modularity optimization to get the raw 2D embedding.
    embedding = gradient_ascent_modularity_unsupervised(
        symmetric_matrix,
        k=2,
        eta=1.0,
        iterations=100,
        seed=42
    )

    
    if embedding.shape[0] > 1: 
        scaler = StandardScaler()
        embedding = scaler.fit_transform(embedding)

    # 3. Visualization 
    plt.figure(figsize=(6, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1])

    for i, feat in enumerate(feature_names):
        plt.text(embedding[i, 0] + 0.05, embedding[i, 1] + 0.05, feat)

    plt.title(plot_title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.axis('equal') 
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()