import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def compute_cosine_similarity_matrix(matrix, top_k=1000, self_loop=False, verbose=1):
    """
    Compute cosine similarity matrix and filter to keep only top-k similar items for each item.
    
    Args:
        matrix: numpy.ndarray or scipy.sparse.csr_matrix
            User-item interaction matrix
        top_k: int
            Number of top similar items to keep for each item
        self_loop: bool
            Whether to include self-similarity in the result (diagonal)
        verbose: int
            Verbosity level. If > 0, print progress information
            
    Returns:
        filtered_similarity_matrix: torch.Tensor
            Tensor containing the top-k similarities for each item
    """
    if verbose > 0:
        print('Computing cosine similarity by top-k...')
    
    # Convert to scipy sparse matrix if it's not already
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    
    if not sp.issparse(matrix):
        sparse_matrix = csr_matrix(matrix)
    else:
        sparse_matrix = matrix.copy()
    
    # Convert to binary
    sparse_matrix.data = (sparse_matrix.data > 0).astype(int)
    
    # Compute sparse cosine similarity (output will be sparse)
    similarity_matrix = cosine_similarity(sparse_matrix, dense_output=False)
    
    if verbose > 0:
        print('Cosine similarity computed.')
    
    # If self_loop is False, set the diagonal to zero
    if self_loop:
        similarity_matrix.setdiag(1)
    else:
        similarity_matrix.setdiag(0)
    
    # Prepare to filter top K values
    filtered_data = []
    filtered_rows = []
    filtered_cols = []
    
    if verbose > 0:
        print('Filtering top-k values...')
        pbar = tqdm(range(similarity_matrix.shape[0]), 
                    bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', 
                    ascii="░❯")
        pbar.set_description(f"Preparing cosine similarity matrix | Top-K: {top_k}")
    else:
        pbar = range(similarity_matrix.shape[0])
    
    for i in pbar:
        # Get the non-zero elements in the i-th row
        row = similarity_matrix.getrow(i).tocoo()
        if row.nnz == 0:
            continue
        
        # Extract indices and values of the row
        row_data = row.data
        row_indices = row.col
        
        # Sort indices based on similarity values (in descending order) and select top K
        if row.nnz > top_k:
            top_k_idx = np.argsort(-row_data)[:top_k]
        else:
            top_k_idx = np.argsort(-row_data)
        
        # Store the top K similarities
        filtered_data.extend(row_data[top_k_idx])
        filtered_rows.extend([i] * len(top_k_idx))
        filtered_cols.extend(row_indices[top_k_idx])
    
    # Construct the final filtered sparse matrix
    filtered_similarity_matrix = coo_matrix(
        (filtered_data, (filtered_rows, filtered_cols)), 
        shape=similarity_matrix.shape
    )
    
    # Clean up to free memory
    del sparse_matrix, similarity_matrix
    del filtered_data, filtered_rows, filtered_cols
    
    # Convert to torch tensor for later use
    similarity_csr = filtered_similarity_matrix.tocsr()
    similarity_tensor = torch.FloatTensor(similarity_csr.toarray())
    
    return similarity_tensor

def get_top_k_similar_users(similarity_matrix, k=10, exclude_self=True):
    """
    Get the top-k similar users for each user.
    
    Args:
        similarity_matrix: torch.Tensor
            User similarity matrix
        k: int
            Number of similar users to retrieve
        exclude_self: bool
            Whether to exclude the user itself from the similar users
            
    Returns:
        top_k_users: torch.Tensor
            Tensor of shape [n_users, k] containing indices of top-k similar users
        top_k_similarities: torch.Tensor
            Tensor of shape [n_users, k] containing similarity values for top-k similar users
    """
    n_users = similarity_matrix.shape[0]
    device = similarity_matrix.device
    
    if exclude_self:
        # Set diagonal to minimum value to exclude self
        similarity_matrix = similarity_matrix.clone()
        similarity_matrix.fill_diagonal_(-float('inf'))
    
    # Get top-k similar users
    top_k_similarities, top_k_users = torch.topk(similarity_matrix, k=k, dim=1)
    
    return top_k_users, top_k_similarities

def aggregate_similar_users_old(user_vectors, full_user_vectors, top_k_similar_users, top_k_similarities, gamma=0.7, temperature=0.5):
    """
    Aggregate user vectors with their similar users' vectors using weighted average.
    
    Args:
        user_vectors: torch.Tensor
            Current user vectors to aggregate
        full_user_vectors: torch.Tensor
            Complete user-item interaction matrix for all users
        top_k_similar_users: torch.Tensor
            Indices of the top-k similar users for ALL users
        top_k_similarities: torch.Tensor
            Similarity scores of the top-k similar users for ALL users
        gamma: float
            Weight for the user's own vector, (1-gamma) will be weight for similar users
        temperature: float
            Temperature parameter for softmax when weighting similarities
            
    Returns:
        aggregated_vectors: torch.Tensor
            Aggregated user vectors
    """
    device = user_vectors.device
    batch_size = user_vectors.size(0)
    vector_dim = user_vectors.size(1)
    top_k = top_k_similar_users.size(1)
    
    # Initialize tensor to hold weighted contributions from similar users
    similar_vectors = torch.zeros_like(user_vectors)
    
    # Process each user vector in the batch
    for i in range(batch_size):
        # Get the current user's similar users
        similar_indices = top_k_similar_users[i]
        
        # Get similarity scores for the current user's similar users
        similarities = top_k_similarities[i]
        
        # Apply temperature scaling and softmax for weighted averaging
        weights = torch.softmax(similarities / temperature, dim=0)
        
        # Get vectors for all similar users
        similar_user_vectors = full_user_vectors[similar_indices]
        
        # Apply weighted averaging: multiply each vector by its weight and sum
        # Shape: [top_k, 1] * [top_k, vector_dim] -> [top_k, vector_dim]
        weighted_vectors = weights.view(-1, 1) * similar_user_vectors
        
        # Sum up the weighted vectors
        similar_vectors[i] = torch.sum(weighted_vectors, dim=0)
    
    # Combine original vectors with weighted similar vectors
    aggregated_vectors = gamma * user_vectors + (1 - gamma) * similar_vectors
    
    return aggregated_vectors

# def aggregate_similar_users(user_vectors, full_user_vectors, top_k_similar_users, top_k_similarities, gamma=0.7, temperature=0.5):
#     """
#     Aggregate user vectors with their similar users' vectors using classic KNN user-based CF approach.
    
#     Args:
#         user_vectors: torch.Tensor
#             Current user vectors to aggregate
#         full_user_vectors: torch.Tensor
#             Complete user-item interaction matrix for all users
#         top_k_similar_users: torch.Tensor
#             Indices of the top-k similar users for ALL users
#         top_k_similarities: torch.Tensor
#             Similarity scores of the top-k similar users for ALL users
#         gamma: float
#             Weight for user's own vector, (1-gamma) will be weight for similar users
#         temperature: float
#             Temperature parameter for similarity weighting (lower = more emphasis on high similarities)
            
#     Returns:
#         aggregated_vectors: torch.Tensor
#             Aggregated user vectors
#     """
#     device = user_vectors.device
#     batch_size = user_vectors.size(0)
#     vector_dim = user_vectors.size(1)
#     top_k = top_k_similar_users.size(1)
    
#     # Initialize tensor to hold KNN predictions
#     knn_predictions = torch.zeros_like(user_vectors)
    
#     # Process each user vector in the batch
#     for i in range(batch_size):
#         # Get the current user's similar users
#         similar_indices = top_k_similar_users[i]
        
#         # Get similarity scores for the current user's similar users
#         similarities = top_k_similarities[i]
        
#         # Apply temperature scaling to similarities (temperature < 1 emphasizes higher similarities)
#         scaled_similarities = similarities / temperature
        
#         # Classic KNN approach uses normalized similarities as weights
#         # We'll clip to ensure no negative similarities are used
#         similarity_weights = torch.clamp(scaled_similarities, min=0)
        
#         # Normalize weights to sum to 1 (if any positive similarities exist)
#         weight_sum = similarity_weights.sum()
#         if weight_sum > 0:
#             similarity_weights = similarity_weights / weight_sum
        
#         # Get vectors for all similar users
#         similar_user_vectors = full_user_vectors[similar_indices]
        
#         # For each item, predict rating as weighted average of similar users' ratings
#         # Only consider items where similar users have expressed interest
#         weighted_sum = torch.zeros(vector_dim, device=device)
        
#         for j in range(top_k):
#             # Get this similar user's vector and similarity weight
#             similar_vector = similar_user_vectors[j]
#             weight = similarity_weights[j]
            
#             # Apply weighted contribution
#             weighted_sum += weight * similar_vector
        
#         # Store the KNN prediction for this user
#         knn_predictions[i] = weighted_sum
    
#     # Combine the user's own vector with the KNN prediction using gamma
#     # This is a common approach to balance personalization with collaborative info
#     aggregated_vectors = gamma * user_vectors + (1 - gamma) * knn_predictions
    
#     return aggregated_vectors