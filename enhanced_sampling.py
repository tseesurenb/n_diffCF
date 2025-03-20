import torch
import numpy as np


def enhanced_batch_sampling(user_indices, top_k_similar_users, batch_size):
    """
    Optimized version of enhanced batch sampling with reduced redundancy.
    
    Args:
        user_indices: torch.Tensor
            Indices of the target users in the current batch
        top_k_similar_users: torch.Tensor
            Tensor of shape [n_users, k] containing indices of top-k similar users for all users
        batch_size: int
            Original batch size
            
    Returns:
        enhanced_indices: torch.Tensor
            Combined tensor of target users and their similar users, without duplicates
        index_mapping: dict
            Mapping from original indices to positions in the enhanced batch
    """
    # Get top-k similar users for each user in the batch
    batch_similar_users = top_k_similar_users[user_indices]
    
    # Combine target users with their similar users
    all_indices = torch.cat([
        user_indices.view(-1, 1),  # Shape: [batch_size, 1]
        batch_similar_users        # Shape: [batch_size, top_k]
    ], dim=1).view(-1)  # Flatten to 1D tensor
    
    # Remove duplicates while preserving order
    enhanced_indices, inverse_indices = torch.unique(all_indices, return_inverse=True)
    
    # Create a mapping from original indices to positions in the enhanced batch
    # Vectorized approach to avoid loops
    index_mapping = {}
    
    # Find positions of each user index in the enhanced batch
    for i, idx in enumerate(user_indices):
        idx_item = idx.item()
        mask = (enhanced_indices == idx)
        if mask.any():
            index_mapping[idx_item] = torch.where(mask)[0].item()
    
    # Do the same for similar users, but only once for each unique user
    flat_similar = batch_similar_users.view(-1)
    unique_similar, _ = torch.unique(flat_similar, return_inverse=True)
    
    for sim_idx in unique_similar:
        sim_idx_item = sim_idx.item()
        if sim_idx_item not in index_mapping:
            mask = (enhanced_indices == sim_idx)
            if mask.any():
                index_mapping[sim_idx_item] = torch.where(mask)[0].item()
    
    return enhanced_indices, index_mapping

def enhance_p_sample(diffusion, model, x_start, steps, batch_indices, sampling_noise=False):
    """
    Optimized enhanced sampling process with vectorized operations.
    
    Args:
        diffusion: EnhancedGaussianDiffusion
            The diffusion model
        model: nn.Module
            The prediction model
        x_start: torch.Tensor
            The starting point for sampling
        steps: int
            Number of steps to sample
        batch_indices: torch.Tensor
            Indices of the target users in the current batch
        sampling_noise: bool
            Whether to add noise during sampling
            
    Returns:
        final_results: torch.Tensor
            The aggregated predictions for the original batch
    """
    # Get model device
    device = next(model.parameters()).device
    
    # If not using similarity, just use the regular p_sample
    if not hasattr(diffusion, 'use_similarity') or not diffusion.use_similarity or diffusion.top_k_similar_users is None:
        return diffusion._p_sample_original(model, x_start.to(device), steps, sampling_noise)
    
    # Create an enhanced batch with target users and their neighbors
    enhanced_indices, index_mapping = enhanced_batch_sampling(
        batch_indices, 
        diffusion.top_k_similar_users, 
        x_start.shape[0]
    )
    
    # Get the interaction vectors for the enhanced batch
    # Ensure vectors are on the same device as the model
    enhanced_vectors = diffusion.full_user_vectors[enhanced_indices].to(device)
    
    # Run the diffusion sampling process on the enhanced batch
    enhanced_results = diffusion._p_sample_original(model, enhanced_vectors, steps, sampling_noise)
    
    # Pre-allocate final results tensor on the correct device
    final_results = torch.zeros_like(x_start, device=device)
    
    # Vectorize the aggregation where possible
    for i, idx in enumerate(batch_indices):
        idx_item = idx.item()
        target_idx = index_mapping[idx_item]
        target_vector = enhanced_results[target_idx]
        
        # Get similar users' indices and vectors
        similar_indices = diffusion.top_k_similar_users[idx]
        
        # Create a mask of which similar users are in the enhanced batch
        sim_in_batch = torch.tensor([sim_idx.item() in index_mapping for sim_idx in similar_indices], 
                                   device=device)
        
        # Pre-allocate the similar vectors tensor on the correct device
        similar_vectors = torch.zeros(
            (diffusion.top_k, enhanced_results.shape[1]), 
            device=device
        )
        
        # Get enhanced indices for similar users in batch
        for j, sim_idx in enumerate(similar_indices):
            sim_idx_item = sim_idx.item()
            if sim_idx_item in index_mapping:
                enhanced_idx = index_mapping[sim_idx_item]
                similar_vectors[j] = enhanced_results[enhanced_idx]
            else:
                # Get from original vectors if not in batch, ensuring it's on the right device
                similar_vectors[j] = diffusion.full_user_vectors[sim_idx].to(device)
        
        # Get normalized similarity weights
        weights = torch.softmax(diffusion.top_k_similarities[idx].to(device) / diffusion.temperature, dim=0).view(-1, 1)
        
        # Weighted average and final combination
        similar_contribution = torch.sum(weights * similar_vectors, dim=0)
        final_results[i] = diffusion.gamma * target_vector + (1 - diffusion.gamma) * similar_contribution
    
    return final_results