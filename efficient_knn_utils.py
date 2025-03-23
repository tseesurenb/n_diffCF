import torch
import numpy as np
from tqdm import tqdm

def batch_knn_aggregation(predictions, batch_indices, top_k_similar_users, top_k_similarities, full_user_vectors, gamma=0.7, temperature=0.5):
    """
    Efficient batch implementation of KNN aggregation after diffusion prediction.
    
    Args:
        predictions: torch.Tensor
            Diffusion model predictions for the batch
        batch_indices: torch.Tensor
            Indices of users in the current batch
        top_k_similar_users: torch.Tensor
            Pre-computed top-k similar users for all users
        top_k_similarities: torch.Tensor
            Pre-computed similarity scores for top-k users
        full_user_vectors: torch.Tensor
            Full user-item interaction matrix
        gamma: float
            Weight for user's own prediction vs similar users' predictions
        temperature: float
            Temperature parameter for softmax when weighting similarities
            
    Returns:
        aggregated_predictions: torch.Tensor
            KNN-enhanced predictions
    """
    device = predictions.device
    batch_size = predictions.size(0)
    
    # Create a mapping from user indices to their positions in the batch
    batch_pos_map = {idx.item(): pos for pos, idx in enumerate(batch_indices)}
    
    # Initialize output tensor
    aggregated_predictions = torch.zeros_like(predictions)
    
    # Process each user in the batch
    for i in range(batch_size):
        user_idx = batch_indices[i].item()
        user_pred = predictions[i]
        
        # Get top-k similar users and their similarities
        similar_users = top_k_similar_users[user_idx]
        similarity_scores = top_k_similarities[user_idx]
        
        # Apply temperature scaling and softmax for weighted averaging
        weights = torch.softmax(similarity_scores / temperature, dim=0).to(device)
        
        # Initialize tensor to hold similar users' predictions
        similar_preds = torch.zeros((len(similar_users), predictions.size(1)), device=device)
        
        # For each similar user, get their prediction if available in batch
        for j, sim_idx in enumerate(similar_users):
            sim_idx = sim_idx.item()
            if sim_idx in batch_pos_map:
                # Similar user is in the current batch
                similar_preds[j] = predictions[batch_pos_map[sim_idx]]
            else:
                # Similar user not in batch, use raw vector as approximation
                # In a complete solution, we would maintain a prediction cache
                similar_preds[j] = full_user_vectors[sim_idx].to(device)
        
        # Calculate weighted average of similar users' predictions
        similar_contribution = torch.sum(weights.unsqueeze(1) * similar_preds, dim=0)
        
        # Combine user's own prediction with similar users' predictions
        aggregated_predictions[i] = gamma * user_pred + (1 - gamma) * similar_contribution
    
    return aggregated_predictions

def efficient_batch_sampling(diffusion, model, batch, steps, batch_indices, sampling_noise=False):
    """
    Efficient two-stage sampling: first run diffusion, then apply KNN.
    
    Args:
        diffusion: EnhancedGaussianDiffusionKNN
            The diffusion model
        model: nn.Module
            The prediction model
        batch: torch.Tensor
            The batch of user data
        steps: int
            Number of diffusion steps
        batch_indices: torch.Tensor
            Indices of users in the current batch
        sampling_noise: bool
            Whether to add noise during sampling
            
    Returns:
        enhanced_predictions: torch.Tensor
            KNN-enhanced predictions
    """
    # 1. Run standard diffusion process
    diffusion_predictions = diffusion._p_sample_original(model, batch, steps, sampling_noise)
    
    # 2. Apply KNN aggregation if enabled
    if diffusion.use_similarity and batch_indices is not None and diffusion.top_k_similar_users is not None:
        return batch_knn_aggregation(
            diffusion_predictions,
            batch_indices,
            diffusion.top_k_similar_users,
            diffusion.top_k_similarities,
            diffusion.full_user_vectors,
            diffusion.gamma,
            diffusion.temperature
        )
    
    # If KNN is disabled, return standard diffusion predictions
    return diffusion_predictions

def precompute_user_predictions(diffusion, model, data_loader, n_users, batch_size):
    """
    Precompute diffusion predictions for all users to improve KNN aggregation.
    
    Args:
        diffusion: EnhancedGaussianDiffusionKNN
            The diffusion model
        model: nn.Module
            The prediction model
        data_loader: DataLoader
            Loader for user data
        n_users: int
            Total number of users
        batch_size: int
            Batch size
            
    Returns:
        all_predictions: torch.Tensor
            Predictions for all users
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Initialize tensor to hold all predictions
    all_predictions = torch.zeros((n_users, diffusion.full_user_vectors.size(1)), device=device)
    
    # Track progress with tqdm
    pbar = tqdm(total=n_users, desc="Precomputing user predictions")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Get batch indices
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + len(batch), n_users)
            batch_indices = torch.arange(batch_start, batch_end, device=device)
            
            # Move batch to device
            batch = batch.to(device)
            
            # Run diffusion without KNN to get base predictions
            predictions = diffusion._p_sample_original(
                model, 
                batch, 
                diffusion.sampling_steps, 
                diffusion.sampling_noise
            )
            
            # Store predictions
            all_predictions[batch_start:batch_end] = predictions
            
            # Update progress bar
            pbar.update(batch_end - batch_start)
    
    pbar.close()
    return all_predictions