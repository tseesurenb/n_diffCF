import torch
import numpy as np
from evaluate_utils import computeTopNAccuracy
from tqdm import tqdm

def enhanced_evaluate(diffusion, model, data_loader, data_te, mask_his, topN):
    """
    Optimized enhanced evaluation function with batch processing and vectorized operations.
    
    Args:
        diffusion: EnhancedGaussianDiffusion
            The diffusion model
        model: nn.Module
            The prediction model
        data_loader: DataLoader
            Loader for the user interaction data
        data_te: scipy.sparse.csr_matrix
            Test data matrix
        mask_his: scipy.sparse.csr_matrix
            Mask for historical interactions (train + validation)
        topN: list
            List of N values for top-N evaluation
            
    Returns:
        test_results: tuple
            Tuple of evaluation metrics
    """
    model.eval()
    e_N = mask_his.shape[0]
    e_idxlist = list(range(e_N))
    
    # Pre-compute target items for all users to avoid repeated conversions
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    # Pre-allocate array for predictions
    max_k = max(topN)
    predict_items = [[] for _ in range(e_N)]
    
    # Use GPU batch processing
    with torch.no_grad():
        # Show progress bar for better monitoring
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            # Get the indices of users in this batch
            batch_start = batch_idx * data_loader.batch_size
            batch_end = min(batch_start + data_loader.batch_size, e_N)
            batch_size = batch_end - batch_start
            user_indices = e_idxlist[batch_start:batch_end]
            batch_indices = torch.tensor(user_indices, device=batch.device)
            
            # Move batch to device
            batch = batch.to(batch.device)
            
            # Get historical data for masking (move to GPU for faster operations)
            his_data = mask_his[user_indices].toarray()
            his_data_tensor = torch.tensor(his_data, device=batch.device)
            
            # Enhanced sampling using batch indices (optimized)
            if hasattr(diffusion, 'use_enhanced_sampling') and diffusion.use_enhanced_sampling:
                prediction = diffusion.p_sample(
                    model, 
                    batch, 
                    diffusion.sampling_steps, 
                    diffusion.sampling_noise,
                    batch_indices=batch_indices
                )
            else:
                # Fallback to regular sampling if enhanced sampling is not available
                prediction = diffusion.p_sample(
                    model, 
                    batch, 
                    diffusion.sampling_steps, 
                    diffusion.sampling_noise
                )
            
            # Apply historical interaction mask (vectorized)
            prediction.masked_fill_(his_data_tensor > 0, -float('inf'))

            # Get top-k items efficiently
            _, indices = torch.topk(prediction, max_k)
            indices_cpu = indices.cpu().numpy()
            
            # Store predictions for each user
            for i, user_idx in enumerate(user_indices):
                predict_items[user_idx] = indices_cpu[i].tolist()
    
    # Use vectorized accuracy computation
    test_results = computeTopNAccuracy(target_items, predict_items, topN)

    return test_results

def enhanced_evaluate_old(diffusion, model, data_loader, data_te, mask_his, topN):
    """
    Enhanced evaluation function that uses batch indices for improved sampling.
    
    Args:
        diffusion: EnhancedGaussianDiffusion
            The diffusion model
        model: nn.Module
            The prediction model
        data_loader: DataLoader
            Loader for the user interaction data
        data_te: scipy.sparse.csr_matrix
            Test data matrix
        mask_his: scipy.sparse.csr_matrix
            Mask for historical interactions (train + validation)
        topN: list
            List of N values for top-N evaluation
            
    Returns:
        test_results: tuple
            Tuple of evaluation metrics
    """
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Get the indices of users in this batch
            batch_start = batch_idx * data_loader.batch_size
            batch_end = min(batch_start + len(batch), e_N)
            batch_indices = torch.tensor(e_idxlist[batch_start:batch_end], device=batch.device)
            
            # Get historical data for masking
            his_data = mask_his[e_idxlist[batch_start:batch_end]].copy()
            
            # Move batch to device
            batch = batch.to(batch.device)
            
            # Enhanced sampling using batch indices
            prediction = diffusion.p_sample(
                model, 
                batch, 
                diffusion.sampling_steps, 
                diffusion.sampling_noise,
                batch_indices=batch_indices
            )
            
            # Apply historical interaction mask
            prediction[his_data.nonzero()] = -np.inf

            # Get top-N items
            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    # Compute evaluation metrics
    from evaluate_utils import computeTopNAccuracy
    test_results = computeTopNAccuracy(target_items, predict_items, topN)

    return test_results