import torch
import numpy as np
from utils import computeTopNAccuracy
import sys
from tqdm import tqdm

def enhanced_evaluate(diffusion, model, data_loader, data_te, mask_his, topN):
    """
    Enhanced evaluation function that uses the DM+KNN approach.
    
    Args:
        diffusion: EnhancedGaussianDiffusionKNN
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
    e_N = mask_his.shape[0] # Number of users
    e_idxlist = list(range(e_N)) # List of user indices
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Pre-compute target items for all users to avoid repeated conversions
    target_items = [] # these are the ground truth items
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    # Pre-allocate array for predictions
    max_k = max(topN)
    predict_items = [[] for _ in range(e_N)]
    
    # Calculate total number of batches for progress tracking
    total_batches = len(data_loader)
    
    # Use GPU batch processing
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Calculate and display progress
            progress = (batch_idx / total_batches) * 100
            sys.stdout.write(f"\rEvaluating: {progress:.1f}%")
            sys.stdout.flush()
            
            # Get the indices of users in this batch
            batch_start = batch_idx * data_loader.batch_size
            batch_end = min(batch_start + data_loader.batch_size, e_N)
            batch_size = batch_end - batch_start
            user_indices = e_idxlist[batch_start:batch_end]
            batch_indices = torch.tensor(user_indices, device=device)
            
            # Move batch to device
            batch = batch.to(device)
            
            # Get historical data for masking (move to GPU for faster operations)
            his_data = mask_his[user_indices].toarray()
            his_data_tensor = torch.tensor(his_data, device=device)
            
            # Call the diffusion p_sample method with batch indices
            # This will run diffusion + KNN as defined in our enhanced model
            prediction = diffusion.p_sample(
                model, 
                batch, 
                diffusion.sampling_steps, 
                diffusion.sampling_noise,
                batch_indices=batch_indices
            )
            
            # Apply historical interaction mask (vectorized)
            prediction[his_data_tensor > 0] = -float('inf')

            # Get top-k items efficiently
            _, indices = torch.topk(prediction, max_k)
            indices_cpu = indices.cpu().numpy()
            
            # Store predictions for each user
            for i, user_idx in enumerate(user_indices):
                predict_items[user_idx] = indices_cpu[i].tolist()
    
    # Print 100% completion and new line
    sys.stdout.write("\rEvaluating: 100.0%\n")
    sys.stdout.flush()
    
    # Use computeTopNAccuracy to match existing code
    test_results = computeTopNAccuracy(target_items, predict_items, topN)

    return test_results