import torch
import numpy as np

def enhanced_evaluate(diffusion, model, data_loader, data_te, mask_his, topN):
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