import numpy as np
import bottleneck as bn
import torch
import math

def computeTopNAccuracy_old(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))
        
    return precision, recall, NDCG, MRR

def computeTopNAccuracy(ground_truth, predicted_indices, topN):
    """
    Vectorized implementation of top-N accuracy metrics computation.
    
    Args:
        ground_truth: list of lists
            Ground truth item indices for each user
        predicted_indices: list of lists
            Predicted item indices for each user
        topN: list
            List of N values for top-N evaluation
            
    Returns:
        tuple: (precision, recall, ndcg, mrr) for each N in topN
    """
    n_users = len(ground_truth)
    max_k = max(topN)
    
    # Pre-allocate result arrays
    precision = np.zeros((n_users, len(topN)))
    recall = np.zeros((n_users, len(topN)))
    ndcg = np.zeros((n_users, len(topN)))
    mrr = np.zeros(n_users)
    
    # Convert predictions to numpy array for faster operations
    pred_array = np.array(predicted_indices)[:, :max_k]
    
    # Create a mask of relevant items
    for i in range(n_users):
        if len(ground_truth[i]) == 0:
            continue
            
        # Create set of ground truth items for faster lookup
        gt_set = set(ground_truth[i])
        
        # Create hit array - 1 if item is relevant, 0 otherwise
        hit_array = np.zeros(max_k)
        for j in range(max_k):
            if j < len(pred_array[i]) and pred_array[i][j] in gt_set:
                hit_array[j] = 1
        
        # Calculate MRR - find index of first hit
        if np.sum(hit_array) > 0:
            first_hit_idx = np.argmax(hit_array)
            mrr[i] = 1.0 / (first_hit_idx + 1)
        
        # Calculate metrics for each N
        for k_idx, k in enumerate(topN):
            # Precision and recall are straightforward
            hit_k = hit_array[:k].sum()
            precision[i, k_idx] = hit_k / k
            recall[i, k_idx] = hit_k / len(gt_set)
            
            # For NDCG, we need the discounted gains
            dcg = np.sum(hit_array[:k] / np.log2(np.arange(2, k + 2)))
            
            # Ideal DCG - perfect ranking of all relevant items
            ideal_rank_len = min(len(gt_set), k)
            idcg = np.sum(1.0 / np.log2(np.arange(2, ideal_rank_len + 2)))
            
            ndcg[i, k_idx] = dcg / idcg if idcg > 0 else 0
    
    # Average across users and round
    precision_avg = np.round(np.mean(precision, axis=0), 4)
    recall_avg = np.round(np.mean(recall, axis=0), 4)
    ndcg_avg = np.round(np.mean(ndcg, axis=0), 4)
    mrr_avg = np.round(np.mean(mrr), 4)
    
    # Convert to lists for compatibility
    precision_list = precision_avg.tolist()
    recall_list = recall_avg.tolist()
    ndcg_list = ndcg_avg.tolist()
    mrr_list = [mrr_avg] * len(topN)
    
    return precision_list, recall_list, ndcg_list, mrr_list


def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))


    