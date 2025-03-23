import os
import argparse
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import itertools
import json
from pathlib import Path

# Import our modified components
from models.enhanced_gaussian_diffusion_knn import EnhancedGaussianDiffusionKNN, ModelMeanType
from models.DNN import DNN
from enhanced_evaluation import enhanced_evaluate
import data_utils
from user_similarity import precompute_similarity_data
import utils
from tqdm import tqdm
from efficient_knn_utils import efficient_batch_sampling

# Set command line arguments for Jupyter notebook environment
sys.argv = ['diffusion_training.py', '--dataset', 'ml-1m', '--data_path', 'data/', 
           '--batch_size', '400', '--epochs', '350', '--use_similarity', 'True',
           '--enable_hyperparameter_search', 'True',
           '--top_k', '30', '--gamma', '0.6', '--temperature', '1.0', '--lr', '0.0001', '--steps', '40', '--noise_scale', '0.0001', '--noise_min', '0.005', '--noise_max', '0.01',]


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id):
    """Initialize workers with different random seeds."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='choose the dataset')
    parser.add_argument('--data_path', type=str, default='data/', help='load data path')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=150, help='upper epoch limit')
    parser.add_argument('--early_stop_patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--topN', type=str, default='[10, 20]')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
    parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
    
    # Params for the model
    parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
    parser.add_argument('--dims', type=str, default='[200, 600]', help='the dims for the DNN')
    parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
    parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

    # Params for diffusion
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

    # Params for user similarity
    parser.add_argument('--use_similarity', type=bool, default=True, help='use user similarity aggregation')
    parser.add_argument('--top_k', type=int, default=35, help='number of similar users for aggregation')
    parser.add_argument('--gamma', type=float, default=0.7, help='weight for user vector vs similar users')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for weighted averaging')
    
    # Hyperparameter search
    parser.add_argument('--enable_hyperparameter_search', type=bool, default=False, help='Enable hyperparameter search')
    parser.add_argument('--search_top_k', type=str, default='[20, 30, 40]', help='Search values for top_k')
    parser.add_argument('--search_gamma', type=str, default='[0.5, 0.6, 0.7, 0.8]', help='Search values for gamma')
    parser.add_argument('--search_temperature', type=str, default='[0.5, 1.0, 1.5]', help='Search values for temperature')
    parser.add_argument('--search_lr', type=str, default='[0.0001, 0.0003, 0.001]', help='Search values for learning rate')
    parser.add_argument('--search_steps', type=str, default='[5, 10, 20, 40]', help='Search values for diffusion steps')
    parser.add_argument('--search_noise_scale', type=str, default='[0.00001, 0.0001, 0.001]', help='Search values for noise scale')
    
    return parser.parse_args()

def train_and_evaluate(args, train_data, valid_y_data, test_y_data, n_user, n_item, run=None):
    """Train and evaluate model with given hyperparameters"""
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create data loaders
    train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, 
                           shuffle=True, num_workers=0, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    mask_tv = train_data + valid_y_data

    # Build Enhanced Gaussian Diffusion with KNN
    if args.mean_type == 'x0':
        mean_type = ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)

    diffusion = EnhancedGaussianDiffusionKNN(
        mean_type=mean_type, 
        noise_schedule=args.noise_schedule,
        noise_scale=args.noise_scale, 
        noise_min=args.noise_min, 
        noise_max=args.noise_max, 
        steps=args.steps, 
        device=device,
        top_k=args.top_k,
        gamma=args.gamma,
        use_similarity=args.use_similarity,
        temperature=args.temperature
    ).to(device)
    
    # Set sampling parameters
    diffusion.sampling_steps = args.sampling_steps
    diffusion.sampling_noise = args.sampling_noise

    print(f"Using hyperparameters: top_k={args.top_k}, gamma={args.gamma}, temperature={args.temperature}, "
          f"lr={args.lr}, steps={args.steps}, noise_scale={args.noise_scale}")
    
    # Precompute or load similarity data from cache
    similarity_matrix, top_k_similar_users, top_k_similarities = precompute_similarity_data(
        train_data, top_k=max(50, args.top_k), save_path='./cache/'  # Use a larger top_k for cache to support various search values
    )
    
    # Use the precomputed data with the dedicated method
    diffusion.set_precomputed_similarity_data(
        similarity_matrix,
        top_k_similar_users,
        top_k_similarities,
        torch.FloatTensor(train_data.A).to(device)
    )
    
    # Build MLP
    out_dims = eval(args.dims) + [n_item]
    in_dims = out_dims[::-1]
    model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    param_num = 0
    mlp_num = sum([param.nelement() for param in model.parameters()])
    diff_num = sum([param.nelement() for param in diffusion.parameters()])  # Should be 0
    param_num = mlp_num + diff_num
    
    # Training metrics tracking
    all_epochs = []
    train_losses = []
    valid_recalls = []
    test_recalls = []
    
    best_recall, best_epoch = -100, 0
    best_results = None
    best_test_results = None
    
    # Initialize tqdm for epoch-level progress bar
    pbar = tqdm(range(1, args.epochs + 1), 
                bar_format='{l_bar}{bar}{r_bar}',
                desc="Training progress")

    for epoch in pbar:
        if epoch - best_epoch >= args.early_stop_patience:
            pbar.set_description(f"Early stopping at epoch {epoch}")
            break
        
        model.train()
        start_time = time.time()
        
        batch_count = 0
        total_loss = 0.0
        
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            # Get batch indices for enhanced sampling and training
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + len(batch), n_user)
            batch_indices = torch.arange(batch_start, batch_end, device=device)
            
            batch = batch.to(device)
            batch_count += 1
            optimizer.zero_grad()
            
            # Since KNN is now a post-processing step, we use standard diffusion training
            # without mixing KNN during the training process
            losses = diffusion.training_losses(model, batch, None, args.reweight)
                
            loss = losses["loss"].mean()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Update the progress bar description with the final loss
        pbar.set_description(f"Epoch {epoch:03d} | Loss {total_loss:.4f}")

        # Evaluation every 5 epochs
        if epoch % 5 == 0:
            # Use the enhanced evaluation function with KNN
            print("\nEvaluating model...")
            valid_results = enhanced_evaluate(
                diffusion, model, test_loader, valid_y_data, train_data, eval(args.topN)
            )
            test_results = enhanced_evaluate(
                diffusion, model, test_loader, test_y_data, mask_tv, eval(args.topN)
            )
            
            # Format evaluation results
            valid_recall = valid_results[1][1]  # recall@20
            test_recall = test_results[1][1]
            valid_ndcg = valid_results[2][1]  # NDCG@20
            test_ndcg = test_results[2][1]
            
            if run is not None:
                run.log({
                    "epoch": epoch,
                    "Recall/Valid": valid_recall, 
                    "Recall/Test": test_recall,
                    "NDCG/Valid": valid_ndcg,
                    "NDCG/Test": test_ndcg,
                    "Loss/Total": total_loss,
                    "top_k": args.top_k,
                    "gamma": args.gamma,
                    "temperature": args.temperature,
                    "lr": args.lr,
                    "steps": args.steps,
                    "noise_scale": args.noise_scale
                })
            
            # Store metrics for plotting
            all_epochs.append(epoch)
            train_losses.append(total_loss)
            valid_recalls.append(valid_recall)
            test_recalls.append(test_recall)
            
            # Update the progress bar with the metrics
            pbar.set_postfix(valid_recall=valid_recall, test_recall=test_recall,
                           valid_ndcg=valid_ndcg, test_ndcg=test_ndcg)
            
            # Check for the best results and save model if improved
            if valid_recall > best_recall:  # recall@20 as selection
                best_recall, best_epoch = valid_recall, epoch
                best_results = valid_results
                best_test_results = test_results
                
                # Save the model with the best performance
                config_str = f"k{args.top_k}_g{args.gamma}_t{args.temperature}_lr{args.lr}_s{args.steps}_ns{args.noise_scale}"
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                model_save_path = f'{args.save_path}{args.dataset}_dm_knn_post_{config_str}_epoch{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_recall': best_recall,
                    'config': {
                        'top_k': args.top_k,
                        'gamma': args.gamma,
                        'use_similarity': args.use_similarity,
                        'temperature': args.temperature,
                        'lr': args.lr,
                        'steps': args.steps,
                        'noise_scale': args.noise_scale
                    }
                }, model_save_path)
                
                print(f"Saved best model at epoch {epoch} with recall {best_recall:.4f}")

    # Final results for this hyperparameter set
    print('='*54)
    print(f"Training complete. Best Epoch {best_epoch:03d}")
    utils.print_results(None, best_results, best_test_results)
    
    return {
        'hyperparameters': {
            'top_k': args.top_k,
            'gamma': args.gamma,
            'temperature': args.temperature,
            'lr': args.lr,
            'steps': args.steps,
            'noise_scale': args.noise_scale
        },
        'best_epoch': best_epoch,
        'best_valid_recall': best_recall,
        'best_test_recall': best_test_results[1][1] if best_test_results else None,
        'best_valid_ndcg': best_results[2][1] if best_results else None,
        'best_test_ndcg': best_test_results[2][1] if best_test_results else None,
        'valid_results': best_results,
        'test_results': best_test_results
    }

def main():
    """Main function to run the DM+KNN integrated model with hyperparameter search."""
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Parse arguments
    args = parse_args()
    
    # Import wandb here to avoid loading it if not needed
    try:
        import wandb
        # Start a new wandb run
        run = wandb.init(
            project="diffRec",
            config={
                "learning_rate": args.lr,
                "architecture": "DM+KNN-Post",
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "top_k": args.top_k,
                "gamma": args.gamma,
                "temperature": args.temperature,
                "use_similarity": args.use_similarity,
                "hyperparameter_search": args.enable_hyperparameter_search
            },
        )
    except ImportError:
        print("WandB not installed, proceeding without logging")
        run = None
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load data
    train_path = os.path.join(args.data_path, args.dataset, 'train_list.npy')
    valid_path = os.path.join(args.data_path, args.dataset, 'valid_list.npy')
    test_path = os.path.join(args.data_path, args.dataset, 'test_list.npy')
    
    try:
        train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
        print(f"Loaded dataset with {n_user} users and {n_item} items.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please make sure the dataset exists in the specified path.")
        sys.exit(1)
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    if args.enable_hyperparameter_search:
        print("\n" + "="*50)
        print("Starting hyperparameter grid search...")
        print("="*50)
        
        # Parse the hyperparameter search space
        top_k_values = eval(args.search_top_k)
        gamma_values = eval(args.search_gamma)
        temperature_values = eval(args.search_temperature)
        lr_values = eval(args.search_lr)
        steps_values = eval(args.search_steps)
        noise_scale_values = eval(args.search_noise_scale)
        
        print(f"Search space:\n"
              f"- top_k: {top_k_values}\n"
              f"- gamma: {gamma_values}\n"
              f"- temperature: {temperature_values}\n"
              f"- learning rate: {lr_values}\n"
              f"- diffusion steps: {steps_values}\n"
              f"- noise scale: {noise_scale_values}")
        
        # Generate all combinations of hyperparameters
        param_combinations = list(itertools.product(
            top_k_values, gamma_values, temperature_values, lr_values, steps_values, noise_scale_values
        ))
        
        print(f"Total combinations to search: {len(param_combinations)}")
        
        # Store all results
        all_results = []
        
        # Run grid search with progress bar
        search_pbar = tqdm(param_combinations, desc="Hyperparameter search progress")
        
        for top_k, gamma, temperature, lr, steps, noise_scale in search_pbar:
            # Update args with current hyperparameters
            args.top_k = top_k
            args.gamma = gamma
            args.temperature = temperature
            args.lr = lr
            args.steps = steps
            args.noise_scale = noise_scale
            
            config_str = f"k{top_k}_g{gamma}_t{temperature}_lr{lr}_s{steps}_ns{noise_scale}"
            search_pbar.set_description(f"Training with {config_str}")
            
            # Train and evaluate with current hyperparameters
            result = train_and_evaluate(args, train_data, valid_y_data, test_y_data, n_user, n_item, run)
            all_results.append(result)
            
            # Log the result to wandb
            if run is not None:
                run.log({
                    "grid_search/best_valid_recall": result['best_valid_recall'],
                    "grid_search/best_test_recall": result['best_test_recall'],
                    "grid_search/best_valid_ndcg": result['best_valid_ndcg'],
                    "grid_search/best_test_ndcg": result['best_test_ndcg'],
                    "grid_search/best_epoch": result['best_epoch'],
                    "grid_search/config": config_str
                })
        
        # Sort results by validation recall
        all_results.sort(key=lambda x: x['best_valid_recall'], reverse=True)
        
        # Save all results to file
        with open(f'results/hyperparameter_search_results_{args.dataset}.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print best results
        print("\n" + "="*50)
        print("Hyperparameter search completed")
        print("="*50)
        print("Top 5 configurations:")
        
        for i, result in enumerate(all_results[:5]):
            print(f"{i+1}. {result['hyperparameters']} - Valid Recall@20: {result['best_valid_recall']:.4f}, Test Recall@20: {result['best_test_recall']:.4f}")
        
        # Plot summary of results
        plt.figure(figsize=(15, 10))
        
        # Top 10 configurations by validation recall
        top_configs = all_results[:10]
        config_names = [f"k{r['hyperparameters']['top_k']}_g{r['hyperparameters']['gamma']}" for r in top_configs]
        valid_recalls = [r['best_valid_recall'] for r in top_configs]
        test_recalls = [r['best_test_recall'] for r in top_configs]
        
        plt.subplot(1, 2, 1)
        x = np.arange(len(config_names))
        width = 0.35
        plt.bar(x - width/2, valid_recalls, width, label='Valid Recall@20')
        plt.bar(x + width/2, test_recalls, width, label='Test Recall@20')
        plt.xlabel('Configuration')
        plt.ylabel('Recall@20')
        plt.title('Top 10 Hyperparameter Configurations')
        plt.xticks(x, config_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Heatmap of top_k vs gamma (using average of best validation recalls)
        plt.subplot(1, 2, 2)
        heatmap_data = {}
        for result in all_results:
            k = result['hyperparameters']['top_k']
            g = result['hyperparameters']['gamma']
            key = (k, g)
            if key not in heatmap_data:
                heatmap_data[key] = []
            heatmap_data[key].append(result['best_valid_recall'])
        
        # Average the values
        for key in heatmap_data:
            heatmap_data[key] = np.mean(heatmap_data[key])
        
        # Create the heatmap
        unique_k = sorted(set(k for k, _ in heatmap_data.keys()))
        unique_g = sorted(set(g for _, g in heatmap_data.keys()))
        heatmap = np.zeros((len(unique_k), len(unique_g)))
        
        for i, k in enumerate(unique_k):
            for j, g in enumerate(unique_g):
                if (k, g) in heatmap_data:
                    heatmap[i, j] = heatmap_data[(k, g)]
        
        plt.imshow(heatmap, cmap='viridis')
        plt.colorbar(label='Avg. Valid Recall@20')
        plt.xticks(np.arange(len(unique_g)), unique_g)
        plt.yticks(np.arange(len(unique_k)), unique_k)
        plt.xlabel('Gamma')
        plt.ylabel('Top_k')
        plt.title('Heatmap of top_k vs gamma')
        
        # Add text annotations to the heatmap
        for i in range(len(unique_k)):
            for j in range(len(unique_g)):
                plt.text(j, i, f"{heatmap[i, j]:.3f}", 
                       ha="center", va="center", color="w" if heatmap[i, j] > np.mean(heatmap) else "black")
        
        plt.tight_layout()
        plt.savefig(f'results/hyperparameter_search_{args.dataset}.png')
        print(f"Search results visualization saved to results/hyperparameter_search_{args.dataset}.png")
        
        # Use the best hyperparameters for final model
        best_config = all_results[0]['hyperparameters']
        print("\n" + "="*50)
        print(f"Best hyperparameters found: {best_config}")
        print("="*50)
        
        # Update args with best hyperparameters
        args.top_k = best_config['top_k']
        args.gamma = best_config['gamma']
        args.temperature = best_config['temperature']
        args.lr = best_config['lr']
        args.steps = best_config['steps']
        args.noise_scale = best_config['noise_scale']
        
        # Train a final model with the best hyperparameters
        print("\nTraining final model with best hyperparameters...")
        
    else:
        print("\n" + "="*50)
        print("Starting training with specified hyperparameters...")
        print("="*50)
    
    # Train final model (either with best hyperparameters from search or specified ones)
    final_result = train_and_evaluate(args, train_data, valid_y_data, test_y_data, n_user, n_item, run)
    
    # Plot final training progress
    plt.figure(figsize=(12, 10))
    
    # Create plot from the stored epoch data in train_and_evaluate
    plt.subplot(2, 1, 1)
    plt.plot(final_result.get('all_epochs', []), final_result.get('train_losses', []))
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Epoch')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(final_result.get('all_epochs', []), final_result.get('valid_recalls', []), label='Validation Recall@20')
    plt.plot(final_result.get('all_epochs', []), final_result.get('test_recalls', []), label='Test Recall@20')
    plt.xlabel('Epoch')
    plt.ylabel('Recall@20')
    plt.title('Recall@20 vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # Create a configuration string for the filename
    config_str = f"k{args.top_k}_g{args.gamma}_t{args.temperature}"
    
    plt.tight_layout()
    plt.savefig(f'results/training_progress_dm_knn_post_{config_str}.png')
    print(f"Training progress plot saved to training_progress_dm_knn_post_{config_str}.png")
    
    print("\nExperiment completed successfully!")
    
    if run is not None:
        run.finish()
    
    return final_result

if __name__ == "__main__":
    main()