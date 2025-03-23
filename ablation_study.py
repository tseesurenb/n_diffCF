import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import custom modules
from models.enhanced_gaussian_diffusion_knn import EnhancedGaussianDiffusion, ModelMeanType
from models.DNN import DNN
from enhanced_evaluation import enhanced_evaluate
import data_utils

def run_ablation_study(args, train_data, valid_y_data, test_y_data, n_user, n_item, device):
    """Run ablation studies on gamma and temperature parameters."""
    gamma_values = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    temperature_values = [0.1, 0.5, 1.0, 2.0]
    
    # Create data loaders
    train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
    test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    mask_tv = train_data + valid_y_data
    
    # Setup for results collection
    gamma_results = {'recall': [], 'ndcg': []}
    temp_results = {'recall': [], 'ndcg': []}
    
    # Setup model architecture
    out_dims = eval(args.dims) + [n_item]
    in_dims = out_dims[::-1]
    
    # Study gamma parameter
    print("\n===== GAMMA ABLATION STUDY =====")
    fixed_temp = 0.5  # Fix temperature
    for gamma in gamma_values:
        print(f"\nTesting with gamma = {gamma}, temperature = {fixed_temp}")
        
        # Create model
        model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)
        
        # Create diffusion with current gamma
        if args.mean_type == 'x0':
            mean_type = ModelMeanType.START_X
        else:
            mean_type = ModelMeanType.EPSILON
            
        diffusion = EnhancedGaussianDiffusion(
            mean_type=mean_type, 
            noise_schedule=args.noise_schedule,
            noise_scale=args.noise_scale, 
            noise_min=args.noise_min, 
            noise_max=args.noise_max, 
            steps=args.steps, 
            device=device,
            top_k=args.top_k,
            gamma=gamma,
            use_similarity=args.use_similarity,
            temperature=fixed_temp
        ).to(device)
        
        diffusion.sampling_steps = args.sampling_steps
        diffusion.sampling_noise = args.sampling_noise
        
        # Initialize similarity matrix
        diffusion.initialize_similarity_matrix(train_data)
        
        # Load the best model if available
        import glob
        model_files = glob.glob(f'{args.save_path}{args.dataset}_similarity_best_model_*.pth')
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            print(f"Loading model: {latest_model}")
            checkpoint = torch.load(latest_model, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        test_results = enhanced_evaluate(
            diffusion, model, test_loader, test_y_data, mask_tv, eval(args.topN)
        )
        
        # Store results
        gamma_results['recall'].append(test_results[1][1])  # recall@20
        gamma_results['ndcg'].append(test_results[2][1])    # ndcg@20
        
        print(f"Results: Recall@20 = {test_results[1][1]}, NDCG@20 = {test_results[2][1]}")
    
    # Study temperature parameter
    print("\n===== TEMPERATURE ABLATION STUDY =====")
    fixed_gamma = 0.7  # Fix gamma
    for temp in temperature_values:
        print(f"\nTesting with gamma = {fixed_gamma}, temperature = {temp}")
        
        # Create model
        model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)
        
        # Create diffusion with current temperature
        if args.mean_type == 'x0':
            mean_type = ModelMeanType.START_X
        else:
            mean_type = ModelMeanType.EPSILON
            
        diffusion = EnhancedGaussianDiffusion(
            mean_type=mean_type, 
            noise_schedule=args.noise_schedule,
            noise_scale=args.noise_scale, 
            noise_min=args.noise_min, 
            noise_max=args.noise_max, 
            steps=args.steps, 
            device=device,
            top_k=args.top_k,
            gamma=fixed_gamma,
            use_similarity=args.use_similarity,
            temperature=temp
        ).to(device)
        
        diffusion.sampling_steps = args.sampling_steps
        diffusion.sampling_noise = args.sampling_noise
        
        # Initialize similarity matrix
        diffusion.initialize_similarity_matrix(train_data)
        
        # Load the best model if available
        import glob
        model_files = glob.glob(f'{args.save_path}{args.dataset}_similarity_best_model_*.pth')
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            print(f"Loading model: {latest_model}")
            checkpoint = torch.load(latest_model, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        test_results = enhanced_evaluate(
            diffusion, model, test_loader, test_y_data, mask_tv, eval(args.topN)
        )
        
        # Store results
        temp_results['recall'].append(test_results[1][1])  # recall@20
        temp_results['ndcg'].append(test_results[2][1])    # ndcg@20
        
        print(f"Results: Recall@20 = {test_results[1][1]}, NDCG@20 = {test_results[2][1]}")
    
    # Plot gamma results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(gamma_values, gamma_results['recall'], 'o-', linewidth=2)
    plt.xlabel('Gamma')
    plt.ylabel('Recall@20')
    plt.title('Effect of Gamma on Recall@20')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(gamma_values, gamma_results['ndcg'], 'o-', linewidth=2)
    plt.xlabel('Gamma')
    plt.ylabel('NDCG@20')
    plt.title('Effect of Gamma on NDCG@20')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('gamma_ablation_study.png')
    plt.show()
    
    # Plot temperature results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(temperature_values, temp_results['recall'], 'o-', linewidth=2)
    plt.xlabel('Temperature')
    plt.ylabel('Recall@20')
    plt.title('Effect of Temperature on Recall@20')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(temperature_values, temp_results['ndcg'], 'o-', linewidth=2)
    plt.xlabel('Temperature')
    plt.ylabel('NDCG@20')
    plt.title('Effect of Temperature on NDCG@20')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('temperature_ablation_study.png')
    plt.show()
    
    print("Ablation study results saved to gamma_ablation_study.png and temperature_ablation_study.png")