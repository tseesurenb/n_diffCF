import enum
import math
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp
from user_similarity import compute_cosine_similarity_matrix, get_top_k_similar_users

class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class EnhancedGaussianDiffusionKNN(nn.Module):
    def __init__(
        self, 
        mean_type, 
        noise_schedule, 
        noise_scale, 
        noise_min, 
        noise_max,
        steps, 
        device, 
        history_num_per_term=10, 
        beta_fixed=True,
        top_k=10,
        gamma=0.7,
        use_similarity=True,
        temperature=0.5
    ):
        """
        Enhanced Gaussian Diffusion with KNN integration for post-diffusion recommendation.
        
        Args:
            mean_type: ModelMeanType
                Type of prediction model (predict x0 or noise)
            noise_schedule: str
                Schedule for noise addition
            noise_scale: float
                Scale of noise
            noise_min: float
                Minimum noise value
            noise_max: float
                Maximum noise value
            steps: int
                Number of diffusion steps
            device: torch.device
                Device to use
            history_num_per_term: int
                Number of history terms to keep for each timestep
            beta_fixed: bool
                Whether to fix beta for first step
            top_k: int
                Number of similar users to consider
            gamma: float
                Weight for user's own vector vs similar users
            use_similarity: bool
                Whether to use user similarity aggregation
            temperature: float
                Temperature parameter for softmax when weighting similar users
        """
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device
        self.top_k = top_k
        self.gamma = gamma
        self.use_similarity = use_similarity
        self.temperature = temperature

        self.history_num_per_term = history_num_per_term
        self.Lt_history = th.zeros(steps, history_num_per_term, dtype=th.float64).to(device)
        self.Lt_count = th.zeros(steps, dtype=int).to(device)
        
        # User similarity matrix will be initialized later
        self.similarity_matrix = None
        self.top_k_similar_users = None
        self.top_k_similarities = None
        self.full_user_vectors = None

        if noise_scale != 0.:
            self.betas = th.tensor(self.get_betas(), dtype=th.float64).to(self.device)
            if beta_fixed:
                self.betas[0] = 0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
                # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

            self.calculate_for_diffusion()

        super(EnhancedGaussianDiffusionKNN, self).__init__()
        
    def initialize_similarity_matrix(self, user_item_matrix):
        """
        Initialize the user similarity matrix.
        
        Args:
            user_item_matrix: scipy.sparse.csr_matrix or torch.Tensor
                User-item interaction matrix
        """
        if not self.use_similarity:
            return
            
        # Store the full user-item matrix
        if isinstance(user_item_matrix, sp.csr_matrix):
            self.full_user_vectors = th.FloatTensor(user_item_matrix.toarray()).to(self.device)
        else:
            self.full_user_vectors = user_item_matrix.clone().to(self.device)
            
        print("Computing user similarity matrix...")
        self.similarity_matrix = compute_cosine_similarity_matrix(user_item_matrix).to(self.device)
        
        print(f"Finding top-{self.top_k} similar users for each user...")
        self.top_k_similar_users, self.top_k_similarities = get_top_k_similar_users(
            self.similarity_matrix, k=self.top_k, exclude_self=True
        )
        print("User similarity initialization complete!")
    
    def set_precomputed_similarity_data(self, similarity_matrix, top_k_similar_users, top_k_similarities, full_user_vectors):
        """
        Set precomputed similarity data instead of calculating from scratch.
        
        Args:
            similarity_matrix: torch.Tensor
                Precomputed similarity matrix
            top_k_similar_users: torch.Tensor
                Precomputed top-k similar users
            top_k_similarities: torch.Tensor
                Precomputed similarity scores for top-k users
            full_user_vectors: torch.Tensor
                Full user-item interaction matrix
        """
        if not self.use_similarity:
            print("Warning: Similarity data provided but use_similarity is set to False.")
            return
            
        self.similarity_matrix = similarity_matrix.to(self.device)
        self.top_k_similar_users = top_k_similar_users.to(self.device)
        self.top_k_similarities = top_k_similarities.to(self.device)
        self.full_user_vectors = full_user_vectors.to(self.device)
        print("Precomputed similarity data set successfully.")
        
    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
            self.steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        elif self.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = th.cat([th.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # alpha_{t-1}
        self.alphas_cumprod_next = th.cat([self.alphas_cumprod[1:], th.tensor([0.0]).to(self.device)]).to(self.device)  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = th.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * th.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    def p_sample(self, model, x_start, steps, sampling_noise=False, batch_indices=None):
        """
        Enhanced sample method with proper KNN integration after diffusion process.
        
        Args:
            model: nn.Module
                The model to sample from
            x_start: torch.Tensor
                The starting point for sampling
            steps: int
                Number of steps to sample
            sampling_noise: bool
                Whether to add noise during sampling
            batch_indices: torch.Tensor, optional
                Indices of the users in the batch, needed for enhanced sampling
                
        Returns:
            x_t: torch.Tensor
                The sampled tensor with KNN-enhanced predictions
        """
        # First perform standard diffusion sampling without any KNN aggregation
        predictions = self._p_sample_original(model, x_start, steps, sampling_noise)
        
        # Only apply KNN aggregation after diffusion if enabled
        if self.use_similarity and batch_indices is not None and self.top_k_similar_users is not None:
            # Apply KNN aggregation on the diffusion predictions
            return self._apply_knn_aggregation(predictions, batch_indices)
        
        return predictions
    
    def _p_sample_original(self, model, x_start, steps, sampling_noise=False):
        """
        Original p_sample method without KNN integration.
        
        Args:
            model: nn.Module
                The model to sample from
            x_start: torch.Tensor
                The starting point for sampling
            steps: int
                Number of steps to sample
            sampling_noise: bool
                Whether to add noise during sampling
                
        Returns:
            x_t: torch.Tensor
                The sampled tensor
        """
        assert steps <= self.steps, "Too many steps in inference."
        
        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t

        for i in indices:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(model, x_t, t)
            
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
                
        return x_t
    
    def _apply_knn_aggregation_self(self, predictions, batch_indices):
        batch_size = predictions.size(0)
        device = predictions.device
        
        # Initialize tensor for aggregated predictions
        aggregated_predictions = th.zeros_like(predictions)
        
        for i in range(batch_size):
            user_idx = batch_indices[i]
            user_pred = predictions[i]
            
            # Get similar users for this user
            similar_indices = self.top_k_similar_users[user_idx]
            similar_weights = self.top_k_similarities[user_idx]
            
            # Create extended arrays that include the user itself
            extended_indices = th.cat([th.tensor([user_idx], device=device), similar_indices])
            extended_weights = th.cat([th.tensor([1.0], device=device), similar_weights])
            
            # Apply normalization (either simple normalization or softmax)
            # Simple normalization:
            # weights = extended_weights / extended_weights.sum()
            # Or softmax:
            weights = th.softmax(extended_weights, dim=0)
            
            # Create tensor to hold all predictions (including the user's own)
            all_preds = th.zeros((len(extended_indices), predictions.size(1)), device=device)
            all_preds[0] = user_pred  # User's own prediction
            
            # Retrieve the predictions for similar users
            for j, sim_idx in enumerate(similar_indices):
                # Find prediction for similar user (same logic as before)
                match_indices = (batch_indices == sim_idx).nonzero(as_tuple=True)
                
                if len(match_indices[0]) > 0:
                    sim_batch_idx = match_indices[0][0]
                    all_preds[j+1] = predictions[sim_batch_idx]
                else:
                    all_preds[j+1] = self.full_user_vectors[sim_idx]
            
            # Apply weighted average of all predictions
            aggregated_predictions[i] = th.sum(weights.unsqueeze(1) * all_preds, dim=0)
        
        return aggregated_predictions
    
    def _apply_knn_aggregation(self, predictions, batch_indices):
        """
        Apply KNN aggregation on the diffusion predictions.
        
        This is the key implementation of your idea: applying KNN aggregation
        after the diffusion process has recovered user preferences.
        
        Args:
            predictions: torch.Tensor
                Recovered user vectors from diffusion process
            batch_indices: torch.Tensor
                Indices of users in the current batch
                
        Returns:
            aggregated_predictions: torch.Tensor
                KNN-enhanced predictions
        """
        batch_size = predictions.size(0)
        device = predictions.device
        
        # Initialize tensor for aggregated predictions
        aggregated_predictions = th.zeros_like(predictions)
        
        for i in range(batch_size):
            user_idx = batch_indices[i]
            user_pred = predictions[i]
            
            # Get similar users for this user
            similar_indices = self.top_k_similar_users[user_idx]
            similar_weights = self.top_k_similarities[user_idx]
                        
            # Replace this line
            #weights = th.softmax(similar_weights / self.temperature, dim=0)

            # With this
            weights = similar_weights / (similar_weights.sum() + 1e-10)  # normalize with small epsilon
            
            # Create tensor to hold similar users' predictions
            similar_preds = th.zeros((self.top_k, predictions.size(1)), device=device)
            
            # Retrieve the predictions for similar users that are in the current batch
            for j, sim_idx in enumerate(similar_indices):
                # Check if similar user is in current batch
                match_indices = (batch_indices == sim_idx).nonzero(as_tuple=True)
                
                if len(match_indices[0]) > 0:
                    # Similar user is in the batch, use its prediction
                    sim_batch_idx = match_indices[0][0]
                    similar_preds[j] = predictions[sim_batch_idx]
                else:
                    # Similar user not in batch, use its original vector (less optimal)
                    # In a full implementation, we would need to run predictions for all users
                    # or maintain a cache of predictions
                    similar_preds[j] = self.full_user_vectors[sim_idx]
            
            # Apply weighted average of similar users' predictions
            similar_contribution = th.sum(weights.unsqueeze(1) * similar_preds, dim=0)
            
            # Combine user's own prediction with similar users' predictions
            aggregated_predictions[i] = self.gamma * user_pred + (1 - self.gamma) * similar_contribution
        
        return aggregated_predictions
    
    def training_losses(self, model, x_start, batch_indices=None, reweight=False, knn_weight=0.3):
        """
        Modified training loss calculation that integrates KNN during the training process.
        
        Args:
            model: nn.Module
                The prediction model
            x_start: torch.Tensor
                The starting point for diffusion (user-item interaction vectors)
            batch_indices: torch.Tensor
                Indices of users in the current batch
            reweight: bool
                Whether to reweight loss terms
            knn_weight: float
                Weight for the KNN component in the combined loss
                
        Returns:
            terms: dict
                Dictionary containing loss terms
        """
        batch_size, device = x_start.size(0), x_start.device
        
        # Sample timesteps
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        noise = th.randn_like(x_start)
        
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        model_output = model(x_t, ts)
        
        if self.mean_type == ModelMeanType.START_X:
            target = x_start
        elif self.mean_type == ModelMeanType.EPSILON:
            target = noise
        else:
            raise ValueError(f"Unknown mean type: {self.mean_type}")

        assert model_output.shape == target.shape == x_start.shape

        # Standard diffusion MSE loss
        mse = mean_flat((target - model_output) ** 2)

        if reweight:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), th.ones_like(weight), weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = th.where((ts == 0), th.ones_like(weight), weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.ones_like(mse)

        # Standard diffusion loss
        standard_loss = weight * loss
        
        # Add KNN component to the loss if enabled and batch indices are provided
        if self.use_similarity and batch_indices is not None and self.top_k_similar_users is not None:
            # Use the current model to predict x_0 from x_t
            if self.mean_type == ModelMeanType.START_X:
                # Model directly predicts x_0
                x_0_pred = model_output
            else:  # EPSILON
                # Convert epsilon prediction to x_0
                x_0_pred = self._predict_xstart_from_eps(x_t, ts, model_output)
            
            # Detach to avoid backpropagating through diffusion process
            x_0_pred_detached = x_0_pred.detach()
            
            # Apply KNN aggregation to the predicted x_0
            knn_predictions = self._apply_knn_aggregation(x_0_pred_detached, batch_indices)
            
            # Calculate KNN loss: how well does the KNN-aggregated prediction match the target?
            knn_mse = mean_flat((target - knn_predictions) ** 2)
            knn_loss = weight * knn_mse
            
            # Combine standard diffusion loss with KNN loss
            combined_loss = (1 - knn_weight) * standard_loss + knn_weight * knn_loss
            terms["standard_loss"] = standard_loss
            terms["knn_loss"] = knn_loss
            terms["loss"] = combined_loss
        else:
            # If KNN not enabled, just use standard diffusion loss
            terms["loss"] = standard_loss
        
        # Update Lt_history & Lt_count using the final loss
        for t, l in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = l.detach()
            else:
                self.Lt_history[t, self.Lt_count[t]] = l.detach()
                self.Lt_count[t] += 1

        terms["loss"] /= pt
        return terms

    def training_losses_old(self, model, x_start, batch_indices=None, reweight=False):
        """
        Training loss calculation - no KNN during training process.
        
        Args:
            model: nn.Module
                The prediction model
            x_start: torch.Tensor
                The starting point for diffusion (user-item interaction vectors)
            batch_indices: torch.Tensor
                Indices of users in the current batch
            reweight: bool
                Whether to reweight loss terms
                
        Returns:
            terms: dict
                Dictionary containing loss terms
        """
        batch_size, device = x_start.size(0), x_start.device
        
        # Sample timesteps
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        noise = th.randn_like(x_start)
        
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        model_output = model(x_t, ts)
        
        if self.mean_type == ModelMeanType.START_X:
            target = x_start
        elif self.mean_type == ModelMeanType.EPSILON:
            target = noise
        else:
            raise ValueError(f"Unknown mean type: {self.mean_type}")

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)

        if reweight:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), th.ones_like(weight), weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = th.where((ts == 0), th.ones_like(weight), weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.ones_like(mse)

        terms["loss"] = weight * loss
        
        # Update Lt_history & Lt_count
        for t, l in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = l.detach()
            else:
                self.Lt_history[t, self.Lt_count[t]] = l.detach()
                self.Lt_count[t] += 1

        terms["loss"] /= pt
        return terms
    
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)"""
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B, )
        model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        """Sample timesteps for training."""
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')
            
            Lt_sqrt = th.sqrt(th.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / th.sum(Lt_sqrt)
            pt_all *= 1- uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = th.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt
        
        elif method == 'uniform':  # uniform sampling
            t = th.randint(0, self.steps, (batch_size,), device=device).long()
            pt = th.ones_like(t).float()

            return t, pt
            
        else:
            raise ValueError
    
    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


# Helper functions

def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))