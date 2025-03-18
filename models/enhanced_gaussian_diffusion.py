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

class EnhancedGaussianDiffusion(nn.Module):
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
        Enhanced Gaussian Diffusion with user similarity integration.
        
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

        super(EnhancedGaussianDiffusion, self).__init__()
        
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
            
        self.similarity_matrix = similarity_matrix
        self.top_k_similar_users = top_k_similar_users
        self.top_k_similarities = top_k_similarities
        self.full_user_vectors = full_user_vectors
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
        Enhanced sample method that uses the original implementation but with batch indices.
        
        This is a wrapper that decides whether to use the enhanced sampling or the original
        based on whether batch_indices are provided.
        
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
                The sampled tensor
        """
        # If batch indices are provided and similarity is enabled, use enhanced sampling
        if batch_indices is not None and self.use_similarity and self.top_k_similar_users is not None:
            from enhanced_sampling import enhance_p_sample
            return enhance_p_sample(self, model, x_start, steps, batch_indices, sampling_noise)
        else:
            # Otherwise use the original sampling method
            return self._p_sample_original(model, x_start, steps, sampling_noise)
    
    def _p_sample_original(self, model, x_start, steps, sampling_noise=False):
        """
        Original sampling method without the enhanced batch processing.
        
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
        assert steps <= self.steps, "Too much steps in inference."
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
                
            # # Apply user similarity aggregation at the final step, only in the original method
            # if i == 0 and self.use_similarity and self.similarity_matrix is not None:
            #     # Only apply similarity aggregation at the final denoising step
            #     x_t = self.apply_similarity_aggregation(x_t)
                
        return x_t
    
    # def apply_similarity_aggregation(self, user_vectors):
    #     """
    #     Apply similarity-based aggregation to user vectors.
        
    #     Args:
    #         user_vectors: torch.Tensor
    #             The user vectors to aggregate
                
    #     Returns:
    #         aggregated_vectors: torch.Tensor
    #             The aggregated user vectors
    #     """
    #     if not self.use_similarity or self.similarity_matrix is None:
    #         return user_vectors
            
    #     # Use the top-k similar users to aggregate vectors
    #     return aggregate_similar_users(
    #         user_vectors, 
    #         self.full_user_vectors,  # Use the stored full user-item matrix
    #         self.top_k_similar_users, 
    #         self.top_k_similarities, 
    #         gamma=self.gamma,
    #         temperature=self.temperature
    #     )
    
    def training_losses(self, model, x_start, reweight=False):
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        noise = th.randn_like(x_start)
        
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        model_output = model(x_t, ts)
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)

        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.tensor([1.0] * len(target)).to(device)

        terms["loss"] = weight * loss
        
        # update Lt_history & Lt_count
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError

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