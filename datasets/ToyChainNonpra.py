# %%
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from torchvision import datasets, transforms
from torch import nn
import os

# %%

class ToyNonParanormalLoader:
    def __init__(self, batch_size=32, device='cuda'):
        self.batch_size = batch_size
        self.device = device
        
        # --- 1. Load or Generate Precision Matrix (Latent Structure) ---
        if os.path.exists('./glasso_loader_Theta_true.npy'):
            prec = np.load('./glasso_loader_Theta_true.npy')
            prec = torch.tensor(prec, device=self.device, dtype=torch.float32)
            print("Loaded precision matrix from file.")
        else:
            print("Precision matrix file not found. Generating a synthetic CHAIN graph.")
            dim = 30
            prec = torch.eye(dim, device=self.device)
            # Create a chain structure: (i, i+1)
            for i in range(dim - 1):
                prec[i, i+1] = 0.5
                prec[i+1, i] = 0.5
                prec[i, i] = 1.2 # Ensure diagonal dominance
            prec[dim-1, dim-1] = 1.2
            
        self.dim = prec.size(0)
        
        # Visualize the Latent Precision Matrix
        plt.figure(figsize=(5, 4))
        plt.imshow(prec.cpu().numpy(), cmap = 'viridis')
        plt.title("Latent Precision Matrix (Gaussian Level)")
        plt.colorbar()
        plt.show()

        

        self.Theta = prec.clone()
        self.Sigma = torch.inverse(prec)
        
        # --- 2. Generate Latent Gaussian Data Z ---
        # Z ~ N(0, Sigma)
        latent_dist = torch.distributions.MultivariateNormal(
            torch.zeros(self.dim, device=self.device), self.Sigma
        )
        
        raw_train_z = latent_dist.sample((1024*2,))
        raw_test_z = latent_dist.sample((1000,))
        
        # --- 3. Apply Non-Paranormal Transformation ---
        # We apply a monotonic non-linear function: f(z) = sign(z) * |z|^3
        # This preserves the copula structure (graph) but changes marginals to be heavy-tailed.
        print("Applying Cubic Transformation (Non-Paranormal)...")
        
        power = 3.0
        self.xdata = torch.sign(raw_train_z) * torch.abs(raw_train_z).pow(power)
        self.xtestdata = torch.sign(raw_test_z) * torch.abs(raw_test_z).pow(power)

        # --- 4. Standardize ---
        # Neural networks struggle with the raw scale of cubic data, so we normalize
        # to mean=0, std=1. This does not undo the non-Gaussianity (kurtosis remains).
        mean = self.xdata.mean(dim=0, keepdim=True)
        std = self.xdata.std(dim=0, keepdim=True)
        
        self.xdata = (self.xdata - mean) / (std + 1e-6)
        self.xtestdata = (self.xtestdata - mean) / (std + 1e-6)
        
        print(f"Non-Paranormal Dataset: dim={self.dim}, batch={batch_size}, device={device}")

    def _process_batch(self, batch1, batch2):
        """
        Helper function to ensure Train and Test batches undergo 
        the EXACT same transformation and corruption logic.
        """
        # Randomly select one dimension to mask for each item in the batch
        indices = torch.randint(0, self.dim, (batch1.size(0),), device=self.device)
        mask = torch.zeros((batch1.size(0), self.dim), device=self.device)
        mask.scatter_(1, indices.unsqueeze(1), 1.0)
        
        # Create a batch index range on the correct device for indexing
        batch_idx = torch.arange(batch1.size(0), device=self.device)

        # Flip a coin for each item to potentially mask the next dimension
        # Only consider indices that are not the last dimension
        should_mask_next = (torch.rand(batch1.size(0), device=self.device) < 0.5) & (indices < self.dim - 1)
        
        # Use device-aware indexing
        mask[batch_idx[should_mask_next], indices[should_mask_next] + 1] = 1.0
        
        # flip a coin for each item to potentially mask the previous dimension
        should_mask_prev = (torch.rand(batch1.size(0), device=self.device) < 0.5) & (indices > 0)
        mask[batch_idx[should_mask_prev], indices[should_mask_prev] - 1] = 1.0
        
        return batch1, batch2, mask 

    def get_batch(self):
        """Returns a random batch."""
        batch1 = self.xdata[torch.randint(0, self.xdata.size(0) - self.batch_size, (self.batch_size,), device=self.device)]
        batch2 = self.xdata[torch.randint(0, self.xdata.size(0) - self.batch_size, (self.batch_size,), device=self.device)]
        return self._process_batch(batch1, batch2)

    def get_test_batch(self):
        """Returns a random batch from the TESTING set."""
        testbatch1 = self.xtestdata[torch.randint(0, self.xtestdata.size(0), (self.batch_size,), device=self.device)]
        testbatch2 = self.xtestdata[torch.randint(0, self.xtestdata.size(0), (self.batch_size,), device=self.device)]
        return self._process_batch(testbatch1, testbatch2)

# --- Usage Example ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader = ToyNonParanormalLoader(batch_size=99, device=device)

    # Get Train Batch
    v1, v2, mask = loader.get_batch()
    print(f"Train Batch: {v1.shape, v2.shape}")

    # Check Non-Gaussianity
    print("\nVisualizing Marginals (Should be non-Gaussian)...")
    plt.figure(figsize=(10, 4))
    
    # 1. Histogram of Data
    plt.subplot(1, 2, 1)
    data_np = v1[:, 0].cpu().numpy() # Take first dimension
    plt.hist(data_np, bins=30, density=True, alpha=0.7, color='orange', label='Transformed Data')
    
    # Overlay standard normal for comparison
    x = np.linspace(-3, 3, 100)
    plt.plot(x, 1/np.sqrt(2*np.pi) * np.exp(-0.5*x**2), 'k--', label='Standard Gaussian')
    plt.title("Marginal Distribution (Non-Gaussian)")
    plt.legend()
    
    # 2. Pairwise Scatter (Visualizing the Copula)
    plt.subplot(1, 2, 2)
    # Find two connected nodes to visualize
    adj = loader.Theta.cpu().numpy()
    np.fill_diagonal(adj, 0)
    r, c = np.where(np.abs(adj) > 0.1)
    if len(r) > 0:
        idx1, idx2 = r[0], c[0]
        plt.scatter(v1[:, idx1].cpu().numpy(), v1[:, idx2].cpu().numpy(), alpha=0.5, s=5)
        plt.title(f"Joint Distribution (Dim {idx1} vs {idx2})")
        plt.xlabel(f"X_{idx1}")
        plt.ylabel(f"X_{idx2}")
    else:
        plt.text(0.5, 0.5, "No Edges Found")
        
    plt.tight_layout()
    plt.show()
# %%
