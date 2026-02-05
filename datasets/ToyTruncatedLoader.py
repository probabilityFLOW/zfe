# %%
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from torchvision import datasets, transforms
from torch import nn
import os
import networkx as nx

# %%

class ToyTruncatedLoader:
    def __init__(self, batch_size=32, device='cuda'):
        self.batch_size = batch_size
        self.device = device
        
        # --- 1. Load or Generate Precision Matrix ---
        if os.path.exists('./glasso_loader_Theta_true.npy'):
            prec = np.load('./glasso_loader_Theta_true.npy')
            prec = torch.tensor(prec, device=self.device, dtype=torch.float32)
            print("Loaded precision matrix from file.")
        else:
            print("Precision matrix file not found. Generating a synthetic CHAIN graph.")
            dim = 16
            prec = torch.eye(dim, device=self.device)
            # Create a chain structure
            for i in range(dim - 1):
                prec[i, i+1] = 0.5
                prec[i+1, i] = 0.5
                prec[i, i] = 1.25
            prec[dim-1, dim-1] = 1.25

        self.dim = prec.size(0)
        self.Theta = prec.clone()
        self.Sigma = torch.inverse(prec)

        # --- 2. Setup Base Distribution ---
        self.mu = torch.ones(self.dim, device=self.device) * 0.0  # Shift mean to reduce rejections
        
        self.base_dist = torch.distributions.MultivariateNormal(self.mu, self.Sigma)

        # --- 3. Generate Truncated Data ---
        print(f"Generating Truncated Gaussian Data (All Dim > -0.5)...")
        self.xdata = self._generate_data(n_samples=1024*2)
        self.xtestdata = self._generate_data(n_samples=1000)
        
        print(f"Truncated Dataset: dim={self.dim}, batch={batch_size}, device={device}")

    def _generate_data(self, n_samples):
        """
        Generates samples and rejects any that have a value <= -0.5 in ANY dimension.
        """
        collected = []
        counts = 0
        
        # Limit infinite loops if rejection rate is too high
        attempts = 0
        max_attempts = 5000000

        while counts < n_samples and attempts < max_attempts:
            # Oversample to account for rejections
            raw = self.base_dist.sample((4096*100,))
            
            # --- TRUNCATION CONDITION ---
            # Keep sample only if ALL dimensions are > -1.0
            mask_valid = (raw > -.75).all(dim=1)
            
            valid_samples = raw[mask_valid]
            
            if valid_samples.size(0) > 0:
                collected.append(valid_samples)
                counts += valid_samples.size(0)
            
            attempts += 1
            print(valid_samples.size(0), " valid samples collected. Total so far:", counts, end='\r')
        if len(collected) == 0:
            raise RuntimeError("Rejection sampling failed. Try increasing mu_val.")

        # Concatenate and trim
        full_data = torch.cat(collected, dim=0)
        return full_data[:n_samples]

    def _process_batch(self, batch1, batch2):
        indices = torch.randint(0, self.dim, (batch1.size(0),), device=self.device)
        mask = torch.zeros((batch1.size(0), self.dim), device=self.device)
        mask.scatter_(1, indices.unsqueeze(1), 1.0)
        
        batch_idx = torch.arange(batch1.size(0), device=self.device)
        should_mask_next = (torch.rand(batch1.size(0), device=self.device) < 0.5) & (indices < self.dim - 1)
        mask[batch_idx[should_mask_next], indices[should_mask_next] + 1] = 1.0
        
        should_mask_prev = (torch.rand(batch1.size(0), device=self.device) < 0.5) & (indices > 0)
        mask[batch_idx[should_mask_prev], indices[should_mask_prev] - 1] = 1.0
        
        return batch1, batch2, mask 

    def get_batch(self):
        batch1 = self.xdata[torch.randint(0, self.xdata.size(0) - self.batch_size, (self.batch_size,), device=self.device)]
        batch2 = self.xdata[torch.randint(0, self.xdata.size(0) - self.batch_size, (self.batch_size,), device=self.device)]
        return self._process_batch(batch1, batch2)

    def get_test_batch(self):
        testbatch1 = self.xtestdata[torch.randint(0, self.xtestdata.size(0), (self.batch_size,), device=self.device)]
        testbatch2 = self.xtestdata[torch.randint(0, self.xtestdata.size(0), (self.batch_size,), device=self.device)]
        return self._process_batch(testbatch1, testbatch2)

# --- Usage Example ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader = ToyTruncatedLoader(batch_size=99, device=device)

    # 1. Verify Truncation
    print(f"Mean values of dataset: {loader.xdata.mean(dim=0)} (Should be > -0.5)")
    
    v1, _, _ = loader.get_batch()
    
    plt.figure(figsize=(8, 4))
    data_np = v1[:, 0].cpu().numpy()
    plt.hist(data_np, bins=40, alpha=0.7, color='purple', label='Data')
    plt.axvline(x=-0.5, color='red', linestyle='--', linewidth=2, label='Truncation Line (-0.5)')
    plt.title("Marginal Distribution (Dim 0)")
    plt.legend()
    plt.show()

    print(f"\nMinimum value in dataset: {loader.xdata.min().item():.4f} (Should be > -0.5)")
# %%