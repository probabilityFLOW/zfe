# %%
import numpy as np
import math
import torch
from torchvision import datasets, transforms
from torch import nn

# %%
class LoudQuietLoader:
    def __init__(self, batch_size=32, dim=64, device='cuda'):
        self.batch_size = batch_size
        self.dim = dim
        self.device = device
        
        print(f"Loud/Quiet Data Loader: dim={dim}, batch_size={batch_size}, device={device}")
        
    def _gendata(self, batch_size):
        """
        Generates the SHARED Latent Factors.
        Returns tensor of shape (batch_size, 2).
        Column 0: Loud Feature 'c' (Binary {-1, 1})
        Column 1: Quiet Feature 's' (Binary {-1, 1})
        """
        # 1. Sample Latents
        c = torch.randint(0, 2, (batch_size, 1), device=self.device).float() * 2 - 1 # {-1, 1}
        s = torch.randint(0, 2, (batch_size, 1), device=self.device).float() * 2 - 1 # {-1, 1}
        
        # Concatenate them into a single latent vector
        latents = torch.cat([c, s], dim=1)
        return latents

    def _process_batch(self, latents):
        """
        Takes shared latents and produces two views (Positive Pairs).
        Each view gets the shared signals + independent noise.
        """
        N = latents.shape[0]
        
        # Initialize empty views with adequate dimension
        # We need at least dim=2 to hold the signals. 
        # The rest is noise.
        view1 = torch.zeros(N, self.dim, device=self.device)
        view2 = torch.zeros(N, self.dim, device=self.device)
        
        c = latents[:, 0]
        s = latents[:, 1]
        
        # --- Inject LOUD Feature (Magnitude 5.0) at Index 0 ---
        view1[:, 0] = c * 10.0
        view2[:, 0] = c * 10.0
        
        # --- Inject QUIET Feature (Magnitude 1.0) at Index 1 ---
        view1[:, 1] = s * 2.0
        view2[:, 1] = s * 2.0
        
        # --- Add Independent Gaussian Noise to ALL dimensions ---
        # This acts as the "nuisance" and ensures the trivial solution 
        # isn't just matching exact floating point values.
        view1 = view1 + torch.randn_like(view1)
        view2 = view2 + torch.randn_like(view2)
        
        # Return view1, view2, and the latents (labels) for visualization
        return view1, view2, latents

    def get_batch(self):
        """Returns a random batch of Positive Pairs."""
        # 1. Generate one set of shared latents
        latents = self._gendata(self.batch_size)
        
        # 2. Create two views from these latents
        return self._process_batch(latents)

    def get_test_batch(self):
        """Returns a random batch for testing (same logic)."""
        latents = self._gendata(self.batch_size)
        return self._process_batch(latents)

# --- Usage Example ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use dim=10 to match the previous experiment explanation
    loader = LoudQuietLoader(batch_size=5, dim=10, device=device)

    # Get Train Batch
    view1, view2, labels = loader.get_batch()
    
    print(f"\n--- Batch Info ---")
    print(f"View Shapes: {view1.shape}")
    print(f"Loud Feature (Col 0) Magnitude: ~5.0")
    print(f"Quiet Feature (Col 1) Magnitude: ~1.0")
    
    print(f"\n--- View 1 (First 2 Rows) ---")
    print(view1[:2])
    print(f"\n--- View 2 (First 2 Rows) ---")
    print(view2[:2])
    print(f"\n--- Shared Labels (Loud, Quiet) ---")
    print(labels[:2])
# %%