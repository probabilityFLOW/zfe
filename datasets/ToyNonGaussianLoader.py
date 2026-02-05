# %%
# %%
from sklearn.covariance import GraphicalLassoCV
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.covariance import GraphicalLasso

class ToyNonGaussianLoader:
    def __init__(self, batch_size=32, dim=64, device='cuda'):
        self.batch_size = batch_size
        self.dim = dim
        self.device = device
        
        side_len = int(math.sqrt(dim))
        assert side_len * side_len == dim, "dim must be a perfect square for a lattice grid"

        # --- 1. Construct the Precision Matrix (Lattice Structure) ---
        prec = torch.zeros((dim, dim), device=self.device)
        epsilon = 0.1 

        for r in range(side_len):
            for c in range(side_len):
                curr_idx = r * side_len + c
                neighbors = 0
                
                # Connect to Down Neighbor
                if r + 1 < side_len:
                    down_idx = (r + 1) * side_len + c
                    prec[curr_idx, down_idx] = 1
                    prec[down_idx, curr_idx] = 1
                    neighbors += 1
                    
                # Connect to Right Neighbor
                if c + 1 < side_len:
                    right_idx = r * side_len + (c + 1)
                    prec[curr_idx, right_idx] = 1
                    prec[right_idx, curr_idx] = 1
                    neighbors += 1
                    
                # Just counting neighbors for diagonal dominance
                if r > 0: neighbors += 1
                if c > 0: neighbors += 1

                prec[curr_idx, curr_idx] = neighbors + epsilon
        
        self.Theta = prec.clone()
        self.Sigma = torch.inverse(prec)
        
        # --- 2. Generate Non-Gaussian Data ---
        # Z ~ N(0, Sigma)
        self.latent_dist = torch.distributions.MultivariateNormal(
            torch.zeros(self.dim, device=self.device), self.Sigma
        )
        
        raw_train_z = self.latent_dist.sample((50000,))
        raw_test_z = self.latent_dist.sample((1000,))

        # X = Z^3 (Cubic Transform to break linearity)
        self.xdata = torch.sign(raw_train_z) * torch.abs(raw_train_z).pow(3.0)
        self.xtestdata = torch.sign(raw_test_z) * torch.abs(raw_test_z).pow(3.0)

        # Standardize
        mean = self.xdata.mean(dim=0, keepdim=True)
        std = self.xdata.std(dim=0, keepdim=True)
        self.xdata = (self.xdata - mean) / std
        self.xtestdata = (self.xtestdata - mean) / std
        
        print(f"Non-Gaussian Toy dataset (Cubic Transform) with dim={dim}, batch={batch_size} on {device}")
        

    def _process_batch(self, batch1, batch2):
        """
        Helper function to apply masking.
        """
        mask = torch.zeros((batch1.size(0), self.dim), device=self.device)
        
        # FIX: Create the batch range index on the correct device
        batch_idx_range = torch.arange(batch1.size(0), device=self.device)

        # Lattice logic for masking patches
        side_len = int(math.sqrt(self.dim))
        idxi = torch.randint(0, side_len, (batch1.size(0),), device=self.device)
        idxj = torch.randint(0, side_len, (batch1.size(0),), device=self.device)
        indices = idxi * side_len + idxj
        
        # Indices for neighbors
        indices_iplus1 = torch.clamp(idxi + 1, max=side_len - 1) * side_len + idxj
        indices_jplus1 = idxi * side_len + torch.clamp(idxj + 1, max=side_len - 1)
        indices_iminus1 = torch.clamp(idxi - 1, min=0) * side_len + idxj
        indices_jminus1 = idxi * side_len + torch.clamp(idxj - 1, min=0)
        
        # Apply mask to center
        mask[batch_idx_range, indices] = 1.0
        
        # Apply mask to neighbors probabilistically
        rand_vals = torch.rand(batch1.size(0), device=self.device)
        
        # Right
        mask_r = (rand_vals < 0.5) & (idxj < side_len - 1)
        # Fix: Use batch_idx_range (on GPU) instead of torch.arange (on CPU)
        mask[batch_idx_range[mask_r], indices_jplus1[mask_r]] = 1.0
        
        # Down
        mask_d = (rand_vals < 0.5) & (idxi < side_len - 1)
        mask[batch_idx_range[mask_d], indices_iplus1[mask_d]] = 1.0
        
        # Left
        mask_l = (rand_vals < 0.5) & (idxj > 0)
        mask[batch_idx_range[mask_l], indices_jminus1[mask_l]] = 1.0
        
        # Up
        mask_u = (rand_vals < 0.5) & (idxi > 0)
        mask[batch_idx_range[mask_u], indices_iminus1[mask_u]] = 1.0

        # Next dimension (linear index)
        mask_next = (rand_vals < 0.5) & (indices < self.dim - 1)
        mask[batch_idx_range[mask_next], indices[mask_next] + 1] = 1.0

        return batch1, batch2, mask 

    def get_batch(self):
        idx = torch.randint(0, self.xdata.size(0) - self.batch_size, (self.batch_size,), device=self.device)
        batch1 = self.xdata[idx]
        batch2 = self.xdata[torch.randint(0, self.xdata.size(0) - self.batch_size, (self.batch_size,), device=self.device)]
        return self._process_batch(batch1, batch2)

    def get_test_batch(self):
        idx = torch.randint(0, self.xtestdata.size(0), (self.batch_size,), device=self.device)
        testbatch1 = self.xtestdata[idx]
        testbatch2 = self.xtestdata[torch.randint(0, self.xtestdata.size(0), (self.batch_size,), device=self.device)]
        return self._process_batch(testbatch1, testbatch2)

# --- Usage & Verification ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use 9 dimensions (3x3 grid) for clearer visualization
    loader = ToyNonGaussianLoader(batch_size=128, dim=9, device=device)

    # 1. Plot Pairwise Joint Distribution (The "Banana" or "Star" check)
    X_numpy = loader.xdata.cpu().numpy()
    
    # Find two connected nodes (e.g., node 0 and node 1 are usually connected in lattice)
    # Check true precision to be sure
    adj = loader.Theta.cpu().numpy()
    np.fill_diagonal(adj, 0)
    rows, cols = np.where(adj > 0)
    idx1, idx2 = rows[0], cols[0] # First connected pair found
    
    

    plt.figure(figsize=(15, 5))
    
    # Plot A: Scatter of Connected Variables
    plt.subplot(1, 3, 1)
    plt.scatter(X_numpy[:, idx1], X_numpy[:, idx2], alpha=0.3, s=1)
    plt.title(f"Joint Dist: Node {idx1} vs {idx2}\n(Connected - Non-Linear!)")
    plt.xlabel(f"Node {idx1}")
    plt.ylabel(f"Node {idx2}")
    
    # Plot B: Scatter of Unconnected Variables (for contrast)
    # Find unconnected pair
    unconnected = np.where(adj[idx1] == 0)[0]
    if len(unconnected) > 0:
        idx_un = unconnected[-1] # Pick last unconnected
        plt.subplot(1, 3, 2)
        plt.scatter(X_numpy[:, idx1], X_numpy[:, idx_un], alpha=0.3, s=1)
        plt.title(f"Joint Dist: Node {idx1} vs {idx_un}\n(Unconnected)")
    
    # Plot C: Glasso Failure
    plt.subplot(1, 3, 3)
    print("Fitting Gaussian Graphical Lasso on Non-Gaussian Data...")
    
    # GraphicalLassoCV tries to find the best fit, but will likely struggle
    glasso = GraphicalLassoCV(cv=3, max_iter=2000)
    try:
        glasso.fit(X_numpy[:2000]) # Fit on subset for speed
        Theta_est = glasso.precision_
        np.fill_diagonal(Theta_est, 0)
        
        # Visualize recovered graph
        G_est = nx.from_numpy_array(np.abs(Theta_est) > 0.1) # Threshold
        pos = nx.spring_layout(G_est)
        nx.draw(G_est, pos, node_size=20, node_color='red', edge_color='gray', width=0.5)
        plt.title("Gaussian Glasso Recovery\n(Messy due to Non-Linearity)")
    except Exception as e:
        print(f"Glasso failed: {e}")
        plt.text(0.5, 0.5, "Glasso Convergence Failed", ha='center')

    plt.tight_layout()
    plt.show()
# %%