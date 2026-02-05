# %%
import numpy as np
import math
import torch
from torchvision import datasets, transforms
from torch import nn
# %%

class ToyLoader:
    def __init__(self, batch_size=32, dim = 64, device='cuda'):
        self.batch_size = batch_size
        self.dim = dim
        self.device = device
        
        side_len = int(math.sqrt(dim))
        assert side_len * side_len == dim, "dim must be a perfect square for a lattice grid"

        prec = torch.zeros((dim, dim), device=self.device)

        # We use an offset (epsilon) to ensure Positive Definiteness
        # The diagonal must be > sum of absolute off-diagonals
        epsilon = 0.1 

        for r in range(side_len):
            for c in range(side_len):
                # 1. Calculate linear index for current node (r, c)
                curr_idx = r * side_len + c
                
                # Count neighbors to determine diagonal strength automatically
                # (Like your original code: boundary nodes have smaller diagonals)
                neighbors = 0
                
                # 2. Connect to Down Neighbor (r+1, c)
                if r + 1 < side_len:
                    down_idx = (r + 1) * side_len + c
                    prec[curr_idx, down_idx] = 1
                    prec[down_idx, curr_idx] = 1
                    neighbors += 1
                    
                # 3. Connect to Right Neighbor (r, c+1)
                if c + 1 < side_len:
                    right_idx = r * side_len + (c + 1)
                    prec[curr_idx, right_idx] = 1
                    prec[right_idx, curr_idx] = 1
                    neighbors += 1
                    
                # 4. Check Up and Left (just for neighbor count, links already made)
                if r > 0: neighbors += 1
                if c > 0: neighbors += 1

                # 5. Set Diagonal
                # Your original code used (neighbors + 0.5) roughly
                # 2 neighbors -> diag 2.0? Actually, strict dominance is safer.
                # Let's use: Diagonal = Num_Neighbors + epsilon
                prec[curr_idx, curr_idx] = neighbors + epsilon
        
        for row in prec:
            print(' '.join([f"{val.item():.1f}" for val in row]))
            
        self.Theta = prec.clone()
        self.Sigma = torch.inverse(prec)
        
        self.xdata = torch.distributions.MultivariateNormal(
                    torch.zeros(self.dim, device=self.device), self.Sigma).sample((2048,))
        
        self.xtestdata = torch.distributions.MultivariateNormal(
                    torch.zeros(self.dim, device=self.device), self.Sigma).sample((1000,))
        
        
        print("Toy dataset with dim =", dim, "and batch size =", batch_size, "on", device)
        

    def _process_batch(self, batch1, batch2):
        """
        Helper function to ensure Train and Test batches undergo 
        the EXACT same transformation and corruption logic.
        """
        mask = torch.zeros((batch1.size(0), self.dim), device=self.device)
        # select random patches on the lattice to mask
        side_len = int(math.sqrt(self.dim))
        idxi = torch.randint(0, side_len, (batch1.size(0),), device=self.device)
        idxj = torch.randint(0, side_len, (batch1.size(0),), device=self.device)
        indices = idxi * side_len + idxj
        indices_iplus1 = torch.clamp(idxi + 1, max=side_len - 1) * side_len + idxj
        indices_jplus1 = idxi * side_len + torch.clamp(idxj + 1, max=side_len - 1)
        indices_iminus1 = torch.clamp(idxi - 1, min=0) * side_len + idxj
        indices_jminus1 = idxi * side_len + torch.clamp(idxj - 1, min=0)
        
        mask[torch.arange(batch1.size(0), device=self.device), indices] = 1.0
        # roll a dice to also mask the right neighbor with 50% chance
        should_mask_next = (torch.rand(batch1.size(0), device=self.device) < 0.5) & (idxj < side_len - 1)
        mask[torch.arange(batch1.size(0), device=self.device)[should_mask_next], indices_jplus1[should_mask_next]] = 1.0
        # roll a dice to also mask the down neighbor with 50% chance
        should_mask_next = (torch.rand(batch1.size(0), device=self.device) < 0.5) & (idxi < side_len - 1)
        mask[torch.arange(batch1.size(0), device=self.device)[should_mask_next], indices_iplus1[should_mask_next]] = 1.0
        # roll a dice to also mask the left neighbor with 50% chance
        should_mask_next = (torch.rand(batch1.size(0), device=self.device) < 0.5) & (idxj > 0)
        mask[torch.arange(batch1.size(0), device=self.device)[should_mask_next], indices_jminus1[should_mask_next]] = 1.0
        # roll a dice to also mask the up neighbor with 50% chance
        should_mask_next = (torch.rand(batch1.size(0), device=self.device) < 0.5) & (idxi > 0)
        mask[torch.arange(batch1.size(0), device=self.device)[should_mask_next], indices_iminus1[should_mask_next]] = 1.0

        # Flip a coin for each item to potentially mask the next dimension
        # Only consider indices that are not the last dimension
        should_mask_next = (torch.rand(batch1.size(0), device=self.device) < 0.5) & (indices < self.dim - 1)
        mask[torch.arange(batch1.size(0), device=self.device)[should_mask_next], indices[should_mask_next] + 1] = 1.0

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
    loader = ToyLoader(batch_size=99, dim = 16, device=device)

    # Get Train Batch
    v1, v2, mask = loader.get_batch()
    print(f"Train Batch: {v1.shape, v2.shape}")
    print(v1*mask)
    print(v2*(1-mask))
    print(f"Index: {mask[:5, :].cpu().numpy()}")

    # Get Test Batch
    test_v1, test_v2, test_idx = loader.get_test_batch()
    print(f"Test Batch:  {test_v1.shape, test_v2.shape}")
    print(f"Index: {test_idx[:5, :].cpu().numpy()}")

    # Visualize the adjancency structure in the precision matrix as a graph
    import matplotlib.pyplot as plt
    import networkx as nx
    Theta = loader.Sigma.inverse().cpu().numpy(); np.fill_diagonal(Theta, 0)
    Theta = Theta > 1e-5  # Threshold for visualization
    G = nx.from_numpy_array(Theta)
    plt.figure(figsize=(6,6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Graphical Model Structure from Precision Matrix")
    plt.show()


# %%
