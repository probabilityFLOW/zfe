# %%
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from torchvision import datasets, transforms
from torch import nn
# %%

class SNP500Loader:
    def __init__(self, batch_size=32, device='cuda'):
        self.batch_size = batch_size
        self.device = device
        
        # load SNP500 processed data
        data_tensor = torch.load('data/snp500_processed_returns.pt', map_location=self.device)
        data_tensor = data_tensor.T
        self.dim = data_tensor.size(1)
        self.xdata = data_tensor.to(self.device)
        
        print("SNP dataset with n_samples =", self.xdata.size(0), "and dim =", self.dim)
        # test data to none
        self.xtestdata = None
         
    def _process_batch(self, batch1, batch2):
            """
            Helper function to ensure Train and Test batches undergo 
            the EXACT same transformation and corruption logic.
            Now masks a random block of 5 consecutive elements.
            """
            batch_size = batch1.size(0)
            mask_length = 5
            
            # Ensure dimension is large enough
            if self.dim < mask_length:
                raise ValueError(f"Input dimension {self.dim} is too small for a {mask_length}-element mask.")

            # 1. Randomly select a start index for each item in the batch.
            #    Must be between 0 and dim - 5 to ensure the window fits.
            start_indices = torch.randint(0, self.dim - mask_length + 1, (batch_size, 1), device=self.device)

            # 2. Create offsets [0, 1, 2, 3, 4]
            offsets = torch.arange(mask_length, device=self.device).unsqueeze(0) # Shape (1, 5)

            # 3. Calculate the actual indices to mask: start + offset
            #    Broadcasting: (batch, 1) + (1, 5) -> (batch, 5)
            mask_indices = start_indices + offsets

            # 4. Create the mask
            mask = torch.zeros((batch_size, self.dim), device=self.device)
            
            # 5. Scatter 1.0 into the selected positions
            #    src is broadcasted to match mask_indices shape
            mask.scatter_(1, mask_indices, 1.0)
            
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
    loader = SNP500Loader(batch_size=99, device=device)

    # Get Train Batch
    v1, v2, mask = loader.get_batch()
    print(f"Train Batch: {v1.shape, v2.shape}")
    print(v1*mask)
    print(v2*(1-mask))
    print(f"Index: {mask[:5, :].cpu().numpy()}")


# %%
