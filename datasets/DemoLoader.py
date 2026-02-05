# %%
import numpy as np
import math
import torch
from torchvision import datasets, transforms
from torch import nn
# %%

class DemoLoader:
    def __init__(self, batch_size=32, dim = 64, device='cuda'):
        self.batch_size = batch_size
        self.dimx = dim//2
        self.dimy = dim//2
        self.device = device
        
        self.ydata = torch.randn(50_000, self.dimy, device=self.device)
        self.xdata = self.ydata + 0.5 * torch.randn(50_000, self.dimx, device=self.device)
        
        self.ytestdata = torch.randn(10_000, self.dimy, device=self.device)
        self.xtestdata = self.ytestdata + 0.5 * torch.randn(10_000, self.dimx, device=self.device)
        
        self.zdata = torch.cat([self.xdata, self.ydata], dim=1)
        self.ztestdata = torch.cat([self.xtestdata, self.ytestdata], dim=1)
        
        print("Toy dataset with dim =", dim, "and batch size =", batch_size, "on", device)
        

    def _process_batch(self, batch1, batch2):
        """
        Helper function to ensure Train and Test batches undergo 
        the EXACT same transformation and corruption logic.
        """
                
        return batch1, batch2, torch.zeros(1, device=self.device)

    def _gendata(self, batch_size):
        ydata = torch.randn(batch_size, self.dimy, device=self.device)
        xdata = ydata + 0.5 * torch.randn(batch_size, self.dimx, device=self.device)
        zdata = torch.cat([xdata, ydata], dim=1)
        
        return zdata
        
    def get_batch(self):
        """Returns a random batch."""
        # batch1 = self.zdata[torch.randint(0, self.xdata.size(0), (self.batch_size,), device=self.device)]
        # batch2 = self.zdata[torch.randint(0, self.xdata.size(0), (self.batch_size,), device=self.device)]
        
        batch1 = self._gendata(self.batch_size)
        batch2 = self._gendata(self.batch_size)
        
        return self._process_batch(batch1, batch2)

    def get_test_batch(self):
        """Returns a random batch from the TESTING set."""
        # testbatch1 = self.ztestdata[torch.randint(0, self.xtestdata.size(0), (self.batch_size,), device=self.device)]
        # testbatch2 = self.ztestdata[torch.randint(0, self.xtestdata.size(0), (self.batch_size,), device=self.device)]
        
        testbatch1 = self._gendata(self.batch_size)
        testbatch2 = self._gendata(self.batch_size)
        
        return self._process_batch(testbatch1, testbatch2)

# --- Usage Example ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader = DemoLoader(batch_size=99, dim = 16, device=device)

    # Get Train Batch
    batch1, batch2, _ = loader.get_batch()
    print(f"Train Batch: {batch1.shape, batch2.shape}")
    print(batch1)
    print(batch2)

    # Get Test Batch
    batch1, batch2, _ = loader.get_test_batch()
    print(f"Test Batch: {batch1.shape, batch2.shape}")


# %%
