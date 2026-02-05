# %%
import torch
from torchvision import datasets, transforms

class FastMNISTLoader:
    def __init__(self, batch_size=32, device='cuda', root='./data'):
        self.batch_size = batch_size
        self.device = device
        
        print("Loading FashionMNIST to memory...")
        
        # --- 1. Load Training Data ---
        raw_train = datasets.FashionMNIST(root=root, train=True, download=True)
        # Convert to Float, Scale, Add Channel Dim, Move to GPU
        self.data = raw_train.data.float().unsqueeze(1).to(device) / 255.0
        self.labels = raw_train.targets.to(device)
        
        # --- 2. Load Testing Data ---
        raw_test = datasets.FashionMNIST(root=root, train=False, download=True)
        self.test_data = raw_test.data.float().unsqueeze(1).to(device) / 255.0
        self.test_labels = raw_test.targets.to(device)

        # --- 3. Resize Images ---
        self.transform = torch.nn.Sequential(
            transforms.Resize((32, 32)),
        ).to(device)
        
        # pass trainingdata and testing data through transform once to cache
        self.data = self.transform(self.data)
        self.test_data = self.transform(self.test_data)
        
        self._init_augmentations()
        
        print(f"Loaded {len(self.data)} Train images and {len(self.test_data)} Test images to {device}")

    def _init_augmentations(self):
        # self.augment = transforms.Compose([
        #     transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
        #     transforms.RandomHorizontalFlip(),
        #     ])
        
        # no transform
        # self.augment = torch.nn.Sequential().to(self.device)
        
        self.augment = None

    def _process_batch(self, x_clean, labels):
        """
        Helper function to ensure Train and Test batches undergo 
        the EXACT same transformation and corruption logic.
        """
        view1 = x_clean.clone()
        if self.augment is not None:
            view2 = self.augment(x_clean)
        else:
            view2 = x_clean.clone()
        
        # Corrupt view2 with missing pixels (Same logic as before)
        missingratio = torch.rand(1).item()
        mask = (torch.rand_like(view2) > missingratio).float()
        view2 = view2 * mask + torch.rand_like(view2) * (1 - mask)
        
        # Flatten
        view1 = view1.view(self.batch_size, -1)
        view2 = view2.view(self.batch_size, -1)
        
        return view1, view2, labels

    def get_batch(self):
        """Returns a random batch from the TRAINING set."""
        indices = torch.randint(0, len(self.data), (self.batch_size,), device=self.device)
        x_clean = self.data[indices]
        labels = self.labels[indices]
        
        return self._process_batch(x_clean, labels)

    def get_test_batch(self):
        """Returns a random batch from the TESTING set."""
        indices = torch.randint(0, len(self.test_data), (self.batch_size,), device=self.device)
        x_clean = self.test_data[indices]
        labels = self.test_labels[indices]
        
        return self._process_batch(x_clean, labels)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader = FastMNISTLoader(batch_size=64, device=device)

    v1, v2, y = loader.get_batch()
    print(f"Train Batch: {v1.shape}")

    # Get Test Batch
    test_v1, test_v2, test_y = loader.get_test_batch()
    print(f"Test Batch:  {test_v1.shape}")
# %%
