# %%
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class PatchedCIFARLoader:
    def __init__(self, patched=True, batch_size=32, device='cuda', root='./data'):
        self.batch_size = batch_size
        self.device = device
        self.patched = patched
        self.patch_size = 10
        
        print("Loading CIFAR-10 to memory...")
        
        # --- 1. Load Training Data ---
        raw_train = datasets.CIFAR10(root=root, train=True, download=True)
        # CIFAR is already [N, 32, 32, 3]. Convert to [N, 3, 32, 32], Scale to [0,1]
        self.data = torch.tensor(raw_train.data).permute(0, 3, 1, 2).float().to(device) / 255.0
        self.labels = torch.tensor(raw_train.targets).to(device)
        
        # --- 2. Load Testing Data ---
        raw_test = datasets.CIFAR10(root=root, train=False, download=True)
        self.test_data = torch.tensor(raw_test.data).permute(0, 3, 1, 2).float().to(device) / 255.0
        self.test_labels = torch.tensor(raw_test.targets).to(device)

        # --- 3. Generate FIXED Random Patches (The "Loud" Shortcut) ---
        # Each image ID gets a unique, permanent RGB patch (8x8 pixels).
        torch.manual_seed(42) 
        # Shape: [N, 3, 8, 8]
        self.patch_data = torch.rand(self.data.size(0), 3, self.patch_size, self.patch_size, device=self.device)
        self.patch_test_data = torch.rand(self.test_data.size(0), 3, self.patch_size, self.patch_size, device=self.device)
        
        self._init_augmentations()
        
        print(f"Loaded {len(self.data)} Train images and {len(self.test_data)} Test images to {device}")

    def _init_augmentations(self):
        # Standard SimCLR spatial augmentations
        self.augment = torch.nn.Sequential(
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ).to(self.device)

    def _apply_patch(self, views, patches):
        """
        Overwrites the top-left corner of the views with the unique patches.
        views: [B, 3, 32, 32]
        patches: [B, 3, 8, 8]
        """
        # We clone to avoid modifying in place if needed, though usually safe here
        out = views.clone()
        # Overwrite top-left corner
        out[:, :, :self.patch_size, :self.patch_size] = patches
        return out

    def _process_batch(self, x_clean, labels, patches):
        """
        Applies Gradient Starvation Logic:
        1. Dim the Object (Quiet)
        2. Augment
        3. Apply Patch (Loud)
        4. Add Noise (Distraction)
        """
        # --- 1. DIM THE OBJECT (Quiet Feature) ---
        # We suppress the natural image so gradients from the object are tiny.
        x_quiet = x_clean * 1
        
        # --- 2. AUGMENT ---
        view1 = self.augment(x_quiet)
        view2 = self.augment(x_quiet)
        
        # --- 3. APPLY PATCH (Loud Feature) ---
        if self.patched:
            # We apply the patch AFTER augmentation to ensure it's always visible 
            # in the same location (e.g., top-left) acting as a perfect shortcut.
            view1 = self._apply_patch(view1, patches)
            view2 = self._apply_patch(view2, patches)
        
        # --- 4. ADD NOISE ---
        # Noise (0.4) is louder than the object (0.2) but quieter than the patch (1.0).
        noise_level = 0.01
        view1 = view1 + torch.randn_like(view1) * noise_level
        view2 = view2 + torch.randn_like(view2) * noise_level
        
        # Clamp to valid image range [0, 1]
        view1 = torch.clamp(view1, 0, 1)
        view2 = torch.clamp(view2, 0, 1)
        
        # Flatten (Matching your previous interface)
        view1 = view1.view(self.batch_size, -1)
        view2 = view2.view(self.batch_size, -1)
        
        return view1, view2, labels

    def get_batch(self):
        """Returns a random batch from the TRAINING set."""
        indices = torch.randint(0, len(self.data), (self.batch_size,), device=self.device)
        
        x_clean = self.data[indices]
        labels = self.labels[indices]
        patches = self.patch_data[indices]
        
        return self._process_batch(x_clean, labels, patches)

    def get_test_batch(self):
        """Returns a random batch from the TESTING set."""
        indices = torch.randint(0, len(self.test_data), (self.batch_size,), device=self.device)
        
        x_clean = self.test_data[indices]
        labels = self.test_labels[indices]
        patches = self.patch_test_data[indices]
        
        return self._process_batch(x_clean, labels, patches)
    
    def get_label_text(self, label):
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return classes[label]

# --- Usage & Visualization ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize Loader
    loader = PatchedCIFARLoader(batch_size=8, device=device)

    # Get Train Batch
    v1, v2, y = loader.get_batch()
    print(f"Train Batch Shape (Flattened): {v1.shape}") 
    
    # Visualize
    # Reshape back to image dimensions for plotting
    v1_imgs = v1.view(-1, 3, 32, 32)
    v2_imgs = v2.view(-1, 3, 32, 32)
    
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
    
    # Move to CPU and Permute [C, H, W] -> [H, W, C]
    v1_cpu = v1_imgs.cpu().permute(0, 2, 3, 1).numpy()
    v2_cpu = v2_imgs.cpu().permute(0, 2, 3, 1).numpy()
    
    # Remove whitespace
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    for i in range(8):
        # Plot View 1
        axes[0, i].imshow(v1_cpu[i])
        axes[0, i].axis('off')
        
        # Plot View 2
        axes[1, i].imshow(v2_cpu[i])
        axes[1, i].axis('off')
        
    plt.show()
# %%