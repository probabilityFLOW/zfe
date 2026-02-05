# %%
import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm # For progress bar

class FastFashionLoader:
    def __init__(self, csv_file, root_dir, batch_size=32, img_size=64, device='cuda', 
                 label_col='masterCategory', cache_path='./data/fashion_cache.pt'):
        self.batch_size = batch_size
        self.device = device
        
        print(f"Initializing Fast Loader on {device}...")

        # --- CACHE LOGIC START ---
        if os.path.exists(cache_path):
            print(f"Found cached dataset at {cache_path}. Loading directly...")
            saved_data = torch.load(cache_path)
            
            # Verify image size matches what we asked for
            # saved_data['images'] is (N, 3, H, W)
            if saved_data['images'].shape[-1] != img_size:
                print(f"Warning: Cached image size ({saved_data['images'].shape[-1]}) "
                      f"does not match requested size ({img_size}). Ignoring cache.")
                # Fall through to raw loading...
            else:
                self.images = saved_data['images'].to(device)
                self.labels = saved_data['labels'].to(device)
                self.classes = saved_data['classes']
                print(f"Loaded {len(self.images)} images from cache.")
                self._init_augmentations() # Helper method to setup augs
                return
        # --- CACHE LOGIC END ---

        # If we reach here, we are loading from scratch
        print("Cache not found (or invalid). Loading from raw images...")
        
        # 1. Read CSV
        df = pd.read_csv(csv_file, on_bad_lines='skip')
        
        # 2. Create Label Mapping
        self.classes = sorted(df[label_col].astype(str).unique().tolist())
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 3. Pre-allocate Lists
        images_list = []
        labels_list = []
        
        load_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), # Converts to [0, 1]
        ])

        print(f"Processing {len(df)} images into RAM...")
        
        valid_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_id = str(row['id'])
            # Handle float ids if they occur
            if img_id.endswith('.0'): img_id = img_id[:-2]
                
            img_path = os.path.join(root_dir, img_id + ".jpg")
            label_text = str(row[label_col])
            
            if not os.path.exists(img_path):
                continue
                
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    tensor_img = load_transform(img)
                    images_list.append(tensor_img)
                    
                labels_list.append(class_to_idx[label_text])
                valid_count += 1
                
            except Exception:
                pass

        if valid_count == 0:
            raise RuntimeError("No images were loaded! Check your paths.")

        # Stack tensors (CPU for now to save to disk)
        print(f"Stacking {valid_count} images...")
        tensor_images = torch.stack(images_list)
        tensor_labels = torch.tensor(labels_list, dtype=torch.long)

        # --- SAVE CACHE ---
        print(f"Saving dataset to {cache_path}...")
        torch.save({
            'images': tensor_images,
            'labels': tensor_labels,
            'classes': self.classes
        }, cache_path)
        print("Save complete.")

        # Move to GPU
        print(f"Moving to GPU VRAM...")
        self.images = tensor_images.to(device)
        self.labels = tensor_labels.to(device)
        
        self._init_augmentations()
        
        print(f"Success! VRAM used: {self.images.element_size() * self.images.nelement() / 1024**2:.2f} MB")

    def _init_augmentations(self):
            """Initializes GPU-based augmentations."""
            self.augment = torch.nn.Sequential(
                # transforms.RandomResizedCrop(size=self.images.shape[2], scale=(0.5, 1.0)),
                # transforms.RandomHorizontalFlip()
            ).to(self.device)

    def get_batch(self):
        # 1. Random Indices
        indices = torch.randint(0, len(self.images), (self.batch_size,), device=self.device)
        
        # 2. Slice (Instant)
        x_batch = self.images[indices]
        y_batch = self.labels[indices]
        
        # 3. Augment
        view1 = self.augment(x_batch)
        view2 = self.augment(x_batch)

        missingratio = torch.rand(1).item()
        # missingratio = 0.3
        # corrupt view2 with missing pixels
        mask = (torch.rand_like(view2) > missingratio).float()
        view2 = view2 * mask + torch.rand_like(view2) * (1 - mask)
        
        # 4. Flatten
        view1 = view1.view(self.batch_size, -1)
        view2 = view2.view(self.batch_size, -1)
        
        return view1, view2, y_batch

    def get_test_batch(self):
        # there is no test set, so just return a batch
        print("Warning: No separate test set available. Returning a random batch.")
        return self.get_batch()

    def get_label_text(self, label_idx):
        return self.classes[label_idx]

# # --- Usage ---
# # %%
# # Ensure paths are correct
# csv_path = './archive/styles.csv'
# img_dir = './archive/images/'

# # Initialize Once
# loader = FastFashionLoader(
#     csv_file=csv_path, 
#     root_dir=img_dir, 
#     batch_size=64, 
#     img_size=32, # Keep this small to fit in VRAM!
#     device='cuda' if torch.cuda.is_available() else 'cpu'
# )
# # %%

# # Training Loop
# for i in range(10):
#     view1, view2, labels = loader.get_batch()
#     print(f"Batch {i}: {view1.shape} (Range: {view1.min():.2f} to {view1.max():.2f})")
#     # print(f"Labels: {labels}")
#     print(f"view1: {view1.shape}, view2: {view2.shape}, labels: {labels.shape}")

# # %%
# view1batch, view2batch, labelbatch = loader.get_batch()
# from matplotlib import pyplot as plt

# plt.figure(figsize=(12, 6))
# for i in range(8):
#     plt.subplot(2, 8, i + 1)
#     plt.imshow(view1batch[i].view(1, 32, 32).permute(1, 2, 0).cpu())
#     plt.axis('off')
#     if i == 0:
#         plt.title("View 1 (Anchor)")
        
#     plt.subplot(2, 8, i + 9)
#     plt.imshow(view2batch[i].view(1, 32, 32).permute(1, 2, 0).cpu())
#     plt.axis('off')
#     if i == 0:
#         plt.title("View 2 (Positive)")
# plt.suptitle("Self-Supervised Learning Pairs from FastFashionDataset")
# plt.tight_layout()
# plt.show()