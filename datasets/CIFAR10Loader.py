# %%
import os
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

class CIFAR10Loader:
    def __init__(self, root_dir='./data', batch_size=32, img_size=32, device='cuda', 
                 cache_path='./data/cifar10_full_cache.pt'):
        self.batch_size = batch_size
        self.device = device
        self.img_size = img_size
        
        print(f"Initializing CIFAR-10 Loader on {device}...")

        # --- CACHE LOGIC ---
        if os.path.exists(cache_path):
            print(f"Found cached dataset at {cache_path}. Loading directly...")
            saved_data = torch.load(cache_path, map_location='cpu')
            
            if saved_data['train_images'].shape[-1] != img_size:
                print(f"Warning: Cache size mismatch. Re-loading...")
            else:
                self.data = saved_data['train_images'].to(device)
                self.labels = saved_data['train_labels'].to(device)
                self.test_data = saved_data['test_images'].to(device)
                self.test_labels = saved_data['test_labels'].to(device)
                self.classes = saved_data['classes']
                print(f"Loaded {len(self.data)} train and {len(self.test_data)} test images.")
                self._init_augmentations() 
                return

        # --- RAW LOADING (Train + Test) ---
        print("Cache not found. Loading Train & Test sets from raw...")
        
        load_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        def process_dataset(is_train):
            ds = datasets.CIFAR10(root=root_dir, train=is_train, download=True)
            imgs_list, lbls_list = [], []
            print(f"Processing {'Train' if is_train else 'Test'} set...")
            for img, label in tqdm(ds):
                imgs_list.append(load_transform(img))
                lbls_list.append(label)
            return torch.stack(imgs_list), torch.tensor(lbls_list, dtype=torch.long), ds.classes

        train_imgs, train_lbls, classes = process_dataset(True)
        test_imgs, test_lbls, _ = process_dataset(False)

        print(f"Saving combined cache to {cache_path}...")
        torch.save({
            'train_images': train_imgs,
            'train_labels': train_lbls,
            'test_images': test_imgs,
            'test_labels': test_lbls,
            'classes': classes
        }, cache_path)

        self.data = train_imgs.to(device)
        self.labels = train_lbls.to(device)
        self.test_data = test_imgs.to(device)
        self.test_labels = test_lbls.to(device)
        self.classes = classes
        
        self._init_augmentations()
        print("Success! Data loaded.")

    def _init_augmentations(self):
        self.augment = torch.nn.Sequential(
            transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            # transforms.RandomGrayscale(p=0.5),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
        ).to(self.device)

    def get_batch(self):
        """Returns training batch with corruption."""
        indices = torch.randint(0, len(self.data), (self.batch_size,), device=self.device)
        x_batch = self.data[indices]
        y_batch = self.labels[indices]
        
        return self._process_views(x_batch, y_batch)

    def get_test_batch(self):
        """Returns testing batch with the same corruption logic as training."""
        indices = torch.randint(0, len(self.test_data), (self.batch_size,), device=self.device)
        x_batch = self.test_data[indices]
        y_batch = self.test_labels[indices]
        
        return self._process_views(x_batch, y_batch)

    def _process_views(self, x_batch, y_batch):
        """Helper to apply the View1/View2 + Corruption logic shared by both methods."""
        # 1. Create Views
        view1 = x_batch.clone()
        view2 = self.augment(x_batch)

        # # 2. Corruption Logic
        # missingratio = torch.rand(1).item()
        # mask = (torch.rand_like(view2) > missingratio).float()
        # view2 = view2 * mask + torch.rand_like(view2) * (1 - mask)
        
        # 3. Flatten
        view1 = view1.view(self.batch_size, -1)
        view2 = view2.reshape(self.batch_size, -1)
        
        return view1, view2, y_batch

    def get_label_text(self, label_idx):
        if isinstance(label_idx, torch.Tensor):
            label_idx = label_idx.item()
        return self.classes[label_idx]
    
# testing code
if __name__ == "__main__":
    loader = CIFAR10Loader(batch_size=16, img_size=32, device='cpu')
    x, y, label = loader.get_batch()
    print("Batch X shape:", x.shape)
    print("Batch Y shape:", y.shape)
    for i in range(loader.batch_size):
        print("Label for image", i, ":", loader.get_label_text(label[i]))
# %%