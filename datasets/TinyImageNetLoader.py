import os
import torch
import requests
import zipfile
import io
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class TinyImageNetLoader:
    def __init__(self, root_dir='./data', batch_size=32, img_size=64, device='cuda', 
                 cache_path='./data/tiny_imagenet_cache.pt'):
        self.batch_size = batch_size
        self.device = device
        self.img_size = img_size
        self.root_dir = root_dir
        self.dataset_path = os.path.join(root_dir, 'tiny-imagenet-200')
        
        print(f"Initializing Tiny ImageNet Loader on {device}...")

        # --- CACHE LOGIC ---
        if os.path.exists(cache_path):
            print(f"Found cached dataset at {cache_path}. Loading directly...")
            saved_data = torch.load(cache_path, map_location='cpu')
            
            if saved_data['train_images'].shape[-1] != img_size:
                print(f"Warning: Cache size mismatch. Re-loading...")
            else:
                self.images = saved_data['train_images'].to(device)
                self.labels = saved_data['train_labels'].to(device)
                self.test_images = saved_data['test_images'].to(device)
                self.test_labels = saved_data['test_labels'].to(device)
                self.classes = saved_data['classes']
                print(f"Loaded {len(self.images)} train and {len(self.test_images)} test images.")
                self._init_augmentations() 
                return

        # --- RAW LOADING (Train + Test) ---
        print("Cache not found. Checking for raw data...")
        self._ensure_dataset_exists()
        
        # --- DEBUG PRINT ---
        print(f"Checking directory: {self.dataset_path}")
        if os.path.exists(os.path.join(self.dataset_path, 'train')):
            classes_found = os.listdir(os.path.join(self.dataset_path, 'train'))
            print(f"Found {len(classes_found)} class folders in 'train'.")
            if len(classes_found) > 0:
                first_class = classes_found[0]
                first_class_path = os.path.join(self.dataset_path, 'train', first_class, 'images')
                num_imgs = len(os.listdir(first_class_path))
                print(f"Class '{first_class}' contains {num_imgs} images.")
        else:
            print("ERROR: 'train' folder not found!")
        # -------------------
        
        # Load Metadata (mappings from folder names to Class IDs and Text)
        self.wnids, self.id_to_cls, self.classes = self._load_metadata()
        
        load_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        # Custom processor for Tiny ImageNet structure
        def process_dataset(is_train):
            imgs_list, lbls_list = [], []
            
            if is_train:
                print("Processing Train set...")
                # Train data is in train/n01443537/images/n01443537_0.JPEG
                train_dir = os.path.join(self.dataset_path, 'train')
                
                # Iterate over class folders based on our sorted wnids list
                for class_idx, wnid in enumerate(tqdm(self.wnids)):
                    class_dir = os.path.join(train_dir, wnid, 'images')
                    if not os.path.isdir(class_dir): continue
                    
                    for filename in os.listdir(class_dir):
                        if filename.endswith('.JPEG'):
                            path = os.path.join(class_dir, filename)
                            img = Image.open(path).convert('RGB')
                            imgs_list.append(load_transform(img))
                            lbls_list.append(class_idx)
            else:
                print("Processing Test (Val) set...")
                # Val data is flat in val/images/, labels in val/val_annotations.txt
                val_img_dir = os.path.join(self.dataset_path, 'val', 'images')
                val_anno_path = os.path.join(self.dataset_path, 'val', 'val_annotations.txt')
                
                # Parse annotations: "val_0.JPEG   n01443537   ..."
                val_map = {}
                with open(val_anno_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        val_map[parts[0]] = parts[1] # filename -> wnid

                # Load images
                val_filenames = sorted(os.listdir(val_img_dir))
                for filename in tqdm(val_filenames):
                    if filename.endswith('.JPEG') and filename in val_map:
                        path = os.path.join(val_img_dir, filename)
                        wnid = val_map[filename]
                        class_idx = self.id_to_cls[wnid] # Convert wnid to int
                        
                        img = Image.open(path).convert('RGB')
                        imgs_list.append(load_transform(img))
                        lbls_list.append(class_idx)

            return torch.stack(imgs_list), torch.tensor(lbls_list, dtype=torch.long)

        train_imgs, train_lbls = process_dataset(True)
        print(f"Train set: {train_imgs.shape[0]} images.")
        test_imgs, test_lbls = process_dataset(False)
        print(f"Test set: {test_imgs.shape[0]} images.")

        print(f"Saving combined cache to {cache_path}...")
        torch.save({
            'train_images': train_imgs,
            'train_labels': train_lbls,
            'test_images': test_imgs,
            'test_labels': test_lbls,
            'classes': self.classes
        }, cache_path)

        self.images = train_imgs.to(device)
        self.labels = train_lbls.to(device)
        self.test_images = test_imgs.to(device)
        self.test_labels = test_lbls.to(device)
        
        self._init_augmentations()
        print("Success! Data loaded.")

    def _ensure_dataset_exists(self):
        """Downloads and extracts Tiny ImageNet if not found."""
        if not os.path.exists(self.dataset_path):
            print("Downloading Tiny ImageNet-200...")
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            os.makedirs(self.root_dir, exist_ok=True)
            
            r = requests.get(url, stream=True)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            print("Extracting...")
            z.extractall(self.root_dir)
            print("Download complete.")

    def _load_metadata(self):
        """Parses wnids.txt and words.txt to create class mappings."""
        wnids_path = os.path.join(self.dataset_path, 'wnids.txt')
        words_path = os.path.join(self.dataset_path, 'words.txt')
        
        # 1. Load WNIDs (determines the integer index 0-199)
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f]
        
        id_to_cls = {wnid: i for i, wnid in enumerate(wnids)}
        
        # 2. Load Words (human readable names)
        wnid_to_text = {}
        with open(words_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    wnid_to_text[parts[0]] = parts[1]
        
        # 3. Create list of class names in order 0-199
        classes = [wnid_to_text.get(wnid, wnid) for wnid in wnids]
        
        return wnids, id_to_cls, classes

    def _init_augmentations(self):
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
        ])

    def get_batch(self):
        """Returns training batch with corruption."""
        indices = torch.randint(0, len(self.images), (self.batch_size,), device=self.device)
        x_batch = self.images[indices]
        y_batch = self.labels[indices]
        
        return self._process_views(x_batch, y_batch)

    def get_test_batch(self):
        """Returns testing batch with the same corruption logic as training."""
        indices = torch.randint(0, len(self.test_images), (self.batch_size,), device=self.device)
        x_batch = self.test_images[indices]
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