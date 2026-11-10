import os
from typing import Tuple, Union

import numpy as np
from continuum.datasets import ImageFolderDataset, _ContinuumDataset
from continuum.download import download, untar
from continuum.tasks import TaskType
from torchvision import transforms
from torchvision import datasets as tv_datasets
from shutil import move, rmtree
import tarfile

from continuum.datasets import ImageFolderDataset, _ContinuumDataset
from continuum.scenarios import ClassIncremental
import os
from PIL import Image


import os
import numpy as np
from PIL import Image
from torchvision import transforms
from continuum.datasets import _ContinuumDataset

class ContinuumCLRSDataset(_ContinuumDataset):
    """
    CLRS (Remote Sensing) dataset with Continuum compatibility.
    Supports class-incremental learning for satellite/airborne imagery.
    Dataset is assumed to exist locally at "../../online_cl/CLRS".
    """
    
    def __init__(self, root="../../online_cl/datasets", train=True, download=False):

        """
        Args:
            root (str): Path to existing CLRS dataset (default: "../../online_cl/CLRS")
            train (bool): If True, loads training split (80%), else test (20%)
            download (bool): Not used (kept for API consistency)
        """

        #root = "../../online_cl/datasets/clrs25/CLRS" ## replace with your datapath

        root = os.path.join(root, "CLRS")
        super().__init__(data_path=root, train=train, download=False)
        self.root = os.path.expanduser(root)
        self.train = train
        
        # Remote sensing classes (25 classes as per your folder structure)
        self.classes = sorted([
            d for d in os.listdir(self.root) 
            if os.path.isdir(os.path.join(self.root, d))
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Verify dataset structure
        self._verify_structure()
        
        # Load data paths and labels
        self.image_paths, self.labels = self._load_data()
        
        # Create train/test split
        self._make_train_test_split()
        
        # Remote sensing specific transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),

        ])

    def _verify_structure(self):
        """Verify the dataset folder structure."""
        if not os.path.exists(self.root):
            raise RuntimeError(f"CLRS dataset not found at: {self.root}")
            
        if len(self.classes) != 25:
            raise RuntimeError(f"Expected 25 classes, found {len(self.classes)}")

    def _load_data(self):
        """Load image paths and labels from class folders."""
        image_paths = []
        labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(self.class_to_idx[class_name])
        
        return image_paths, np.array(labels)

    def _make_train_test_split(self):
        """Create reproducible 80/20 train-test split."""
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(self.image_paths))
        split_idx = int(0.8 * len(indices))
        self.indices = indices[:split_idx] if self.train else indices[split_idx:]

    def get_data(self):
        """Return data in Continuum format: (images, labels, None)."""
        images = []
        labels = []
        
        for idx in self.indices:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            images.append(np.array(img))
            labels.append(self.labels[idx])
        
        return np.stack(images), np.array(labels), None

    def __getitem__(self, idx):
        """Get single item for DataLoader."""
        img_path = self.image_paths[self.indices[idx]]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[self.indices[idx]]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

    def __len__(self):
        return len(self.indices)