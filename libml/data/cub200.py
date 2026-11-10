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
class CUB200(_ContinuumDataset):
    #"""
        # self.url = 'https://data.deepai.org/CUB200(2011).zip'
        # self.filename = 'CUB200(2011).zip'

        # fpath = os.path.join(root, self.filename)
        # if not os.path.isfile(fpath):
        #     if not download:
        #        raise RuntimeError('Dataset not found. You can use download=True to download it')
        #     else:
        #         print('Downloading from '+self.url)
        #         download_url(self.url, root, filename=self.filename)

        # if not os.path.exists(os.path.join(root, 'CUB_200_2011')):
        #     import zipfile
        #     zip_ref = zipfile.ZipFile(fpath, 'r')
        #     zip_ref.extractall(root)
        #     zip_ref.close()

        #     import tarfile
        #     tar_ref = tarfile.open(os.path.join(root, 'CUB_200_2011.tgz'), 'r')
        #     tar_ref.extractall(root)
        #     tar_ref.close()

        #     self.split()
        
        # if self.train:
        #     fpath = os.path.join(root, 'CUB_200_2011', 'train')

        # else:
        #     fpath = os.path.join(root, 'CUB_200_2011', 'test')
    # url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    # train_subset_url = "https://gist.githubusercontent.com/gqk/e127fe18bf179bdcbdf5e29a8c1ae523/raw/train_list.txt"
    # test_subset_url = "https://gist.githubusercontent.com/gqk/e127fe18bf179bdcbdf5e29a8c1ae523/raw/val_list.txt"

    url = 'https://data.deepai.org/CUB200(2011).zip'
    num_classes = 200
    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    # def _download(self):
    #     """Download and extract the CUB-200 dataset."""
    #     tgz_file = os.path.join(self.data_path, "images.tgz")  # Correct file name
    #     extract_path = os.path.join(self.data_path, "CUB_200_2011")

    #     # Download the dataset if it doesn't exist
    #     if not os.path.exists(tgz_file):
    #         print("Downloading CUB-200 dataset...")
    #         download(self.url, self.data_path)

    #     # Extract the dataset if it hasn't been extracted
    #     if not os.path.exists(extract_path):
    #         print("Extracting CUB-200 dataset...")
    #         with tarfile.open(tgz_file, 'r:gz') as tar:
    #             tar.extractall(path=self.data_path)

    #     # Verify that the images folder exists and contains files
    #     images_folder = os.path.join(extract_path, "images")
    #     if not os.path.exists(images_folder) or not os.listdir(images_folder):
    #         raise FileNotFoundError(
    #             f"The images folder is empty or missing. "
    #             "Ensure the dataset is downloaded and extracted correctly."
    #         )
    # def _download(self):
    #     """Download and extract the CUB-200 dataset."""

    #     self.url = 'https://data.deepai.org/CUB200(2011).zip'
    #     path = os.path.join(self.data_path, "CUB_200_2011")
    #     if not os.path.exists(path):
    #         if not os.path.exists(f"{path}.tgz"):
    #             print("Downloading CUB-200 dataset...")
    #             download(self.url, self.data_path)
    #         print("Extracting CUB-200 dataset...")
    #         untar(f"{path}.tgz")
        #self.split()

    def _download(self):

        self.url = 'https://data.deepai.org/CUB200(2011).zip'
        filename = 'CUB200(2011).zip'
        fpath = os.path.join(self.data_path, filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                #download_url(self.url, self.data_path, filename=self.filename)
                download(self.url, self.data_path)

        if not os.path.exists(os.path.join(self.data_path, 'CUB_200_2011')):
            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(self.data_path)
            zip_ref.close()

            import tarfile
            tar_ref = tarfile.open(os.path.join(self.data_path, 'CUB_200_2011.tgz'), 'r')
            tar_ref.extractall(self.data_path)
            tar_ref.close()
        train_folder = os.path.join(self.data_path, "CUB_200_2011", "train")

        if not os.path.exists(train_folder):
            self.split()
        else:
            print("train data already exist")

        # path = os.path.join(self.data_path, "imagenet-r")
        # if not os.path.exists(path):
        #     if not os.path.exists(f"{path}.tar"):
        #         download(self.url, self.data_path)
        #     untar(f"{path}.tar")

        # filename = "val_list.txt"
        # self.subset_url = self.test_subset_url
        # if self.train:
        #     filename = "train_list.txt"
        #     self.subset_url = self.train_subset_url
        # self.data_subset = os.path.join(self.data_path, filename)
        # if not os.path.exists(self.data_subset):
        #     print("Downloading subset indexes...", end=" ")
        #     download(self.subset_url, self.data_path)
        #     print("Done!")
 

    def split(self):
        """Split the dataset into train and test folders using the provided split files."""
        train_folder = os.path.join(self.data_path, "CUB_200_2011", "train")
        test_folder = os.path.join(self.data_path, "CUB_200_2011", "test")

        
        # if os.path.exists(train_folder):
        #     return
        #     rmtree(train_folder)
        
        # if os.path.exists(test_folder):
        #     return
        #     rmtree(test_folder)
        os.makedirs(train_folder)
        os.makedirs(test_folder)

        # Paths to the split files
        images_file = os.path.join(self.data_path, "CUB_200_2011/CUB_200_2011", "images.txt")
        split_file = os.path.join(self.data_path, "CUB_200_2011/CUB_200_2011", "train_test_split.txt")

        # Read the split files and organize images into train/test folders
        with open(images_file, 'r') as images, open(split_file, 'r') as splits:
            for image_line, split_line in zip(images, splits):
                image_path = image_line.strip().split(' ')[-1]
                class_name = image_path.split('/')[0]
                src = os.path.join(self.data_path, "CUB_200_2011/CUB_200_2011", "images", image_path)

                if split_line.strip().split(' ')[-1] == '1':  # Train split
                    dst_folder = os.path.join(train_folder, class_name)
                else:  # Test split
                    dst_folder = os.path.join(test_folder, class_name)

                os.makedirs(dst_folder, exist_ok=True)
                dst = os.path.join(dst_folder, os.path.basename(image_path))
                move(src, dst)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        """Load the dataset."""
        if self.train:
            data_folder = os.path.join(self.data_path, "CUB_200_2011", "train")
        else:
            data_folder = os.path.join(self.data_path, "CUB_200_2011", "test")

        x, y = [], []
        for class_name in os.listdir(data_folder):
            class_folder = os.path.join(data_folder, class_name)
            if not os.path.isdir(class_folder):
                continue
            class_id = int(class_name.split('.')[0]) - 1  # Convert to 0-based index
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                x.append(image_path)
                y.append(class_id)

        x, y = np.array(x), np.array(y)
        return x, y, None