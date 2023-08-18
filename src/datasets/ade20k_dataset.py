import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import os
from glob import glob
from typing import List
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


object_mapping = {
    "wall":0,
    "bed":1,
    "floor":2,
    "table":3,
    "lamp":4,
    "ceiling":5,
    "painting":6,
    "windowpane":7,
    "pillow":8,
    "curtain":9,
    "cushion":10,
    "door":11,
    "chair":12,
    "cabinet":13,
    "chest":14,
    "mirror":15,
    "rug":16,
    "armchair":17,
    "book":18,
    "sconce":19,
    "plant":20,
    "wardrobe":21,
    "clock":22,
    "light":23,
    "flower":24,
    "vase":25,
    "fan":26,
    "box":27,
    "shelf":28,
    "television":29,
}



class ADE20KDataset(Dataset):
    def __init__(self, data_dir, object_name, data_ids=None, train=True, mask_size=256):
        self.object_name = object_name
        self.train = train
        self.mask_size = mask_size
        self.images_paths = sorted(glob(os.path.join(data_dir, "*.jpg")))
        self.masks_paths = sorted(glob(os.path.join(data_dir, "*.npy")))
        if data_ids is not None:
            self.images_paths = self.images_paths[data_ids[0]:data_ids[1]]
            self.masks_paths = self.masks_paths[data_ids[0]:data_ids[1]]
        
        self.train_transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(),
            # A.RandomScale((0.5, 2), always_apply=True),
            A.GaussianBlur(blur_limit=(1, 31)),
            A.RandomResizedCrop(256, 256, (0.3, 1)),
            A.Rotate((-10, 10), border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            ToTensorV2()
        ])
        self.test_transform = A.Compose([
            A.Resize(256, 256),
            ToTensorV2()
        ])
        aux_images_paths = []
        aux_masks_paths = []
        for idx, mask_path in enumerate(self.masks_paths):
            if object_mapping[object_name] in np.load(mask_path):
                aux_images_paths.append(self.images_paths[idx])
                aux_masks_paths.append(mask_path)
        self.images_paths = aux_images_paths
        self.masks_paths = aux_masks_paths

    def __getitem__(self, idx):
        image = Image.open(self.images_paths[idx])
        # image = transforms.functional.resize(image, size=(512, 512))
        # image = transforms.functional.to_tensor(image)
        mask = np.where(np.load(self.masks_paths[idx])==object_mapping[self.object_name], 1, 0)
        if self.train:
            # image = transforms.functional.resize(image, 256)
            original_mask_size = np.where(mask > 0, 1, 0).sum()
            mask_is_included = False
            while not mask_is_included:
                result = self.train_transform(image=np.array(image), mask=mask)
                # mask = torch.as_tensor(result["mask"])
                if np.where(result["mask"] > 0, 1, 0).sum() / original_mask_size > 0.3:
                    mask_is_included = True
                    
            image = result["image"]
            mask = torch.as_tensor(result["mask"])
            # result = self.train_transform(image=np.array(image), mask=final_mask)
            # image = result["image"]
            # mask = torch.as_tensor(result["mask"])
            mask = \
                torch.nn.functional.interpolate(mask[None, None, ...].type(torch.float), self.mask_size,
                                                mode="nearest")[0, 0]
            return image / 255, mask
        # image = transforms.functional.resize(image, 256)  # because the original image size is 1024 but the mask is 512
        result = self.test_transform(image=np.array(image), mask=mask)
        image = result["image"]
        mask = result["mask"]
        # mask = torch.nn.functional.interpolate(torch.as_tensor(result["mask"])[None, None, ...], 256)[0, 0]
        return image / 255, mask

    def __len__(self):
        return len(self.images_paths)


class ADE20KDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data_dir: str = "./data",
            test_data_dir: str = "./data",
            object_name: str="wall",
            batch_size: int = 1,
            mask_size: int = 256,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.object_name = object_name
        self.batch_size = batch_size
        self.mask_size = mask_size

    def setup(self, stage: str):
        if stage == "fit":

            self.train_dataset = ADE20KDataset(
                data_dir=self.train_data_dir,
                object_name=self.object_name,
                data_ids=[0, 50],
                train=True,
                mask_size=self.mask_size
            )
            self.val_dataset = ADE20KDataset(
                data_dir=self.train_data_dir,
                object_name=self.object_name,
                data_ids=[50, 86],
                train=False,
                mask_size=256
            )
        elif stage == "test":
            self.test_dataset = ADE20KDataset(
                data_dir=self.test_data_dir,
                object_name=self.object_name,
                train=False,
                mask_size=256
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=3, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=3, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=3, shuffle=False)
