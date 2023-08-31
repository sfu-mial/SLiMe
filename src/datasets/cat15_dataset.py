import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from glob import glob
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


part_mapping = {
    "background":0,
    "back":1,
    "belly":2,
    "chest":3,
    "leg":4,
    "paw":5,
    "head":6,
    "ear":7,
    "eye":8,
    "mouth":9,
    "tongue":10,
    "tail":11,
    "nose":12,
    "whiskers":13,
    "neck":14,
}



class CAT15Dataset(Dataset):
    def __init__(self, data_dir, part_name, transform, data_ids=None, train=True, mask_size=256):
        self.part_name = part_name
        self.transform = transform
        self.train = train
        self.mask_size = mask_size
        self.images_paths = sorted(glob(os.path.join(data_dir, "*.png")))
        self.masks_paths = sorted(glob(os.path.join(data_dir, "*.npy")))
        if data_ids is not None:
            self.images_paths = self.images_paths[data_ids[0]:data_ids[1]]
            self.masks_paths = self.masks_paths[data_ids[0]:data_ids[1]]
        
        aux_images_paths = []
        aux_masks_paths = []
        for idx, mask_path in enumerate(self.masks_paths):
            if part_mapping[part_name] in np.load(mask_path):
                aux_images_paths.append(self.images_paths[idx])
                aux_masks_paths.append(mask_path)
        self.images_paths = aux_images_paths
        self.masks_paths = aux_masks_paths

    def __getitem__(self, idx):
        image = Image.open(self.images_paths[idx])
        # image = transforms.functional.resize(image, size=(512, 512))
        # image = transforms.functional.to_tensor(image)
        mask = np.where(np.load(self.masks_paths[idx])==part_mapping[self.part_name], 1, 0)
        if self.train:
            # image = transforms.functional.resize(image, 256)
            original_mask_size = np.where(mask > 0, 1, 0).sum()
            mask_is_included = False
            while not mask_is_included:
                result = self.transform(image=np.array(image), mask=mask)
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
        result = self.transform(image=np.array(image), mask=mask)
        image = result["image"]
        mask = result["mask"]
        # mask = torch.nn.functional.interpolate(torch.as_tensor(result["mask"])[None, None, ...], 256)[0, 0]
        return image / 255, mask

    def __len__(self):
        return len(self.images_paths)


class CAT15DataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data_dir: str = "./data",
            test_data_dir: str = "./data",
            part_name: str="background",
            batch_size: int = 1,
            mask_size: int = 256,
            min_crop_ratio: float = 0.5,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.part_name = part_name
        self.batch_size = batch_size
        self.mask_size = mask_size

        self.train_transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(),
            # A.RandomScale((0.5, 2), always_apply=True),
            A.GaussianBlur(blur_limit=(1, 11)),
            A.RandomResizedCrop(256, 256, (min_crop_ratio, 1)),
            A.Rotate((-30, 30), border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            ToTensorV2()
        ])
        self.test_transform = A.Compose([
            A.Resize(256, 256),
            ToTensorV2()
        ])

    def setup(self, stage: str):
        if stage == "fit":

            self.train_dataset = CAT15Dataset(
                data_dir=self.train_data_dir,
                part_name=self.part_name,
                transform=self.train_transform,
                data_ids=[0, 27],
                train=True,
                mask_size=self.mask_size
            )
            self.val_dataset = CAT15Dataset(
                data_dir=self.train_data_dir,
                part_name=self.part_name,
                transform=self.test_transform,
                data_ids=[27, 30],
                train=False,
                mask_size=256
            )
        elif stage == "test":
            self.test_dataset = CAT15Dataset(
                data_dir=self.test_data_dir,
                part_name=self.part_name,
                transform=self.test_transform,
                train=False,
                mask_size=256
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=3, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=3, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=3, shuffle=False)
