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
    def __init__(self, data_dir, transform, data_ids=None, train=True, mask_size=256):
        self.transform = transform
        self.train = train
        self.mask_size = mask_size
        self.images_paths = sorted(glob(os.path.join(data_dir, "*.png")))
        if data_ids is not None:
            self.images_paths = self.images_paths[data_ids[0]:data_ids[1]]
        
        # aux_images_paths = []
        # aux_masks_paths = []
        # for idx, mask_path in enumerate(self.masks_paths):
        #     if part_mapping[part_name] in np.load(mask_path):
        #         aux_images_paths.append(self.images_paths[idx])
        #         aux_masks_paths.append(mask_path)
        # self.images_paths = aux_images_paths
        # self.masks_paths = aux_masks_paths

    def __getitem__(self, idx):
        image = Image.open(self.images_paths[idx])
        mask = np.load(self.images_paths[idx].replace("png", "npy"))
        result = self.transform(image=np.array(image), mask=mask)
        image = result["image"]
        mask = result["mask"]
        if self.train:
            small_mask = \
                torch.nn.functional.interpolate(mask[None, None, ...].type(torch.float), self.mask_size,
                                                mode="nearest")[0, 0]
            return image / 255, mask, small_mask
        return image / 255, mask

    def __len__(self):
        return len(self.images_paths)


class CAT15DataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data_dir: str = "./data",
            test_data_dir: str = "./data",
            batch_size: int = 1,
            mask_size: int = 256,
            min_crop_ratio: float = 0.5,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.mask_size = mask_size

        self.train_transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(),
            # A.RandomScale((0.5, 2), always_apply=True),
            A.GaussianBlur(blur_limit=(1, 9)),
            A.RandomResizedCrop(256, 256, (min_crop_ratio, 1)),
            A.CLAHE(),
            A.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.1, hue=0.1),
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
                transform=self.train_transform,
                data_ids=[5, 30],
                train=True,
                mask_size=self.mask_size
            )
            self.val_dataset = CAT15Dataset(
                data_dir=self.train_data_dir,
                transform=self.test_transform,
                data_ids=[0, 5],
                train=False,
                mask_size=256
            )
        elif stage == "test":
            self.test_dataset = CAT15Dataset(
                data_dir=self.test_data_dir,
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
