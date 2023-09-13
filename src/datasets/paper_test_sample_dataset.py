import os
from typing import Tuple
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

car_part_color_mappings = {
    "body": [255, 52, 255],
    "light": [255, 74, 70],
    "plate": [0, 137, 65],
    "wheel": [255, 255, 0],
    "window": [28, 230, 255],
}

horse_part_color_mappings = {
    "head": [255, 255, 0],  # head
    "leg": [0, 0, 255],  # leg
    "neck+torso": [255, 0, 0],  # neck+torso
    "tail": [0, 255, 0],  # tail
}


class PaperTestSampleDataset(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        transform,
        train=True,
        train_mask_size=128,
        test_mask_size=128,
        object_name="car",
        parts_to_return=(
            "background",
            "body",
        ),
    ):
        self.image_dirs = sorted(glob(os.path.join(images_dir, "*")))
        self.mask_dirs = sorted(glob(os.path.join(masks_dir, "*")))
        self.transform = transform
        self.train = train
        self.train_mask_size = train_mask_size
        self.test_mask_size = test_mask_size
        self.object_name = object_name
        self.parts_to_return = parts_to_return

    def __getitem__(self, idx):
        image = Image.open(self.image_dirs[idx])
        mask = Image.open(self.mask_dirs[idx])
        mask = np.array(mask)

        # only for horse
        if self.object_name == "horse":
            mask = np.where(mask > np.array([200, 200, 200])[None, None, ...], 255, 0)

        if self.object_name == "car":
            part_mapping = car_part_color_mappings
        else:
            part_mapping = horse_part_color_mappings
        if self.parts_to_return == ["whole"]:
            final_mask = np.where(np.sum(mask, axis=2) > 0, 1, 0)
        else:
            final_mask = np.zeros_like(mask[:, :, 0])
            for idx, part_name in enumerate(self.parts_to_return[1:]):
                final_mask = np.where(
                    np.all(
                        mask == np.array(part_mapping[part_name])[None, None, ...], 2
                    ),
                    idx + 1,
                    final_mask,
                )

        result = self.transform(image=np.array(image), mask=final_mask)
        image = result["image"]
        mask = result["mask"]
        if self.train:
            test_mask = torch.nn.functional.interpolate(
                mask[None, None, ...].type(torch.float),
                self.test_mask_size,
                mode="nearest",
            )[0, 0]
            train_mask = torch.nn.functional.interpolate(
                mask[None, None, ...].type(torch.float),
                self.train_mask_size,
                mode="nearest",
            )[0, 0]
            return image / 255, test_mask, train_mask
        else:
            test_mask = torch.nn.functional.interpolate(
                mask[None, None, ...].type(torch.float),
                self.test_mask_size,
                mode="nearest",
            )[0, 0]
            return image / 255, test_mask

    def __len__(self):
        return len(self.image_dirs)


class PaperTestSampleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        object_name: str = "car",
        parts_to_return: Tuple[str] = ("body",),
        images_dir: str = "./data",
        masks_dir: str = "./data",
        train_mask_size: int = 128,
        test_mask_size: int = 512,
    ):
        super().__init__()
        self.object_name = object_name
        self.parts_to_return = parts_to_return
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.train_mask_size = train_mask_size
        self.test_mask_size = test_mask_size

    def setup(self, stage: str):
        if stage == "fit":
            train_transform = A.Compose(
                [
                    A.LongestMaxSize(512),
                    A.PadIfNeeded(
                        512, 512, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
                    ),
                    A.HorizontalFlip(),
                    # A.RandomScale((0.5, 2), always_apply=True),
                    A.RandomResizedCrop(512, 512, (0.2, 1)),
                    A.Rotate(
                        (-10, 10),
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=0,
                    ),
                    ToTensorV2(),
                ]
            )

            self.train_dataset = PaperTestSampleDataset(
                images_dir=self.images_dir,
                masks_dir=self.masks_dir,
                transform=train_transform,
                train=True,
                train_mask_size=self.train_mask_size,
                test_mask_size=self.test_mask_size,
                parts_to_return=self.parts_to_return,
                object_name=self.object_name,
            )
        if stage == "test":
            test_transform = A.Compose(
                [
                    A.Resize(512, 512),
                    # A.SmallestMaxSize(512),
                    ToTensorV2(),
                ]
            )
            self.test_dataset = PaperTestSampleDataset(
                images_dir=self.images_dir,
                masks_dir=self.masks_dir,
                transform=test_transform,
                train=False,
                train_mask_size=self.train_mask_size,
                test_mask_size=self.test_mask_size,
                parts_to_return=self.parts_to_return,
                object_name=self.object_name,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=1, num_workers=3, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=3, shuffle=False)
