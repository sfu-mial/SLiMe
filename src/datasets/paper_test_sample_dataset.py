import math
import os

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


car_part_color_mappings = [
    # [255, 52, 255],  # body
    # [255, 74, 70],  # light
    # [0, 137, 65],  # plate
    # [255, 255, 0],  # wheel
    [28, 230, 255]  # window
]

horse_part_color_mappings = [
    [255, 255, 0],  # head
    [0, 0, 255],  # leg
    [255, 0, 0],  # neck+torso
    [0, 255, 0],  # tail
]


class PaperTestSampleDataset(Dataset):
    def __init__(self, images_dir, masks_dir, train=True, mask_size=128, zero_pad_test_output=False):
        self.image_dirs = sorted(glob(os.path.join(images_dir, "*")))
        self.mask_dirs = sorted(glob(os.path.join(masks_dir, "*")))
        self.train = train
        self.mask_size = mask_size
        self.train_transform = A.Compose([
            A.LongestMaxSize(512),
            A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, value=0,
                          mask_value=0),
            A.HorizontalFlip(),
            A.RandomScale((0.5, 2), always_apply=True),
            A.RandomResizedCrop(512, 512, (0.5, 1)),
            A.Rotate((-10, 10), border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            ToTensorV2()
        ])
        if zero_pad_test_output:
            self.test_transform = A.Compose([
                A.LongestMaxSize(512),
                A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, value=0,
                              mask_value=0),
                ToTensorV2()
            ])
        else:
            self.test_transform = A.Compose([
                A.Resize(512, 512),
                # A.SmallestMaxSize(512),
                ToTensorV2()
            ])

    def __getitem__(self, idx):
        image = Image.open(self.image_dirs[idx])
        # image = transforms.functional.resize(image, size=512)
        # image = transforms.functional.to_tensor(image)


        mask = Image.open(self.mask_dirs[idx])
        # mask = transforms.functional.resize(mask, size=512,
        #                                           interpolation=transforms.InterpolationMode.NEAREST)
        mask = np.array(mask)

        # only for horse
        # mask = np.where(mask > np.array([200, 200, 200])[None, None, ...], 1, 0)

        # # Zero Padding
        # c, h, w = mask.shape
        # start = math.ceil((max(w, h) - min(w, h))/2)
        # aux_mask = torch.zeros(3, max(w, h), max(w, h))
        # aux_image = torch.zeros(3, max(w, h), max(w, h))
        # if w > h:
        #     aux_mask[:, start:start+min(w, h), 0:] = mask
        #     aux_image[:, start:start + min(w, h), 0:] = image
        # else:
        #     aux_mask[:, 0:, start:start + min(w, h)] = mask
        #     aux_image[:, 0:, start:start + min(w, h)] = image
        # mask = aux_mask
        # image = aux_image

        # image = transforms.functional.resize(image, size=512)
        # mask = transforms.functional.resize(mask, size=512,
        #                                     interpolation=transforms.InterpolationMode.NEAREST)
        final_mask = np.zeros(mask.shape[:2])
        for idx, part_color in enumerate(car_part_color_mappings):
            final_mask += np.where(np.all(mask == np.array(part_color)[None, None, ...], 2), idx+1, 0)
        # final_mask = transforms.functional.resize(final_mask, size=512, interpolation=transforms.InterpolationMode.NEAREST)

        if self.train:
            result = self.train_transform(image=np.array(image), mask=final_mask)
            image = result["image"]
            mask = torch.as_tensor(result["mask"])
            # crop_size = torch.randint(low=400, high=512, size=(1,)).item()
            # crop_params = transforms.RandomCrop.get_params(image, (crop_size, crop_size))
            # image = transforms.functional.crop(image, *crop_params)
            # final_mask = transforms.functional.crop(final_mask, *crop_params)
            #
            # degree = transforms.RandomRotation.get_params([0, 45])
            # image = transforms.functional.rotate(image, degree)
            # final_mask = transforms.functional.rotate(final_mask, degree)
            # image = transforms.functional.resize(image, size=512)
            # final_mask = transforms.functional.resize(final_mask, size=self.mask_size,
            #                                           interpolation=transforms.InterpolationMode.NEAREST)
            mask = \
                torch.nn.functional.interpolate(mask[None, None, ...].type(torch.float), self.mask_size,
                                                mode="nearest")[0, 0]
        else:
            result = self.test_transform(image=np.array(image), mask=final_mask)
            image = result["image"]
            mask = torch.as_tensor(result["mask"])

        return image / 255, mask

    def __len__(self):
        return len(self.image_dirs)


class PaperTestSampleDataModule(pl.LightningDataModule):
    def __init__(
            self,
            test_images_dir: str = "./data",
            test_masks_dir: str = "./data",
            mask_size: int = 128,
            zero_pad_test_output: bool = False,
    ):
        super().__init__()
        self.test_images_dir = test_images_dir
        self.test_masks_dir = test_masks_dir
        self.mask_size = mask_size
        self.zero_pad_test_output = zero_pad_test_output

    def setup(self, stage: str):
        self.train_dataset = PaperTestSampleDataset(
            images_dir=self.test_images_dir,
            masks_dir=self.test_masks_dir,
            train=True,
            mask_size=self.mask_size,
        )
        self.test_dataset = PaperTestSampleDataset(
            images_dir=self.test_images_dir,
            masks_dir=self.test_masks_dir,
            train=False,
            zero_pad_test_output=self.zero_pad_test_output,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers=8, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=8, shuffle=False)
