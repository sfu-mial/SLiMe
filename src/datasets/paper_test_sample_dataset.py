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
    'body': [255, 52, 255],
    'light': [255, 74, 70],
    'plate': [0, 137, 65],
    'wheel': [255, 255, 0],
    'window': [28, 230, 255]
}

horse_part_color_mappings = {
    'head': [255, 255, 0], # head
    'leg': [0, 0, 255],  # leg
    'neck+torso': [255, 0, 0],  # neck+torso
    'tail': [0, 255, 0],  # tail
}


class PaperTestSampleDataset(Dataset):
    def __init__(self, images_dir, masks_dir, train=True, mask_size=128, zero_pad_test_output=False, object_name='car', part_names=('background', 'body', )):
        self.image_dirs = sorted(glob(os.path.join(images_dir, "*")))
        self.mask_dirs = sorted(glob(os.path.join(masks_dir, "*")))
        self.train = train
        self.mask_size = mask_size
        self.object_name = object_name
        self.part_names = part_names
        self.train_transform = A.Compose([
            A.LongestMaxSize(512),
            A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, value=0,
                          mask_value=0),
            A.HorizontalFlip(),
            # A.RandomScale((0.5, 2), always_apply=True),
            A.RandomResizedCrop(512, 512, (0.2, 1)),
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
        if self.object_name == "horse":
            mask = np.where(mask > np.array([200, 200, 200])[None, None, ...], 255, 0)

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
        # final_mask = np.zeros(mask.shape[:2])
        # for idx, part_color in enumerate(car_part_color_mappings):
        if self.object_name == 'car':
            part_mapping = car_part_color_mappings
        else:
            part_mapping = horse_part_color_mappings
        if self.part_names == ['whole']:
            final_mask = np.where(np.sum(mask, axis=2) > 0, 1, 0)
        else:
            final_mask = np.zeros_like(mask[:, :, 0])
            for idx, part_name in enumerate(self.part_names):
                final_mask = np.where(np.all(mask == np.array(part_mapping[part_name])[None, None, ...], 2), idx+1, final_mask)
        # final_mask = transforms.functional.resize(final_mask, size=512, interpolation=transforms.InterpolationMode.NEAREST)

        if self.train:
            mask_is_included = False
            while not mask_is_included:
                result = self.train_transform(image=np.array(image), mask=final_mask)
                # mask = torch.as_tensor(result["mask"])
                if np.where(result["mask"] > 0, 1, 0).sum() > 2000:
                    mask_is_included = True
            image = result["image"]
            mask = torch.as_tensor(result["mask"])
            mask = \
                torch.nn.functional.interpolate(mask[None, None, ...].type(torch.float), self.mask_size, mode="nearest")[0, 0]
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
            object_name: str = 'car',
            part_names: Tuple[str] = ("body", ),
            images_dir: str = "./data",
            masks_dir: str = "./data",
            mask_size: int = 128,
            zero_pad_test_output: bool = False,
    ):
        super().__init__()
        self.object_name = object_name
        self.part_names = part_names
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.mask_size = mask_size
        self.zero_pad_test_output = zero_pad_test_output

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = PaperTestSampleDataset(
                images_dir=self.images_dir,
                masks_dir=self.masks_dir,
                train=True,
                mask_size=self.mask_size,
                part_names=self.part_names,
                object_name=self.object_name,
            )
        if stage == 'test':
            self.test_dataset = PaperTestSampleDataset(
                images_dir=self.images_dir,
                masks_dir=self.masks_dir,
                train=False,
                zero_pad_test_output=self.zero_pad_test_output,
                part_names=self.part_names,
                object_name=self.object_name,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers=3, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=3, shuffle=False)
