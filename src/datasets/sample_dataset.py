import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from src.utils import get_random_crop_coordinates


class SampleDataset(Dataset):
    def __init__(
        self,
        image_dirs,
        mask_dirs,
        train=True,
        mask_size=512,
        num_parts=1,
        min_crop_ratio=0.5,
    ):
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs
        self.train = train
        self.mask_size = mask_size
        self.num_parts = num_parts
        self.min_crop_ratio = min_crop_ratio
        self.train_transform_1 = A.Compose(
            [
                A.Resize(512, 512),
                A.HorizontalFlip(),
                A.GaussianBlur(blur_limit=(1, 5)),
            ]
        )

        self.train_transform_2 = A.Compose(
            [
                A.Resize(512, 512),
                # A.CLAHE(),
                A.ToGray(),
                A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.9, hue=0.9),
                A.Rotate(
                    (-30, 30), border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
                ),
                ToTensorV2(),
            ]
        )
        self.current_part_idx = 0
        self.test_transform = A.Compose([A.Resize(512, 512), ToTensorV2()])

    def __getitem__(self, idx):
        image = Image.open(self.image_dirs[idx])
        if self.train:
            mask = np.array(Image.open(self.mask_dirs[idx]))
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            result = self.train_transform_1(image=np.array(image), mask=mask)
            image = result["image"]
            mask = result["mask"]
            original_mask_size = np.where(mask == self.current_part_idx + 1, 1, 0).sum()
            mask_is_included = False
            while not mask_is_included:
                x_start, x_end, y_start, y_end = get_random_crop_coordinates(
                    (self.min_crop_ratio, 1), 512, 512
                )
                aux_mask = mask[y_start:y_end, x_start:x_end]
                if (
                    original_mask_size == 0
                    or np.where(aux_mask == self.current_part_idx + 1, 1, 0).sum()
                    / original_mask_size
                    > 0.3
                ):
                    mask_is_included = True
            image = image[y_start:y_end, x_start:x_end]
            result = self.train_transform_2(image=image, mask=aux_mask)
            mask, image = result["mask"], result["image"]
            mask = torch.nn.functional.interpolate(
                mask[None, None, ...].type(torch.float),
                self.mask_size,
                mode="nearest",
            )[0, 0]
            self.current_part_idx += 1
            self.current_part_idx = self.current_part_idx % self.num_parts
            return image / 255, mask
        else:
            if self.mask_dirs is not None:
                mask = np.array(Image.open(self.mask_dirs[idx]))
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                result = self.test_transform(image=np.array(image), mask=mask)
                mask = result["mask"]
                mask = torch.nn.functional.interpolate(
                    mask[None, None, ...].type(torch.float),
                    self.mask_size,
                    mode="nearest",
                )[0, 0]
            else:
                result = self.test_transform(image=np.array(image))
                mask = 0
            image = result["image"]
            return image / 255, mask

    def __len__(self):
        return len(self.image_dirs)


class SampleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        src_image_dirs: str = "./data",
        target_image_dir: str = "./data",
        src_mask_dirs: str = "./data",
        batch_size: int = 1,
        train_mask_size: int = 256,
        test_mask_size: int = 256,
        num_parts: int = 1,
        min_crop_ratio: float = 0.5,
    ):
        super().__init__()
        self.src_image_dirs = src_image_dirs
        self.target_image_dir = target_image_dir
        self.src_mask_dirs = src_mask_dirs
        self.batch_size = batch_size
        self.train_mask_size = train_mask_size
        self.test_mask_size = test_mask_size
        self.num_parts = num_parts
        self.min_crop_ratio = min_crop_ratio

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = SampleDataset(
                image_dirs=self.src_image_dirs[0 : len(self.src_image_dirs) // 2],
                mask_dirs=self.src_mask_dirs[0 : len(self.src_mask_dirs) // 2],
                train=True,
                mask_size=self.train_mask_size,
                num_parts=self.num_parts,
                min_crop_ratio=self.min_crop_ratio,
            )
            self.val_dataset = SampleDataset(
                image_dirs=self.src_image_dirs[len(self.src_image_dirs) // 2 :],
                mask_dirs=self.src_mask_dirs[len(self.src_mask_dirs) // 2 :],
                train=False,
                mask_size=self.test_mask_size,
            )
        # elif stage == 'test':
        #     self.test_dataset = SampleDataset(
        #         image_dirs=self.target_image_dir,
        #         mask_dirs=None,
        #         train=False,
        #     )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False
        )

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)
