import random
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import os
from glob import glob
import re
from tqdm import tqdm
from typing import List, Tuple
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


part_names_mapping = {
    "eye": "eye",
    "mouth": "mouth",
    "nose": "nose",
    "brow": "brow",
    "ear": "ear",
    "lip": "mouth",
    "skin": "skin",
    "neck": "neck",
    "cloth": "cloth",
    "hair": "hair",

}

non_skin_part_names = ["eye", "mouth", "nose", "brow", "ear"]


class CelebaHQDataset(Dataset):
    def __init__(self, images_dir, masks_dir, idx_mapping_file, parts_to_return, data_ids=None, file_names_file_path=None, train=True, mask_size=256, zero_pad_test_output=False):
        self.images_paths = []
        self.masks_paths = []
        self.parts_to_return = parts_to_return
        self.train = train
        self.mask_size = mask_size
        self.data_ids = data_ids
        self.return_whole = False
        if self.parts_to_return[1] == 'whole':
            self.parts_to_return = ['background','brow','cloth','ear','eye','hair','mouth','neck','nose','skin']
            self.return_whole = True
        mapping_dict = {}
        with open(idx_mapping_file) as file:
            mappings = file.readlines()[1:]
        for mapping in mappings:
            idx, _, file_name = mapping.strip().split()
            mapping_dict[file_name] = idx
        with open(file_names_file_path) as file:
            files_names = file.readlines()
        if data_ids is not None:
            files_names = files_names[min(self.data_ids):max(self.data_ids) + 1]
        for file_name in tqdm(files_names):
            file_name = file_name.strip()
            file_name = mapping_dict[file_name]
            self.images_paths.append(os.path.join(images_dir, f"{file_name}.jpg"))
            file_index = int(file_name.split(".")[0])
            mask_folder_idx = file_index // 2000
            masks_paths = glob(os.path.join(masks_dir, str(mask_folder_idx), f"{file_name.zfill(5)}_*.png"))
            part_data_paths = {}
            for path in masks_paths:
                part_name = re.findall("\.*_([a-z|A-Z]+).png", path)[0]
                if part_names_mapping.get(part_name, False):
                    part_name = part_names_mapping[part_name]
                    part_paths = part_data_paths.get(part_name, [])
                    part_paths.append(path)
                    part_data_paths[part_name] = part_paths
                    if part_name in non_skin_part_names:
                        part_paths = part_data_paths.get("non_skin", [])
                        part_paths.append(path)
                        part_data_paths["non_skin"] = part_paths

            self.masks_paths.append(part_data_paths)
        if data_ids is not None:
            aux_images_paths = []
            aux_masks_paths = []
            for id in data_ids:
                aux_images_paths.append(self.images_paths[id-min(self.data_ids)])
                aux_masks_paths.append(self.masks_paths[id-min(self.data_ids)])
            self.images_paths = aux_images_paths
            self.masks_paths = aux_masks_paths

        if zero_pad_test_output:
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
            self.test_transform = A.Compose([
                A.LongestMaxSize(512),
                A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, value=0,
                              mask_value=0),
                ToTensorV2()
            ])
        else:
            self.train_transform = A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(),
                # A.RandomScale((0.5, 2), always_apply=True),
                A.GaussianBlur(blur_limit=(11, 31)),
                A.RandomResizedCrop(512, 512, (0.3, 1)),
                A.Rotate((-10, 10), border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                ToTensorV2()
            ])
            self.test_transform = A.Compose([
                A.Resize(512, 512),
                ToTensorV2()
            ])

    def __getitem__(self, idx):
        image = Image.open(self.images_paths[idx])
        # image = transforms.functional.resize(image, size=(512, 512))
        # image = transforms.functional.to_tensor(image)
        masks_paths = self.masks_paths[idx]
        final_mask = np.zeros((512, 512), dtype=np.float64)
        for idx, part_name in enumerate(self.parts_to_return):
            if part_name == "skin":
                non_skin_mask = 0
                for path in masks_paths["non_skin"]:
                    mask = np.array(Image.open(path))[:, :, 0] / 255
                    non_skin_mask += mask
                non_skin_mask = np.where(non_skin_mask > 0, 1, 0)
                skin_mask = np.array(Image.open(masks_paths[part_name][0]))[:, :, 0] / 255
                aux_mask = np.where(non_skin_mask > 0, 0, skin_mask)
            else:
                aux_mask = 0
                if part_name in masks_paths:
                    for path in masks_paths[part_name]:
                        mask = np.array(Image.open(path))[:, :, 0] / 255
                        aux_mask += mask
                    aux_mask = np.where(aux_mask > 0, 1, 0)
            if not isinstance(aux_mask, int):
                final_mask = np.where(aux_mask > 0, idx, final_mask)
        if self.return_whole:
            final_mask = np.where(final_mask>0, 1, 0)
        if self.train:
            image = transforms.functional.resize(image, 512)  # because the original image size is 1024 but the mask is 512
            original_mask_size = np.where(final_mask > 0, 1, 0).sum()
            mask_is_included = False
            while not mask_is_included:
                result = self.train_transform(image=np.array(image), mask=final_mask)
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
        image = transforms.functional.resize(image, 512)  # because the original image size is 1024 but the mask is 512
        result = self.test_transform(image=np.array(image), mask=final_mask)
        image = result["image"]
        mask = torch.as_tensor(result["mask"])
        return image / 255, mask

    def __len__(self):
        return len(self.images_paths)


class CelebaHQDataModule(pl.LightningDataModule):
    def __init__(
            self,
            images_dir: str = "./data",
            masks_dir: str = "./data",
            idx_mapping_file: str = "./data",
            test_file_names_file_path: str = "./data",
            train_file_names_file_path: str = "./data",
            val_file_names_file_path: str = "./data",
            part_names: List[str]=("background", "eye", "mouth", "nose", "brow", "ear", "skin", "neck",
                                              "cloth", "hair"),
            batch_size: int = 1,
            mask_size: int = 256,
            train_data_ids: List[int] = [i for i in range(10)],
            val_data_ids: List[int] = [i for i in range(10)],
            zero_pad_test_output: bool = False,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.idx_mapping_file = idx_mapping_file
        self.test_file_names_file_path = test_file_names_file_path
        self.train_file_names_file_path = train_file_names_file_path
        self.val_file_names_file_path = val_file_names_file_path
        self.part_names = part_names
        self.batch_size = batch_size
        self.mask_size = mask_size
        self.train_data_ids = train_data_ids
        self.val_data_ids = val_data_ids
        self.zero_pad_test_output = zero_pad_test_output

    def setup(self, stage: str):
        if stage == "fit":

            self.train_dataset = CelebaHQDataset(
                images_dir=self.images_dir,
                masks_dir=self.masks_dir,
                idx_mapping_file=self.idx_mapping_file,
                parts_to_return=self.part_names,
                data_ids=self.train_data_ids,
                file_names_file_path=self.train_file_names_file_path,
                train=True,
                mask_size=self.mask_size,
            )
            self.val_dataset = CelebaHQDataset(
                images_dir=self.images_dir,
                masks_dir=self.masks_dir,
                idx_mapping_file=self.idx_mapping_file,
                parts_to_return=self.part_names,
                data_ids=self.val_data_ids,
                file_names_file_path=self.val_file_names_file_path,
                train=False,
            )
        elif stage == "test":
            self.test_dataset = CelebaHQDataset(
                images_dir=self.images_dir,
                masks_dir=self.masks_dir,
                idx_mapping_file=self.idx_mapping_file,
                parts_to_return=self.part_names,
                file_names_file_path=self.test_file_names_file_path,
                train=False,
                zero_pad_test_output=self.zero_pad_test_output
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=3, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=3, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=3, shuffle=False)
