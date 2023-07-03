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
from src.utils import get_square_cropping_coords
from typing import List

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
    def __init__(self, images_dir, masks_dir, idx_mapping_file, parts_to_return, train_data_ids=[i for i in range(10)], file_names_file_path=None, num_crops=1, train=True, mask_size=256, final_min_crop_size=256):
        self.images_paths = []
        self.masks_paths = []
        self.parts_to_return = parts_to_return
        self.num_crops = num_crops
        self.train = train
        self.mask_size = mask_size
        self.final_min_crop_size = final_min_crop_size
        self.train_data_ids = train_data_ids
        mapping_dict = {}
        with open(idx_mapping_file) as file:
            mappings = file.readlines()[1:]
        for mapping in mappings:
            idx, _, file_name = mapping.strip().split()
            mapping_dict[file_name] = idx
        with open(file_names_file_path) as file:
            files_names = file.readlines()
        if train:
            files_names = files_names[:max(self.train_data_ids) + 1]
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
        if train:
            aux_images_paths = []
            aux_masks_paths = []
            for id in train_data_ids:
                aux_images_paths.append(self.images_paths[id])
                aux_masks_paths.append(self.masks_paths[id])
            self.images_paths = aux_images_paths
            self.masks_paths = aux_masks_paths

    # def set_parts_to_return(self, parts_to_return):

        # if self.train:
        #     assert len(parts_to_return) == 2 and "background" in parts_to_return, "pass correct `parts_to_return`"

    def __getitem__(self, idx):
        idx = idx // self.num_crops
        image = Image.open(self.images_paths[idx])
        image = transforms.functional.resize(image, size=(512, 512))
        image = transforms.functional.to_tensor(image)
        masks_paths = self.masks_paths[idx]
        final_mask = torch.zeros(512, 512, dtype=torch.float)
        for idx, part_name in enumerate(self.parts_to_return):
            if part_name == "skin":
                non_skin_mask = 0
                for path in masks_paths["non_skin"]:
                    mask = transforms.functional.to_tensor(Image.open(path))[0]
                    non_skin_mask += mask
                non_skin_mask = torch.where(non_skin_mask > 0, 1, 0)
                skin_mask = transforms.functional.to_tensor(Image.open(masks_paths[part_name][0]))[0]
                aux_mask = torch.where(non_skin_mask > 0, 0, skin_mask)
            else:
                aux_mask = 0
                if part_name in masks_paths:
                    for path in masks_paths[part_name]:
                        mask = transforms.functional.to_tensor(Image.open(path))[0]
                        aux_mask += mask
                    aux_mask = torch.where(aux_mask > 0, 1, 0)
            if not isinstance(aux_mask, int):
                final_mask = torch.where(aux_mask > 0, idx, final_mask)
        final_mask = torch.nn.functional.interpolate(final_mask[None, None, ...].type(torch.float), 512, mode="nearest")[0, 0]
        if self.train:
            coords = []
            if self.num_crops > 1:
                x_start, x_end, y_start, y_end, crop_size = get_square_cropping_coords(torch.where(final_mask > 0, 1, 0))
                for square_size in torch.linspace(crop_size, 512, self.num_crops + 1)[1:-1]:
                    x_start, x_end, y_start, y_end, crop_size = get_square_cropping_coords(torch.where(final_mask>0, 1, 0),
                                                                                           square_size=int(square_size))
                    coords.append([y_start, y_end, x_start, x_end])
            coords.append([0, 512, 0, 512])
            random_idx = random.randint(0, self.num_crops - 1)
            y_start, y_end, x_start, x_end = coords[random_idx]
            cropped_mask = final_mask[y_start:y_end, x_start:x_end]
            cropped_mask = torch.nn.functional.interpolate(cropped_mask[None, None, ...], size=512, mode="nearest")[
                0, 0]
            cropped_image = image[:, y_start:y_end, x_start:x_end]
            cropped_image = torch.nn.functional.interpolate(cropped_image[None, ...], size=512, mode="bilinear")[0]
            mask_in_crop = False
            if torch.all(cropped_mask == 0):
                mask_in_crop = True
                y = x = 0
                w = h = 512
                final_cropped_mask = cropped_mask
            while not mask_in_crop:
                crop_size = int((self.final_min_crop_size + torch.rand(1) * (512 - self.final_min_crop_size)).item())
                y, x, h, w = transforms.RandomCrop.get_params(cropped_image, output_size=(crop_size, crop_size))
                final_cropped_mask = cropped_mask[y:y + h, x:x + w]
                if torch.sum(torch.where(final_cropped_mask>0, 1, 0)) > 512:
                    mask_in_crop = True
            final_cropped_image = cropped_image[:, y:y + h, x:x + w]
            final_cropped_image = \
            torch.nn.functional.interpolate(final_cropped_image[None, ...], size=512, mode="bilinear")[0]
            final_cropped_mask = \
            torch.nn.functional.interpolate(final_cropped_mask[None, None, ...], size=self.mask_size, mode="nearest")[
                0, 0]
            # final_cropped_mask = torch.where(final_cropped_mask > 0, 1., 0.)
            return final_cropped_image, final_cropped_mask, final_cropped_mask / 10
        return image, final_mask

    def __len__(self):
        return self.num_crops * len(self.images_paths)


class CelebaHQDataModule(pl.LightningDataModule):
    def __init__(
            self,
            images_dir: str = "./data",
            masks_dir: str = "./data",
            idx_mapping_file: str = "./data",
            test_file_names_file_path: str = "./data",
            non_test_file_names_file_path: str = "./data",
            train_num_crops: int = 1,
            train_parts_to_return: List[str]=("background", "eye", "mouth", "nose", "brow", "ear", "skin", "neck",
                                              "cloth", "hair"),
            test_parts_to_return: List[str] = ("background", "eye", "mouth", "nose", "brow", "ear", "skin", "neck",
                                                "cloth", "hair"),
            batch_size: int = 1,
            mask_size: int = 256,
            train_data_ids: List[int] = (i for i in range(10))
    ):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.idx_mapping_file = idx_mapping_file
        self.test_file_names_file_path = test_file_names_file_path
        self.non_test_file_names_file_path = non_test_file_names_file_path
        self.train_num_crops = train_num_crops
        self.train_parts_to_return = train_parts_to_return
        self.test_parts_to_return = test_parts_to_return
        self.batch_size = batch_size
        self.mask_size = mask_size
        self.train_data_ids = train_data_ids

    def setup(self, stage: str):
        if stage == "fit":

            self.train_dataset = CelebaHQDataset(
                images_dir=self.images_dir,
                masks_dir=self.masks_dir,
                idx_mapping_file=self.idx_mapping_file,
                parts_to_return=self.train_parts_to_return,
                train_data_ids=self.train_data_ids,
                file_names_file_path=self.non_test_file_names_file_path,
                num_crops=self.train_num_crops,
                train=True,
                mask_size=self.mask_size,
            )
        elif stage == "test":
            self.test_dataset = CelebaHQDataset(
                images_dir=self.images_dir,
                masks_dir=self.masks_dir,
                idx_mapping_file=self.idx_mapping_file,
                parts_to_return=self.test_parts_to_return,
                file_names_file_path=self.test_file_names_file_path,
                train=False,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)
