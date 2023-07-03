from typing import List
from src.utils import get_square_cropping_coords, adjust_bbox_coords, get_bbox_data
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional
from PIL import Image
from scipy import io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import random

part_mapping = {
    'dog': {
        'head': 'head',
        'nose': 'nose',
        'torso': 'torso',
        'lfleg': 'leg',
        'lfpa': 'paw',
        'rfleg': 'leg',
        'rfpa': 'paw',
        'rbleg': 'leg',
        'rbpa': 'paw',
        'tail': 'tail',
        'neck': 'neck',
        'lbleg': 'leg',
        'lbpa': 'paw',
        'lear': 'ear',
        'rear': 'ear',
        'leye': 'eye',
        'reye': 'eye',
        'muzzle': 'muzzle'
    },
    'car': {
        'backside': 'body',
        'bliplate': 'plate',
        'door_1': 'body',
        'door_2': 'body',
        'door_3': 'body',
        'fliplate': 'plate',
        'frontside': 'body',
        'headlight_1': 'light',
        'headlight_2': 'light',
        'headlight_3': 'light',
        'headlight_4': 'light',
        'headlight_5': 'light',
        'headlight_6': 'light',
        'leftmirror': 'body',
        'leftside': 'body',
        'rightmirror': 'body',
        'rightside': 'body',
        'roofside': 'body',
        'wheel_1': 'wheel',
        'wheel_2': 'wheel',
        'wheel_3': 'wheel',
        'wheel_4': 'wheel',
        'wheel_5': 'wheel',
        'window_1': 'window',
        'window_2': 'window',
        'window_3': 'window',
        'window_4': 'window',
        'window_5': 'window',
        'window_6': 'window',
        'window_7': 'window',
    },

    "horse": {
        'head': 'head',
        'lear': 'head',
        'leye': 'head',
        'muzzle': 'head',
        'neck': 'neck+torso',
        'rear': 'head',
        'reye': 'head',
        'lbho': 'leg',
        'lfho': 'leg',
        'rbho': 'leg',
        'rfho': 'leg',
        'lblleg': 'leg',
        'lbuleg': 'leg',
        'lflleg': 'leg',
        'lfuleg': 'leg',
        'rblleg': 'leg',
        'rbuleg': 'leg',
        'rflleg': 'leg',
        'rfuleg': 'leg',
        'tail': 'tail',
        'torso': 'neck+torso',
    }
}


def get_file_dirs(annotation_file):
    ann_file_path = f"/home/aliasgahr/Downloads/part_segmentation/trainval/Annotations_Part/{annotation_file}"
    img_file_path = f"/home/aliasgahr/Downloads/part_segmentation/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/JPEGImages/{annotation_file.replace('mat', 'jpg')}"

    return ann_file_path, img_file_path


object_size_thresh={"car": 50 * 50, "horse": 32 * 32, "dog": 32 * 32}


class PascalVOCPartDataset(Dataset):
    def __init__(self, data_file_ids_file, object_name_to_return, part_names_to_return, object_size_thresh=50*50,
                 train=True, train_data_ids=(0,), num_crops=1, mask_size=256,
                 remove_overlapping_objects=False, object_overlapping_threshold=0.05, blur_background=False, fill_background_with_black=True, final_min_crop_size=256):
        super().__init__()
        self.train = train
        self.train_data_ids = train_data_ids
        self.num_crops = num_crops
        self.mask_size = mask_size
        self.blur_background = blur_background
        self.fill_background_with_black = fill_background_with_black
        self.counter = 0
        self.data = []
        self.part_names_to_return = part_names_to_return
        self.current_idx = None
        self.final_min_crop_size = final_min_crop_size
        counter_ = 0
        with open(data_file_ids_file) as file:
            data_file_ids = file.readlines()
        progress_bar = tqdm(data_file_ids)
        progress_bar.set_description("preparing the data...")
        for line in progress_bar:
            ann_file_id, is_class_data = line.strip().split()
            if is_class_data == "-1":
                continue
            ann_file_path, img_file_path = get_file_dirs(f"{ann_file_id}.mat")
            anns = io.loadmat(ann_file_path)['anno'][0, 0][1]
            num_objects = anns.shape[1]
            object_ids_to_remove = []
            if remove_overlapping_objects:
                object_bbox_masks = []
                for object_id in range(num_objects):
                    num_parts = anns[0, object_id][3].shape[1]
                    if num_parts == 0:
                        continue
                    object_name = anns[0, object_id][0][0]
                    if object_name != object_name_to_return:
                        continue
                    object_mask = anns[0, object_id][2]
                    x_min, x_max, y_min, y_max = get_bbox_data(object_mask)
                    width, height = (x_max - x_min), (y_max - y_min)
                    object_bbox_size = width * height
                    if object_bbox_size < object_size_thresh:
                        continue
                    object_bbox_mask = np.zeros_like(object_mask)
                    object_bbox_mask[y_min:y_max, x_min:x_max] = 1
                    object_bbox_masks.append(object_bbox_mask)
                if len(object_bbox_masks) > 0:
                    object_bbox_masks = np.stack(object_bbox_masks, axis=0)
                    object_bbox_masks = np.reshape(object_bbox_masks, (object_bbox_masks.shape[0], -1))
                    intersection = object_bbox_masks @ object_bbox_masks.T
                    a = np.stack(
                        [np.sum(object_bbox_masks, axis=1), np.ones(object_bbox_masks.shape[0])],
                        axis=1)
                    b = np.concatenate([np.ones((1, object_bbox_masks.shape[0])),
                                        np.sum(object_bbox_masks, axis=1)[None, ...]], axis=0)
                    union = a @ b
                    iou = (intersection / (union - intersection + 1e-7))
                    ys, xs = np.where(np.triu(iou, 1) >= object_overlapping_threshold)
                    object_ids_to_remove += np.concatenate([ys, xs]).tolist()
            if len(object_ids_to_remove) > 0:
                print(object_ids_to_remove)
            for object_id in range(num_objects):
                if object_id in object_ids_to_remove:
                    continue
                num_parts = anns[0, object_id][3].shape[1]
                if num_parts == 0:
                    continue
                object_name = anns[0, object_id][0][0]
                if object_name != object_name_to_return:
                    continue
                # data = self.data.get(object_name, [])
                object_mask = anns[0, object_id][2]
                x_min, x_max, y_min, y_max = get_bbox_data(object_mask)
                width, height = (x_max - x_min), (y_max - y_min)
                object_bbox_size = width * height
                if object_bbox_size < object_size_thresh:
                    continue

                # object_data = self.data.get(object_name, {})
                aux_part_data = {}
                includes_part = False
                for part_id in range(num_parts):
                    part_name = anns[0, object_id][3][0][part_id][0][0]
                    aux_data = aux_part_data.get(part_mapping[object_name][part_name], [])
                    aux_data.append(part_id)
                    aux_part_data[part_mapping[object_name][part_name]] = aux_data
                    if part_mapping[object_name][part_name] != "body":
                        aux_data = aux_part_data.get("non_body", [])
                        aux_data.append(part_id)
                        aux_part_data["non_body"] = aux_data
                    if part_mapping[object_name][part_name] in self.part_names_to_return:
                        includes_part = True
                if not includes_part:
                    continue
                m_h, m_w = object_mask.shape
                if width < height:
                    x_min, x_max = adjust_bbox_coords(height, width, x_min, x_max, m_w)

                elif width > height:
                    y_min, y_max = adjust_bbox_coords(width, height, y_min, y_max, m_h)

                if x_min < 0 or y_min < 0:
                    counter_ += 1
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)

                self.data.append({
                    "file_name": f"{ann_file_id}.mat",
                    "object_id": object_id,
                    "part_data": aux_part_data,
                    "bbox": [x_min, y_min, x_max, y_max],
                })

        print("number of images with changed aspect ratio: ", counter_)
        if train_data_ids != (0,):
            selected_data = []
            for id in train_data_ids:
                selected_data.append(self.data[id])
            self.data = selected_data

    def __getitem__(self, idx):
        idx = idx // self.num_crops
        data = self.data[idx]
        file_name, object_id, part_data, bbox = data["file_name"], data["object_id"], data["part_data"], data["bbox"]
        ann_file_path, img_file_path = get_file_dirs(file_name)
        image = Image.open(img_file_path)
        data = io.loadmat(ann_file_path)
        anns = data['anno'][0, 0][1]
        if self.blur_background:
            object_mask = anns[0, object_id][2]
            blurred_image = transforms.functional.gaussian_blur(image, 101, 10)
            image = Image.fromarray(np.where(object_mask[..., None] > 0, np.array(image), np.array(blurred_image)).astype(np.uint8))
        if self.fill_background_with_black:
            object_mask = anns[0, object_id][2]
            image = Image.fromarray((np.array(image) * np.where(object_mask[..., None] > 0, 1, 0)).astype(np.uint8))
        body_mask = 0
        non_body_mask = 0
        for idx, part_name in enumerate(self.part_names_to_return):
            part_ids = part_data.get(part_name, None)
            if part_ids is not None:
                if part_name == "body":
                    for part_id in part_ids:
                        body_mask = np.where(anns[0, object_id][3][0][part_id][1] > 0, idx, body_mask)
                    if part_data.get("non_body", False):
                        for part_id in part_data["non_body"]:
                            body_mask = np.where(anns[0, object_id][3][0][part_id][1] > 0, 0, body_mask)
                else:
                    for part_id in part_ids:
                        non_body_mask = np.where(anns[0, object_id][3][0][part_id][1] > 0, idx, non_body_mask)
        mask = non_body_mask + body_mask

        image = image.crop(bbox)
        mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        if self.train:
            image = transforms.functional.resize(image, (512, 512))
            image = transforms.functional.to_tensor(image)
            mask = transforms.functional.resize(torch.as_tensor(mask)[None, ...], size=(512, 512),
                                                interpolation=transforms.InterpolationMode.NEAREST)[0].type(torch.uint8)
            coords = []
            if self.num_crops > 1:
                x_start, x_end, y_start, y_end, crop_size = get_square_cropping_coords(torch.where(mask>0, 1, 0))
                for square_size in torch.linspace(crop_size, 512, self.num_crops + 1)[1:-1]:
                    x_start, x_end, y_start, y_end, crop_size = get_square_cropping_coords(torch.where(mask>0, 1, 0),
                                                                                           square_size=int(square_size))
                    coords.append([y_start, y_end, x_start, x_end])
            coords.append([0, 512, 0, 512])
            random_idx = random.randint(0, self.num_crops - 1)
            y_start, y_end, x_start, x_end = coords[random_idx]
            cropped_mask = mask[y_start:y_end, x_start:x_end]
            cropped_mask = \
                torch.nn.functional.interpolate(cropped_mask[None, None, ...], size=512, mode="nearest")[
                    0, 0].type(torch.float)
            cropped_image = image[:, y_start:y_end, x_start:x_end]
            cropped_image = torch.nn.functional.interpolate(cropped_image[None, ...], size=512, mode="bilinear")[0]
            mask_in_crop = False
            while not mask_in_crop:
                crop_size = int((self.final_min_crop_size + torch.rand(1) * (512 - self.final_min_crop_size)).item())
                y, x, h, w = transforms.RandomCrop.get_params(cropped_image, output_size=(crop_size, crop_size))
                final_cropped_mask = cropped_mask[y:y + h, x:x + w]
                if torch.sum(torch.where(final_cropped_mask > 0, 1, 0)) > 512:
                    mask_in_crop = True
            final_cropped_image = cropped_image[:, y:y + h, x:x + w]
            final_cropped_image = \
            torch.nn.functional.interpolate(final_cropped_image[None, ...], size=512, mode="bilinear")[0]
            final_cropped_mask = \
            torch.nn.functional.interpolate(final_cropped_mask[None, None, ...], size=self.mask_size, mode="nearest")[
                0, 0]
            return final_cropped_image, final_cropped_mask, final_cropped_mask / 10
        else:
            image = transforms.functional.resize(image, 512)
            image = transforms.functional.to_tensor(image)
            mask = transforms.functional.resize(torch.as_tensor(mask)[None, ...], size=512,
                                                interpolation=transforms.InterpolationMode.NEAREST)[0].type(torch.uint8)
            return image, mask

    def __len__(self):
        return self.num_crops * len(self.data)


class PascalVOCPartDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data_file_ids_file: str = "./data",
            val_data_file_ids_file: str = "./data",
            object_name: str = "car",
            train_part_names: List[str] = [""],
            test_part_names: List[str] = [""],
            train_num_crops: int = 1,
            batch_size: int = 1,
            train_data_ids: List[int] = (2,),
            mask_size: int = 256,
            blur_background: bool = False,
            fill_background_with_black: bool = False,
            remove_overlapping_objects: bool = False,
            object_overlapping_threshold: float = 0.05,
    ):
        super().__init__()
        self.train_data_file_ids_file = train_data_file_ids_file
        self.val_data_file_ids_file = val_data_file_ids_file
        self.object_name = object_name
        self.train_part_names = train_part_names
        self.test_part_names = test_part_names
        self.train_num_crops = train_num_crops
        self.batch_size = batch_size
        self.train_data_ids = train_data_ids
        self.mask_size = mask_size
        self.blur_background = blur_background
        self.fill_background_with_black = fill_background_with_black
        self.remove_overlapping_objects = remove_overlapping_objects
        self.object_overlapping_threshold = object_overlapping_threshold

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = PascalVOCPartDataset(
                data_file_ids_file=self.train_data_file_ids_file,
                object_name_to_return=self.object_name,
                part_names_to_return=self.train_part_names,
                object_size_thresh=object_size_thresh[self.object_name],
                train=True,
                train_data_ids=self.train_data_ids,
                num_crops=self.train_num_crops,
                mask_size=self.mask_size,
                remove_overlapping_objects=self.remove_overlapping_objects,
                object_overlapping_threshold=self.object_overlapping_threshold,
                blur_background=self.blur_background,
                fill_background_with_black=self.fill_background_with_black
            )
        elif stage == "test":
            self.test_dataset = PascalVOCPartDataset(
                data_file_ids_file=self.val_data_file_ids_file,
                object_name_to_return=self.object_name,
                part_names_to_return=self.test_part_names,
                object_size_thresh=object_size_thresh[self.object_name],
                train=False,
                remove_overlapping_objects=self.remove_overlapping_objects,
                object_overlapping_threshold=self.object_overlapping_threshold,
                blur_background=self.blur_background,
                fill_background_with_black=self.fill_background_with_black
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1, shuffle=False)
