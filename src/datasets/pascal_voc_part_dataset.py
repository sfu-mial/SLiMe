from typing import Tuple

import cv2
import os
from src.utils import adjust_bbox_coords, get_bbox_data
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional
from PIL import Image
from scipy import io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from .paper_test_sample_dataset import PaperTestSampleDataset


import albumentations as A
from albumentations.pytorch import ToTensorV2

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


def get_file_dirs(annotation_file, ann_file_base_dir, images_base_dir):
    ann_file_path = os.path.join(ann_file_base_dir, annotation_file)
    img_file_path = os.path.join(images_base_dir, annotation_file.replace('mat', 'jpg'))

    return ann_file_path, img_file_path


object_size_thresh={"car": 50 * 50, "horse": 32 * 32, "dog": 32 * 32}


class PascalVOCPartDataset(Dataset):
    def __init__(
            self,
            ann_file_base_dir,
            images_base_dir,
            data_file_ids_file,
            object_name_to_return,
            part_names_to_return,
            object_size_thresh=50*50,
            train=True,
            train_data_ids=(0,),
            mask_size=512,
            remove_overlapping_objects=False,
            object_overlapping_threshold=0.05,
            blur_background=False,
            fill_background_with_black=True,
            final_min_crop_size=512,
            single_object=True,
            adjust_bounding_box=False,
            zero_pad_test_output=False
    ):
        super().__init__()
        self.ann_file_base_dir = ann_file_base_dir
        self.images_base_dir = images_base_dir
        self.train = train
        self.train_data_ids = train_data_ids
        self.mask_size = mask_size
        self.blur_background = blur_background
        self.fill_background_with_black = fill_background_with_black
        self.counter = 0
        self.data = []
        self.part_names_to_return = part_names_to_return
        self.return_whole = False
        if self.part_names_to_return[1] == 'whole':
            if object_name_to_return == 'car':
                self.part_names_to_return = ['background', 'body', 'light', 'plate', 'wheel', 'window']
            elif object_name_to_return == 'horse':
                self.part_names_to_return = ['background', 'head', 'neck+torso', 'leg', 'tial']
            self.return_whole = True
        self.current_idx = None
        self.final_min_crop_size = final_min_crop_size
        self.single_object = single_object
        self.zero_pad_test_output = zero_pad_test_output
        counter_ = 0
        with open(data_file_ids_file) as file:
            data_file_ids = file.readlines()
        progress_bar = tqdm(data_file_ids)
        progress_bar.set_description("preparing the data...")
        for line in progress_bar:
            ann_file_id, is_class_data = line.strip().split()
            if is_class_data == "-1":
                continue
            ann_file_path, img_file_path = get_file_dirs(f"{ann_file_id}.mat", ann_file_base_dir=ann_file_base_dir, images_base_dir=images_base_dir)
            anns = io.loadmat(ann_file_path)['anno'][0, 0][1]
            num_objects = anns.shape[1]
            object_ids_to_remove = []
            if remove_overlapping_objects:
                object_bbox_masks = []
                for object_id in range(num_objects):
                    object_mask = anns[0, object_id][2]
                    x_min, x_max, y_min, y_max = get_bbox_data(object_mask)
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
            if self.single_object:
                aux_part_data = {}
                data = []
                for object_id in range(num_objects):
                    if object_id in object_ids_to_remove:
                        continue
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

                    includes_part = False
                    for part_id in range(num_parts):
                        part_name = anns[0, object_id][3][0][part_id][0][0]
                        aux_data = aux_part_data.get(part_mapping[object_name][part_name], [])
                        aux_data.append((object_id, part_id))
                        aux_part_data[part_mapping[object_name][part_name]] = aux_data
                        if part_mapping[object_name][part_name] != "body":
                            aux_data = aux_part_data.get("non_body", [])
                            aux_data.append((object_id, part_id))
                            aux_part_data["non_body"] = aux_data
                        if part_mapping[object_name][part_name] in self.part_names_to_return:
                            includes_part = True
                    if not includes_part:
                        continue
                    if adjust_bounding_box:
                        m_h, m_w = object_mask.shape
                        if width < height:
                            x_min, x_max = adjust_bbox_coords(height, width, x_min, x_max, m_w)

                        elif width > height:
                            y_min, y_max = adjust_bbox_coords(width, height, y_min, y_max, m_h)

                        if x_min < 0 or y_min < 0:
                            counter_ += 1
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)

                    data.append({
                        "file_name": f"{ann_file_id}.mat",
                        "object_id": object_id,
                        "bbox": [x_min, y_min, x_max, y_max],
                    })
                for i in range(len(data)):
                    data[i]["part_data"] = aux_part_data
                    self.data.append(data[i])

            else:
                whole_objects_mask = 0
                includes_part = False
                objects_data = {}
                for object_id in range(num_objects):
                    if object_id in object_ids_to_remove:
                        continue
                    num_parts = anns[0, object_id][3].shape[1]
                    if num_parts == 0:
                        continue
                    object_name = anns[0, object_id][0][0]
                    if object_name != object_name_to_return:
                        continue
                    object_mask = anns[0, object_id][2]
                    whole_objects_mask += object_mask
                    x_min, x_max, y_min, y_max = get_bbox_data(object_mask)
                    width, height = (x_max - x_min), (y_max - y_min)
                    object_bbox_size = width * height
                    if object_bbox_size < object_size_thresh:
                        continue
                    aux_part_data = {}
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
                    objects_data[object_id] = aux_part_data
                if not includes_part:
                    continue
                x_min, x_max, y_min, y_max = get_bbox_data(np.where(whole_objects_mask > 0, 1, 0))
                self.data.append({
                    "file_name": f"{ann_file_id}.mat",
                    "part_data": objects_data,
                    "bbox": [x_min, y_min, x_max, y_max],
                })
        print("number of images with changed aspect ratio: ", counter_)
        if train_data_ids != (0,):
            selected_data = []
            for id in train_data_ids:
                selected_data.append(self.data[id])
            self.data = selected_data

        if zero_pad_test_output:
            self.train_transform = A.Compose([
                # A.Resize(512, 512),
                A.LongestMaxSize(512),
                A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, value=0,
                              mask_value=0),
                A.HorizontalFlip(),
                # A.RandomScale((0.5, 2), always_apply=True),
                A.RandomResizedCrop(512, 512, (0.8, 1)),
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
                # A.LongestMaxSize(512),
                # A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, value=0,
                #               mask_value=0),
                A.HorizontalFlip(),
                # A.RandomScale((0.5, 2), always_apply=True),
                A.GaussianBlur(blur_limit=(1, 31)),
                # A.Persepective(scale=(0.05, 0.1), pad_mode=cv2.BORDER_REPLICATE),
                A.RandomResizedCrop(512, 512, (0.4, 1)),
                A.Rotate((-10, 10), border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                ToTensorV2()
            ])
            self.test_transform = A.Compose([
                # A.SmallestMaxSize(512),
                A.Resize(512, 512),
                ToTensorV2()
            ])

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.single_object:
            file_name, object_id, part_data, bbox = data["file_name"], data["object_id"], data["part_data"], data["bbox"]
        else:
            file_name, part_data, bbox = data["file_name"], data["part_data"], data["bbox"]
        ann_file_path, img_file_path = get_file_dirs(file_name, ann_file_base_dir=self.ann_file_base_dir, images_base_dir=self.images_base_dir)
        image = Image.open(img_file_path)
        data = io.loadmat(ann_file_path)
        anns = data['anno'][0, 0][1]
        if self.single_object:
            body_mask = 0
            non_body_mask = 0
            whole_mask = 0
            for idx, part_name in enumerate(self.part_names_to_return):
                part_ids = part_data.get(part_name, None)
                if part_ids is not None:
                    if part_name == "body":
                        for object_id, part_id in part_ids:
                            body_mask = np.where(anns[0, object_id][3][0][part_id][1] > 0, idx, body_mask)
                            whole_mask += anns[0, object_id][2]  # object mask
                        if part_data.get("non_body", False):
                            for object_id, part_id in part_data["non_body"]:
                                body_mask = np.where(anns[0, object_id][3][0][part_id][1] > 0, 0, body_mask)
                    else:
                        for object_id, part_id in part_ids:
                            non_body_mask = np.where(anns[0, object_id][3][0][part_id][1] > 0, idx, non_body_mask)
                            whole_mask += anns[0, object_id][2]  # object mask
            mask = non_body_mask + body_mask
            whole_mask = np.where(whole_mask > 0, 1, 0)
        else:
            whole_mask = 0
            body_mask = 0
            non_body_mask = 0
            for object_id in part_data:
                for idx, part_name in enumerate(self.part_names_to_return):
                    part_ids = part_data[object_id].get(part_name, None)
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
                object_mask = anns[0, object_id][2]
                whole_mask += object_mask
            mask = non_body_mask + body_mask

        if self.blur_background:
            blurred_image = transforms.functional.gaussian_blur(image, 101, 10)
            image = Image.fromarray(np.where(whole_mask[..., None] > 0, np.array(image), np.array(blurred_image)).astype(np.uint8))
        if self.fill_background_with_black:
            image = Image.fromarray((np.array(image) * np.where(whole_mask[..., None] > 0, 1, 0)).astype(np.uint8))

        image = image.crop(bbox)
        mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if self.return_whole:
            mask = np.where(mask > 0, 1, 0)
        if self.train:
            mask_is_included = False
            original_mask_size = np.where(mask > 0, 1, 0).sum()
            while not mask_is_included:
                result = self.train_transform(image=np.array(image), mask=mask)
                # mask = torch.as_tensor(result["mask"])
                if np.where(result["mask"] > 0, 1, 0).sum() / original_mask_size > 0.3:
                    mask_is_included = True
            image = result["image"]
            mask = torch.as_tensor(result["mask"])
            mask = \
                torch.nn.functional.interpolate(mask[None, None, ...].type(torch.float), self.mask_size, mode="nearest")[0, 0]
            return image/255, mask
        else:
            result = self.test_transform(image=np.array(image), mask=mask)
            image = result["image"]
            mask = torch.as_tensor(result["mask"])
            return image/255, mask

    def __len__(self):
        return len(self.data)


class PascalVOCPartDataModule(pl.LightningDataModule):
    def __init__(
            self,
            ann_file_base_dir: str,
            images_base_dir: str,
            car_test_data_dir: str,
            train_data_file_ids_file: str = "./data",
            val_data_file_ids_file: str = "./data",
            object_name: str = "car",
            part_names: Tuple[str] = [""],
            batch_size: int = 1,
            train_data_ids: Tuple[int] = (2,),
            val_data_ids: Tuple[int] = (2,),
            mask_size: int = 256,
            blur_background: bool = False,
            fill_background_with_black: bool = False,
            remove_overlapping_objects: bool = False,
            object_overlapping_threshold: float = 0.05,
            final_min_crop_size: int = 512,
            single_object: bool = True,
            adjust_bounding_box: bool = False,
            zero_pad_test_output: bool = False
    ):
        super().__init__()
        self.ann_file_base_dir = ann_file_base_dir
        self.images_base_dir = images_base_dir
        self.car_test_data_dir = car_test_data_dir
        self.train_data_file_ids_file = train_data_file_ids_file
        self.val_data_file_ids_file = val_data_file_ids_file
        self.object_name = object_name
        self.part_names = part_names
        self.batch_size = batch_size
        self.train_data_ids = train_data_ids
        self.val_data_ids = val_data_ids
        self.mask_size = mask_size
        self.blur_background = blur_background
        self.fill_background_with_black = fill_background_with_black
        self.remove_overlapping_objects = remove_overlapping_objects
        self.object_overlapping_threshold = object_overlapping_threshold
        self.final_min_crop_size = final_min_crop_size
        self.single_object = single_object
        self.adjust_bounding_box = adjust_bounding_box
        self.zero_pad_test_output = zero_pad_test_output

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = PascalVOCPartDataset(
                ann_file_base_dir=self.ann_file_base_dir,
                images_base_dir=self.images_base_dir,
                data_file_ids_file=self.train_data_file_ids_file,
                object_name_to_return=self.object_name,
                part_names_to_return=self.part_names,
                object_size_thresh=object_size_thresh[self.object_name],
                train=True,
                train_data_ids=self.train_data_ids,
                mask_size=self.mask_size,
                remove_overlapping_objects=self.remove_overlapping_objects,
                object_overlapping_threshold=self.object_overlapping_threshold,
                blur_background=self.blur_background,
                fill_background_with_black=self.fill_background_with_black,
                final_min_crop_size=self.final_min_crop_size,
                single_object=self.single_object,
                adjust_bounding_box=self.adjust_bounding_box,
            )
            self.val_dataset = PascalVOCPartDataset(
                ann_file_base_dir=self.ann_file_base_dir,
                images_base_dir=self.images_base_dir,
                data_file_ids_file=self.train_data_file_ids_file,
                object_name_to_return=self.object_name,
                part_names_to_return=self.part_names,
                object_size_thresh=object_size_thresh[self.object_name],
                train=False,
                train_data_ids=self.val_data_ids,
                remove_overlapping_objects=self.remove_overlapping_objects,
                object_overlapping_threshold=self.object_overlapping_threshold,
                blur_background=self.blur_background,
                fill_background_with_black=self.fill_background_with_black,
                single_object=self.single_object,
                adjust_bounding_box=self.adjust_bounding_box,
                zero_pad_test_output=self.zero_pad_test_output,
            )
        elif stage == "test":
            if self.object_name == 'horse':
                self.test_dataset = PascalVOCPartDataset(
                    ann_file_base_dir=self.ann_file_base_dir,
                    images_base_dir=self.images_base_dir,
                    data_file_ids_file=self.val_data_file_ids_file,
                    object_name_to_return=self.object_name,
                    part_names_to_return=self.part_names,
                    object_size_thresh=object_size_thresh[self.object_name],
                    train=False,
                    remove_overlapping_objects=self.remove_overlapping_objects,
                    object_overlapping_threshold=self.object_overlapping_threshold,
                    blur_background=self.blur_background,
                    fill_background_with_black=self.fill_background_with_black,
                    single_object=self.single_object,
                    adjust_bounding_box=self.adjust_bounding_box,
                    zero_pad_test_output=self.zero_pad_test_output,
                )
            elif self.object_name == 'car':
                if self.fill_background_with_black:
                    image_dir = os.path.join(self.car_test_data_dir, 'image_no_bg')
                else:
                    image_dir = os.path.join(self.car_test_data_dir, 'image_bg')
                self.test_dataset = PaperTestSampleDataset(
                    image_dir,
                    os.path.join(self.car_test_data_dir, 'gt_mask'),
                    train=False,
                    part_names=self.part_names[1:],
                    object_name=self.object_name,
                )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)
