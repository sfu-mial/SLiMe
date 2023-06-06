import math
from copy import deepcopy

import os
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional
from PIL import Image
from scipy import io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

part_mapping = {
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

    'head': 'head',
    'lear': 'head',
    'leye': 'head',
    'muzzle': 'head',
    'neck': 'neck',
    'rear': 'head',
    'reye': 'head',
    'lbho': 'leg',
    'lfho': 'leg',
    'rbho': 'leg',
    'rfho': 'leg',
    'lblleg': 'log',
    'lbuleg': 'leg',
    'lflleg': 'leg',
    'lfuleg': 'leg',
    'rblleg': 'leg',
    'rbuleg': 'leg',
    'rflleg': 'leg',
    'rfuleg': 'leg',
    'tail': 'tail',
    'torso': 'torso',
}


def get_file_dirs(annotation_file):
    ann_file_path = f"/home/aliasgahr/Downloads/part_segmentation/trainval/Annotations_Part/{annotation_file}"
    img_file_path = f"/home/aliasgahr/Downloads/part_segmentation/VOCtrainval_03-May-2010/VOCdevkit/VOC2010/JPEGImages/{annotation_file.replace('mat', 'jpg')}"

    return ann_file_path, img_file_path


def get_bbox_data(mask):
    ys, xs = np.where(mask == 1)
    return xs.min(), xs.max(), ys.min(), ys.max()


def adjust_bbox_coords(long_side_length, short_side_length, short_side_coord_min, short_side_coord_max, short_side_original_length):
    margin = (long_side_length - short_side_length) / 2
    short_side_coord_min -= math.ceil(margin)
    short_side_coord_max += math.floor(margin)
    short_side_coord_max -= min(0, short_side_coord_min)
    short_side_coord_min = max(0, short_side_coord_min)
    short_side_coord_min += min(short_side_original_length - short_side_coord_max, 0)
    short_side_coord_max = min(short_side_original_length, short_side_coord_max)

    return short_side_coord_min, short_side_coord_max


class PascalVOCPartDataset(Dataset):
    def __init__(self, ann_file_names, object_size_thresh={"car": 50 * 50, "horse": 32 * 32}, train=True, train_data_id=0, num_crops=1, mask_size=256, fill_background_with_black=False, remove_overlapping_objects=False, object_overlapping_threshold=0.05):
        super().__init__()
        self.train = train
        self.train_data_id = train_data_id
        self.num_crops = num_crops
        self.mask_size = mask_size
        self.fill_background_with_black = fill_background_with_black
        self.crop_ratio = None
        self.counter = 0
        self.dataset_len = None
        self.data = {}
        self.object_name = None
        self.part_names = None
        self.current_idx = None
        self.object_part_data = []
        counter_ = 0
        progress_bar = tqdm(ann_file_names)
        progress_bar.set_description("preparing the data...")
        for ann_file_name in progress_bar:
            ann_file_path, img_file_path = get_file_dirs(ann_file_name)
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
                    if object_name not in ["car", "horse"]:
                        continue
                    object_mask = anns[0, object_id][2]
                    x_min, x_max, y_min, y_max = get_bbox_data(object_mask)
                    width, height = (x_max - x_min), (y_max - y_min)
                    object_bbox_size = width * height
                    if object_bbox_size < object_size_thresh[object_name]:
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
                if object_name not in ["car", "horse"]:
                    continue
                object_mask = anns[0, object_id][2]
                x_min, x_max, y_min, y_max = get_bbox_data(object_mask)
                width, height = (x_max - x_min), (y_max - y_min)
                object_bbox_size = width * height
                if object_bbox_size < object_size_thresh[object_name]:
                    continue

                object_data = self.data.get(object_name, {})
                aux_part_data = {}
                for part_id in range(num_parts):
                    part_name = anns[0, object_id][3][0][part_id][0][0]
                    aux_data = aux_part_data.get(part_mapping[part_name], [])
                    aux_data.append(part_id)
                    aux_part_data[part_mapping[part_name]] = aux_data

                m_h, m_w = object_mask.shape
                if width < height:
                    x_min, x_max = adjust_bbox_coords(height, width, x_min, x_max, m_w)

                elif width > height:
                    y_min, y_max = adjust_bbox_coords(width, height, y_min, y_max, m_h)

                if x_min < 0 or y_min < 0:
                    counter_ += 1
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)

                for part_name in aux_part_data.keys():
                    part_data = object_data.get(part_name, [])
                    part_data.append({
                        "file_name": ann_file_name,
                        "object_id": object_id,
                        "part_ids": aux_part_data[part_name],
                        "bbox": [x_min, y_min, x_max, y_max],
                    })
                    object_data[part_name] = part_data
                self.data[object_name] = object_data
        print("number of images with changed aspect ratio: ", counter_)

    def get_object_names(self):
        return sorted(list(self.data.keys()))

    def get_object_part_names(self, object_name_to_get):
        return sorted(list(self.data[object_name_to_get].keys()))

    def setup(self, object_name, part_name):
        self.object_part_data = self.data[object_name][part_name]

    def set_dataset_len(self, dataset_len):
        self.dataset_len = dataset_len

    def set_num_crops(self, num_crops):
        self.num_crops = num_crops

    def set_train_data_id(self, train_data_id):
        self.train_data_id = train_data_id

    def set_crop_ratio(self, crop_ratio):
        self.crop_ratio = crop_ratio

    def set_train(self, train):
        self.train = train

    def set_fill_background_with_black(self, fill_background_with_black):
        self.fill_background_with_black = fill_background_with_black

    def __getitem__(self, idx):
        if self.train:
            idx = self.train_data_id
        else:
            idx = idx // self.num_crops
            if self.counter == 0:
                self.current_idx = idx
            self.counter += 1
            self.counter = self.counter % self.num_crops
            idx = self.current_idx

        data = self.object_part_data[idx]
        file_name, object_id, part_ids, bbox = data["file_name"], data["object_id"], data["part_ids"], data["bbox"]
        ann_file_path, img_file_path = get_file_dirs(file_name)
        image = Image.open(img_file_path)
        data = io.loadmat(ann_file_path)
        anns = data['anno'][0, 0][1]
        if self.fill_background_with_black:
            object_mask = anns[0, object_id][2]
            image = Image.fromarray((np.array(image)*np.where(object_mask[..., None] > 0, 1, 0)).astype(np.uint8))
        mask = 0
        for part_id in part_ids:
            mask += anns[0, object_id][3][0][part_id][1]
        mask = np.where(mask > 0, 1, 0)
        mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        image = image.crop(bbox)
        image = transforms.functional.resize(image, (int(512/self.crop_ratio), int(512/self.crop_ratio)))
        mask = transforms.functional.resize(Image.fromarray((mask * 255).astype(np.uint8)), (int(512/self.crop_ratio), int(512/self.crop_ratio)))
        y, x, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 512))
        image = transforms.functional.to_tensor(image)
        mask = transforms.functional.to_tensor(mask)
        if self.train:
            cropped_image = image[:, y:y + h, x:x + w]
            cropped_mask = mask[:, y:y + h, x:x + w]
            cropped_mask = torch.nn.functional.interpolate(cropped_mask[None, ...], self.mask_size)[0, 0]
            return cropped_image, cropped_mask / 10, cropped_mask / 10
        else:
            mask = torch.nn.functional.interpolate(mask[None, ...], self.mask_size)[0, 0]
            cropped_image = image[:, y:y + h, x:x + w]
            return cropped_image, y, x, mask, image

    def __len__(self):
        if self.dataset_len is not None:
            return self.dataset_len
        return self.num_crops * len(self.object_part_data)


class PascalVOCPartDataModule(pl.LightningDataModule):
    def __init__(
            self,
            annotations_files_dir: str = "./data",
            object_name: str = "car",
            part_name: str = "window",
            train_num_crops: int = 1,
            test_num_crops: int = 1,
            batch_size: int = 1,
            train_crop_ratio: float = 1.,
            test_crop_ratio: float = 1.,
            train_data_id: int = 2,
            mask_size: int = 256,
            fill_background_with_black: bool = False,
            remove_overlapping_objects: bool = False,
            object_overlapping_threshold: float = 0.05,
    ):
        super().__init__()
        self.annotations_files_dir = annotations_files_dir
        self.object_name = object_name
        self.part_name = part_name
        self.train_num_crops = train_num_crops
        self.test_num_crops = test_num_crops
        self.batch_size = batch_size
        self.train_crop_ratio = train_crop_ratio
        self.test_crop_ratio = test_crop_ratio
        self.train_data_id = train_data_id
        self.mask_size = mask_size
        self.fill_background_with_black = fill_background_with_black
        self.remove_overlapping_objects = remove_overlapping_objects
        self.object_overlapping_threshold = object_overlapping_threshold

    def setup(self, stage: str):
        self.train_dataset = PascalVOCPartDataset(
            ann_file_names=os.listdir(self.annotations_files_dir),
            mask_size=self.mask_size,
            fill_background_with_black=False,
            remove_overlapping_objects=self.remove_overlapping_objects,
            object_overlapping_threshold=self.object_overlapping_threshold,
        )
        self.train_dataset.setup(self.object_name, self.part_name)
        self.test_dataset = deepcopy(self.train_dataset)

        self.train_dataset.set_crop_ratio(self.train_crop_ratio)
        self.train_dataset.set_train_data_id(self.train_data_id)
        self.train_dataset.set_dataset_len(self.train_num_crops)
        self.train_dataset.set_num_crops(self.train_num_crops)
        self.train_dataset.set_train(True)

        self.test_dataset.set_crop_ratio(self.test_crop_ratio)
        self.test_dataset.set_num_crops(self.test_num_crops)
        self.test_dataset.set_train(False)
        self.test_dataset.set_fill_background_with_black(self.fill_background_with_black)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1, shuffle=False)
