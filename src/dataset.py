from typing import Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


class SampleDataset(Dataset):
    def __init__(self, image_dir, segmentation_dir, location1, location2, num_crops=1, crop_ratio=0.93, train=True):
        self.segmentation, self.eroded_segmentation = None, None
        self.emulated_attention_map1, self.emulated_attention_map2 = None, None
        self.crop_size = 512
        self.num_cropped = num_crops
        self.crop_ratio = crop_ratio
        image = Image.open(image_dir)
        w, h = image.size
        cropped_image_size = min(w, h)
        self.train = train
        if w != h:
            image = transforms.functional.center_crop(image, (cropped_image_size, cropped_image_size))
            if train:
                if w < h:
                    location1[0] = location1[0] - (h - w) // 2
                    location2[0] = location2[0] - (h - w) // 2
                else:
                    location1[1] = location1[1] - (w - h) // 2
                    location2[1] = location2[1] - (w - h) // 2
                assert 0 <= location1[0] < cropped_image_size and 0 <= location1[1] < cropped_image_size and 0 <= location2[
                    0] < cropped_image_size and 0 <= location2[
                           1] < cropped_image_size, "points are out of image after center cropping"
        self.image = transforms.functional.resize(image, size=(int(512 / crop_ratio), int(512 / crop_ratio)))

        if segmentation_dir is not None:
            segmentation = Image.open(segmentation_dir)
            # kernel_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            # segmentation = cv2.erode(cv2.dilate(np.array(segmentation), kernel_), kernel_, iterations=1)  # fill holes

            # segmentation = cv2.GaussianBlur(cv2.erode(segmentation, kernel, iterations=1), (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
            # segmentation = cv2.erode(segmentation, kernel, iterations=1)

            # segmentation = Image.fromarray(segmentation)
            self.segmentation = transforms.functional.resize(segmentation, size=(int(512 / crop_ratio), int(512 / crop_ratio)),
                                             interpolation=transforms.InterpolationMode.NEAREST)
        elif location1 is not None and location2 is not None:
            point_y1, point_x1 = int(int(512 / crop_ratio) * location1[0] / cropped_image_size), int(
                int(512 / crop_ratio) * location1[1] / cropped_image_size)
            point_y2, point_x2 = int(int(512 / crop_ratio) * location2[0] / cropped_image_size), int(
                int(512 / crop_ratio) * location2[1] / cropped_image_size)
            gaussian_radius = 100
            x, y = torch.meshgrid(torch.linspace(-1, 1, gaussian_radius * 2),
                                  torch.linspace(-1, 1, gaussian_radius * 2), indexing="xy")
            d = torch.sqrt(x * x + y * y)
            sigma, mu = 1 / 5, 0
            gaussian_dist = torch.exp(-(torch.pow(d - mu, 2) / (2.0 * sigma ** 2)))
            emulated_attention_map_size = int(512 / crop_ratio)
            self.emulated_attention_map1 = torch.zeros((emulated_attention_map_size, emulated_attention_map_size))
            self.emulated_attention_map1[
            max(0, point_y1 - gaussian_radius): min(emulated_attention_map_size, point_y1 + gaussian_radius),
            max(0, point_x1 - gaussian_radius): min(emulated_attention_map_size,
                                                    point_x1 + gaussian_radius)] = gaussian_dist[max(0,
                                                                                                     gaussian_radius - point_y1):gaussian_radius * 2 - max(
                0, gaussian_radius + point_y1 - emulated_attention_map_size), max(0,
                                                                                  gaussian_radius - point_x1):gaussian_radius * 2 - max(
                0, gaussian_radius + point_x1 - emulated_attention_map_size)]
            self.emulated_attention_map2 = torch.zeros((emulated_attention_map_size, emulated_attention_map_size))
            self.emulated_attention_map2[
            max(0, point_y2 - gaussian_radius): min(emulated_attention_map_size, point_y2 + gaussian_radius),
            max(0, point_x2 - gaussian_radius): min(emulated_attention_map_size,
                                                    point_x2 + gaussian_radius)] = gaussian_dist[max(0,
                                                                                                     gaussian_radius - point_y2):gaussian_radius * 2 - max(
                0, gaussian_radius + point_y2 - emulated_attention_map_size), max(0,
                                                                                  gaussian_radius - point_x2):gaussian_radius * 2 - max(
                0,
                gaussian_radius + point_x2 - emulated_attention_map_size)]

    def __getitem__(self, idx):
        y, x, h, w = transforms.RandomCrop.get_params(self.image, output_size=(self.crop_size, self.crop_size))
        image = transforms.functional.to_tensor(self.image)
        cropped_image = image[:, y:y + h, x:x + w]
        if self.train:
            if self.emulated_attention_map1 is not None and self.emulated_attention_map2 is not None:
                emulated_attention_map1 = \
                torch.nn.functional.interpolate(self.emulated_attention_map1[y:y + h, x:x + w][None, None, ...], 32,
                                                mode="bilinear")[0, 0]
                emulated_attention_map2 = \
                torch.nn.functional.interpolate(self.emulated_attention_map2[y:y + h, x:x + w][None, None, ...], 32,
                                                mode="bilinear")[0, 0]
                return cropped_image, emulated_attention_map1, emulated_attention_map2, y, x, image
            if self.segmentation is not None:
                segmentation = self.segmentation.crop((x, y, x+512, y+512))
                segmentation = transforms.functional.resize(segmentation, size=(256, 256),
                                                            interpolation=transforms.InterpolationMode.NEAREST)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                segmentation = cv2.erode(cv2.dilate(np.array(segmentation), kernel, iterations=1), kernel, iterations=1)
                # segmentation = cv2.GaussianBlur(segmentation, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
                # segmentation = (segmentation - segmentation.min()) / (segmentation.max() - segmentation.min())
                segmentation = transforms.functional.to_tensor(segmentation)[0].type(torch.float)
                return cropped_image, segmentation/10, segmentation/10, y, x, image
        else:
            return cropped_image, y, x, image

    def __len__(self):
        return self.num_cropped


class SampleDataModule(pl.LightningDataModule):
    def __init__(
            self,
            src_image_dir: str = "./data",
            target_image_dir: str = "./data",
            src_segmentation_dir: str = "./data",
            location1: Tuple = (5, 6),  # [y, x]
            location2: Tuple = (5, 6),  # [y, x]
            num_crops: int = 1,
            batch_size: int = 1,
            crop_ratio: float = 0.93,
    ):
        super().__init__()
        self.src_image_dir = src_image_dir
        self.target_image_dir = target_image_dir
        self.src_segmentation_dir = src_segmentation_dir
        self.location1 = location1
        self.location2 = location2
        self.num_crops = num_crops
        self.batch_size = batch_size
        self.crop_ratio = crop_ratio

    def setup(self, stage: str):
        self.train_dataset = SampleDataset(
            image_dir=self.src_image_dir,
            segmentation_dir=self.src_segmentation_dir,
            location1=self.location1,
            location2=self.location2,
            num_crops=self.num_crops,
            crop_ratio=self.crop_ratio,
            train=True,
        )
        self.test_dataset = SampleDataset(
            image_dir=self.target_image_dir,
            segmentation_dir=None,
            location1=None,
            location2=None,
            num_crops=self.num_crops,
            crop_ratio=self.crop_ratio,
            train=False,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)
