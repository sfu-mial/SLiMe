from typing import Tuple
import random
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


class SampleDataset(Dataset):
    def __init__(self, image_dirs, segmentation_dirs, location1, location2, num_crops=1, crop_ratio=0.93, train=True, mask_size=256):
        self.segmentations = None
        self.emulated_attention_map1, self.emulated_attention_map2 = None, None
        self.crop_size = 512
        self.num_cropped = num_crops
        self.crop_ratio = crop_ratio
        self.train = train
        self.mask_size = mask_size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

        images = []
        for image_dir in image_dirs:
            image = Image.open(image_dir)
            w, h = image.size
            cropped_image_size = min(w, h)
            if w != h:
                image = transforms.functional.center_crop(image, (cropped_image_size, cropped_image_size))
                if train:
                    if w < h:
                        location1[0] = location1[0] - (h - w) // 2
                        location2[0] = location2[0] - (h - w) // 2
                    else:
                        location1[1] = location1[1] - (w - h) // 2
                        location2[1] = location2[1] - (w - h) // 2
                    assert 0 <= location1[0] < cropped_image_size and 0 <= location1[1] < cropped_image_size and 0 <= \
                           location2[
                               0] < cropped_image_size and 0 <= location2[
                               1] < cropped_image_size, "points are out of image after center cropping"
            images.append(transforms.functional.resize(image, size=(int(512 / crop_ratio), int(512 / crop_ratio))))
        self.images = images

        if segmentation_dirs is not None:
            segmentations = []
            for segmentation_dir in segmentation_dirs:
                segmentation = Image.open(segmentation_dir)
                # kernel_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
                # segmentation = cv2.erode(cv2.dilate(np.array(segmentation), kernel_), kernel_, iterations=1)  # fill holes

                # segmentation = cv2.GaussianBlur(cv2.erode(segmentation, kernel, iterations=1), (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
                # segmentation = cv2.erode(segmentation, kernel, iterations=1)

                # segmentation = Image.fromarray(segmentation)
                segmentations.append(transforms.functional.resize(segmentation,
                                                                 size=(int(512 / crop_ratio), int(512 / crop_ratio)),
                                                                 interpolation=transforms.InterpolationMode.NEAREST))
            self.segmentations = segmentations
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
        image = self.images[idx % len(self.images)]
        if self.train:
            crop_size = random.randint(int(self.crop_size*0.9), self.crop_size)
        else:
            crop_size = 512
        y, x, h, w = transforms.RandomCrop.get_params(image, output_size=(crop_size, crop_size))
        cropped_image = image.crop((x, y, x+w, y+h))
        cropped_image = transforms.functional.resize(cropped_image, size=512)
        cropped_image = transforms.functional.to_tensor(cropped_image)
        image = transforms.functional.to_tensor(image)
        # cropped_image = image[:, y:y + h, x:x + w]
        if self.train:
            if self.emulated_attention_map1 is not None and self.emulated_attention_map2 is not None:
                emulated_attention_map1 = \
                    torch.nn.functional.interpolate(self.emulated_attention_map1[y:y + h, x:x + w][None, None, ...], 32,
                                                    mode="bilinear")[0, 0]
                emulated_attention_map2 = \
                    torch.nn.functional.interpolate(self.emulated_attention_map2[y:y + h, x:x + w][None, None, ...], 32,
                                                    mode="bilinear")[0, 0]
                return cropped_image, emulated_attention_map1, emulated_attention_map2, y, x, image
            if self.segmentations is not None:
                segmentation = self.segmentations[idx % len(self.images)]
                cropped_segmentation = segmentation.crop((x, y, x + w, y + h))
                cropped_segmentation = transforms.functional.resize(cropped_segmentation, size=(self.mask_size, self.mask_size),
                                                            interpolation=transforms.InterpolationMode.NEAREST)
                cropped_segmentation = cv2.erode(cv2.dilate(np.array(cropped_segmentation), self.kernel, iterations=1), self.kernel, iterations=1)
                # cropped_segmentation = cv2.GaussianBlur(cropped_segmentation, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
                cropped_segmentation = transforms.functional.to_tensor(cropped_segmentation)[0].type(torch.float)
                return cropped_image, cropped_segmentation / 10, cropped_segmentation
        else:
            return cropped_image, y, x, 0, image

    def __len__(self):
        return self.num_cropped * len(self.images)


class SampleDataModule(pl.LightningDataModule):
    def __init__(
            self,
            src_image_dirs: str = "./data",
            target_image_dir: str = "./data",
            src_segmentation_dirs: str = "./data",
            location1: Tuple = (5, 6),  # [y, x]
            location2: Tuple = (5, 6),  # [y, x]
            train_num_crops: int = 1,
            test_num_crops: int = 1,
            batch_size: int = 1,
            train_crop_ratio: float = 0.93,
            test_crop_ratio: float = 0.93,
            mask_size: int = 256
    ):
        super().__init__()
        self.src_image_dirs = src_image_dirs
        self.target_image_dir = target_image_dir
        self.src_segmentation_dirs = src_segmentation_dirs
        self.location1 = location1
        self.location2 = location2
        self.train_num_crops = train_num_crops
        self.test_num_crops = test_num_crops
        self.batch_size = batch_size
        self.train_crop_ratio = train_crop_ratio
        self.test_crop_ratio = test_crop_ratio
        self.mask_size = mask_size

    def setup(self, stage: str):
        self.train_dataset = SampleDataset(
            image_dirs=self.src_image_dirs,
            segmentation_dirs=self.src_segmentation_dirs,
            location1=self.location1,
            location2=self.location2,
            num_crops=self.train_num_crops,
            crop_ratio=self.train_crop_ratio,
            train=True,
            mask_size=self.mask_size,
        )
        self.test_dataset = SampleDataset(
            image_dirs=self.target_image_dir,
            segmentation_dirs=None,
            location1=None,
            location2=None,
            num_crops=self.test_num_crops,
            crop_ratio=self.test_crop_ratio,
            train=False,
            mask_size=self.mask_size
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)
