import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from typing import Tuple


class SampleDataset(Dataset):
    def __init__(self, image_dir, location, num_crops=1, crop_ratio=0.93):
        self.image = transforms.functional.resize(Image.open(image_dir), size=(int(512/crop_ratio), int(512/crop_ratio)))

        point_y, point_x = location
        gaussian_radius = 28
        x, y = torch.meshgrid(torch.linspace(-1, 1, gaussian_radius*2), torch.linspace(-1, 1, gaussian_radius*2), indexing="xy")
        d = torch.sqrt(x * x + y * y)
        sigma, mu = 1, 0
        gaussian_dist = torch.exp(-(torch.pow(d - mu, 2) / (2.0 * sigma ** 2)))
        self.emulated_attention_map = torch.zeros((512, 512))
        self.emulated_attention_map[max(0, point_y-gaussian_radius): min(512, point_y+gaussian_radius), max(0, point_x-gaussian_radius): min(512, point_x+gaussian_radius)] = gaussian_dist[max(0, gaussian_radius-point_y):100-max(0, gaussian_radius+point_y-512), max(0, gaussian_radius-point_x):100-max(0, gaussian_radius+point_x-512)]
        self.emulated_attention_map = torch.nn.functional.interpolate(self.emulated_attention_map[None, None, ...], size=(int(512/crop_ratio), int(512/crop_ratio)), mode="bilinear")[0, 0]
        self.crop_size = 512
        self.num_cropped = num_crops

    def __getitem__(self, idx):
        y, x, h, w = transforms.RandomCrop.get_params(self.image, output_size=(self.crop_size, self.crop_size))

        image = transforms.functional.to_tensor(self.image)
        cropped_image = image[:, y:y+h, x:x+w]

        emulated_attention_map = self.emulated_attention_map[y:y+h, x:x+w]

        return cropped_image, emulated_attention_map, y, x, image

    def __len__(self):
        return self.num_cropped


class SampleDataModule(pl.LightningDataModule):
    def __init__(
            self,
            src_image_dir: str = "./data",
            target_image_dir: str = "./data",
            location: Tuple = (5, 6),  # [y, x]
            num_crops: int = 1,
            batch_size: int = 1,
            crop_ratio: float = 0.93,
    ):
        super().__init__()
        self.src_image_dir = src_image_dir
        self.target_image_dir = target_image_dir
        self.location = location
        self.num_crops = num_crops
        self.batch_size = batch_size
        self.crop_ratio = crop_ratio

    def setup(self, stage: str):
        self.train_dataset = SampleDataset(
            image_dir=self.src_image_dir,
            location=self.location,
            num_crops=self.num_crops,
            crop_ratio=self.crop_ratio,
        )
        self.test_dataset = SampleDataset(
            image_dir=self.target_image_dir,
            location=self.location,
            num_crops=self.num_crops,
            crop_ratio=self.crop_ratio,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)

