import random
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import get_square_cropping_coords


class SampleDataset(Dataset):
    def __init__(self, image_dirs, segmentation_dirs, num_crops=1, train=True, mask_size=256):
        self.image_dirs = image_dirs
        self.segmentation_dirs = segmentation_dirs
        self.num_cropped = num_crops
        self.train = train
        self.mask_size = mask_size

    def __getitem__(self, idx):
        image = Image.open(self.image_dirs[idx % len(self.image_dirs)])
        w, h = image.size
        image = transforms.functional.center_crop(image, (min(w, h), min(w, h)))
        image = transforms.functional.resize(image, size=512)
        image = transforms.functional.to_tensor(image)
        if self.train:
            segmentation = Image.open(self.segmentation_dirs[idx % len(self.image_dirs)])
            w, h = segmentation.size
            segmentation = transforms.functional.center_crop(segmentation, (min(w, h), min(w, h)))
            segmentation = transforms.functional.resize(segmentation, size=512)
            segmentation = transforms.functional.to_tensor(segmentation)[0].type(torch.float)
            segmentation = torch.where(segmentation >= 0.5, 1., 0.)
            coords = []
            if self.num_cropped > 1:
                x_start, x_end, y_start, y_end, crop_size = get_square_cropping_coords(segmentation)
                # coords.append([y_start, y_end, x_start, x_end])
                for square_size in torch.linspace(crop_size, 512, self.num_cropped + 1)[1:-1]:
                    x_start, x_end, y_start, y_end, crop_size = get_square_cropping_coords(segmentation,
                                                                                           square_size=square_size)
                    coords.append([y_start, y_end, x_start, x_end])
            coords.append([0, 512, 0, 512])
            random_idx = random.randint(0, self.num_cropped-1)
            y_start, y_end, x_start, x_end = coords[random_idx]
            cropped_segmentation = segmentation[y_start:y_end, x_start:x_end]
            cropped_segmentation = torch.nn.functional.interpolate(cropped_segmentation[None, None, ...], size=self.mask_size, mode="nearest")[0, 0]
            cropped_image = image[:, y_start:y_end, x_start:x_end]
            cropped_image = torch.nn.functional.interpolate(cropped_image[None, ...], size=512, mode="bilinear")[0]
            return cropped_image, cropped_segmentation, cropped_segmentation / 10
        else:
            return image, 0

    def __len__(self):
        if self.train:
            return self.num_cropped * len(self.image_dirs)
        return len(self.image_dirs)


class SampleDataModule(pl.LightningDataModule):
    def __init__(
            self,
            src_image_dirs: str = "./data",
            target_image_dir: str = "./data",
            src_segmentation_dirs: str = "./data",
            train_num_crops: int = 1,
            test_num_crops: int = 1,
            batch_size: int = 1,
            mask_size: int = 256
    ):
        super().__init__()
        self.src_image_dirs = src_image_dirs
        self.target_image_dir = target_image_dir
        self.src_segmentation_dirs = src_segmentation_dirs
        self.train_num_crops = train_num_crops
        self.test_num_crops = test_num_crops
        self.batch_size = batch_size
        self.mask_size = mask_size

    def setup(self, stage: str):
        self.train_dataset = SampleDataset(
            image_dirs=self.src_image_dirs,
            segmentation_dirs=self.src_segmentation_dirs,
            num_crops=self.train_num_crops,
            train=True,
            mask_size=self.mask_size,
        )
        self.test_dataset = SampleDataset(
            image_dirs=self.target_image_dir,
            segmentation_dirs=None,
            num_crops=self.test_num_crops,
            train=False,
            mask_size=self.mask_size
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)
