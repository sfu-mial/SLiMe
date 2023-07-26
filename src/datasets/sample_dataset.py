import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class SampleDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, num_crops=1, train=True, mask_size=256, final_min_crop_size=256):
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs
        self.num_crops = num_crops
        self.train = train
        self.mask_size = mask_size
        self.final_min_crop_size = final_min_crop_size
        self.train_transform = A.Compose([
            A.Resize(512, 512),
            # A.LongestMaxSize(512),
            # A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, value=0,
            #               mask_value=0),
            A.HorizontalFlip(),
            # A.RandomScale((0.5, 2), always_apply=True),
            A.GaussianBlur(blur_limit=(11, 31)),
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
        image = Image.open(self.image_dirs[idx])
        if self.train:
            mask = Image.open(self.mask_dirs[idx])
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
            if self.mask_dirs is not None:
                mask = Image.open(self.mask_dirs[idx])
                result = self.test_transform(image=np.array(image), mask=mask)
                mask = result['mask']
            else:
                result = self.test_transform(image=np.array(image))
                mask = 0
            image = result["image"]
            return image/255, mask
        # # w, h = image.size
        # # image = transforms.functional.center_crop(image, (min(w, h), min(w, h)))
        # image = transforms.functional.resize(image, size=(512, 512))
        # image = transforms.functional.to_tensor(image)
        # if self.train:
        #     mask = Image.open(self.mask_dirs[idx])
        #     # w, h = mask.size
        #     # mask = transforms.functional.center_crop(mask, (min(w, h), min(w, h)))
        #     mask = transforms.functional.resize(mask, size=(512, 512))
        #     mask = transforms.functional.to_tensor(mask).sum(dim=0)
        #     mask = torch.where(mask >= 0.5, 1., 0.)
        #     rotation_degree = transforms.RandomRotation.get_params([0, 45])
        #     image = transforms.functional.rotate(image, rotation_degree)
        #     mask = transforms.functional.rotate(mask[None, ...], rotation_degree)[0]
        #     coords = []
        #     if self.num_crops > 1:
        #         x_start, x_end, y_start, y_end, crop_size = get_square_cropping_coords(mask)
        #         # coords.append([y_start, y_end, x_start, x_end])
        #         for square_size in torch.linspace(crop_size, 512, self.num_crops + 1)[1:-1]:
        #             x_start, x_end, y_start, y_end, crop_size = get_square_cropping_coords(mask,
        #                                                                                    square_size=int(square_size))
        #             coords.append([y_start, y_end, x_start, x_end])
        #     coords.append([0, 512, 0, 512])
        #     random_idx = random.randint(0, self.num_crops-1)
        #     y_start, y_end, x_start, x_end = coords[random_idx]
        #     cropped_mask = mask[y_start:y_end, x_start:x_end]
        #     cropped_mask = torch.nn.functional.interpolate(cropped_mask[None, None, ...], size=512, mode="nearest")[0, 0]
        #     cropped_image = image[:, y_start:y_end, x_start:x_end]
        #     cropped_image = torch.nn.functional.interpolate(cropped_image[None, ...], size=512, mode="bilinear")[0]
        #     mask_in_crop = False
        #     while not mask_in_crop:
        #         crop_size = int((self.final_min_crop_size + torch.rand(1) * (512-self.final_min_crop_size)).item())
        #         y, x, h, w = transforms.RandomCrop.get_params(cropped_image, output_size=(crop_size, crop_size))
        #         final_cropped_mask = cropped_mask[y:y+h, x:x+w]
        #         if torch.any(final_cropped_mask == 1):
        #             mask_in_crop = True
        #     final_cropped_image = cropped_image[:, y:y+h, x:x+w]
        #     final_cropped_image = torch.nn.functional.interpolate(final_cropped_image[None, ...], size=512, mode="bilinear")[0]
        #     final_cropped_mask = torch.nn.functional.interpolate(final_cropped_mask[None, None, ...], size=self.mask_size, mode="nearest")[0, 0]

        #     return final_cropped_image, final_cropped_mask
        # else:
        #     return image, 0

    def __len__(self):
        return len(self.image_dirs)


class SampleDataModule(pl.LightningDataModule):
    def __init__(
            self,
            src_image_dirs: str = "./data",
            target_image_dir: str = "./data",
            src_mask_dirs: str = "./data",
            train_num_crops: int = 1,
            batch_size: int = 1,
            mask_size: int = 256
    ):
        super().__init__()
        self.src_image_dirs = src_image_dirs
        self.target_image_dir = target_image_dir
        self.src_mask_dirs = src_mask_dirs
        self.train_num_crops = train_num_crops
        self.batch_size = batch_size
        self.mask_size = mask_size

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = SampleDataset(
                image_dirs=self.src_image_dirs[0:1],
                mask_dirs=self.src_mask_dirs[0:1],
                train=True,
                mask_size=self.mask_size,
            )
            self.val_dataset = SampleDataset(
                image_dirs=self.src_image_dirs[1:2],
                mask_dirs=self.src_mask_dirs[1:2],
                train=False,
            )
        elif stage == 'test':
            self.test_dataset = SampleDataset(
                image_dirs=self.target_image_dir,
                mask_dirs=None,
                train=False,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)
