from pycocotools.coco import COCO

import random
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from glob import glob
import re
from tqdm import tqdm
from src.utils import get_square_cropping_coords
from typing import List


class COCODataset(Dataset):
    def __init__(self, ann_file_path, fold=0):
        coco = COCO(ann_file_path)
        cats = coco.loadCats(coco.getCatIds())
        fold_categories = cats[fold::4]
        self.data = []
        self.masks = []
        for category in fold_categories:
            catId = coco.getCatIds(catNms=category['name'])
            imgIds = coco.getImgIds(catIds=catId)
            imgs = coco.loadImgs(imgIds)
            for img in imgs:
                self.data.append([catId, img['file_name']])
                annId = coco.getAnnIds(imgIds=img['id'], catIds=catId, iscrowd=None)
                coco.loadAnns(annId)