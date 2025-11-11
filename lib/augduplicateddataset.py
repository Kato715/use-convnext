#データセットを拡張するための処理
from os.path import join

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class AugDuplicatedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dir,
        image_paths,
        targets,
        duplicate_aug: list = None,
        aug=None,
        channel_first=False,
        torgb=True,
    ):
        self.dir = dir
        self.image_paths = image_paths
        self.targets = targets
        self.duplicate_aug = duplicate_aug
        self.aug = aug
        self.channel_first = channel_first
        self.torgb = torgb

    def __len__(self):
        return len(self.image_paths) * (len(self.duplicate_aug) + 1)

    def __getitem__(self, idx):
        image_idx = idx % len(self.image_paths)
        target = self.targets[image_idx]
        image = Image.open(join(self.dir, self.image_paths[image_idx]))

        if self.torgb:
            image = image.convert("RGB")

        image = np.array(image)

        if self.duplicate_aug is not None and idx > (len(self.image_paths) - 1):
            aug_idx = (idx // len(self.image_paths)) - 1
            aug_method = self.duplicate_aug[aug_idx]
            augmented = aug_method(image=image)
            image = augmented["image"]

        if self.aug is not None:
            augmented = self.aug(image=image)
            image = augmented["image"]

        if self.channel_first:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.tensor(image),
            "targets": torch.tensor(target),
        }
