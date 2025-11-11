import torch
import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, targets, augmentations=None, channel_first=False, torgb=True):

        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.channel_first = channel_first
        self.torgb = torgb

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        targets = self.targets[idx]
        image = Image.open(self.image_paths[idx])

        if self.torgb:
            image = image.convert("RGB")

        image = np.array(image)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        if self.channel_first:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.tensor(image),
            "targets": torch.tensor(targets),
        }
