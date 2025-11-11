import albumentations as A
from albumentations.augmentations.blur.transforms import *
from albumentations.augmentations.geometric.rotate import *
from albumentations.augmentations.geometric.transforms import *
from albumentations.augmentations.transforms import *
from albumentations.core.transforms_interface import *


def crop_top(img, **kwargs):
    return img[: (19 * img.shape[0]) // 20, :, :]

def make_list(augs: list) -> list:
    avail_methods = {
        "SmallestMaxSize": A.SmallestMaxSize,
        "CenterCrop": A.CenterCrop,
        "Normalize": Normalize,
        "CLAHE": CLAHE,
        "GaussianBlur": GaussianBlur,
        "Sharpen": A.augmentations.transforms.Sharpen,
        "RandomBrightnessContrast": RandomBrightnessContrast,
        "Equalize": Equalize,
        "HorizontalFlip": HorizontalFlip,
        "VerticalFlip": VerticalFlip,
        "Flip": Flip,
        "NoOp": NoOp,
        "Transpose": Transpose,
        "Rotate": Rotate,
        "RandomRotate90": RandomRotate90,
        "Perspective": Perspective,
        "ElasticTransform": ElasticTransform,
        "GridDistortion": GridDistortion,
        "ShiftScaleRotate": ShiftScaleRotate,
        "Defocus": Defocus,
        "crop_top": "crop_top",
        "multiplicativenoise": A.MultiplicativeNoise
    }

    aug_list = []

    for aug in augs:
        name = aug["method"]
        params = aug.get("params", None)

        if name not in avail_methods:
            continue

        if avail_methods[name] == "crop_top":
            aug_list.insert(0, A.Lambda(image=crop_top, name="crop_top"))
        else:
            method = (
                avail_methods[name](**params)
                if params is not None
                else avail_methods[name]()
            )
            aug_list.append(method)

    print(f"augmentation list: {aug_list}")

    return aug_list

def compose(aug: list) -> A.Compose:
    return A.Compose(make_list(aug))
