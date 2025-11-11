# /nas.dbms/kato/workspace/kato-polycam/src/lib_randy/augmentations.py
# albumetationsのバージョンが違っても大丈夫．


import warnings
import albumentations as A


def _warn(msg: str):
    warnings.warn(msg)


def crop_top(img, **kwargs):
    return img[: (19 * img.shape[0]) // 20, :, :]


def _get_transform(cls_name: str):
    """
    Albumentations v1/v2 両対応: 可能な限りトップレベル A.<Class> から取得。
    一部クラスが見つからない場合は互換候補にフォールバック。
    """
    t = getattr(A, cls_name, None)
    if t is not None:
        return t

    if cls_name == "Sharpen":
        alt = getattr(A, "UnsharpMask", None)
        if alt is not None:
            _warn("[augmentations] 'Sharpen' が見つからないため 'UnsharpMask' に切替えます。")
            return alt

    if cls_name == "NoOp":
        def _noop_factory(**kwargs):
            return A.Lambda(image=lambda x, **k: x, name="NoOp")
        _warn("[augmentations] 'NoOp' が見つからないため Lambda 恒等変換に切替えます。")
        return _noop_factory

    if cls_name == "Defocus":
        _warn("[augmentations] 'Defocus' が見つからないため、この変換はスキップします。")
        return None

    _warn(f"[augmentations] '{cls_name}' が見つからないため、この変換はスキップします。")
    return None


def make_list(augs: list) -> list:
    avail_methods = {
        "SmallestMaxSize": "SmallestMaxSize",
        "CenterCrop": "CenterCrop",
        "Normalize": "Normalize",
        "CLAHE": "CLAHE",
        "GaussianBlur": "GaussianBlur",
        "Sharpen": "Sharpen",
        "RandomBrightnessContrast": "RandomBrightnessContrast",
        "Equalize": "Equalize",
        "HorizontalFlip": "HorizontalFlip",
        "VerticalFlip": "VerticalFlip",
        "Flip": "Flip",
        "NoOp": "NoOp",
        "Transpose": "Transpose",
        "Rotate": "Rotate",
        "RandomRotate90": "RandomRotate90",
        "Perspective": "Perspective",
        "ElasticTransform": "ElasticTransform",
        "GridDistortion": "GridDistortion",
        "ShiftScaleRotate": "ShiftScaleRotate",
        "Defocus": "Defocus",
        "crop_top": "crop_top",
        "multiplicativenoise": "MultiplicativeNoise",
        "Affine" : "Affine"
    }

    aug_list = []

    for aug in augs:
        name = aug["method"]
        params = aug.get("params", None)

        if name not in avail_methods:
            _warn(f"[augmentations] 未対応の method: {name} をスキップします。")
            continue

        if avail_methods[name] == "crop_top":
            aug_list.insert(0, A.Lambda(image=crop_top, name="crop_top"))
            continue

        cls_name = avail_methods[name]
        transform_cls = _get_transform(cls_name)
        if transform_cls is None:
            continue

        try:
            method = transform_cls(**params) if params is not None else transform_cls()
        except TypeError:
            method = transform_cls(**(params or {}))

        aug_list.append(method)

    print(f"augmentation list: {aug_list}")
    return aug_list


def compose(aug: list) -> A.Compose:
    return A.Compose(make_list(aug))
