import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import toml
import torch


class Dict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config(object):
    @staticmethod
    def __load__(data):
        if type(data) is dict:
            return Config.load_dict(data)
        elif type(data) is list:
            return Config.load_list(data)
        else:
            return data

    @staticmethod
    def load_dict(data: dict):
        result = Dict()

        for key, value in data.items():
            result[key] = Config.__load__(value)

        return result

    @staticmethod
    def load_list(data: list):
        result = [Config.__load__(item) for item in data]

        return result

    @staticmethod
    def load_json(path: str):
        with open(path, "r") as f:
            result = Config.__load__(json.loads(f.read()))

        return result

    @staticmethod
    def load_toml(path: str):
        with open(path, "r") as f:
            result = Config.__load__(toml.load(f))

        return result


def confusion_matrix(y: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    y_labels = np.unique(y)
    y_pred_labels = np.unique(y_pred)

    matrix = pd.DataFrame(
        np.zeros((len(y_labels), len(y_pred_labels))),
        index=y_labels,
        columns=y_pred_labels,
        dtype=int,
    )

    for c, p in zip(y, y_pred):
        matrix.loc[c, p] += 1

    return matrix


def get_targets(loader: torch.utils.data.DataLoader, dtype=None):
    iterator = iter(loader)
    dtype = torch.uint8 if dtype is None else dtype
    targets = torch.empty(0, dtype=dtype)

    while (batch := next(iterator, None)) is not None:
        targets = torch.cat((targets, batch["targets"]))

    return targets


def get_model_filename(conf) -> str:
    # return f"{conf.model.id}-{conf.model.type}-{conf.model.size[0]}x{conf.model.size[1]}-f{conf.datasets.train.val_fold}.pth"
    return f"{conf.model.id}_{conf.model.type}_{conf.model.size[0]}_{conf.model.size[1]}_{conf.datasets.train.val_fold}.bin"


def count_files(path: Path, recursive=False):
    pattern = "**/*" if recursive else "*"

    return sum(1 for f in path.glob(pattern) if f.is_file())


def avoid_overwriting(path: Path):
    counter = 1
    orig_stem = path.stem

    while path.exists():
        new_name = f"{orig_stem}_{counter:02}{path.suffix}"
        path = path.parent / new_name
        counter += 1

    return path
