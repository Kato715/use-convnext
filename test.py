#!/usr/bin/env python3
# train_randyで訓練したモデルをテストする．

import sys
import torch
import pandas as pd
import numpy as np
import albumentations as A
import lib.augmentations
import json

from pprint import pprint
from sklearn.metrics import classification_report
from os.path import join
from lib import engine, models
from lib.dataset import Dataset
from lib.utils import Dict, Config, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from tabulate import tabulate

def load_model(conf: Dict):
    model = models.select(conf.model.type, conf.model.n_classes)
    model_path = join(conf.model.dir,
                      f"{conf.model.id}_{conf.model.type}_{conf.model.size[0]}_{conf.model.size[1]}_{config.datasets.train.val_fold}.bin")

    model.load_state_dict(torch.load(
       model_path,
       map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else 'cpu'
       )
    )
    model.to(conf.model.device)

    return model

def crop_top_19_20(img, **kwargs):
    return img[: (19 * img.shape[0]) // 20, :, :]


def predict(conf, test_images, model):
    test_targets = np.zeros(len(test_images), dtype=np.int_)

    test_aug = lib.augmentations.compose(conf.datasets.test.aug)

    test_dataset = Dataset(
        image_paths=test_images,
        targets=test_targets,
        augmentations=test_aug,
        channel_first=True,
        torgb=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=conf.DataLoader.batch_size,
        shuffle=False,
        num_workers=conf.DataLoader.num_workers
    )
    

    predictions = engine.predict(model, test_loader, conf.model.device)
    preds = []
    for vp in predictions:
        preds.extend(vp)

    p = np.vstack((predictions))
    return p


def test(conf: Dict, datasets_csv: str):
    df = pd.read_csv(datasets_csv)
    test_images = df.path.values.tolist()

    model = load_model(conf)
    
    preds = predict(conf, test_images, model)
    final_preds = [np.argmax(p) for p in preds]
    
    with open("lib/class_labels.json", "r", encoding="utf-8") as f:
        name_dic = json.load(f)
    target_list = list(name_dic.keys())
    print(target_list)
    

    le = LabelEncoder()
    le.classes_ = np.load(conf.datasets.label_encoder)
    labels_num = list(range(conf.model.n_classes))
    labels = le.inverse_transform(labels_num)

    

    labels_tem = list(set(df.class_num.values) | set(final_preds))
    label_target = le.inverse_transform(labels_tem)

    labels_target = []
    for tar in label_target:
        if len(tar) > 4:
            tar = tar[:4]
        elif len(tar) < 4:
            tar = tar.ljust(4, "　")
        labels_target.append(tar)
    
    class_num_values_name = np.array([target_list[label] for label in df.class_num.values])
    final_preds_name = np.array([target_list[label] for label in final_preds])

    classification = classification_report(df.class_num.values, final_preds, target_names=labels_target, digits=4)
    classification_matrix = confusion_matrix(class_num_values_name, final_preds_name)
    print("classification_report")
    print(classification)
    print("Confusion Matrix (row/col = act/pred)")
    print(tabulate(classification_matrix, headers='keys', tablefmt='pretty'))

    classification = classification_report(df.class_num.values, final_preds, target_names=labels_target, digits=4,output_dict=True)
    classification_df = pd.DataFrame(classification).transpose()
    confusion_df = pd.DataFrame(classification_matrix)

    state = datasets_csv.split("/")[-1].split("_")[0]
    
    # Excelに保存
    with pd.ExcelWriter(f"result_data/{conf.model.id}_{state}_classification_confusion.xlsx") as writer:
        classification_df.to_excel(writer, sheet_name="Classification Report")
        confusion_df.to_excel(writer, sheet_name="Confusion Matrix")

    # 間違った画像のパスを出力
    mis_path = []
    for i, pre in enumerate(final_preds):
        if df.class_num.values[i] != pre:
            mis_path.append((df.path[i],labels[pre]))
    return mis_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Path[s] of configuration file[s] needed.")
        exit(1)

    for arg in sys.argv[1:]:
        print(f'Loading configuration "{arg}"')
        config = Config.load_json(arg)
        print("Configuration")
        pprint(config)

        # config.model.device = "mps"
        config.model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use 1st GPU

        print("Evaluate using testing dataset.")
        pprint(test(config, config.datasets.test.csv))
        print("Evaluate using real dataset.")
        pprint(test(config, config.datasets.real.csv))

