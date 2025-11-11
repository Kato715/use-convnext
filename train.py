#!/usr/bin/env python3
# randyさんのコードを元にしてる．前処理で色んなの使える

import os
import sys
import torch
import numpy as np
import pandas as pd
import albumentations as A
import lib.augmentations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pprint import pprint
from os.path import join
from sklearn import metrics

from lib import engine, models
from lib.augduplicateddataset import AugDuplicatedDataset
from lib.es import EarlyStopping
# from lib_randy.telegram import Telegram
from lib.utils import (
    Dict,
    avoid_overwriting,
    count_files,
    get_model_filename,
    get_targets,
    Config,
)
from torchmetrics import Accuracy, F1Score

# import timm


def create_model(conf: Dict) -> tuple:
    model = models.select(conf.model.type, conf.model.n_classes)
    model = model.to(device=conf.model.device)

    model_path = join(conf.model.dir,
                      f"{conf.model.id}_{conf.model.type}_{conf.model.size[0]}_{conf.model.size[1]}_{config.datasets.train.val_fold}.bin")

    return model, model_path

def train(conf: Dict):
    csv_path = conf.datasets.train.csv
    df = pd.read_csv(csv_path, engine="pyarrow") # csvファイル読み込み
    df_train = df[df.kfold != conf.datasets.train.val_fold].reset_index(drop=True) # trainデータセットのパス
    df_val = df[df.kfold == conf.datasets.train.val_fold].reset_index(drop=True) # 検証データセットのパス
    duplicate_aug = lib.augmentations.make_list(conf.datasets.train.duplicate_aug) # データ拡張
    aug = lib.augmentations.compose(conf.datasets.train.aug) # 前処理

    print("Probability augmentation:", aug)
    print("Duplicating augmentation:", duplicate_aug)

    # 画像パス・ラベル・拡張処理をまとめたDatasetオブジェクトを作成
    train_dataset = AugDuplicatedDataset(
        "",
        df_train.path.values,
        df_train.class_num.values,
        duplicate_aug=duplicate_aug,
        aug=aug,
        channel_first=True,
    )

    # バッチ単位で取り出せるようにするdata loaderの設定
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=conf.DataLoader.batch_size,
        num_workers=conf.DataLoader.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    # 学習中のモデルを評価する
    val_dataset = AugDuplicatedDataset(
        "",
        df_val.path.values,
        df_val.class_num.values,
        duplicate_aug=duplicate_aug,
        aug=aug,
        channel_first=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=conf.DataLoader.batch_size,
        num_workers=conf.DataLoader.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    print("Original train size:", len(df_train))
    print("Augmented train size:", len(train_dataset))

    model, model_path = create_model(conf) # modelとmodelのパスを用意


    # ハイパーパラメータの設定
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=conf.optimizer.learning_rate, # パラメータをどの程度更新するかを決める係数
        betas=conf.optimizer.betas, # 勾配の係数
        eps=conf.optimizer.epsilon, # 勾配の分母が0に近づくのを防ぐ．パラメータの更新量が大きくなりすぎるのを防ぐ
        weight_decay=conf.optimizer.weight_decay, # 重みが大きくなりすぎないようにするペナルティの項
    )

    # 学習率を調整するschedulerの作成
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=conf.scheduler.patience, mode=conf.scheduler.mode
    )

    es = EarlyStopping(patience=conf.scheduler.patience, mode=conf.scheduler.mode) # あるエポック数更新されなかったら学習を終了する
    accuracy_metric = Accuracy(task="multiclass", num_classes=conf.model.n_classes)
    if conf.scheduler.mode == "max": # accuracyで更新するか決める時はmax，lossで決めるときはmin
        es_flag = True
    else:
        es_flag = False

    train_losses, val_losses, accs, lrs = [], [], [], [] # 記録ようにリストを作成

    for epoch in range(conf.training.epoch):
        training_loss = engine.train_fn(
            model, train_loader, optimizer, conf.model.device
        ) # 訓練
        preds, val_loss = engine.evaluate(model, val_loader, conf.model.device) # 検証

        print("Tra. loss:", training_loss)
        print("Val. loss:", val_loss)

        # 推移を表にするために記録
        train_losses.append(float(training_loss))
        val_losses.append(float(val_loss))
        lrs.append(float(optimizer.param_groups[0]["lr"]))

        pred_list = []

        # 全バッチの出力を一つのリストにする
        for vp in preds:
            pred_list.extend(vp)

        preds = [torch.argmax(p) for p in pred_list] # 各サンプルの予測クラスIDを取得
        preds = np.vstack(preds).ravel() # すべてのサンプルの予測クラスIDを1次元配列で得る

        accuracy = float(
            accuracy_metric(
                torch.tensor(preds), # 予測ラベル
                get_targets(val_loader), # 正解ラベルを取得
            ) # 予測と正解を比較して精度を計算
        ) # 出力をfloat型に変換
        accs.append(accuracy)

        # es_flagでaccuracyを使うかlossを使うか決める
        if es_flag:
            scheduler.step(accuracy)
            es(accuracy, model, model_path)
        else:
            scheduler.step(val_loss)
            es(val_loss, model, model_path)

        if es.early_stop:
            print("Early stop")
            break

        print(f"Model: {conf.model.type}, epoch: {epoch}, acc: {accuracy}")

    # 記録したlossやaccuracyをcsv形式で保存する
    fold = conf.datasets.train.val_fold
    base = f"{conf.model.id}_{conf.model.type}_fold{fold}"
    folder = f"result_data/history"
    os.makedirs(folder, exist_ok=True)
    hist_csv = join(folder, f"{base}_history.csv")
    loss_png = join(folder, f"{base}_loss.png")
    acc_png  = join(folder, f"{base}_acc.png")

    epochs = list(range(1, len(train_losses) + 1))
    pd.DataFrame({
        "epoch": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "accuracy": accs[:len(epochs)],
        "lr": lrs[:len(epochs)],
    }).to_csv(hist_csv, index=False)

    # Loss曲線
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title(base)
    plt.tight_layout()
    plt.savefig(loss_png, dpi=200)
    plt.close()

    if len(accs) >= 1:
        plt.figure()
        plt.plot(epochs, accs[:len(epochs)], label="Accuracy")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title(base)
        plt.tight_layout()
        plt.savefig(acc_png, dpi=200)
        plt.close()


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Path[s] of configuration file[s] needed.")
        exit(1)

    for arg in sys.argv[1:]:
        print(f'Loading configuration "{arg}"')
        config = Config.load_json(arg)
        print("Configuration")
        pprint(config)
        
        config.model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use 1st GPU
        train(config)
