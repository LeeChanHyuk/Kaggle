import os
import json
import glob
import random
import collections
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

local_path = 'D:/Training_seg_png' # dataset path
train_df = pd.read_csv(local_path + '/train_labels_task1.csv')

# 결국에는, 환자 한 폴더당 3개의 종류에 따라 Normalized된 이미지를 불러서, 3개를 각각 채널로 삼아서 Network의 Input으로 설정하고
# Inference 또한 똑같은 방식으로 진행한다.
def load_image_normalization(path):
    data = cv2.imread(path, 0)
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data) # 여기서 왜 / (np.max(data) - np.min(data) 로 하지 않았지?
    data = (data * 255).astype(np.uint8)
    return data

submission = pd.read_csv(local_path + "/sample_submission.csv")
# submission.to_csv("submission.csv", index=False)

package_path = "./EfficientNet-PyTorch-master"
import sys
sys.path.append(package_path)

import time

import torch
from torch import nn
from torch.utils import data as torch_data
from sklearn import model_selection as sk_model_selection
from torch.nn import functional as torch_functional
import efficientnet_pytorch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from sklearn.model_selection import StratifiedKFold

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed(42)

df = pd.read_csv(local_path + "/train_labels_task1.csv")
df_train, df_valid = sk_model_selection.train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["MGMT_value"],
)
class DataRetriever(torch_data.Dataset): # 특정 Patient의 index의 폴더 내에서 9개의 영상을 추출하고 그 평균을 내어서 더해준다.
    # 결국의 반환값에는 3종류의 평균값 영상을 3채널로 삼고, target-value를 반환하게 된다.
    def __init__(self, paths, targets):
        self.paths = paths
        self.targets = targets
        self.start_value = 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        _id = self.paths[index]
        patient_path = local_path+f"/BraTS2021_{str(_id).zfill(5)}/"
        channels = []
        for t in ("FLAIR", "T1", "T1CE", "T2"): # "T2w"
            t_path = []
            for path in os.listdir(os.path.join(patient_path, t)):
                t_path.append(os.path.join(patient_path,t,path))
            pathss = t_path[1].split()[-1]
            channel = []
            # for i in range(start, end + 1):
            # classify the min pixel in the images
            min_index=154
            max_index=0
            for index, image_path in enumerate(t_path):
                img = cv2.imread(image_path, 0)
                max_val = np.max(img)
                if max_val>0:
                    if index<min_index:
                        min_index = index
                    if index>max_index:
                        max_index = index
            edge_num = []
            edge_num.append(min_index+((max_index-min_index)//3))
            edge_num.append(min_index+((max_index-min_index)//3 * 2))
            start_num = 0
            end_num = 0
            if self.start_value == 0:
                start_num = min_index
                end_num = edge_num[0]
            elif self.start_value == 1:
                start_num = edge_num[0]
                end_num = edge_num[1]
            else:
                start_num = edge_num[1]
                end_num = max_index
            for i in range(start_num, end_num):
                channel.append(cv2.resize(load_image_normalization(t_path[i]), (256, 256)) / 255)
            channel = np.mean(channel, axis=0)
            channels.append(channel)

        y = torch.tensor(self.targets[index], dtype=torch.float)

        return {"X": torch.tensor(channels).float(), "y": y}
    def start_value_up(self):
        self.start_value += 1

train_data_retriever = DataRetriever(
    df_train["BraTS21ID"].values,
    df_train["MGMT_value"].values,
)

valid_data_retriever = DataRetriever(
    df_valid["BraTS21ID"].values,
    df_valid["MGMT_value"].values,
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b7')
        n_features = self.net._fc.in_features
        self.conv  = torch.nn.Conv2d(4, 3, kernel_size=3, stride=1, bias=True, groups=1)
        self.batch_norm = nn.BatchNorm2d(3, momentum=0.01, eps=1e-3)
        self.relu = nn.ReLU()
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)

    def forward(self, x):
        x = F.pad(x, [1,1,1,1])
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        out = self.net(x)
        return out

class LossMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, val):
        self.n += 1
        # incremental update
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg


class AccMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().astype(int)
        y_pred = y_pred.cpu().numpy() >= 0
        last_n = self.n
        self.n += len(y_true)
        true_count = np.sum(y_true == y_pred)
        # incremental update
        self.avg = true_count / self.n + last_n / self.n * self.avg

class Trainer:
    def __init__(
        self,
        model,
        device,
        optimizer,
        criterion,
        loss_meter,
        score_meter
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_meter = loss_meter
        self.score_meter = score_meter

        self.best_valid_score = -np.inf
        self.n_patience = 0

        self.messages = {
            "epoch": "[Epoch {}: {}] loss: {:.5f}, score: {:.5f}, time: {} s",
            "checkpoint": "The score improved from {:.5f} to {:.5f}. Save model to '{}'",
            "patience": "\nValid score didn't improve last {} epochs."
        }

    def fit(self, epochs, train_loader, valid_loader, save_path, patience):
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_score, train_time = self.train_epoch(train_loader)
            valid_loss, valid_score, valid_time = self.valid_epoch(valid_loader)

            self.info_message(
                self.messages["epoch"], "Train", n_epoch, train_loss, train_score, train_time
            )

            self.info_message(
                self.messages["epoch"], "Valid", n_epoch, valid_loss, valid_score, valid_time
            )

            if True:
             if self.best_valid_score < valid_score:
                self.info_message(
                    self.messages["checkpoint"], self.best_valid_score, valid_score, save_path
                )
                self.best_valid_score = valid_score
                self.save_model(n_epoch, save_path)
                self.n_patience = 0
            else:
                self.n_patience += 1

            if self.n_patience >= patience:
                self.info_message(self.messages["patience"], patience)
                break

    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        train_loss = self.loss_meter()
        train_score = self.score_meter()

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)

            loss = self.criterion(outputs, targets)
            loss.backward()

            train_loss.update(loss.detach().item())
            train_score.update(targets, outputs.detach())

            self.optimizer.step()

            _loss, _score = train_loss.avg, train_score.avg
            message = 'Train Step {}/{}, train_loss: {:.5f}, train_score: {:.5f}'
            self.info_message(message, step, len(train_loader), _loss, _score, end="\r")

        return train_loss.avg, train_score.avg, int(time.time() - t)

    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        valid_loss = self.loss_meter()
        valid_score = self.score_meter()

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                valid_loss.update(loss.detach().item())
                valid_score.update(targets, outputs)

            _loss, _score = valid_loss.avg, valid_score.avg
            message = 'Valid Step {}/{}, valid_loss: {:.5f}, valid_score: {:.5f}'
            self.info_message(message, step, len(valid_loader), _loss, _score, end="\r")

        return valid_loss.avg, valid_score.avg, int(time.time() - t)

    def save_model(self, n_epoch, save_path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            save_path,
        )

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_retriever = DataRetriever(
    df_train["BraTS21ID"].values,
    df_train["MGMT_value"].values,
)

valid_data_retriever = DataRetriever(
    df_valid["BraTS21ID"].values,
    df_valid["MGMT_value"].values,
)



model = Model()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch_functional.binary_cross_entropy_with_logits

trainer = Trainer(
    model,
    device,
    optimizer,
    criterion,
    LossMeter,
    AccMeter
)

for i in range(3):
    train_loader = torch_data.DataLoader(
        train_data_retriever,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = torch_data.DataLoader(
        valid_data_retriever,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    history = trainer.fit(
        10,
        train_loader,
        valid_loader,
        f"best-model-"+str(i)+".pth",
        100,
    )
    train_data_retriever.start_value_up()
    valid_data_retriever.start_value_up()


models = []
for i in range(5):
    model = Model()
    model.to(device)

    checkpoint = torch.load(f"best-model-{i}.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    models.append(model)

class DataRetriever(torch_data.Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.start_value = 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        _id = self.paths[index]
        patient_path = local_path+f"/BraTS2021_{str(_id).zfill(5)}/"
        channels = []
        for t in ("FLAIR", "T1", "T1CE", "T2"): # "T2w"
            t_path = []
            for path in os.listdir(os.path.join(patient_path, t)):
                t_path.append(os.path.join(patient_path,t,path))
            channel = []
            # for i in range(start, end + 1):
            # classify the min pixel in the images
            min_index=154
            max_index=0
            for index, image_path in enumerate(t_path):
                img = cv2.imread(image_path, 0)
                min_val = np.min(img)
                if min_val>0:
                    if index<min_index:
                        min_index = index
                    if index>max_index:
                        max_index = index
            edge_num = []
            edge_num.append(min_index+((max_index-min_index)//3))
            edge_num.append(min_index+((max_index-min_index)//3 * 2))
            start_num = 0
            end_num = 0
            if self.start_value == 0:
                start_num = min_index
                end_num = edge_num[0]
            elif self.start_value == 1:
                start_num = edge_num[0]
                end_num = edge_num[1]
            else:
                start_num = edge_num[1]
                end_num = max_index
            for i in range(start_num, end_num):
                channel.append(cv2.resize(load_image_normalization(t_path[i]), (256, 256)) / 255)
            channel = np.mean(channel, axis=0)
            channels.append(channel)

        return {"X": torch.tensor(channels).float(), "id": _id}
    def start_value_up(self):
        self.start_value += 1

submission = pd.read_csv(local_path + "/sample_submission.csv")

test_data_retriever = DataRetriever(
    submission["BraTS21ID"].values,
)

test_loader = torch_data.DataLoader(
    test_data_retriever,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)
from sklearn.ensemble import VotingClassifier
y_pred = []
ids = []

def voting(classifier_num, batch):
    for i in range(classifier_num):
        with torch.no_grad():
            tmp_pred = np.zeros((batch["X"].shape[0],))
            for model in models:
                tmp_res = torch.sigmoid(model(batch["X"].to(device))).cpu().numpy().squeeze()
                tmp_pred += tmp_res
            y_pred.extend(tmp_pred / float(classifier_num))
            ids.extend(batch["id"].numpy().tolist())


for e, batch in enumerate(test_loader):
    print(f"{e}/{len(test_loader)}", end="\r")
    voting(3, batch=batch)
submission = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": y_pred})
submission.to_csv("submission.csv", index=False)

plt.figure(figsize=(5, 5))
plt.hist(submission["MGMT_value"]);


