import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms
import torch


def get_eyes_paths(df):
    paths_dir = df.iloc[:, 0][0].split(",")
    if len(paths_dir) > 1:
        left_eye_dir = paths_dir[0]
        right_eye_dir = paths_dir[1]
        return left_eye_dir, right_eye_dir
    else:
        print("Folder: ", df.iloc[:, 3], "does not have both eyes image")


data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop([330, 506]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Resize([330, 506]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class MaratoCustomDataset(Dataset):

    def __init__(self, csv_path, transform=None, target_transform=None):
        df = pd.read_csv(csv_path, index_col=False, header=None, squeeze=True)

        # TODO: mirar què es pot fer en comptes de petarnos tota la fila a lo loco
        df_updated = df.dropna(axis=0)

        self.left_eye_dir = df_updated.iloc[:, 0]
        self.right_eye_dir = df_updated.iloc[:, 1]
        self.img_labels = df_updated.iloc[:, 4]
        self.folder = df_updated.iloc[:, 5]
        self.transform = transforms.Compose([
            transforms.Resize([335, 506]),
            transforms.ToPILImage(),
            transforms.PILToTensor()
        ])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        left_image = read_image(self.left_eye_dir.iloc[idx])
        right_image = read_image(self.right_eye_dir.iloc[idx])
        label = self.img_labels.iloc[idx]
        folder = self.folder.iloc[idx]

        if self.transform:
            left_image = self.transform(left_image)
            left_image = left_image.permute(1, 2, 0)

            right_image = self.transform(right_image)
            right_image = right_image.permute(1, 2, 0)

            composed_image = torch.cat([left_image, right_image], 1)

        if self.target_transform:
            label = self.target_transform(label)
        return composed_image, label, folder
