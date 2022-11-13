import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms


def get_eyes_paths(df):
    paths_dir = df.iloc[:, 0][0].split(",")
    if len(paths_dir) > 1:
        left_eye_dir = paths_dir[0]
        right_eye_dir = paths_dir[1]
        return left_eye_dir, right_eye_dir
    else:
        print("Folder: ", df.iloc[:, 3], "does not have both eyes image")


class MaratoCustomDataset(Dataset):

    def __init__(self, csv_path, transform=None, target_transform=None):
        df = pd.read_csv(csv_path, index_col=False, header=None, squeeze=True)
        self.left_eye_dir = df.iloc[:, 0]
        self.right_eye_dir = df.iloc[:, 1]
        self.img_labels = df.iloc[:, 2]
        self.folder = df.iloc[:, 3]
        self.transform = transforms.Compose([
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
        if self.target_transform:
            label = self.target_transform(label)
        return [left_image, right_image], label, folder
