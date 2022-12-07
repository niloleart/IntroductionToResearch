import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms
import torch
import numpy as np

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop([335, 506]),
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.ToTensor()
        # transforms.Normalize([24.3918, 56.5434, 53.6119], [33.7875, 35.7010, 57.2551])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize([24.3918, 56.5434, 53.6119], [33.7875, 35.7010, 57.2551])
    ]),
}


def compute_data_metrics(dataloader):
    dataloader = dataloader['train']
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _, _ in dataloader:
        b, h, w, c = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 1, 2])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 1, 2])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt((snd_moment - fst_moment) ** 2)
    print(mean)
    print(std)
    return mean, std


def get_data(csv_path):
    columns = ['left_macular', 'right_macular', 'left_color', 'right_color', 'folder', 'label_left', 'label_right']

    df = pd.read_csv(csv_path, sep=",", index_col=False, squeeze=True)

    # TODO: mirar què es pot fer en comptes de petar-nos tota la fila a lo loco
    df = df.dropna(axis=0)

    df = df.loc[:, columns]  # https://builtin.com/data-science/train-test-split
    df.head(len(df))
    features = ['left_macular', 'right_macular', 'left_color', 'right_color', 'folder']
    labels = ['label_left', 'label_right']

    X = df.loc[:, features]
    y = df.loc[:, labels]

    return X, y


def get_data_formatted(X_train, X_test, y_train, y_test):
    X_train_macular = pd.concat((X_train.iloc[:, 0], X_train.iloc[:, 1]), axis=0)
    X_train_color = pd.concat((X_train.iloc[:, 2], X_train.iloc[:, 3]), axis=0)
    X_train_folders = pd.concat((X_train.iloc[:, 4], X_train.iloc[:, 4]), axis=0)
    y_train_concat = pd.concat((y_train.iloc[:, 0], y_train.iloc[:, 1]), axis=0)

    dict_X_train = {'macular': X_train_macular, 'color': X_train_color, 'folder': X_train_folders}
    X_train_out = pd.DataFrame(dict_X_train)

    y_train_dict = {'labels': y_train_concat}
    # Y_train_out = pd.DataFrame(y_train_dict)
    Y_train_out = y_train_concat

    X_test_macular = pd.concat((X_test.iloc[:, 0], X_test.iloc[:, 1]), axis=0)
    X_test_color = pd.concat((X_test.iloc[:, 2], X_test.iloc[:, 3]), axis=0)
    X_test_folders = pd.concat((X_test.iloc[:, 4], X_test.iloc[:, 4]), axis=0)
    y_test_concat = pd.concat((y_test.iloc[:, 0], y_test.iloc[:, 1]), axis=0)

    dict_X_test = {'macular': X_test_macular, 'color': X_test_color, 'folder': X_test_folders}
    X_test_out = pd.DataFrame(dict_X_test)

    y_test_dict = {'labels': y_test_concat}
    # Y_test_out = pd.DataFrame(y_test_dict)
    Y_test_out = y_test_concat

    return X_train_out, X_test_out, Y_train_out, Y_test_out


class MaratoCustomDataset(Dataset):

    def __init__(self, X, y, transform=None, target_transform=None):

        # self.img_eye = X.iloc[:]['macular']
        self.img_eye = X.iloc[:]['color']

        self.folder = X.iloc[:]['folder']
        self.img_labels_not_one_hot = y
        self.img_labels = torch.nn.functional.one_hot(
            torch.from_numpy(
                self.img_labels_not_one_hot.values
            ).to(torch.int64), num_classes=2
        )

        # TODO: s'ha de normalitzar?
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([335, 506]),
            transforms.ToPILImage(),
            transforms.PILToTensor()
        ])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_eye.iloc[idx], 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        label = self.img_labels[idx]
        label_not_one_hot = self.img_labels_not_one_hot.iloc[idx]
        folder = self.folder.iloc[idx]

        if self.transform:
            img = self.transform(img)
            img = img.permute(1, 2, 0)

        if self.target_transform:
            label = self.target_transform(label)
            label_not_one_hot = self.target_transform(label_not_one_hot)
        return img, label_not_one_hot, label, folder
