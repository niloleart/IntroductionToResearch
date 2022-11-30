import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms
import torch

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop([335, 506]),
        transforms.ToTensor(),
        transforms.Normalize([24.3918, 56.5434, 53.6119], [33.7875, 35.7010, 57.2551])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([24.3918, 56.5434, 53.6119], [33.7875, 35.7010, 57.2551])
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


class MaratoCustomDataset(Dataset):

    def __init__(self, csv_path, transform=None, target_transform=None):
        # df = pd.read_csv(csv_path, sep=";", index_col=False, header=None, squeeze=True)
        df = pd.read_csv(csv_path, sep=",", index_col=False, header=None, squeeze=True)

        # TODO: mirar què es pot fer en comptes de petar-nos tota la fila a lo loco
        df_updated = df.dropna(axis=0)

        self.left_eye_dir = df_updated.iloc[:, 0]
        # self.left_eye_dir = df_updated.iloc[:, 2]
        self.right_eye_dir = df_updated.iloc[:, 1]
        # self.right_eye_dir = df_updated.iloc[:, 3]
        self.img_labels_not_one_hot = df_updated.iloc[:, 4]
        self.img_labels = torch.nn.functional.one_hot((torch.tensor(self.img_labels_not_one_hot.values)).type(torch.int64))
        self.folder = df_updated.iloc[:, 5]

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
        # left_image = read_image(self.left_eye_dir.iloc[idx])
        left_image = cv2.imread(self.left_eye_dir.iloc[idx], 1)
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

        # right_image = read_image(self.right_eye_dir.iloc[idx])
        right_image = cv2.imread(self.right_eye_dir.iloc[idx], 1)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
        label = self.img_labels[idx]
        label_not_one_hot = self.img_labels_not_one_hot.iloc[idx]
        folder = self.folder.iloc[idx]

        if self.transform:
            left_image = self.transform(left_image)
            left_image = left_image.permute(1, 2, 0)

            right_image = self.transform(right_image)
            right_image = right_image.permute(1, 2, 0)

            composed_image = torch.cat([left_image, right_image], 1)

        if self.target_transform:
            label = self.target_transform(label)
            label_not_one_hot = self.target_transform(label_not_one_hot)
        return composed_image, label, folder
