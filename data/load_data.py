import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch
import numpy as np


def get_data(csv_path):
    columns = ['left_macular', 'right_macular',
               'left_color', 'right_color',
               'left_octa_3x3_sup', 'right_octa_3x3_sup',
               'left_octa_3x3_deep', 'right_octa_3x3_deep',
               'left_octa_6x6_sup', 'right_octa_6x6_sup',
               'left_octa_6x6_deep', 'right_octa_6x6_deep',
               'folder', 'label_left', 'label_right']

    df = pd.read_csv(csv_path, sep=",", index_col=False).squeeze(1)

    # TODO: mirar què es pot fer en comptes de petar-nos tota la fila a lo loco
    df = df.dropna(axis=0)

    df = df.loc[:, columns]  # https://builtin.com/data-science/train-test-split
    df.head(len(df))
    features = ['left_macular', 'right_macular',
                'left_color', 'right_color',
                'left_octa_3x3_sup', 'right_octa_3x3_sup',
                'left_octa_3x3_deep', 'right_octa_3x3_deep',
                'left_octa_6x6_sup', 'right_octa_6x6_sup',
                'left_octa_6x6_deep', 'right_octa_6x6_deep',
                'folder']
    labels = ['label_left', 'label_right']

    X = df.loc[:, features]
    y = df.loc[:, labels]

    return X, y


def get_data_concat(X, y):
    X_macular = pd.concat((X.iloc[:, 0], X.iloc[:, 1]), axis=0)
    X_color = pd.concat((X.iloc[:, 2], X.iloc[:, 3]), axis=0)
    X_octa_3x3_sup = pd.concat((X.iloc[:, 4], X.iloc[:, 5]), axis=0)
    X_octa_3x3_deep = pd.concat((X.iloc[:, 6], X.iloc[:, 7]), axis=0)
    X_octa_6x6_sup = pd.concat((X.iloc[:, 8], X.iloc[:, 9]), axis=0)
    X_octa_6x6_deep = pd.concat((X.iloc[:, 10], X.iloc[:, 11]), axis=0)
    X_folders = pd.concat((X.iloc[:, 12], X.iloc[:, 12]), axis=0)
    y_train_concat = pd.concat((y.iloc[:, 0], y.iloc[:, 1]), axis=0)

    dict_X = {'macular': X_macular, 'color': X_color,
              'octa_3x3_sup': X_octa_3x3_sup, 'octa_3x3_deep': X_octa_3x3_deep,
              'octa_6x6_sup': X_octa_6x6_sup, 'octa_6x6_deep': X_octa_6x6_deep,
              'folder': X_folders}
    X_out = pd.DataFrame(dict_X)
    Y_out = y_train_concat
    return X_out, Y_out


# Applied to all data
def get_transforms(image_type, image_size, concatenate):
    if concatenate == 'depth' or concatenate == '2d':  # TODO
        if concatenate == '2d':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Resize([image_size, image_size])
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation([-90, 90]),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([image_size, image_size])
            ])
    else:
        if image_type == 'color':
            width = 644
            height = 484
            target_width = int(width * 0.75)

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([height, width]),
                transforms.CenterCrop([height, target_width]),
            ])
        elif image_type == 'octa_3x3_sup':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([image_size, image_size])
            ])
        elif image_type == 'octa_3x3_deep':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([image_size, image_size])
            ])
        elif image_type == 'octa_6x6_sup':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([image_size, image_size])
            ])
        elif image_type == 'octa_6x6_deep':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([image_size, image_size])
            ])
        elif image_type == 'macular':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                #TODO si s'ha de fer servir aquesta veure que passa amb inception i tal
                transforms.CenterCrop([330, 500])
            ])

        else:
            transform = transforms.Compose([
                # transforms.ToTensor(),
                # transforms.Resize([335, 506]),
                # transforms.ToPILImage(),
                # transforms.PILToTensor()
            ])
    return transform


class MaratoCustomDataset(Dataset):
    def __init__(self, X, y, target_transform=None, image_type='octa_6x6_deep', image_size=224, concatenate='2d', augment_individual_images = True):
        self.img_type = image_type
        self.img_concatenation = concatenate
        self.augment_individual_images = augment_individual_images
        self.img_eye = []
        if self.img_concatenation == 'depth' or self.img_concatenation == '2d':
            self.img_eye.append(X.iloc[:]['octa_6x6_deep'])
            self.img_eye.append(X.iloc[:]['octa_3x3_deep'])
            self.img_eye.append(X.iloc[:]['octa_6x6_sup'])
            self.img_eye.append(X.iloc[:]['octa_3x3_deep'])

        elif self.img_concatenation == 'none':
            print("Using " + image_type + "...")
        else:
            print("Choose correct concatenation image type, exiting...")
            exit()

        # self.img_eye = X.iloc[:][self.img_type]

        self.folder = X.iloc[:]['folder']
        self.img_labels_not_one_hot = y
        self.img_labels = torch.nn.functional.one_hot(
            torch.from_numpy(
                self.img_labels_not_one_hot.values
            ).to(torch.int64), num_classes=2
        )

        self.transform_individual = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize([image_size, image_size])
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation([-90, 90]),
        ])

        self.transform = transforms.Compose([
            # get_transforms(self.img_type, image_size, concatenate),
            target_transform
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if self.img_concatenation != 'none':
            #  Not necessary to normalize because toTensor normalizes values between [0, 1]
            img_6x6_deep = cv2.imread(self.img_eye[0].iloc[idx], cv2.IMREAD_GRAYSCALE)
            img_6x6_deep = cv2.resize(img_6x6_deep, (229, 229), interpolation=cv2.INTER_AREA)
            # img_6x6_deep = cv2.normalize(img_6x6_deep, None, -1, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            img_3x3_deep = cv2.imread(self.img_eye[1].iloc[idx], cv2.IMREAD_GRAYSCALE)
            img_3x3_deep = cv2.resize(img_3x3_deep, (229, 229), interpolation=cv2.INTER_AREA)
            # img_3x3_deep = cv2.normalize(img_3x3_deep, None, -1, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            img_6x6_sup = cv2.imread(self.img_eye[2].iloc[idx], cv2.IMREAD_GRAYSCALE)
            img_6x6_sup = cv2.resize(img_6x6_sup, (229, 229), interpolation=cv2.INTER_AREA)
            # img_6x6_sup = cv2.normalize(img_6x6_sup, None, -1, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            img_3x3_sup = cv2.imread(self.img_eye[3].iloc[idx], cv2.IMREAD_GRAYSCALE)
            img_3x3_sup = cv2.resize(img_3x3_sup, (229, 229), interpolation=cv2.INTER_AREA)
            # img_3x3_sup = cv2.normalize(img_3x3_sup, None, -1, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            if self.augment_individual_images:
                img_6x6_deep = np.asarray(self.transform_individual(img_6x6_deep))
                img_3x3_deep = np.asarray(self.transform_individual(img_3x3_deep))
                img_6x6_sup = np.asarray(self.transform_individual(img_6x6_sup))
                img_3x3_sup = np.asarray(self.transform_individual(img_3x3_sup))

            if self.img_concatenation == 'depth':
                img = np.dstack(img_6x6_deep, img_3x3_deep, img_6x6_sup, img_3x3_sup)

            elif self.img_concatenation == '2d':
                img1_hconcat = cv2.hconcat([img_6x6_deep, img_3x3_deep])
                img2_hconcat = cv2.hconcat([img_6x6_sup, img_3x3_sup])
                img = cv2.vconcat([img1_hconcat, img2_hconcat])


        else:
            if self.img_type == 'octa_3x3_sup' or self.img_type == 'octa_3x3_deep' \
                    or self.img_type == 'octa_6x6_sup' or self.img_type == 'octa_6x6_deep':
                img = cv2.imread(self.img_eye.iloc[idx], cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(self.img_eye.iloc[idx], 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.img_labels[idx]
        label_not_one_hot = self.img_labels_not_one_hot.iloc[idx]
        folder = self.folder.iloc[idx]

        if self.transform:
            img = self.transform(img)

        return img, label_not_one_hot, label, folder
