from torchvision.transforms import ToTensor

from data.create_csv import CreateCSV
from data.load_data import MaratoCustomDataset
from data.plot_data import plot_sample_data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# per activar entorn
# source .virtualenvs/pythonProject/bin/activate

# nvidia-smi
# cuda 11.2

# Instalar en remot
# .virtualenvs/pythonProject/bin/pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113


# Instalar en local
# 1) Activar entorn "source environments/pytorch/bin/activate
# 2) Instalar paquets-> pip install XXXXXXXXX

# Perque segueixi executant si es tanca l'ordinador -> tmux ubuntu. Mirar tutorial


CSV_PATH = '/Users/niloleart/PycharmProjects/test.csv'

hparams = {
    'seed': 123,
    'batch_size': 64,
    'num_epochs': 10,
    'test_batch_size': 64,
    'num_classes': 2,
    'learning_rate': 1e-4,
    'validation_split': .2,
    'shuffle_dataset': True
}

rparams = {
    'create_csv': False,
    'plot_data_sample': False,
    'local_mode': True
}


def set_seed(seed):
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    _ = torch.cuda.manual_seed(seed)


def get_dataloaders(dataset):
    # Create data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(hparams['validation_split'] * dataset_size))
    if hparams['shuffle_dataset']:
        np.random.seed(hparams['seed'])
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Create PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=hparams['batch_size'], sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=hparams['batch_size'], sampler=valid_sampler)

    return train_loader, validation_loader


def get_csv_path():
    if rparams['local_mode']:
        return CSV_PATH
    else:
        return ''


def main():
    # Creates a CSV file with "Row1=full img path", "Row2 = label" (0 = healthy, 1 = ill)
    if rparams['create_csv']:
        c = CreateCSV()
        annotations_path = c.create_CSV()

    if 'annotations_path' in locals():
        dataset = MaratoCustomDataset(csv_path=annotations_path, transform=ToTensor())
    else:
        dataset = MaratoCustomDataset(csv_path=get_csv_path(), transform=ToTensor())

    if rparams['plot_data_sample']:
        plot_sample_data(dataset)

    set_seed(hparams['seed'])

    train_loader, val_loader = get_dataloaders(dataset)

    train_images, train_labels, train_folders = next(iter(train_loader))


if __name__ == '__main__':
    main()

