from torch import optim, nn
from torch.optim import lr_scheduler
from torchvision.transforms import ToTensor

from data.create_csv import CreateCSV
from data.load_data import MaratoCustomDataset
from data.plot_data import plot_sample_data, plot_losses
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import torchvision.models as models

from train.train import train_model

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


CSV_LOCAL_PATH = '/Users/niloleart/PycharmProjects/test.csv'
CSV_REMOTE_PATH = '/home/niloleart/images_paths_and_labels.csv'

hparams = {
    'seed': 123,
    'batch_size': 2,
    'num_epochs': 10,
    'test_batch_size': 2,
    'num_classes': 2,
    'learning_rate': 1e-4,
    'validation_split': .2,
    'test_split': .2,
    'shuffle_dataset': True
}

rparams = {
    'create_csv': False,
    'plot_data_sample': False,
    'local_mode': False,
    'do_train': True
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

    dataloaders = {'train': train_loader, 'val': validation_loader}
    dataset_sizes = {'train': len(train_indices), 'val': len(val_indices)}

    return dataloaders, dataset_sizes


def get_csv_path():
    if rparams['local_mode']:
        return CSV_LOCAL_PATH
    else:
        return CSV_REMOTE_PATH


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pretrained_model():
    device = get_device()

    pretrained_model = models.vgg16(pretrained=True)
    pretrained_model.to(device)
    feature_extractor = pretrained_model.features
    print(feature_extractor)

    for layer in feature_extractor[:24]:  # Freeze layers 0 to 23
        for param in layer.parameters():
            param.requires_grad = False

    for layer in feature_extractor[24:]:  # Train layers 24 to 30
        for param in layer.parameters():
            param.requires_grad = True

    # TODO: canviar tamany entrada i veure si sigmoid o q fem
    feature_classifier = nn.Sequential(
        nn.Linear(158720, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    feature_classifier.to(device)

    model = nn.Sequential(
        feature_extractor,
        nn.Flatten(),
        feature_classifier
    )
    model.to(device)

    loss_fn = nn.BCELoss()

    optimizer_ft = optim.Adam(model.parameters(), lr=hparams['learning_rate'])

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return model, loss_fn, optimizer_ft, exp_lr_scheduler


def train(dataset):
    set_seed(hparams['seed'])

    data_loaders, dataset_sizes = get_dataloaders(dataset)

    model_ft, loss_fn, optimizer_ft, exp_lr_scheduler = get_pretrained_model()

    train_acc, train_loss, val_acc, val_loss = train_model(model_ft, loss_fn, optimizer_ft, exp_lr_scheduler, data_loaders, dataset_sizes)

    plot_losses(train_acc, val_acc, train_loss, val_loss)


def main():
    # Creates a CSV file with "Row1=full img path", "Row2 = label" (0 = healthy, 1 = ill)
    if rparams['create_csv']:
        c = CreateCSV()
        annotations_path = c.create_CSV(rparams['local_mode'])

    if 'annotations_path' in locals():
        dataset = MaratoCustomDataset(csv_path=annotations_path, transform=ToTensor())
    else:
        dataset = MaratoCustomDataset(csv_path=get_csv_path(), transform=ToTensor())

    if rparams['plot_data_sample']:
        plot_sample_data(dataset)

    if rparams['do_train']:
        train(dataset)


if __name__ == '__main__':
    main()
