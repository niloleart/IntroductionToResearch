import copy
import time

from torch import optim, nn
from torch.optim import lr_scheduler
from torchvision.transforms import ToTensor

from data.create_csv import CreateCSV
from data.load_data import MaratoCustomDataset
from data.plot_data import plot_sample_data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import torchvision.models as models

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
    'local_mode': True,
    'do_train': False
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def get_pretrained_model():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return model_ft, criterion, optimizer_ft, exp_lr_scheduler


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    bes_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # TODO: passar les dues imatges
            for inputs, labels, _ in dataloaders[phase]:

                inputs = inputs[0].to(device)
                inputs = torch.permute(inputs, (0, 3, 1, 2))

                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.float())
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train(dataset):
    set_seed(hparams['seed'])

    train_loaders, dataset_sizes = get_dataloaders(dataset)

    model_ft, criterion, optimizer_ft, exp_lr_scheduler = get_pretrained_model()

    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, train_loaders, dataset_sizes)


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
