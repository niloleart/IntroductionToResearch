from torch import optim, nn
from torch.optim import lr_scheduler
from torchvision.models import VGG16_Weights
from torchvision.transforms import ToTensor

from data.create_csv import CreateCSV
from data.load_data import MaratoCustomDataset, compute_data_metrics, get_data, data_transforms
from data.plot_data import plot_sample_data, plot_losses
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
import torchvision.models as models
from sklearn.model_selection import train_test_split

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
    'batch_size': 8,
    'num_epochs': 30,
    'num_classes': 2,
    'learning_rate': 1e-3,
    'learning_rate_fine_tune': 0.0005,
    'validation_split': .2,
    'test_split': .0,
    'shuffle_dataset': True
}

rparams = {
    'create_csv': False,
    'plot_data_sample': True,
    'local_mode': False,
    'do_train': True
}


def set_seed(seed):
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    _ = torch.cuda.manual_seed(seed)


def get_dataloaders(dataset):
    # Create data indices for training and validation splits:
    dataloaders = {
        'train':
            torch.utils.data.DataLoader(dataset['train'],
                                        hparams['batch_size'],
                                        shuffle=False),
        'val':
            torch.utils.data.DataLoader(dataset['val'],
                                        hparams['batch_size'],
                                        shuffle=False)
    }

    return dataloaders


def get_csv_path():
    if rparams['local_mode']:
        return CSV_LOCAL_PATH
    else:
        return CSV_REMOTE_PATH


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def get_VGG_classifier():
    return nn.Sequential(
        nn.Linear(158720, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 2),  # TODO: és un 1 amb CE? O un 2? Un 1 per BCEWithLogits, un 2 per CELoss
        nn.Sigmoid()  # No fa falta amb BCE, per dues classes
        # nn.Softmax()  # per multiclass (https://discuss.pytorch.org/t/vgg-output-layer-no-softmax/9273/6)
    )


def get_VGG16(device):
    pretrained_model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    feature_extractor = pretrained_model.features

    feature_extractor.to(device)

    for layer in feature_extractor[:24]:  # Freeze layers 0 to 23
        for param in layer.parameters():
            param.requires_grad = False

    for layer in feature_extractor[24:]:  # Train layers 24 to 30
        for param in layer.parameters():
            param.requires_grad = True

    feature_classifier = get_VGG_classifier()

    feature_classifier.to(device)

    model = nn.Sequential(
        feature_extractor,
        nn.Flatten(),
        feature_classifier
    )
    model.to(device)
    return model


def get_ResNet50(device):
    pretrained_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = pretrained_model.fc.in_features

    # for param in pretrained_model.parameters():
    #     param.requires_grad = False

    for module, param in zip(pretrained_model.modules(), pretrained_model.parameters()):
        if isinstance(module, nn.BatchNorm2d):
            param.requires_grad = False

    pretrained_model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 2),  # TODO: és un 1 amb CE? O un 2? Un 1 per BCEWithLogits, un 2 per CELoss
        nn.Sigmoid()  # No fa falta amb BCE, per dues classes
        # nn.Softmax()  # per multiclass (https://discuss.pytorch.org/t/vgg-output-layer-no-softmax/9273/6)
    )
    pretrained_model.to(device)

    return pretrained_model


def get_pretrained_model(dataset):
    device = get_device()

    # model = get_VGG16(device)
    model = get_ResNet50(device)

    labels = dataset.img_labels_not_one_hot

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels.values)
    loss_fn = nn.BCELoss(weight=torch.tensor(class_weights)).to(device)

    optimizer_ft = optim.Adam(model.parameters(), lr=hparams['learning_rate'])  # TODO: veure que s'ha de fer amb això
    # optimizer_ft = optim.Adam(model.fc.parameters(), lr=hparams['learning_rate'])  #  aquí és només si mantenim les capes q fan feat extraction sempre frozen


    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return model, loss_fn, optimizer_ft, exp_lr_scheduler


def train(dataset):
    # set_seed(hparams['seed'])

    data_loaders = get_dataloaders(dataset)

    # mean, std = compute_data_metrics(data_loaders)

    model_ft, loss_fn, optimizer_ft, exp_lr_scheduler = get_pretrained_model(dataset['train'])

    model, train_loss, train_acc, val_loss, val_acc = train_model(model_ft, dataset, data_loaders, loss_fn, optimizer_ft, get_device(), hparams['num_epochs'])

    plot_losses(train_loss, train_acc, val_loss, val_acc)


def main():
    # Creates a CSV file with "Row1=full img path", "Row2 = label" (0 = healthy, 1 = ill)
    if rparams['create_csv']:
        c = CreateCSV()
        annotations_path = c.create_CSV(rparams['local_mode'])

    if 'annotations_path' in locals():
        annotations_path = annotations_path
    else:
        annotations_path = get_csv_path()

    X, y = get_data(annotations_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75)

    train_dataset = MaratoCustomDataset(X_train, y_train, data_transforms['train'])
    test_dataset = MaratoCustomDataset(X_test, y_test, data_transforms['train'])

    dataset = {'train': train_dataset, 'val': test_dataset}

    if rparams['plot_data_sample']:
        plot_sample_data(dataset['train'])

    if rparams['do_train']:
        train(dataset)


if __name__ == '__main__':
    main()
