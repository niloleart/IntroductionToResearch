import os

from torch import optim, nn
from torchvision.transforms import transforms
from data.create_csv import CreateCSV
from data.load_data import MaratoCustomDataset, get_data, get_data_concat
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
import torchvision.models as models
from sklearn.model_selection import train_test_split

from data.plot_data import PlotUtils, plot_losses
from train.train import train_model

# This is a sample Python script.

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
    'num_epochs': 40,
    'num_classes': 2,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'validation_split': .2,
    'test_split': .0,
    'shuffle_dataset': True
}

rparams = {
    'create_csv': False,
    'plot_data_sample': True,
    'local_mode': False,
    'do_train': True,
    'feature_extracting': False,  # False = finetune whole model, True = only update last layers (classifier)
    'model_name': 'alexnet',  # resnet, alexnet, vgg11_bn, squeezenet, densenet
    'plot_curves': True
}


# Not used
def set_seed(seed):
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    _ = torch.cuda.manual_seed(seed)


def get_dataloaders(dataset):
    # Create data indices for training and validation splits:
    print('Creating dataloaders...')
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


def batch_mean_and_sd(dataloader):
    total_sum = torch.tensor([0.0, 0.0, 0.0])
    total_sum_square = torch.tensor([0.0, 0.0, 0.0])

    for images, _, _, _ in dataloader:
        c, h, w = images.shape
        total_sum += images.sum(axis=[1, 2])
        total_sum_square += (images ** 2).sum(axis=[1, 2])

    count = len(dataloader) * h * w
    total_mean = total_sum / count
    total_var = (total_sum_square / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)
    print('mean: ' + str(total_mean))
    print('std: ' + str(total_std) + '\n')

    transform = transforms.Normalize(mean=total_mean, std=total_std)

    return transform


def get_csv_path():
    if rparams['local_mode']:
        return CSV_LOCAL_PATH
    else:
        return CSV_REMOTE_PATH


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, feature_extract=False, num_classes=2, use_pretrained=True):
    print('Initializing models')
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """Resnet18
        """

        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, num_classes)
            # nn.Softmax()
        )
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.Softmax(dim=1)
        )
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def get_optimizer_and_loss(model_ft, dataset, feature_extract=rparams['feature_extracting']):
    model_ft = model_ft.to(get_device())

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # optimizer_ft = optim.SGD(params_to_update, hparams['learning_rate'], momentum=hparams['momentum'])
    optimizer_ft = optim.Adam(params_to_update, hparams[
        'learning_rate'])  # TODO fer servir adam al principi i després SGD per convergir amb un scheduler, mirar com polles, pq ara per SGD no entrena reees!

    labels = dataset.img_labels_not_one_hot

    class_weights = compute_class_weight('balanced', classes=np.unique(np.ravel(labels, order='C')),
                                         y=np.ravel(labels, order='C'))
    # criterion = nn.BCELoss(weight=torch.tensor(class_weights)).to(get_device())
    criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(class_weights)).to(
        get_device())  # TODO: mirar bé si val la pena
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights)).to(get_device())

    # TODO: add lr scheduler!

    return optimizer_ft, criterion


def create_dataset(X, y):
    X_init, y_init = get_data_concat(X, y)
    dataset = MaratoCustomDataset(X_init, y_init, transforms.ToTensor())

    # Plot init dataset
    plot = PlotUtils(dataset)
    plot.plot_samples()

    # Compute mean and std for whole dataset
    print('Computing mean and std for the dataset')
    mean_std_transform = batch_mean_and_sd(dataset)

    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.ToTensor(),
        mean_std_transform,
    ])

    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        mean_std_transform
    ])

    dataset_normalized = MaratoCustomDataset(X_init, y_init, normalize_transform)
    plot.plot_samples(dataset_normalized)
    _ = batch_mean_and_sd(dataset_normalized)


    print('Splitting dataset into train/val (75/25)...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75, shuffle=True)

    X_train, Y_train = get_data_concat(X_train, y_train)
    X_test, Y_test = get_data_concat(X_test, y_test)

    train_dataset = MaratoCustomDataset(X_train, Y_train, train_transform)
    test_dataset = MaratoCustomDataset(X_test, Y_test, normalize_transform)

    dataset = {'train': train_dataset, 'val': test_dataset}
    print('Dataset created. Data is ready!')
    return dataset


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

    dataset = create_dataset(X, y)

    # if rparams['plot_data_sample']:
    #     plot_sample_data(dataset['train'])

    if rparams['do_train']:
        data_loaders = get_dataloaders(dataset)

        model_ft, input_size = initialize_model(rparams['model_name'])
        print(model_ft)
        optimizer_ft, criterion = get_optimizer_and_loss(model_ft, dataset['train'])

        model, train_loss, train_acc, val_loss, val_acc = train_model(model_ft, data_loaders, criterion,
                                                                      optimizer_ft, get_device(), hparams['num_epochs'])

    if rparams['do_train'] and rparams['plot_curves']:
        plot_losses(train_loss, train_acc, val_loss, val_acc)


if __name__ == '__main__':
    main()
