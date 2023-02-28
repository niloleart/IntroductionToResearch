from torch import optim, nn
from torchvision.transforms import transforms

from model import inception
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
dev = None
hparams = {
    'batch_size': 64, #TODO change to 64
    'num_epochs': 10,
    'num_classes': 2,
    'learning_rate': 0.0001,
    'momentum': 0.9,
    'validation_split': .2,
    'test_split': .0
}

rparams = {
    'create_csv': True,
    'plot_data_sample': False,
    'local_mode': False,
    'do_train': True,
    'do_test': True,
    'image_type': 'octa_3x3_sup',  # color, macular, octa_3x3_sup, octa_3x3_deep, octa_6x6_sup,
    'compute_mean_and_std': False,
    'feature_extracting': False,  # False = finetune whole model, True = only update last layers (classifier)
    'model_name': 'inception',
    # use: resnet, inception, alexnet, vgg, squeezenet, densenet -> for: ResNet50, InceptionV3, AlexNet, VGG16_bn
    'plot_curves': True,
    'concatenate': 'depth',  # 'depth' for 4 channel images, '2d' for mosaic
}


# Not use

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

    return total_mean, total_std


def get_csv_path():
    if rparams['local_mode']:
        return CSV_LOCAL_PATH
    else:
        return CSV_REMOTE_PATH


def get_device():
    if torch.cuda.is_available():
        print("GPU is available!")
        device = 'cuda:1'
        # device = 'cpu'
    else:
        print("WARNING! GPU NOT AVAILABLE!")
        device = 'cpu'
    return device
    # return torch.device('cuda:1' if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, feature_extract=False, num_classes=2, use_pretrained=True):
    print('Initializing models')
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """Resnet50
        """

        model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)

        if rparams['image_type'] != 'color' and rparams['image_type'] != 'macular':
            conv_weight = model_ft.conv1.weight
            model_ft.conv1.in_channels = 1
            model_ft.conv1.weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes - 1),
        )
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(weights=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        if rparams['image_type'] != 'color' and rparams['image_type'] != 'macular':
            conv_weight = model_ft.features[0].weight
            model_ft.features[0].in_channels = 1
            model_ft.features[0].weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))

        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, num_classes - 1)
        )
        input_size = 224

    elif model_name == "vgg":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(weights=models.VGG16_BN_Weights)
        set_parameter_requires_grad(model_ft, feature_extract)

        if rparams['image_type'] != 'color' and rparams['image_type'] != 'macular':
            conv_weight = model_ft.features[0].weight
            model_ft.features[0].in_channels = 1
            model_ft.features[0].weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))

        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, num_classes - 1)
        )
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.Softmax()
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
        model_ft = inception.nilception(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        if rparams['image_type'] != 'color' and rparams['image_type'] != 'macular':
            model_ft.transform_input = False

            conv_weight = model_ft.Conv2d_1a_3x3.conv.weight
            model_ft.Conv2d_1a_3x3.conv.in_channels = 1
            model_ft.Conv2d_1a_3x3.conv.weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))

        model_ft.maxpool2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
        set_parameter_requires_grad(model_ft, feature_extract)

        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            # nn.Softmax(dim=-1) #AMB BCE no fa falta per training
        )
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def get_optimizer_and_loss(model_ft, dataset, feature_extract=rparams['feature_extracting']):
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

    # optimizer_ft = optim.SGD(params_to_update, hparams['learning_rate'], momentum=hparams['momentum'], nesterov=False)
    optimizer_ft = optim.Adam(params_to_update, hparams[
        'learning_rate'])  # TODO fer servir adam al principi i després SGD per convergir amb un scheduler, mirar com polles, pq ara per SGD no entrena reees!
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.1)

    labels = dataset.img_labels_not_one_hot

    class_weights = compute_class_weight('balanced', classes=np.unique(np.ravel(labels, order='C')),
                                         y=np.ravel(labels, order='C'))

    # num_positives = torch.tensor(sum(labels == 1), dtype=float)
    # num_negatives = torch.tensor(len(labels) - num_positives, dtype=float)
    # pos_weight = (num_negatives / num_positives)

    # criterion = nn.BCELoss(weight=torch.tensor(class_weights)).to(get_device())
    # criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(class_weights)).to(  # TODO: fer servir una sola neurona
    #     get_device())
    # TODO calcular el valor del tensor per cada tipus d'experiment, no es el mateix en RDR q en DM!
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.4).clone().detach()).to(dev)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights)).to(get_device())

    # TODO: add lr scheduler!

    return optimizer_ft, criterion, scheduler


def create_dataset(X, y, input_size):
    X_init, y_init = get_data_concat(X, y)
    dataset = MaratoCustomDataset(X_init, y_init, transforms.ToTensor(), rparams['image_type'], input_size)

    # Plot init dataset (whole data)
    plot = PlotUtils(dataset)
    plot.plot_samples()

    if rparams['compute_mean_and_std']:
        # Compute mean and std for whole dataset
        print('Computing mean and std for the dataset')
        total_mean, total_std = batch_mean_and_sd(dataset)

        if rparams['image_type'] == 'octa_3x3_sup' or rparams['image_type'] == 'octa_3x3_deep' \
                or rparams['image_type'] == 'octa_6x6_sup' or rparams['image_type'] == 'octa_6x6_deep':
            total_mean = total_mean[0]
            total_std = total_std[0]

        norm_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=total_mean, std=total_std)

        ])

        dataset_normalized = MaratoCustomDataset(X_init, y_init, target_transform=norm_transforms)

        plot.plot_samples(dataset_normalized)
        print('Computing mean and std for the normalized dataset')
        normalized_mean, normalized_std = batch_mean_and_sd(dataset_normalized)

    else:
        if rparams['concatenate'] == '2d':
            total_mean = -0.5041
            total_std = 0.2378
        else:
            if rparams['image_type'] == 'color':
                total_mean = ([0.4292, 0.1894, 0.1048])
                total_std = ([0.2798, 0.1336, 0.0811])

            elif rparams['image_type'] == 'octa_3x3_sup':
                total_mean = 0.2622
                total_std = 0.1776

            elif rparams['image_type'] == 'octa_3x3_deep':
                total_mean = 0.1868
                total_std = 0.1087

            elif rparams['image_type'] == 'octa_6x6_sup':
                total_mean = 0.3539
                total_std = 0.1409

            elif rparams['image_type'] == 'octa_6x6_deep':
                total_mean = 0.2465
                total_std = 0.0847

            elif rparams['image_type'] == 'macular':
                total_mean = ([0.0895, 0.2247, 0.2095])
                total_std = ([0.1765, 0.2882, 0.1578])

    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=total_mean, std=total_std)
    ])

    # print('Splitting dataset into train/val (90/10)...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, train_size=.70, shuffle=True, stratify=y)
    #
    X_train, Y_train = get_data_concat(X_train, y_train)
    X_test, Y_test = get_data_concat(X_test, y_test)
    #
    train_dataset = MaratoCustomDataset(X_train, Y_train, normalize_transform,
                                        rparams['image_type'], input_size, augment_individual_images=True)  # TODO data aug goes here
    # plot.plot_samples(train_dataset)
    #
    test_dataset = MaratoCustomDataset(X_test, Y_test, normalize_transform, rparams['image_type'], input_size, augment_individual_images=False)
    #
    dataset = {'train': train_dataset, 'test': test_dataset}
    # dataset = MaratoCustomDataset(X_init, y_init, normalize_transform, rparams['image_type'], input_size) # TODO data aug goes here
    print('Dataset created. Data is ready!')
    return dataset


def main():
    dev = get_device()
    # Creates a CSV file with "Row1=full img path", "Row2 = label" (0 = healthy, 1 = ill)
    if rparams['create_csv']:
        c = CreateCSV()
        annotations_path = c.create_CSV(rparams['local_mode'])

    if 'annotations_path' in locals():
        annotations_path = annotations_path
    else:
        annotations_path = get_csv_path()

    X, y = get_data(annotations_path)

    model_ft, input_size = initialize_model(rparams['model_name'], rparams['feature_extracting'], num_classes=2)
    # print(model_ft)

    dataset = create_dataset(X, y, input_size)

    if rparams['do_train']:
        optimizer_ft, criterion, scheduler = get_optimizer_and_loss(model_ft, dataset['train'])
        # optimizer_ft, criterion, scheduler = get_optimizer_and_loss(model_ft, dataset)

        # data_loaders = get_dataloaders(dataset)
        data_loaders = 0  # Canvi degut a que estem creant els dataloaders dins de train per fer kfold cros vala

        model, train_loss, train_auc, val_loss, val_auc = train_model(dataset, optimizer_ft, dev,
                                                                      hparams['num_epochs'],
                                                                      rparams['model_name'] == "inception")

    if rparams['do_train'] and rparams['plot_curves']:
        plot_losses(train_loss, train_auc, val_loss, val_auc)

    if rparams['do_test']:
        from train.test_model import test_model
        try:
            test_model(dataset['test'], dev, model)
        except:
            test_model(dataset['test'], dev)


if __name__ == '__main__':
    CUDA_LAUNCH_BLOCKING = 1

    main()
