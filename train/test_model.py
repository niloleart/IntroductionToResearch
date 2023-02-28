import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torchmetrics.classification.auroc import AUROC

from model import inception

model_path = "/home/niloleart/pycharm_projects/projecte/saved_models/model.pth"


def test_model(dataset, device, model=None):
    auroc = AUROC(task="binary").to(device)
    running_auroc = 0
    running_CM = 0
    num_classes = 2
    if model is None:
        model = inception.nilception()
        model.transform_input = False

        conv_weight = model.Conv2d_1a_3x3.conv.weight
        model.Conv2d_1a_3x3.conv.in_channels = 1
        model.Conv2d_1a_3x3.conv.weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))
        model.maxpool2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            # nn.Softmax(dim=-1)
            nn.Sigmoid()
        )

        model.load_state_dict(torch.load(model_path))

    # avoid different results due to batch size and its normalization on BatchNorm layers????
    for module in model.children():
        if type(module) == nn.BatchNorm2d:
            module.track_running_stats = False
        else:
            for child in module.children():
                if type(child) == nn.BatchNorm2d:
                    # isinstance(child, torch.nn.modules.BatchNorm2d)
                    child.track_running_stats = False
                else:
                    for subchild in child.children():
                        if type(subchild) == nn.BatchNorm2d:
                            subchild.track_running_stats = False

    model.eval()
    model.to(device)

    # 20% data is 276 for R-DR diagnosis. Use 276/6=46
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
    running_preds = torch.empty(0,).to(device)
    running_labels = torch.empty(0,).to(device)

    for inputs, labels_raw, labels_one_hot, _ in dataloader:
        inputs = inputs.float()
        inputs = inputs.to(device)
        labels = labels_raw.float().to(device)
        labels_one_hot.to(device)

        outputs = model(inputs)
        # preds = outputs > 0.0

        preds = outputs.argmax(dim=1, keepdim=False)

        # running_auroc += auroc(preds, labels)
        running_CM += confusion_matrix(labels.cpu(), preds.cpu(), labels=[0, 1])
        running_preds = torch.cat((running_preds, preds), dim=0)
        running_labels = torch.cat((running_labels, labels), dim=0)

    # outcomes = {0: 'Healthy', 1: 'Ill'}

    # prediction = outcomes[outputs]

    # auroc = (running_auroc / len(dataloader)) * 100
    auroc = auroc(running_preds, running_labels)
    compute_metrics(running_CM)
    print(auroc.item())


def compute_metrics(CM):
    tn = CM[0][0]
    tp = CM[1][1]
    fp = CM[0][1]
    fn = CM[1][0]

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print('-'*15)
    print('TEST RESULTS:')
    print('Test Sensitivity: {:.4f}, Test Specificity: {:.4f}'.format(sensitivity, specificity))
    print('-' * 15)
    return sensitivity, specificity
