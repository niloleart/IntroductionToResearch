import torch
from torch import nn

from model import inception


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_model(use_pretrained=True, feature_extract=True, num_classes=2, image_type='octa_3x3_sup'):
    """ Inception v3
           Be careful, expects (299,299) sized images and has auxiliary output
           """
    model_ft = inception.nilception(pretrained=use_pretrained)

    set_parameter_requires_grad(model_ft, feature_extract)

    if image_type != 'color' and image_type != 'macular':
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

    return model_ft
