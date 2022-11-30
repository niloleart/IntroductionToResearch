import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_sample_data(data):
    figure = plt.figure(figsize=(8, 12))
    num_rows = 4
    num_cols = 1
    for i in range(1, num_rows*num_cols + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label, folder = data[sample_idx]
        im = np.squeeze(img)
        figure.add_subplot(num_rows, num_cols, i)
        title = ['Folder', str(folder), '-', get_class(label[0]) + '(' + str(int(label[0])) + ')']
        plt.title(' '.join(title))
        plt.xlabel(' Label' + str(label))
        plt.axis("off")
        plt.imshow(im)
    plt.show()


def get_eye(index):
    if index == 0:
        return 'LeftEye'
    elif index == 1:
        return 'RightEye'


def get_class(label):
    if label == 0:
        return 'Healthy'
    elif label == 1:
        return 'Ill'


def plot_losses(train_acc, val_acc, train_loss, val_loss):
    epochs = range(len(train_acc))

    plt.plot(epochs, train_acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
