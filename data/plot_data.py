import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_sample_data(data):
    figure = plt.figure(figsize=(8, 12))
    num_rows = 4
    num_cols = 1
    for i in range(1, num_rows*num_cols + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label_not_one_hot, label, folder = data[sample_idx]
        im = np.squeeze(img)
        figure.add_subplot(num_rows, num_cols, i)
        title = ['Folder', str(folder), '-', get_class(label_not_one_hot) + '(' + str(int(label_not_one_hot)) + ')']
        plt.title(' '.join(title))
        plt.xlabel(' Label' + str(label_not_one_hot))
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


def plot_losses(train_loss, train_acc, val_loss, val_acc):
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
