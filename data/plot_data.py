import matplotlib.pyplot as plt
import numpy as np
import torch


class PlotUtils():
    idx = []
    num_rows = 5
    num_cols = 2
    total = num_rows * num_cols + 1

    def __init__(self, data):
        self.data = data

    def plot_samples(self, data=None):
        if data is not None:
            self.data = data

        figure = plt.figure(figsize=(8, 12))
        for i in range(1, self.total):
            sample_idx = torch.randint(len(self.data), size=(1,)).item()
            img, label_not_one_hot, label, folder = self.data[sample_idx]
            if img.size(dim=0) > 1:
                img = np.squeeze(img)
            figure.add_subplot(self.num_rows, self.num_cols, i)
            title = ['Folder', str(folder), '-',
                     self.get_class(label_not_one_hot) + '(' + str(int(label_not_one_hot)) + ')']
            plt.title(' '.join(title))
            plt.xlabel(' Label' + str(label_not_one_hot))
            plt.axis("off")
            plt.imshow(img.permute(1, 2, 0), cmap='gray')
        plt.show()

    def get_eye(self, index):
        if index == 0:
            return 'LeftEye'
        elif index == 1:
            return 'RightEye'

    def get_class(self, label):
        if label == 0:
            return 'Healthy'
        elif label == 1:
            return 'Ill'


def plot_losses(train_loss, train_acc, train_auc, val_loss, val_acc, val_auc):
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
    plt.figure()

    plt.plot(epochs, train_auc, 'b', label='Training AUC')
    plt.plot(epochs, val_auc, 'r', label='Validation AUC')
    plt.title('Training and validation AUROC')
    plt.legend()

    plt.show()
