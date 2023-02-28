import matplotlib.pyplot as plt
import numpy as np
import torch


class PlotUtils():
    idx = []
    num_rows = 5
    # num_rows = 1
    num_cols = 2
    # num_cols = 1
    total = num_rows * num_cols + 1

    def __init__(self, data):
        self.data = data

    def plot_samples(self, data=None):
        if data is not None:
            self.data = data

        figure = plt.figure(figsize=(8, 12))
        # figure = plt.figure(figsize=(1, 1))
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


def plot_losses(train_loss, train_auc, val_loss, val_auc):

    fig1, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(train_loss[0], label='Train loss fold 1', linestyle="--")
    ax1.plot(train_loss[1], label='Train loss fold 2', linestyle="-.")
    ax1.plot(train_loss[2], label='Train loss fold 3', linestyle=":")
    avg_train_loss = np.average(train_loss, axis=0)
    ax1.plot(avg_train_loss, label='Average train loss', linestyle="-")
    ax1.legend()

    ax2.plot(val_loss[0], label='val loss fold 1', linestyle="--")
    ax2.plot(val_loss[1], label='val loss fold 2', linestyle="-.")
    ax2.plot(val_loss[2], label='val loss fold 3', linestyle=":")
    avg_train_loss = np.average(val_loss, axis=0)
    ax2.plot(avg_train_loss, label='Average val loss', linestyle="-")
    ax2.legend()

    fig2, (ax3, ax4) = plt.subplots(2, 1)
    ax3.plot(train_auc[0], label='Train auc fold 1', linestyle="--")
    ax3.plot(train_auc[1], label='Train auc fold 2', linestyle="-.")
    ax3.plot(train_auc[2], label='Train auc fold 3', linestyle=":")
    avg_train_auc = np.average(train_auc, axis=0)
    ax3.plot(avg_train_auc, label='Average train auc', linestyle="-")
    ax3.legend()

    ax4.plot(val_auc[0], label='val auc fold 1', linestyle="--")
    ax4.plot(val_auc[1], label='val auc fold 2', linestyle="-.")
    ax4.plot(val_auc[2], label='val auc fold 3', linestyle=":")
    avg_val_auc = np.average(val_auc, axis=0)
    ax4.plot(avg_val_auc, label='Average val auc', linestyle="-")
    ax4.legend()

    plt.show()

    plt.show()


