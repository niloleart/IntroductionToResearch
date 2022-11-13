import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_sample_data(data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 2, 3
    num_rows = ((cols * rows) // 2) + 1
    aux = 1
    for i in range(1, num_rows):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        imgs, label, folder = data[sample_idx]
        for imgIdx, img in enumerate(imgs):
            figure.add_subplot(rows, cols, aux)
            im = np.squeeze(img)
            title = ['Folder', str(folder), '-', get_eye(imgIdx), '-', get_class(label) + '(' + str(label) + ')']
            plt.title(' '.join(title))
            plt.xlabel(' Label' + str(label))
            plt.axis("off")
            plt.imshow(im)
            aux += 1
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
