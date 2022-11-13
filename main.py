import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor

from data.create_csv import CreateCSV
from data.load_data import MaratoCustomDataset

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


CREATE_FORMATTED_CSV = True


def plot_sample_data(data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 2, 3
    num_plot = cols*rows+1
    num_rows = (cols*rows)//2
    aux = 1
    for i in range(1, num_rows+1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        imgs, label, folder = data[sample_idx]
        for imgIdx, img in enumerate(imgs):
            figure.add_subplot(rows, cols, aux)
            im = np.squeeze(img)
            title = ['Folder', str(folder)]
            plt.title(' '.join(title))
            plt.xlabel(' Label' + str(label))
            plt.axis("off")
            plt.imshow(im)
            aux += 1
    plt.show()


def main():
    # Creates a CSV file with "Row1=full img path", "Row2 = label" (0 = healthy, 1 = ill)
    if CREATE_FORMATTED_CSV:
        c = CreateCSV()
        annotations_path = c.create_CSV()

    data = MaratoCustomDataset(csv_path=annotations_path, transform=ToTensor())

    plot_sample_data(data)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
