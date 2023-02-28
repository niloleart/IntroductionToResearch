import copy

import sklearn.model_selection
import torch
import time

from torch import nn
from torchmetrics.classification.auroc import AUROC
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix

from data.create_csv import experiment, experiments
from losses.CustomLosses import Loss
from model.get_model import get_model

model_path = "/home/niloleart/pycharm_projects/projecte/saved_models/"


def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
  '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def train_model(dataset, optimizer, device, num_epochs=5, isInception=False, k_folds=3, batch_size=64,
                isTest=False):  # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    since = time.time()

    # TODO compute dinamically
    # if experiment == experiments['DM_diagnosis']:
    #     mbeta = 0.001
    #     msamples_per_class = [914, 22]
    # elif experiment == experiments['DR_diagnosis']:
    #     mbeta = 0.999
    #     msamples_per_class = [590, 324]
    # elif experiment == experiments['R-DR_diagnosis']:
    #     mbeta = 0.999
    #     if isTest:
    #         msamples_per_class = [754, 68]
    #     else:
    #         msamples_per_class = [854, 85]
    if experiment == 0 or experiment == 1 or experiment == 2:
        mbeta = 0.999
        msamples_per_class = [
            2*sum(dataset['train'].img_labels_not_one_hot == 0)/3,
            2*sum(dataset['train'].img_labels_not_one_hot == 1)/3
        ]
    else:
        print("Error")
        exit()

    CM = 0
    train_loss = [[], [], []]
    val_loss = [[], [], []]
    train_auc = [[], [], []]
    val_auc = [[], [], []]

    auroc = AUROC(task="binary").to(device)

    folds_auc = []
    folds_specificity = []
    folds_sensitivity = []


    kfold = KFold(n_splits=k_folds, shuffle=True)  # TODO mirar docu pel shuffle
    # kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

    train_val_dataset = dataset['train']
    Y = train_val_dataset.img_labels_not_one_hot
    X = train_val_dataset.img_eye
    # K-fold Cross Validation model evaluation
    # for fold, (train_ids, val_ids) in enumerate(kfold.split(X[0], Y)):
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_val_dataset)):
    # for fold in (0, 0):
        model = get_model()
        model.to(device)
        best_model_wts = copy.deepcopy(model.state_dict())

        best_auroc = 0.0
        best_specificity = 0.0
        best_sensitivity = 0.0
        best_loss = 0.0

        # For fold results


        # Print
        print(f'--->[FOLD {fold + 1}]<---')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            train_val_dataset,
            batch_size=batch_size, sampler=train_subsampler, shuffle=False)
        valloader = torch.utils.data.DataLoader(
            train_val_dataset,
            batch_size=batch_size, sampler=val_subsampler, shuffle=False)

        # criterion = Loss(
        #     loss_type="focal_loss",
        #     beta=mbeta,
        #     samples_per_class=msamples_per_class,
        #     class_balanced=True,
        # ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.4).clone().detach()).to(device)

        dataloaders = {
            'train': trainloader,
            'val': valloader
        }

        # model.apply(reset_weights())  # TODO: mirar si és necessari

        for epoch in range(num_epochs):
            print()
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_CM = 0
                running_loss = 0.0
                running_preds = torch.empty(0, 1).to(device)
                running_labels = torch.empty(0,).to(device)

                for inputs, labels_raw, labels_one_hot, _ in dataloaders[phase]:
                    inputs = inputs.float()
                    labels = labels_raw.float()
                    # labels = labels_one_hot.float()

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    labels_one_hot.to(device)

                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        if isInception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels.long())
                            loss2 = criterion(aux_outputs, labels.long())
                            loss = loss1 + (0.4 * loss2)
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels.long())

                        preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # _, preds = torch.max(outputs, 1)
                    # running_corrects += torch.sum(preds == torch.argmax(labels)).item()

                    running_CM += confusion_matrix(labels.cpu(), preds.cpu(), labels=[0, 1])

                    running_preds = torch.cat((running_preds, preds), dim=0)
                    running_labels = torch.cat((running_labels, labels), dim=0)
                    running_loss += loss.item() * inputs.size(0)  # Per BCE With logits loss

                epoch_sensitivity, epoch_specificity = compute_metrics(running_CM, phase)
                epoch_auroc = auroc(running_preds, running_labels)
                epoch_loss = running_loss / len(dataloaders[phase].dataset)

                if phase == 'val' \
                        and epoch_sensitivity >= best_sensitivity\
                        and epoch_specificity >= best_specificity:
                        # and epoch_auroc > best_auroc \
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_loss = epoch_loss
                    best_auroc = epoch_auroc
                    best_specificity = epoch_specificity
                    best_sensitivity = epoch_sensitivity

                if phase == 'train':
                    # scheduler.step() # TODO
                    train_auc[fold].append(epoch_auroc.item())
                    train_loss[fold].append(epoch_loss)

                    # Saving the model
                    save_path = f'./model-fold-{fold}.pth'
                    torch.save(model.state_dict(), save_path)
                else:
                    val_auc[fold].append(epoch_auroc.item())
                    val_loss[fold].append(epoch_loss)

                # folds_auc[fold] = 100.0 * epoch_auroc.item()
                # folds_sensitivity[fold] = 100 * epoch_sensitivity
                # folds_specificity[fold] = 100 * epoch_specificity

                print('     {} loss: {:.4f}, AUROC: {:.4f}'.format(phase, epoch_loss, epoch_auroc))
                print('*' * 20)

        folds_auc.append(100.0 * best_auroc.item())
        folds_sensitivity.append(100 * best_sensitivity)
        folds_specificity.append(100 * best_specificity)
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('Best val Loss: {:.4f}, AUROC: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}'
          .format(best_loss, best_auroc, best_sensitivity, best_specificity))

    avg_auc = sum(folds_auc)/len(folds_auc)
    avg_sensitivity = sum(folds_sensitivity)/len(folds_sensitivity)
    avg_specificity = sum(folds_specificity)/len(folds_specificity)

    print('Average over 3 folds: AUROC: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}'
          .format(avg_auc, avg_sensitivity, avg_specificity))

    # load model with best weights and save to path
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_path + "model.pth")

    return model, train_loss, train_auc, val_loss, val_auc

def compute_metrics(CM, phase):
    tn = CM[0][0]
    tp = CM[1][1]
    fp = CM[0][1]
    fn = CM[1][0]

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print('*' * 20)
    print(phase + ' results: ')
    print('     Sensitivity: {:.4f}, Specificity: {:.4f}'.format(sensitivity, specificity))

    return sensitivity, specificity



