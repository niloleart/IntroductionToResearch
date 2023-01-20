import copy

import torch
import time
from torchmetrics.classification.auroc import AUROC
from torchmetrics.classification import BinaryF1Score


def train_model(model, dataloaders, criterion, optimizer, scheduler, device,
                num_epochs=4):  # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    since = time.time()
    train_loss = []
    val_loss = []
    train_acc = []
    train_f1 = []
    val_acc = []
    train_auc = []
    val_auc = []
    val_acc_history = []
    val_f1 = []

    auroc = AUROC(task="binary").to(device)
    f1score = BinaryF1Score().to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auroc = 0.0
    best_f1 = 0.0
    best_loss = 0.0

    model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_auroc = 0
            running_acc = 0
            running_f1score = 0
            running_f1_labels = torch.Tensor().to(device)
            running_f1_preds = torch.Tensor().to(device)

            for inputs, labels_raw, _, _ in dataloaders[phase]:
                inputs = inputs.float()
                labels = labels_raw.float()

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    loss = criterion(outputs, labels.unsqueeze(1))
                    # _, preds = torch.max(outputs, dim=1)
                    preds = outputs > 0.0

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # _, preds = torch.max(outputs, 1)
                # running_corrects += torch.sum(preds == torch.argmax(labels)).item()

                running_auroc += auroc(preds.float(), labels)
                running_loss += loss.item() * inputs.size(0)  # Per BCE With logits loss
                running_acc += (preds == labels).float().mean().item()  # Per calcular acc amb bce with logits loss

                running_f1_labels = torch.cat((running_f1_labels, labels), -1)
                # running_f1_labels.cat(labels)
                # running_f1_preds.cat(preds.reshape(-1))
                running_f1_preds = torch.cat((running_f1_preds, preds.reshape(-1)), -1)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase])  # l'estem calculant per batch
            epoch_auroc = (running_auroc / len(dataloaders[phase])).item()
            epoch_f1score = f1score(running_f1_preds, running_f1_labels)


            if phase == 'val' and epoch_f1score > best_f1:
                best_f1 = epoch_f1score
                best_model_wts = copy.deepcopy(model.state_dict())
                best_loss = epoch_loss
                best_auroc = epoch_auroc
            if phase == 'val':
                val_acc_history.append(epoch_acc)


            if phase == 'train':
                # scheduler.step() # TODO
                train_auc.append(epoch_auroc)
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                train_f1.append(epoch_f1score.item())
            else:
                val_auc.append(epoch_auroc)
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                val_f1.append(epoch_f1score.item())

            print('{} loss: {:.4f}, F1-Score: {:.4f}, Acc: {:.4f}, AUROC: {:.4f}'.format(phase, epoch_loss, epoch_f1score, epoch_acc, epoch_auroc))

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('Best val F1-score: {:.4f}, Loss: {:.4f}, AUROC: {:.4f}'.format(best_f1, best_loss, best_auroc))

    model.load_state_dict(best_model_wts)

    return model, train_loss, train_f1, train_auc, val_loss, val_f1, val_auc
