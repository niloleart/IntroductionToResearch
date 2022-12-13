import copy

import torch
import time
from torchmetrics.classification.auroc import AUROC


def train_model(model, dataloaders, criterion, optimizer, scheduler, device,
                num_epochs=4):  # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    since = time.time()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    train_auc = []
    val_auc = []
    auroc = AUROC(task="binary")
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase])  # l'estem calculant per batch
            epoch_auroc = (running_auroc / len(dataloaders[phase])).item()


            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)


            if phase == 'train':
                # scheduler.step() # TODO
                train_auc.append(epoch_auroc)
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_auc.append(epoch_auroc)
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            print('{} loss: {:.4f}, Acc: {:.4f}, AUROC: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_auroc))

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('Best val acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model, train_loss, train_acc, train_auc, val_loss, val_acc, val_auc
