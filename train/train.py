import copy

import torch
import time


# from torchmetrics.classification import BinarySpecificity, BinaryF1Score
# from torchmetrics.functional import retrieval_recall

def train_model(model, dataloaders, criterion, optimizer, device,
                num_epochs=4):  # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    since = time.time()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels_raw, labels_one_hot, _ in dataloaders[phase]:
                inputs = inputs.float().to(device)
                labels_one_hot = labels_one_hot.float().to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_one_hot)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == torch.argmax(labels_one_hot, 1)).item()

                # recall = retrieval_recall(preds, labels_one_hot)
                # epoch_specificity = BinarySpecificity(preds, labels_one_hot)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            # epoch_specificity = epoch_specificity / len(datasets[phase])
            # epoch_recall = recall / len(datasets[phase])

            # print('{} loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            print('{} loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        print()

        # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_specificity, epoch_recall))
        # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_specificity))

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('Best val acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model, train_loss, train_acc, val_loss, val_acc
