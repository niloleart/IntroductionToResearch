import copy

import torch
import time
# from torchmetrics.classification import BinarySpecificity, BinaryF1Score
# from torchmetrics.functional import retrieval_recall


# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


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

            for inputs, _, labels, _ in dataloaders[phase]:
                inputs = torch.permute(inputs, (0, 3, 1, 2))
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == torch.argmax(labels, 1)).item()

                # recall = retrieval_recall(preds, labels)
                # epoch_specificity = BinarySpecificity(preds, labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            # epoch_specificity = epoch_specificity / len(datasets[phase])
            # epoch_recall = recall / len(datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

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

        print()

        # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_specificity, epoch_recall))
        # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_specificity))

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('Best val acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model, train_loss, train_acc, val_loss, val_acc

# def train_model(model, datasets, dataloaders, criterion, optimizer, device, num_epochs=4):
#     since = time.time()
#     train_loss = []
#     val_loss = []
#     train_acc = []
#     val_acc = []
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch + 1, num_epochs))
#         print('-' * 10)
#
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             for inputs, _, labels, _ in dataloaders[phase]:
#                 inputs = torch.permute(inputs, (0, 3, 1, 2))
#                 inputs = inputs.float().to(device)
#                 labels = labels.float().to(device)
#
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#
#                 if phase == 'train':
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#
#                 _, preds = torch.max(outputs, 1)
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == torch.argmax(labels, 1)).item()
#
#                 # recall = retrieval_recall(preds, labels)
#                 # epoch_specificity = BinarySpecificity(preds, labels)
#
#             epoch_loss = running_loss / len(datasets[phase])
#             epoch_acc = running_corrects / len(datasets[phase])
#             # epoch_specificity = epoch_specificity / len(datasets[phase])
#             # epoch_recall = recall / len(datasets[phase])
#
#             if phase == 'train':
#                 train_loss.append(epoch_loss)
#                 train_acc.append(epoch_acc)
#             else:
#                 val_loss.append(epoch_loss)
#                 val_acc.append(epoch_acc)
#
#             print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
#             # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_specificity, epoch_recall))
#             # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_specificity))
#
#     time_elapsed = time.time() - since
#     print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#     return model, train_loss, train_acc, val_loss, val_acc

# def train_model(model, loss_fn, optimizer, dataloaders, device, num_epochs=25):
#     train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []
#     val_loss = AverageMeter()
#     val_accuracy = AverageMeter()
#     train_loss = AverageMeter()
#     train_accuracy = AverageMeter()
#
#     since = time.time()
#
#     # TRAIN
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss.reset()
#         train_accuracy.reset()
#         train_loop = tqdm(dataloaders['train'], unit=" batches")
#
#         for data, target, _ in train_loop:
#             data = torch.permute(data, (0, 3, 1, 2))
#             train_loop.set_description('[TRAIN] Epoch {}/{}'.format(epoch + 1, num_epochs))
#             data, target = data.float().to(device), target.float().to(device)
#             # target = target.unsqueeze(-1)  # Fa falta per passar de [X] a [X,1]
#
#             optimizer.zero_grad()
#             output = model(data)
#             loss = loss_fn(output, target)
#             loss.backward()
#             optimizer.step()
#
#
#             train_loss.update(loss.item(), n=len(target))
#             pred = output.round()  # get prediction
#             acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
#             train_accuracy.update(acc, n=len(target))
#             train_loop.set_postfix(loss=train_loss.avg, accuracy=train_accuracy.avg)
#
#         train_losses.append(train_loss.avg)
#         train_accuracies.append(train_accuracy.avg)
#
#         # VALIDATION
#         model.eval()
#         val_loss.reset()
#         val_accuracy.reset()
#         val_loop = tqdm(dataloaders['val'], unit=" batches")
#         with torch.no_grad():
#             for data, target, _ in val_loop:
#                 data = torch.permute(data, (0, 3, 1, 2))
#                 val_loop.set_description('[VAL] Epoch {}/{}'.format(epoch + 1, num_epochs))
#                 data, target = data.float().to(device), target.float().to(device)
#                 # target = target.unsqueeze(-1)
#
#                 output = model(data)
#                 loss = loss_fn(output, target)
#                 val_loss.update(loss.item(), n=len(target))
#                 pred = output.round()  # get prediction
#                 acc = pred.eq(target.view_as(pred)).sum().item()/len(target)
#                 val_accuracy.update(acc, n=len(target))
#                 val_loop.set_postfix(loss=val_loss.avg, accuracy=val_accuracy.avg)
#
#         val_losses.append(val_loss.avg)
#         val_accuracies.append(val_accuracy.avg)
#
#     time_elapsed = time.time() - since
#     print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#
#     return train_accuracies, train_losses, val_accuracies, val_losses
