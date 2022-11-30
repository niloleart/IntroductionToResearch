import torch
import copy
import time
from tqdm.notebook import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(model, loss_fn, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()

    since = time.time()

    # TRAIN
    for epoch in range(num_epochs):
        model.train()
        train_loss.reset()
        train_accuracy.reset()
        train_loop = tqdm(dataloaders['train'], unit=" batches")

        for data, target, _ in train_loop:
            data = torch.permute(data, (0, 3, 1, 2))
            train_loop.set_description('[TRAIN] Epoch {}/{}'.format(epoch + 1, num_epochs))
            data, target = data.float().to(device), target.float().to(device)
            # target = target.unsqueeze(-1)  # Fa falta per passar de [X] a [X,1]

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()


            train_loss.update(loss.item(), n=len(target))
            pred = output.round()  # get prediction
            acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
            train_accuracy.update(acc, n=len(target))
            train_loop.set_postfix(loss=train_loss.avg, accuracy=train_accuracy.avg)

        train_losses.append(train_loss.avg)
        train_accuracies.append(train_accuracy.avg)

        # VALIDATION
        model.eval()
        val_loss.reset()
        val_accuracy.reset()
        val_loop = tqdm(dataloaders['val'], unit=" batches")
        with torch.no_grad():
            for data, target, _ in val_loop:
                data = torch.permute(data, (0, 3, 1, 2))
                val_loop.set_description('[VAL] Epoch {}/{}'.format(epoch + 1, num_epochs))
                data, target = data.float().to(device), target.float().to(device)
                # target = target.unsqueeze(-1)

                output = model(data)
                loss = loss_fn(output, target)
                val_loss.update(loss.item(), n=len(target))
                pred = output.round()  # get prediction
                acc = pred.eq(target.view_as(pred)).sum().item()/len(target)
                val_accuracy.update(acc, n=len(target))
                val_loop.set_postfix(loss=val_loss.avg, accuracy=val_accuracy.avg)

        val_losses.append(val_loss.avg)
        val_accuracies.append(val_accuracy.avg)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return train_accuracies, train_losses, val_accuracies, val_losses
