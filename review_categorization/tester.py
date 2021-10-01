import torch
from torch import nn
import numpy as np


def test(net, test_loader, batch_size):
    criterion = nn.BCELoss()
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        net.cuda()

    test_losses = []
    num_correct = 0

    h = net.init_hidden(batch_size)

    net.eval()

    for inputs, labels in test_loader:

        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        h = tuple([each.data for each in h])

        output, h = net(inputs, h)

        test_loss = criterion(output.squeeze(), labels.float())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())

        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = (
            np.squeeze(correct_tensor.numpy())
            if not train_on_gpu
            else np.squeeze(correct_tensor.cpu().numpy())
        )

        num_correct += np.sum(correct)

        test_losses.append(test_loss)
    # stats
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))
