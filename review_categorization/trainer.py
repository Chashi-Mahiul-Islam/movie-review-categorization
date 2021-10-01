from torch import nn, optim
import torch
import numpy as np


def train(net, train_loader, val_loader, batch_size):
    train_on_gpu = torch.cuda.is_available()
    lr = 0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    """ TRIANING AND VALIDATION """
    valid_loss_min = np.Inf

    epochs = 4  # converges in 4
    counter = 0
    print_every = 100
    clip = 5  # gradient clipping

    if train_on_gpu:
        net.cuda()

    net.train()

    # train for some number of epochs
    for e in range(epochs):
        # initialize the hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # creating new variables for the hidden state, otherwise
            # we'd backdrop through the entire training history
            h = tuple([each.data for each in h])

            optimizer.zero_grad()

            output, h = net(inputs, h)

            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            if counter % print_every == 0:
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in val_loader:

                    if train_on_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                    net.train()
                print(
                    "Epoch: {}/{}...".format(e + 1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)),
                )
                if np.mean(val_losses) <= valid_loss_min:
                    print(
                        "Validation loss decreased ({:.6f} --> {:.6f}). Saving Model ...".format(
                            valid_loss_min, val_loss
                        )
                    )
                    torch.save(
                        net.state_dict(), "lstm_model_movie_review_categorization.pt"
                    )
                    valid_loss_min = np.mean(val_losses)
