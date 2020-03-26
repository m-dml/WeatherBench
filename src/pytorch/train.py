import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy


def calc_val_loss(validation_loader, model_forward, device):
    val_loss = 0.0
    with torch.no_grad():
        nb = 0
        for batch in validation_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            val_loss += F.mse_loss(model_forward(inputs), targets)
            nb += 1
    return val_loss / nb


def train_model(model, train_loader, validation_loader, device, model_forward,
                max_epochs=200, max_patience=20, verbose=True):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss, patience = np.inf, max_patience
    best_state_dict = {}

    n_batches = len(train_loader) // train_loader.batch_size
    if not train_loader.drop_last and len(train_loader) > n_batches * train_loader.batch_size:
        n_batches += 1

    training_loss = []  # list (over epochs) of numpy arrays (over minibatches)
    validation_loss = np.zeros(0, dtype=np.float32)

    epoch = 0
    while True:

        epoch += 1
        if epoch > max_epochs:
            break

        if verbose:
            print(f'epoch #{epoch}')

        training_loss.append(np.full(n_batches, np.nan, dtype=np.float32))

        # Train for a single epoch.
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = batch[0].to(device), batch[1].to(device)
            loss = F.mse_loss(model_forward(inputs), targets)
            loss.backward()
            optimizer.step()

            training_loss[epoch - 1][batch_index] = loss.item()  # record training loss
            assert np.isfinite(training_loss[-1])

        # Track convergence on validation set
        validation_loss = np.append(validation_loss, calc_val_loss(validation_loader, model_forward, device))

        if verbose:
            print(f'epoch #{epoch} || loss (last batch) {loss} || validation loss {validation_loss[-1]}')

        if validation_loss[-1] < best_loss:
            patience = max_patience
            best_loss = validation_loss[-1]
            best_state_dict = deepcopy(model.state_dict()) # during early training will save every epoch
        else:
            patience -= 1

        if patience == 0:
            break

    model.load_state_dict(best_state_dict)
    return dict(training_loss=training_loss, validation_loss=validation_loss)