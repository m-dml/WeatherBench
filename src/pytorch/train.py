import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy


def train_model(model, train_loader, validation_loader, device, model_forward,
                results_dir, model_filename,
                verbose=True):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs, max_patience = 200, 20
    best_loss, patience = np.inf, max_patience
    best_state_dict = {}

    epoch = 0
    while True:

        epoch += 1
        if epoch > n_epochs:
            break

        if verbose:
            print(f'epoch #{epoch}')

        # Train for a single epoch.
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch[0].to(device), batch[1].to(device)
            loss = F.mse_loss(model_forward(inputs), targets)
            print('computed minibatch loss')
            loss.backward()
            optimizer.step()

            import pdb
            pdb.set_trace()

        # Track convergence on validation set.
        val_loss = 0
        with torch.no_grad():
            nb = 0
            for batch in validation_loader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                val_loss += F.mse_loss(model_forward(inputs), targets)
                nb += 1
        val_loss /= nb
        print(f'epoch #{epoch} || loss (last batch) {loss} || validation loss {val_loss}')

        if val_loss < best_loss:
            patience = max_patience
            best_loss = val_loss
            best_state_dict = deepcopy(model.state_dict()) # during early training will save every epoch
        else:
            patience -= 1

        if patience == 0:
            model.load_state_dict(best_state_dict)
            break

    torch.save(model.state_dict(), results_dir + model_filename)