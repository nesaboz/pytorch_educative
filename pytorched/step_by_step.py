import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import datetime
from torch.utils.tensorboard import SummaryWriter
import random

RUNS_FOLDER_NAME = 'runs'


class StepByStep(object):
    """
    Main class for training neural network.

    Args:
        model:
        optimizer:
        loss_fn:
        device:

    """

    def __init__(self, model, optimizer, loss_fn):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None
        self.writer = None

        self.total_epochs = 0

        self.losses = []
        self.val_losses = []

        self.pera = 0

    def __str__(self):
        return f"{self.model}, {self.optimizer}, {self.loss_fn}"

    __repr__ = __str__


    def set_loaders(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def perform_train_step(self, batch_x, batch_y):
        self.model.train()

        predictions = self.model(batch_x)
        loss = self.loss_fn(predictions, batch_y)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def perform_val_step(self, batch_x, batch_y):
        self.model.eval()

        predictions = self.model(batch_x)
        loss = self.loss_fn(predictions, batch_y)

        return loss.item()

    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        self.device = device
        self.model.to(self.device)

    def set_tensorboard(self, name, folder=RUNS_FOLDER_NAME):
        # This method allows the user to define a SummaryWriter to interface with TensorBoard
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def train(self, n_epochs):

        self.set_seed()

        for epoch in range(n_epochs):
            self.total_epochs += 1
            train_losses = []
            for batch_x, batch_y in self.train_loader:
                train_losses.append(self.perform_train_step(batch_x, batch_y))
            mini_batch_loss = np.mean(train_losses)
            self.losses.append(mini_batch_loss)

            with torch.no_grad():
                val_losses = []
                for batch_x, batch_y in self.val_loader:
                    val_losses.append(self.perform_val_step(batch_x, batch_y))
                mini_batch_val_loss = np.mean(val_losses)
                self.val_losses.append(mini_batch_val_loss)

            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training': mini_batch_loss}
                if mini_batch_val_loss is not None:
                    scalars.update({'validation': mini_batch_val_loss})
                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)

    def predict(self, x):
        self.model.eval()
        # need to evaluate
        prediction = self.model(torch.tensor(x).float().to(self.device))
        self.model.train()
        return prediction.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training loss', c='b')
        plt.plot(self.val_losses, label='Validation loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_checkpoint(self, filename):
        checkpoint = {
            'total_epochs': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'val_losses': self.val_losses,
            }

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)

        self.total_epochs = checkpoint['total_epochs']

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.losses = checkpoint['losses']
        self.val_losses = checkpoint['val_losses']

        self.model.train()

    def correct(self, x, y, threshold=.5):
        self.model.eval()
        yhat = self.model(x.to(self.device))
        y = y.to(self.device)
        self.model.train()

        # We get the size of the batch and the number of classes
        # (only 1, if it is binary)
        n_samples, n_dims = yhat.shape
        if n_dims > 1:
            # In a multiclass classification, the biggest logit
            # always wins, so we don't bother getting probabilities

            # This is PyTorch's version of argmax,
            # but it returns a tuple: (max value, index of max value)
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims += 1
            # In binary classification, if last layer is not Sigmoid we need to apply one:
            if not (isinstance(self.model, nn.Sequential) and isinstance(self.model[-1], nn.Sigmoid)):
                yhat = torch.sigmoid(yhat)
            predicted = (yhat > threshold).long()

        # How many samples got classified correctly for each class
        result = []
        for c in range(n_dims):
            n_class = (y == c).sum().item()
            n_correct = (predicted[y == c] == c).sum().item()
            result.append((n_correct, n_class))
        return torch.tensor(result)

    @staticmethod
    def loader_apply(loader, func, reduce='sum'):
        results = [func(x, y) for i, (x, y) in enumerate(loader)]
        results = torch.stack(results, axis=0)

        if reduce == 'sum':
            results = results.sum(axis=0)
        elif reduce == 'mean':
            results = results.float().mean(axis=0)

        return results

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def reset_parameters(self):
        """
        CAUTION this does not reset all parameters like in nn.PReLU for example.
        Based on https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        """
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
