import numpy as np
import matplotlib.pyplot as plt
import torch


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

        self.total_epochs = 0

        self.losses = []
        self.val_losses = []

    def __str__(self):
        return f"{self.model}, {self.optimizer}, {self.loss_fn}"

    __repr__ = __str__

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

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

    def train(self, n_epochs):

        self.set_seed()

        for epoch in range(n_epochs):
            self.total_epochs += 1
            train_losses = []
            for batch_x, batch_y in self.train_loader:
                train_losses.append(self.perform_train_step(batch_x, batch_y))
            self.losses.append(np.mean(train_losses))

            with torch.no_grad():
                val_losses = []
                for batch_x, batch_y in self.val_loader:
                    val_losses.append(self.perform_val_step(batch_x, batch_y))
                self.val_losses.append(np.mean(val_losses))

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
