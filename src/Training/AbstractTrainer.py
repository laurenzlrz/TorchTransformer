from abc import ABC, abstractmethod
import torch

class AbstractTrainer(ABC):
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader=None, device='cpu'):
        """
        Initialize the trainer with common components.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model.to(self.device)
        self.logging = False
        self.logging_metrics = [loss_fn]
        self.log = {loss_fn.__class__.__name__: []}

    @abstractmethod
    def train_one_epoch(self):
        """
        Train the model for one epoch. Must be implemented by subclasses.
        """
        pass

    def train(self, epochs):

        for epoch in range(epochs):
            training_loss = self.train_one_epoch()

            if self.val_loader is not None:
                val_loss = self.validate()
                print(f'Epoch: {epoch} || Training Loss: {training_loss:.4f} || Validation Loss: {val_loss:.4f}')
            else:
                print(f'Epoch: {epoch} || Training Loss: {training_loss:.4f}')

        return self.model

    def toggle_logging(self):
        """
        Toggles the logging. If Ture additional metrics are calculated and saved during training
        """
        self.logging = not self.logging

    def add_logging_metric(self, metric):
        """
        Adds additional logging metrics to the logging.
        """
        self.logging_metrics.append(metric)
        self.log[metric.__class__.__name__] = []
    @abstractmethod
    def validate(self):
        """
        Validate the model. Must be implemented by subclasses.
        """
        pass

    def save_checkpoint(self, path):
        """
        Save model checkpoint.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved at {path}")

    def load_checkpoint(self, path):
        """
        Load model checkpoint.
        """
        self.model.load_state_dict(torch.load(path))
        print(f"Checkpoint loaded from {path}")
