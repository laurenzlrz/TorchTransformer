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

    @abstractmethod
    def train_one_epoch(self):
        """
        Train the model for one epoch. Must be implemented by subclasses.
        """
        pass

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
