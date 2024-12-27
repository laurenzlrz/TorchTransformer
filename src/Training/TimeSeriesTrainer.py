from src.Training.AbstractTrainer import AbstractTrainer
import torch.nn as nn


class TimeSeriesTrainer(AbstractTrainer):
    def __init__(self, model, optimizer, train_loader, val_loader=None, device='cpu', loss_fn=nn.MSELoss()):
        super().__init__(model, optimizer, loss_fn, train_loader, val_loader, device)

    def train_one_epoch(self):
        epoch_loss = 0.0

        # Initialize metric storage if logging is enabled
        if self.logging:
            epoch_metrics = {metric.__class__.__name__: 0.0 for metric in self.logging_metrics}

        # Iterate through the dataset
        for X, Y in self.train_loader:
            X, Y = X.to(self.device), Y.to(self.device)
            Y = Y[:, :, 0] #for testin purposes
            predictions = self.model(X)
            loss = self.loss_fn(predictions, Y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update epoch loss
            epoch_loss += loss.detach()

            # Accumulate metrics if logging is enabled
            if self.logging:
                for metric in self.logging_metrics:
                    # Assuming metric is callable and takes (predictions, Y)
                    epoch_metrics[metric.__class__.__name__] += metric(predictions, Y).item()

        # Compute average loss for the epoch
        avg_loss = epoch_loss / len(self.train_loader)

        # Finalize metrics for the epoch
        if self.logging:
            for metric in epoch_metrics:
                epoch_metrics[metric] /= len(self.train_loader)
                self.log['train'][metric].append(epoch_metrics[metric])

        # Return average loss
        return avg_loss

    def validate(self):
        epoch_loss = 0.0

        # Initialize metric storage if logging is enabled
        if self.logging:
            epoch_metrics = {metric.__class__.__name__: 0.0 for metric in self.logging_metrics}

        # Iterate through the dataset
        for X, Y in self.val_loader:
            X, Y = X.to(self.device), Y.to(self.device)
            predictions = self.model(X)
            loss = self.loss_fn(predictions, Y)

            # Update epoch loss
            epoch_loss += loss.detach()

            # Accumulate metrics if logging is enabled
            if self.logging:
                for metric in self.logging_metrics:
                    # Assuming metric is callable and takes (predictions, Y)
                    epoch_metrics[metric.__class__.__name__] += metric(predictions, Y).item()

        # Compute average loss for the epoch
        avg_loss = epoch_loss / len(self.train_loader)

        # Finalize metrics for the epoch
        if self.logging:
            for metric in epoch_metrics:
                epoch_metrics[metric] /= len(self.train_loader)
                self.log['validation'][metric].append(epoch_metrics[metric])

        # Return average loss
        return avg_loss
