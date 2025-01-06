import time
import logging
from pathlib import Path

import torch
import torch.nn as nn


class Trainer(nn.Module):
    """
    Template method for training a network
    """

    def __init__(self, model, dataloader, optimizer, loss_fn, epoch=10, scheduler=None, 
                 logger=None, save_dir=None, loss_freq=20, save_freq=1):
        """
        Args:
            model: The network to be trained.
            dataloader: User-defined dataloader.
            epoch: Total number of training epochs.
            optimizer: The torch optimizer function.
            loss_fn: The torch loss function.
            scheduler: (Optional) The torch scheduler function.
            logger: (Optional) Logger for recording training progress.
            save_dir: (Optional) Path for saving checkpoints.
            loss_freq: Frequency of logging the loss.
            save_freq: Epoch frequency of saving the model.
        """

        super(Trainer, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.epoch = epoch
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.logger = logger
        self.save_dir = save_dir
        self.loss_freq = loss_freq
        self.save_freq = save_freq
        self.total_steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._forward_hook = None

        # Move model to device
        self.model = self.model.to(self.device)

    def set_input(self, data):
        """
        Set the input data for the model.
        """
        if type(self.model).__name__ == 'PatchCraft':
            # data[0] is a list containing two tensor
            self.input = [item.to(self.device) for item in data[0]]
        elif type(self.model).__name__ == 'AIMClassifier':
            self.input = data[0].to(self.device)
            
        self.label = data[1].to(self.device)

    def forward(self):
        """
        Forward pass of the model, with optional hook.
        """

        if self.forward_hook:
            self.output = self.forward_hook(self.input)
        else:
            self.output = self.model(self.input)

    @property
    def forward_hook(self):
        return self._forward_hook

    @forward_hook.setter
    def forward_hook(self, hook):
        self._forward_hook = hook

    def compute_loss(self):
        """
        Compute the loss function.
        """
        self.loss = self.loss_fn(self.output.squeeze(), self.label)

    def optimize_parameters(self):
        """
        Update model parameters using the optimizer.
        """

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def train(self):
        """
        Template method for the training loop.
        """

        self.logger.info(f"Starting training for {self.epoch} epochs...")
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            print(f"Epoch {epoch + 1}/{self.epoch}:")
            epoch_loss = 0.0

            for i, data in enumerate(self.dataloader):
                self.total_steps += 1
                self.set_input(data)
                self.forward()
                self.compute_loss()
                self.optimize_parameters()

                # Accumulate epoch loss for reporting
                epoch_loss += self.loss.item()

                # Log loss periodically
                if self.total_steps % self.loss_freq == 0:
                    self.logger.info(f"Step {self.total_steps}, Loss: {self.loss.item():.4f}")

            # Save model periodically
            if self.save_dir and epoch % self.save_freq == 0:
                print(f"Saving model at step {self.total_steps}")
                self.save_networks(f"epoch_{epoch}")

            # Scheduler step at the end of the epoch
            if self.scheduler:
                self.scheduler.step()

            # Print epoch loss
            print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(self.dataloader):.4f}")
            print(f"Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.2f} seconds.")

        print("Training completed.")

    def save_networks(self, label):
        """
        Save model parameters.
        """
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.save_dir:
            save_path = f"{self.save_dir}/{label}_model.pth"
            torch.save(self.model.state_dict(), save_path)
            self.logger.info(f"Model saved to {save_path}")
