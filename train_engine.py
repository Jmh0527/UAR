import time
import torch
import torch.nn as nn


class Trainer(nn.Module):
    """
    Template method for training a network
    """

    def __init__(self, model, dataloader, epoch=1, optimizer, loss_fn, scheduler=None, 
                 writer=None, save_dir=None, loss_freq=100, save_freq=500):
        """
        Args:
            model: The network to be trained.
            dataloader: User-defined dataloader.
            epoch: Total number of training epochs.
            optimizer: The torch optimizer function.
            loss_fn: The torch loss function.
            scheduler: (Optional) The torch scheduler function.
            writer: (Optional) The TensorBoard writer object.
            save_dir: (Optional) Path for saving checkpoints.
            loss_freq: Frequency of logging the loss.
            save_freq: Frequency of saving the model.
        """

        super(Trainer, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.epoch = epoch
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.writer = writer
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

        self.loss = self.loss_fn(self.output, self.label)

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

        print(f"Starting training for {self.epoch} epochs...")
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
                if self.writer and self.total_steps % self.loss_freq == 0:
                    print(f"Step {self.total_steps}, Loss: {self.loss.item():.4f}")
                    self.writer.add_scalar('loss', self.loss.item(), self.total_steps)

                # Save model periodically
                if self.save_dir and self.total_steps % self.save_freq == 0:
                    print(f"Saving model at step {self.total_steps}")
                    self.save_networks("latest")

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

        if self.save_dir:
            save_path = f"{self.save_dir}/{label}_model.pth"
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
