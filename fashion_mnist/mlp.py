import torch
from torch import nn
from torch.utils.data import DataLoader


class FashionMLP():
    """A model made for the Fashion MNIST data set.
    """

    def __init__(self, hidden_dims: tuple[int, int]) -> None:
        """Initialize model.

        Args:
            hidden_dims (tuple[int, int]): Number of neurons for each hidden layer.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MultiLayerPerceptron(hidden_dims)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, train_data: DataLoader, val_data: DataLoader, epochs: int) -> dict:
        """Train model.

        Args:
            train_data (DataLoader): Training set.
            val_data (DataLoader): Hold-out validation set.
            epochs (int): Number of epochs.

        Returns:
            dict: Training loss, validation loss, and validation accuracy for each epoch.
        """
        history = {"epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": []}

        for t in range(epochs):
            # Training loop
            self.model.train()
            num_batches = len(train_data)
            train_loss = 0
            for X, y in train_data:
                X, y = X.to(self.device), y.to(self.device)

                # Forward pass
                logits = self.model(X)
                probs = nn.Softmax(dim=1)(logits)
                loss = self.criterion(probs, y)
                train_loss += loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Compute average train loss over all batches
            train_loss /= num_batches

            # Compute validation loss and accuracy
            val_loss, val_accuracy = self.test(val_data)

            print(f"Epoch {t+1:>2d}/{epochs}: {train_loss=:.6f}, {val_loss=:.6f}, {val_accuracy=:.4f}")

            # Save epoch stats
            history["epoch"].append(t+1)
            history["train_loss"].append(train_loss.item())
            history["val_loss"].append(val_loss.item())
            history["val_accuracy"].append(val_accuracy.item())

        return history

    def test(self, data: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and accuracy for a given set of data.

        Args:
            data (DataLoader): Inputs, targets.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Loss and accuracy. 
        """
        self.model.eval()
        loss, correct = 0, 0
        with torch.no_grad():
            for X, y in data:
                X, y = X.to(self.device), y.to(self.device)

                logits = self.model(X)
                probs = nn.Softmax(dim=1)(logits)
                pred = torch.argmax(probs, dim=1)

                loss += self.criterion(probs, y)
                correct += (pred == y).type(torch.float).sum()

        loss /= len(data)
        accuracy = correct / len(data.dataset)

        return loss, accuracy

    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Make predictions for the given input features.

        Args:
            X (torch.Tensor): Features.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Prediction of classes and the corresponding probability distributions.
        """
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            logits = self.model(X)
            probs = nn.Softmax(dim=1)(logits)
            pred = torch.argmax(probs, dim=1)

        return pred, probs


class MultiLayerPerceptron(nn.Module):
    """A fully-connected feedforward neural network with two hidden layer.
    """

    def __init__(self, hidden_dims: tuple[int, int]) -> None:
        """Initialize MLP.

        Args:
            hidden_dims (tuple[int, int]): Number of neurons for each hidden layer.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make a single forward pass for a given input.

        Args:
            x (torch.Tensor): Input features

        Returns:
            torch.Tensor: Logits
        """
        return self.layers(x)
