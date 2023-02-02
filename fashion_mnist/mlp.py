import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class FashionMLP():
    """A model made for the Fashion MNIST data set.
    """

    def __init__(self, hidden_dims: tuple[int, int], lr: float = 1e-3) -> None:
        """Initialize model.

        Args:
            hidden_dims (tuple[int, int]): Number of neurons for each hidden layer.
            lr (float, optional): Learnign rate for the Adam optimizer. Defaults to 1e-3.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MultiLayerPerceptron(hidden_dims)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def train(self, data: Dataset, epochs: int = 10, val_split: float = 0.0, batch_size: int = 128) -> dict:
        """Train model.
        Args:
            data (Dataset): Data set with inputs and targets.
            epochs (int, optional): Number of epochs. Defaults to 10.
            val_split (float, optional): Percentage of data to set aside for validation. Defaults to 0.0.
            batch_size (int, optional): How many samples to load per batch. Defaults to 128.

        Returns:
            dict: Training loss/accuracy, validation loss/accuracy for each epoch.
        """
        history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        # Seperate validation set and create dataloader
        if val_split > 0:
            train_data, val_data = random_split(data, [1-val_split, val_split])
            train_loader = DataLoader(train_data, batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size)
        else:
            train_loader = DataLoader(data, batch_size, shuffle=True)

        for t in range(epochs):
            # Training loop
            self.model.train()
            train_loss, correct = 0.0, 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)

                # Forward pass
                logits = self.model(X)
                probs = nn.Softmax(dim=1)(logits)
                loss = self.criterion(probs, y)
                train_loss += loss
                correct += (torch.argmax(probs, dim=1) == y).sum()

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Compute train loss (average) and accuracy over all batches
            train_loss /= len(train_loader)
            train_acc = correct / len(train_loader.dataset)

            # Compute validation loss and accuracy
            if val_split > 0:
                val_loss, val_acc = self.validate(val_loader)
            else:
                val_loss, val_acc = torch.tensor(0.0), torch.tensor(0.0)

            # Save stats
            history["epoch"].append(t+1)
            history["train_loss"].append(train_loss.item())
            history["train_acc"].append(train_acc.item())
            history["val_loss"].append(val_loss.item())
            history["val_acc"].append(val_acc.item())

            print(f"Epoch {t+1:>2d}/{epochs}: {train_loss=:.6f}, {train_acc=:.4f}, {val_loss=:.6f}, {val_acc=:.4f}")

        return history

    def validate(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and accuracy for a given set of data.

        Args:
            data (DataLoader): Inputs, targets.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Loss and accuracy. 
        """
        self.model.eval()
        loss, correct = 0.0, 0.0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)

                logits = self.model(X)
                probs = nn.Softmax(dim=1)(logits)

                loss += self.criterion(probs, y)
                correct += (torch.argmax(probs, dim=1) == y).sum()

        loss /= len(loader)
        accuracy = correct / len(loader.dataset)

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
