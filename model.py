import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Simple MLP network for Q-value approximation."""
    def __init__(self, n_observations, n_actions):
        """
        Args:
            n_observations (int): Dimension of the observation space.
            n_actions (int): Dimension of the action space.
        """
        super(QNetwork, self).__init__()
        # Define the network layers
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """Passes a state batch through the network."""
        # Use ReLU activation functions
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # Output layer (no activation, as these are Q-values)
        return self.layer3(x)