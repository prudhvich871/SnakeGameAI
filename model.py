import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    """A neural network model for Q-learning in the Snake game AI.

    The Linear_QNet class implements a simple feedforward neural network for predicting optimal actions in the game environment.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initialize the neural network with specified layer dimensions.

        Creates a two-layer neural network with ReLU activation for Q-value prediction.

        Args:
            input_dim: Number of input features.
            hidden_dim: Number of neurons in the hidden layer.
            output_dim: Number of possible actions.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_tensor):
        """Perform a forward pass through the neural network.

        Computes the Q-values for the given input state.

        Args:
            input_tensor: The input state tensor.

        Returns:
            Predicted Q-values for each possible action.
        """
        hidden = F.relu(self.fc1(input_tensor))
        output = self.fc2(hidden)
        return output

    def save(self, filename="model.pth"):
        """Save the trained neural network model to a file.

        Stores the model's state dictionary in a specified directory.

        Args:
            filename: Name of the file to save the model. Defaults to 'model.pth'.
        """
        model_dir = "./model"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filepath = os.path.join(model_dir, filename)
        torch.save(self.state_dict(), filepath)


class QTrainer:
    """A Q-learning trainer for neural network-based reinforcement learning.

    The QTrainer class manages the training process for a neural network agent, implementing Q-value updates and optimization.
    """

    def __init__(self, model, learning_rate, discount_factor):
        """Initialize the Q-learning trainer with specified hyperparameters.

        Sets up the neural network model, optimizer, and loss function for training.

        Args:
            model: The neural network model to be trained.
            learning_rate: The learning rate for the optimization algorithm.
            discount_factor: The discount factor for future rewards.
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """Perform a single training step using Q-learning algorithm.

        Updates the neural network's Q-values based on the current experience and predicted future rewards.

        Args:
            state: The current game state.
            action: The action taken.
            reward: The reward received.
            next_state: The resulting state after the action.
            done: Whether the game episode has ended.
        """
        state_tensor = torch.tensor(state, dtype=torch.float)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float)
        action_tensor = torch.tensor(action, dtype=torch.long)
        reward_tensor = torch.tensor(reward, dtype=torch.float)

        if len(state_tensor.shape) == 1:
            state_tensor = torch.unsqueeze(state_tensor, 0)
            next_state_tensor = torch.unsqueeze(next_state_tensor, 0)
            action_tensor = torch.unsqueeze(action_tensor, 0)
            reward_tensor = torch.unsqueeze(reward_tensor, 0)
            done = (done,)

        predicted_q_values = self.model(state_tensor)

        target_q_values = predicted_q_values.clone()
        for idx in range(len(done)):
            new_q_value = reward_tensor[idx]
            if not done[idx]:
                new_q_value = reward_tensor[idx] + self.discount_factor * torch.max(
                    self.model(next_state_tensor[idx])
                )

            target_q_values[idx][torch.argmax(action_tensor[idx]).item()] = new_q_value

        self.optimizer.zero_grad()
        loss = self.criterion(target_q_values, predicted_q_values)
        loss.backward()

        self.optimizer.step()
