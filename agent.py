import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, GridPoint
from model import Linear_QNet, QTrainer
from helper import plot

# Define constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:

    def __init__(self):
        """Initialize a reinforcement learning agent for the Snake game.

        Sets up the agent's learning parameters, neural network model, and experience replay memory for training.
        """

        # Initialize the agent
        self.num_games = 0
        self.epsilon = 0  # randomness
        self.discount_factor = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() if memory exceeds MAX_MEMORY
        self.model = Linear_QNet(11, 256, 3)  # Neural network model
        self.trainer = QTrainer(
            self.model,
            learning_rate=LEARNING_RATE,
            discount_factor=self.discount_factor,
        )

    def get_state(self, game):
        """Construct a comprehensive state representation of the current game environment.

            Generates a feature vector capturing snake position, movement direction, potential dangers, and food location.

        Args:
            game: The current Snake game instance.

        Returns:
            A numpy array representing the game state with binary features.
        """

        # Get the current state of the game
        head = game.snake[0]
        point_left = GridPoint(head.x - 20, head.y)
        point_right = GridPoint(head.x + 20, head.y)
        point_up = GridPoint(head.x, head.y - 20)
        point_down = GridPoint(head.x, head.y + 20)

        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (direction_right and game.is_collision(point_right))
            or (direction_left and game.is_collision(point_left))
            or (direction_up and game.is_collision(point_up))
            or (direction_down and game.is_collision(point_down)),
            # Danger right
            (direction_up and game.is_collision(point_right))
            or (direction_down and game.is_collision(point_left))
            or (direction_left and game.is_collision(point_up))
            or (direction_right and game.is_collision(point_down)),
            # Danger left
            (direction_down and game.is_collision(point_right))
            or (direction_up and game.is_collision(point_left))
            or (direction_right and game.is_collision(point_up))
            or (direction_left and game.is_collision(point_down)),
            # Move direction
            direction_left,
            direction_right,
            direction_up,
            direction_down,
            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Store a game experience in the agent's memory for future learning.

        Logs the current game interaction to enable batch training and experience replay.

        Args:
            state: The current game state.
            action: The action taken.
            reward: The reward received.
            next_state: The resulting game state after the action.
            done: Whether the game episode has ended.
        """

        # Store the experience in memory
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        """Perform batch training on accumulated game experiences.

        Randomly samples experiences from memory and trains the neural network to improve decision-making using experience replay.

        Trains the model using either a full batch or the entire memory if fewer experiences are available.
        """

        # Train the model using a batch of experiences from memory
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train the model using a single experience
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Decide on an action based on the current state
        self.epsilon = 80 - self.num_games  # Decrease randomness over time
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            # Perform a random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Predict the best move using the model
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    """Train an AI agent to play the Snake game using reinforcement learning.

    Manages the training loop for the Snake game AI, including state tracking, action selection, experience replay, and performance monitoring.
    """

    # Train the agent
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record_score = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get the old state
        old_state = agent.get_state(game)

        # Get the move
        final_move = agent.get_action(old_state)

        # Perform the move and get the new state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(old_state, final_move, reward, new_state, done)

        # Remember the experience
        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            # Train long memory and plot the result
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record_score:
                record_score = score
                agent.model.save()

            print("Game", agent.num_games, "Score", score, "Record:", record_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
