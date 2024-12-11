# Building an AI to Play Snake Using Reinforcement Learning

## Problem Description

We aim to develop an AI agent capable of playing the classic Snake game using reinforcement learning (RL). In Snake, the player controls a snake that grows longer as it consumes food while avoiding collisions with walls or itself. The objective is to maximize the score by eating as much food as possible before crashing.

The challenge lies in teaching the agent to determine the optimal sequence of actions that maximize its score. Each move alters the game state, making it a dynamic and complex problem to solve.

---

## Challenges and Uncertainties

### Observability

The game environment can be considered **fully observable**, as the agent has access to all relevant information such as the snake’s position, direction, food location, and obstacles. However, effectively representing this information in the state space is a non-trivial task.

### Complexity

- **Dynamic Environment:** As the snake grows longer, it occupies more of the grid, increasing the risk of collision and limiting movement options.
- **Sequential Decision-Making:** The agent must account for future consequences of its current actions.

### Exploration vs. Exploitation

Balancing exploration (trying new actions) and exploitation (using known strategies) is critical. An epsilon-greedy policy will help manage this tradeoff.

---

## Motivation

The goal is to develop an agent that learns optimal strategies for maximizing the score in Snake. This project serves as a practical application of reinforcement learning concepts, offering insights into dynamic decision-making and state-action representation.

---

## State Space and Action Space

### State Representation

The state is represented as an 11-dimensional vector capturing:

- Danger in three directions: straight, right, and left.
- Current direction: left, right, up, down.
- Relative position of the food: left, right, up, down.

Example:

```
[danger_straight, danger_right, danger_left,
 direction_left, direction_right, direction_up, direction_down,
 food_left, food_right, food_up, food_down]
```

### Action Space

The agent can perform one of four actions:

- Move **left**
- Move **right**
- Move **up**
- Move **down**

---

## Algorithm: Q-Learning

### Rewards

The agent receives rewards based on its actions:

- **+10** for eating food.
- **-10** for colliding with walls or itself.
- **0** otherwise.

### Exploration-Exploitation Strategy

We use an epsilon-greedy policy:

- With probability ε, the agent explores by selecting a random action.
- Otherwise, it exploits by selecting the action with the highest Q-value for the current state.

### Bellman Equation

The Bellman equation updates the Q-values:

```
Q_new(s, a) = Q(s, a) + α [R(s, a) + γ max Q(s', a') - Q(s, a)]
```

Where:

- α: Learning rate
- γ: Discount factor
- R(s, a): Reward for taking action `a` in state `s`
- max Q(s', a'): Maximum Q-value for the next state

---

## Implementation

### Game Environment

- **Library:** Pygame is used to simulate the Snake game.
- **Core Method:** `play_step(action)` updates the game state and returns:
  - Reward
  - Game Over flag
  - Current Score

### Model

We use a deep Q-network (DQN) implemented in PyTorch:

- **Architecture:**
  - Input layer: 11 neurons (state representation)
  - Hidden layers: Fully connected
  - Output layer: 3 neurons (Q-values for left, right, straight actions)
- **Loss Function:** Mean Squared Error (MSE) between predicted and target Q-values.

### General Structure of the Code

The project follows the structure outlined below:

1. **Game Class:**

   - Initializes the Snake game environment.
   - Includes methods such as `reset()` and `play_step(action)`.

2. **Agent Class:**

   - Maintains the RL logic and manages the training loop.
   - Defines `get_state()`, `remember()`, and `train()` functions.

3. **Model Class:**

   - Implements the neural network using PyTorch.
   - Contains methods for forward propagation (`predict`) and model updates (`train`).

4. **Training Script:**

   - Combines the game, agent, and model into a cohesive training loop.
   - Handles hyperparameter tuning, logging, and performance evaluation.

---

## Training Loop

1. **State Initialization:** Retrieve the initial state.
2. **Action Selection:** Use the epsilon-greedy policy to pick an action.
3. **Environment Update:** Perform the action in the game environment.
4. **Reward and State Transition:** Record the reward and transition to the next state.
5. **Experience Replay:** Store the experience (state, action, reward, next state) in memory.
6. **Model Update:** Train the DQN using the stored experiences.

---

## Results and Evaluation

### Success Metrics

- **Score Improvement:** Measure the average score achieved by the agent over multiple games.
- **Convergence:** Observe the stability of Q-values and policy over training episodes.

### Visualizations

Plots of reward trends, exploration rates, and Q-value convergence will help assess performance.

---

## Deliverables

1. **Code Repository:** Includes the game environment, RL algorithm, and trained model.
2. **Report:** This blog post serves as a comprehensive summary of the project.
3. **Trained Model:** The Q-table or DQN model for playing Snake autonomously

---

## Reproducibility

### Software and Hardware Requirements

- **Software:** Python 3.8+, PyTorch, Pygame
- **Hardware:** A machine with at least 8 GB RAM and a GPU (optional but recommended for faster training)

### Steps to Reproduce

1. **Set Up a Virtual Environment:**

   ```bash
   conda create -n pygame_env python=3.8
   conda activate pygame_env
   ```

2. **Install Required Libraries:**

   ```bash
   pip install pygame
   pip install torch torchvision
   pip install matplotlib ipython
   ```

3. **Clone the Repository:**

   ```bash
   git clone https://github.com/prudhvich871/SnakeGameAI
   cd SnakeGameAI
   ```

4. **Run the Training Script:**

   ```bash
   python agent.py
   ```

### Data Sources

- No external data sources are required. The game environment generates all states dynamically during training.

### Detailed Code Comments

Each function and class in the repository includes comments explaining its purpose and functionality. The repository README file provides additional context for setting up and running the project.

### Workflow Summary

1. **Setup Environment:** Prepare the game, agent, and model.
2. **Training Loop:** Train the agent with experience replay and Q-learning.
3. **Evaluation:** Test the model using unseen game episodes.
4. **Visualization:** Plot metrics to evaluate learning progress.
5. **Reproducibility:** Ensure all scripts, configurations, and results are well-documented for easy replication.

---

## Sequence Diagrams

### Linear Q Model Interaction
# Linear QNet UML
![Linear QNet Model UML](https://github.com/prudhvich871/SnakeGameAI/blob/main/diagrams/LinearQtrainerUML.png)

# Linear QNet Sequence
![Linear QNet Sequence](https://github.com/prudhvich871/SnakeGameAI/blob/main/diagrams/linearQSequence.png)


This diagram shows how the agent interacts with the Linear_QNet and QTrainer classes, observing the current state, predicting Q-values, and updating neural network weights.

### Game Logic and Pygame Integration
# Game UML
![Game Logic UML](https://github.com/prudhvich871/SnakeGameAI/blob/main/diagrams/SnakeGameAIUML.png)

# Game Sequence
![Game Logic Sequence](https://github.com/prudhvich871/SnakeGameAI/blob/main/diagrams/SnakeGameAISequence.png)

This sequence diagram illustrates the game logic, including state updates, collision checks, and score tracking.

### Agent Code Interaction
# Agent UML
![Agent UML](https://github.com/prudhvich871/SnakeGameAI/blob/main/diagrams/AgentUML.png)

# Agent Sequence
![Agent Sequence](https://github.com/prudhvich871/SnakeGameAI/blob/main/diagrams/AgentSequence.png)

## Conclusion

This project combines the classic Snake game with reinforcement learning principles to develop an intelligent agent. By providing clear instructions, a well-structured repository, and detailed documentation, we ensure that others can reproduce and build upon our work.

