import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize Pygame
pygame.init()
# Set font for displaying score
font = pygame.font.Font("arial.ttf", 25)
# font = pygame.font.SysFont('arial', 25)


# Define directions for snake movement
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Define a point in the game grid
GridPoint = namedtuple("GridPoint", "x, y")

# Define RGB colors
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (200, 0, 0)
COLOR_BLUE1 = (0, 0, 255)
COLOR_BLUE2 = (0, 100, 255)
COLOR_BLACK = (0, 0, 0)

# Define block size and game speed
BLOCK_SIZE = 20
GAME_SPEED = 40


class SnakeGameAI:
    """A programmatically controlled Snake game environment for AI and machine learning experiments.

    The SnakeGameAI class provides a complete, customizable game simulation with built-in reward mechanisms and state tracking for reinforcement learning.
    """

    def __init__(self, width=640, height=480):
        """Create a new Snake game with configurable display dimensions.

        Initializes the game window, sets up the display, and prepares the initial game state.

        Args:
            width: Width of the game window in pixels. Defaults to 640.
            height: Height of the game window in pixels. Defaults to 480.
        """
        self.width = width
        self.height = height
        # Initialize display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """
        Restore the game to its initial configuration.
        Resets snake position, direction, score, and prepares a fresh game environment for a new episode.
        """
        # Initialize game state
        self.direction = Direction.RIGHT

        # Initialize snake starting position
        self.head = GridPoint(self.width / 2, self.height / 2)
        self.snake = [
            self.head,
            GridPoint(self.head.x - BLOCK_SIZE, self.head.y),
            GridPoint(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        # Initialize score and food
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        # Place food at a random position on the grid
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = GridPoint(x, y)
        # Ensure food is not placed on the snake
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """Execute a single step of the Snake game based on the provided action.

        Manages game progression by moving the snake, checking for collisions, updating score, and rendering the game state.

        Args:
            action: The directional action to be performed by the snake.

        Returns:
            A tuple containing (reward, game_over status, current score) representing the outcome of the game step.
        """

        self.frame_iteration += 1
        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move the snake
        self._move(action)  # Update the head position
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(GAME_SPEED)
        # 6. Return game over and score
        return reward, game_over, self.score

    def is_collision(self, point=None):
        """Detect potential collision scenarios for the snake in the game environment.

        Checks whether the snake has collided with game boundaries or its own body, which would trigger a game-ending condition.

        Args:
            point: Optional specific point to check for collision. Defaults to snake's head.

        Returns:
            Boolean indicating whether a collision has occurred.
        """

        if point is None:
            point = self.head
        # Check if snake hits the boundary
        if (
            point.x > self.width - BLOCK_SIZE
            or point.x < 0
            or point.y > self.height - BLOCK_SIZE
            or point.y < 0
        ):
            return True
        # Check if snake hits itself
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        """Render the current game state on the display screen.

        Updates the visual representation of the snake, food, and current score, refreshing the game window with the latest game information.
        """

        # Fill the display with black color
        self.display.fill(COLOR_BLACK)

        # Draw the snake
        for point in self.snake:
            pygame.draw.rect(
                self.display,
                COLOR_BLUE1,
                pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE),
            )
            pygame.draw.rect(
                self.display, COLOR_BLUE2, pygame.Rect(point.x + 4, point.y + 4, 12, 12)
            )

        # Draw the food
        pygame.draw.rect(
            self.display,
            COLOR_RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        # Display the score
        text = font.render("Score: " + str(self.score), True, COLOR_WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """Translate the input action into snake movement and direction change.

        Determines the snake's new direction and updates its head position based on the provided action.

        Args:
            action: The movement action to be performed by the snake, represented as a binary action vector.
        """

        # Define the possible directions in clockwise order
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = directions.index(self.direction)

        # Determine the new direction based on the action
        if np.array_equal(action, [1, 0, 0]):
            new_direction = directions[idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_direction = directions[next_idx]  # Right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_direction = directions[next_idx]  # Left turn

        self.direction = new_direction

        # Update the head position based on the new direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = GridPoint(x, y)
