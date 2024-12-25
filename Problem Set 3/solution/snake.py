from typing import Dict, List, Optional, Set, Tuple
from mdp import MarkovDecisionProcess
from environment import Environment
from mathutils import Point, Direction
from helpers.mt19937 import RandomGenerator
from helpers.utils import NotImplemented
import json
from dataclasses import dataclass

##############################################
######### Snake Environment ###################
##############################################

"""
Environment Description:
    The snake is a 2D grid world where the snake can move in 4 directions.
    The snake always starts at the center of the level (floor(W/2), floor(H/2)) having a length of 1 and moving LEFT.
    The snake can wrap around the grid.
    The snake can eat apples which will grow the snake by 1.
    The snake can not eat itself.
    You lose if the snake bites itself (the snake head enters a cell occupied by its body).
    The action can not move the snake in the opposite direction of its current direction.
    The action can not move the snake in the same direction 
        i.e. (if moving right don't give an action saying move right).
    Eating an apple increases the reward by 1.
    Losing the game decreases the reward by 100.
"""

# IMPORTANT: This class will be used to store an observation of the snake environment
@dataclass(frozen=True)
class SnakeObservation:
    snake: Tuple[Point]     # The points occupied by the snake body 
                            # where the head is the first point and the tail is the last  
    direction: Direction    # The direction that the snake is moving towards
    apple: Optional[Point]  # The location of the apple. If the game was already won, apple will be None


class SnakeEnv(Environment[SnakeObservation, Direction]):

    rng: RandomGenerator  # A random generator which will be used to sample apple locations

    snake: List[Point]
    direction: Direction
    apple: Optional[Point]

    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        assert width > 1 or height > 1, "The world must be larger than 1x1"
        self.rng = RandomGenerator()
        self.width = width
        self.height = height
        self.snake = []
        self.direction = Direction.LEFT
        self.apple = self.generate_random_apple()

    def generate_random_apple(self) -> Point:
        """
        Generates and returns a random apple position which is not on a cell occupied 
        by the snake's body.
        """
        snake_positions = set(self.snake)
        possible_points = [Point(x, y) for x in range(self.width) for y in range(
            self.height) if Point(x, y) not in snake_positions]
        return possible_points[self.rng.int(0, len(possible_points)-1)]

    # resets the environment and returns the current state
    def reset(self, seed: Optional[int] = None) -> Point:
        """
        Resets the Snake environment to its initial state and returns the starting state.

        Args:
            seed (Optional[int]): An optional integer seed for the random
            number generator used to generate the game's initial state.

        Returns:
            The starting state of the game, represented as a Point object.
        """
        if seed is not None:
            self.rng.seed(seed)  # Initialize random number generator with the seed
        # Position the snake's head in the center of the grid
        center_x, center_y = self.width // 2, self.height // 2
        self.snake = [Point(center_x, center_y)]  # snake with just the head
        self.direction = Direction.LEFT # initial snake movement
        self.apple = self.generate_random_apple()
        # Return the initial game state
        return SnakeObservation(snake=tuple(self.snake), direction=self.direction, apple=self.apple)

    def actions(self) -> List[Direction]:
        """
        Returns a list of the possible actions that can be taken from the current state of the Snake game.

        Returns:
            A list of Directions, representing the possible actions that can be taken from the current state.
        """
        valid_actions = [Direction.NONE]  # Add NONE action by default
        # Allow only perpendicular movements based on current direction
        if self.direction in {Direction.RIGHT, Direction.LEFT}:
            valid_actions.extend([Direction.UP, Direction.DOWN])
        elif self.direction in {Direction.UP, Direction.DOWN}:
            valid_actions.extend([Direction.LEFT, Direction.RIGHT])
        return valid_actions

    # Updates the current state using the given action
    def step(self, action: Direction) -> \
            Tuple[SnakeObservation, float, bool, Dict]:
        """
        Updates the state of the Snake game by applying the given action.

        Args:
            action (Direction): The action to apply to the current state.

        Returns:
            A tuple containing four elements:
            - next_state (SnakeObservation): The state of the game after taking the given action.
            - reward (float): The reward obtained by taking the given action.
            - done (bool): A boolean indicating whether the episode is over.
            - info (Dict): A dictionary containing any extra information. You can keep it empty.
        """
        done = False
        reward = 0
        
        # If no direction (NONE) is given, continue in the current direction
        if action == Direction.NONE:
            action = self.direction  # Maintain the current direction
        self.direction = action # Update the snake's direction with the given action
        head_pos = self.snake[0] + self.direction.to_vector()  # position for the snake's head after action
        head_pos = Point(head_pos.x % self.width, head_pos.y % self.height) # keep the snake within the grid
        # Check if the snake collides with itself (loses)
        if head_pos in self.snake:
            done = True  # Mark the game as finished
            reward = -100  # Assign a large negative reward for losing
            self.snake.pop()  # Remove the tail for consistency
            self.snake.insert(0, head_pos)  # Move the head to the new position
            # Return the updated state, reward, done flag, and empty info
            return SnakeObservation(tuple(self.snake), self.direction, self.apple), reward, done, {}
        # Move the snake by adding the new head position
        self.snake.insert(0, head_pos)
        # Define the winning condition based on grid size
        win_cond = self.width * self.height
        # Check if the snake occupies the entire grid (winning condition)
        if len(self.snake) == win_cond:
            done = True  # Mark the game as finished
            reward = 100  # Assign a large positive reward for winning
        elif head_pos == self.apple:  # Check if the snake eats the apple
            reward = 1  # Assign a small positive reward
            self.apple = self.generate_random_apple()  # Generate a new apple
        else:
            self.snake.pop()  # Remove the tail to maintain the current snake length
        # Return the updated state, reward, done flag, and empty info dictionary
        return SnakeObservation(tuple(self.snake), self.direction, self.apple), reward, done, {}


    ###########################
    #### Utility Functions ####
    ###########################

    def render(self) -> None:
        # render the snake as * (where the head is an arrow < ^ > v) and the apple as $ and empty space as .
        for y in range(self.height):
            for x in range(self.width):
                p = Point(x, y)
                if p == self.snake[0]:
                    char = ">^<v"[self.direction]
                    print(char, end='')
                elif p in self.snake:
                    print('*', end='')
                elif p == self.apple:
                    print('$', end='')
                else:
                    print('.', end='')
            print()
        print()

    # Converts a string to an observation
    def parse_state(self, string: str) -> SnakeObservation:
        snake, direction, apple = eval(str)
        return SnakeObservation(
            tuple(Point(x, y) for x, y in snake), 
            self.parse_action(direction), 
            Point(*apple)
        )
    
    # Converts an observation to a string
    def format_state(self, state: SnakeObservation) -> str:
        snake = tuple(tuple(p) for p in state.snake)
        direction = self.format_action(state.direction)
        apple = tuple(state.apple)
        return str((snake, direction, apple))
    
    # Converts a string to an action
    def parse_action(self, string: str) -> Direction:
        return {
            'R': Direction.RIGHT,
            'U': Direction.UP,
            'L': Direction.LEFT,
            'D': Direction.DOWN,
            '.': Direction.NONE,
        }[string.upper()]
    
    # Converts an action to a string
    def format_action(self, action: Direction) -> str:
        return {
            Direction.RIGHT: 'R',
            Direction.UP:    'U',
            Direction.LEFT:  'L',
            Direction.DOWN:  'D',
            Direction.NONE:  '.',
        }[action]