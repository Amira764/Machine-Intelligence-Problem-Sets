from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json
from helpers.utils import NotImplemented

# This is a class for a generic Value Iteration agent
class ValueIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A]  # The MDP used by this agent for training
    utilities: Dict[S, float]  # The computed utilities
                                # The key is the string representation of the state and the value is the utility
    discount_factor: float  # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        self.utilities = {state: 0 for state in self.mdp.get_states()}  # Initialize all utilities to 0
        self.discount_factor = discount_factor

    # Given a state, compute its utility using the Bellman equation
    # If the state is terminal, return 0
    def compute_bellman(self, state: S) -> float:
        if self.mdp.is_terminal(state):  # check if the state is terminal return 0 if it is
            return 0
        actions = self.mdp.get_actions(state)  # get all possible actions for the given state
        max_utility = float('-inf')  # initialize max utility as negative infinity
        for action in actions:  # go through all possible actions
            expected_utility = 0  # reset utility for the current action
            transitions = self.mdp.get_successor(state, action)  # get the next states with probabilities
            for next_state, prob in transitions.items():  # for each possible next state
                reward = self.mdp.get_reward(state, action, next_state)  # calculate reward for this transition
                expected_utility += prob * (reward + self.discount_factor * self.utilities[next_state])  # bellman equation
            max_utility = max(max_utility, expected_utility)  # keep the highest utility found so far
        return max_utility  # return the maximum utility for the state

    # Applies a single utility update
    # Then returns True if the utilities have converged (the maximum utility change is <= the tolerance)
    def update(self, tolerance: float = 0) -> bool:
        new_utilities = {}  # dictionary to store updated utilities
        max_change = 0  # variable to track the largest change in utilities
        for state in self.mdp.get_states():  # iterate over all possible states
            new_utility = self.compute_bellman(state)  # calculate the new utility for the state
            new_utilities[state] = new_utility  # save the new utility for this state
            max_change = max(max_change, abs(new_utility - self.utilities[state]))  # check the biggest difference
        self.utilities = new_utilities  # update utilities after finishing the iteration
        return max_change <= tolerance  # check if the changes are small enough to consider converged

    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental updates and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None, tolerance: float = 0) -> int:
        iteration_count = 0  # counter for how many iterations were done
        while iterations is None or iteration_count < iterations:  # either keep running or stop after set number of iterations
            converged = self.update(tolerance)  # update utilities and check for convergence
            iteration_count += 1  # increment the iteration counter
            if converged:  # stop early if utilities have converged
                break
        return iteration_count  # return the number of iterations it took to train

    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        if self.mdp.is_terminal(state):  # if the state is terminal there is no action to take
            return None
        actions = self.mdp.get_actions(state)  # get the list of actions for this state
        best_action = None  # store the action with the highest expected utility
        max_utility = float('-inf')  # initialize the max utility as negative infinity
        for action in actions:  # go through each possible action
            expected_utility = 0  # reset the utility for this action
            transitions = self.mdp.get_successor(state, action)  # get the next states with probabilities for the action
            for next_state, prob in transitions.items():  # go through each possible next state
                reward = self.mdp.get_reward(state, action, next_state)  # get the reward for this transition
                expected_utility += prob * (reward + self.discount_factor * self.utilities[next_state])  # calculate utility
            if expected_utility > max_utility:  # if the utility is better than the current best one
                max_utility = expected_utility  # update the max utility
                best_action = action  # save this action as the best one
        return best_action  # return the action that leads to the highest utility

    # Save the utilities to a JSON file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(state): value for state, value in self.utilities.items()}
            json.dump(utilities, f, indent=2, sort_keys=True)

    # Load the utilities from a JSON file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            utilities = json.load(f)
            self.utilities = {self.mdp.parse_state(state): value for state, value in utilities.items()}
