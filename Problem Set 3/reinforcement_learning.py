from typing import Callable, DefaultDict, Dict, Generic, List, Optional, Union
from agents import Agent
from environment import Environment, S, A
from helpers.mt19937 import RandomGenerator
from helpers.utils import NotImplemented

import json
from collections import defaultdict

# The base class for all Reinforcement Learning Agents required for this problem set


class RLAgent(Agent[S, A]):
    rng: RandomGenerator  # A random number generator used for exploration
    actions: List[A]  # A list of all actions that the environment accepts
    discount_factor: float  # The discount factor "gamma"
    epsilon: float  # The exploration probability for epsilon-greedy
    learning_rate: float  # The learning rate "alpha"

    def __init__(self,
                 actions: List[A],
                 discount_factor: float = 0.99,
                 epsilon: float = 0.5,
                 learning_rate: float = 0.01,
                 seed: Optional[int] = None) -> None:
        super().__init__()
        # initialize the random generator with a seed for reproducability
        self.rng = RandomGenerator(seed)
        self.actions = actions
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    # A virtual function that returns the Q-value for a specific state and action
    # This should be overriden by the derived RL agents
    def compute_q(self, env: Environment[S, A], state: S, action: A) -> float:
        return 0

    # Returns true if we should explore (rather than exploit)
    def should_explore(self) -> bool:
        return self.rng.float() < self.epsilon
        
    def act(self, env: Environment[S, A], observation: S, training: bool = False) -> A:
        actions = env.actions()  # get all possible actions from the environment
        # TODO: Return a random action whose index is "self.rng.int(0, len(actions)-1)"
        if training and self.should_explore():  # check if in training mode and exploration should be done
            return actions[self.rng.int(0, len(actions) - 1)]  # pick a random action from the list
        else:
            # TODO: return the action with the maximum q-value as calculated by "compute_q" above
            # if more than one action has the maximum q-value, return the one that appears first in the "actions" list
            q_values = [self.compute_q(env, observation, action) for action in actions]  # calculate q-values for all actions
            max_q = max(q_values)  # find the highest q-value
            return actions[q_values.index(max_q)]  # return the action corresponding to the highest q-value


#############################
#######     SARSA      ######
#############################

# This is a class for a generic SARSA agent

class SARSALearningAgent(RLAgent[S, A]):
    Q: DefaultDict[S, DefaultDict[A, float]]  # The table of the Q values
    # The first key is the string representation of the state
    # The second key is the string representation of the action
    # The value is the Q-value of the given state and action

    def __init__(self,
                 actions: List[A],
                 discount_factor: float = 0.99,
                 epsilon: float = 0.5,
                 learning_rate: float = 0.01,
                 seed: Optional[int] = None) -> None:
        super().__init__(actions, discount_factor, epsilon, learning_rate, seed)
        self.Q = defaultdict(lambda: defaultdict(
            lambda: 0))  # The default Q value is 0

    def compute_q(self, env: Environment[S, A], state: S, action: A) -> float:
        # Return the Q-value of the given state and action
        return self.Q[state][action]
        # NOTE: we cast the state and the action to a string before querying the dictionaries

    # Update the value of Q(state, action) using this transition via the SARSA update rule
    def update(self, env: Environment[S, A], state: S, action: A, reward: float, next_state: S, next_action: Optional[A]):
        # TODO: Complete this function to update Q-table using the SARSA update rule
        # If next_action is None, then next_state is a terminal state in which case, we consider the Q-value of next_state to be 0
        next_q = self.Q[next_state][next_action] if next_action is not None else 0  # use Q-value of next_action or 0 if terminal state
        td_target = reward + self.discount_factor * next_q  # calculate the TD target using SARSA formula
        td_delta = td_target - self.Q[state][action]  # compute the TD error or difference
        self.Q[state][action] += self.learning_rate * td_delta  # update the Q-value for the given state and action

    # Save the Q-table to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            Q = {
                env.format_state(state): {
                    env.format_action(action): value for action, value in state_q.items()
                } for state, state_q in self.Q.items()
            }
            json.dump(Q, f, indent=2, sort_keys=True)

    # load the Q-table from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            Q = json.load(f)
            self.Q = {
                env.parse_state(state): {
                    env.parse_action(action): value for action, value in state_q.items()
                } for state, state_q in Q.items()
            }

#############################
#####   Q-Learning     ######
#############################

# This is a class for a generic Q-learning agent


class QLearningAgent(RLAgent[S, A]):
    Q: DefaultDict[str, DefaultDict[str, float]]  # The table of the Q values
    # The first key is the string representation of the state
    # The second key is the string representation of the action
    # The value is the Q-value of the given state and action

    def __init__(self,
                 actions: List[A],
                 discount_factor: float = 0.99,
                 epsilon: float = 0.5,
                 learning_rate: float = 0.01,
                 seed: Optional[int] = None) -> None:
        super().__init__(actions, discount_factor, epsilon, learning_rate, seed)
        self.Q = defaultdict(lambda: defaultdict(
            lambda: 0))  # The default Q value is 0

    def compute_q(self, env: Environment[S, A], state: S, action: A) -> float:
        # Return the Q-value of the given state and action
        return self.Q[state][action]
        # NOTE: we cast the state and the action to a string before querying the dictionaries

    # Given a state, compute and return the utility of the state using the function "compute_q"
    def compute_utility(self, env: Environment[S, A], state: S) -> float:
        # TODO: Complete this function.
        actions = env.actions()  # get all possible actions in the current state
        q_values = [self.compute_q(env, state, action) for action in actions]  # calculate Q-values for all actions
        return max(q_values) if q_values else 0  # return the max Q-value or 0 if no actions available

    # Update the value of Q(state, action) using this transition via the Q-Learning update rule
    def update(self, env: Environment[S, A], state: S, action: A, reward: float, next_state: S, done: bool):
        # TODO: Complete this function to update Q-table using the Q-Learning update rule
        # If done is True, then next_state is a terminal state in which case, we consider the Q-value of next_state to be 0
        next_max_q = 0 if done else self.compute_utility(env, next_state)  # set next_max_q to 0 if done else get max utility of next_state
        td_target = reward + self.discount_factor * next_max_q  # calculate the TD target using reward and future Q-value
        td_delta = td_target - self.Q[state][action]  # compute the difference between target and current Q-value
        self.Q[state][action] += self.learning_rate * td_delta  # adjust Q-value for state-action pair using learning rate

    # Save the Q-table to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            Q = {
                env.format_state(state): {
                    env.format_action(action): value for action, value in state_q.items()
                } for state, state_q in self.Q.items()
            }
            json.dump(Q, f, indent=2, sort_keys=True)

    # load the Q-table from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            Q = json.load(f)
            self.Q = {
                env.parse_state(state): {
                    env.parse_action(action): value for action, value in state_q.items()
                } for state, state_q in Q.items()
            }

#########################################
#####   Approximate Q-Learning     ######
#########################################
# The type definition for a set of features representing a state
# The key is the feature name and the value is the feature value
Features = Dict[str, float]

# This class takes a state and returns the a set of features


class FeatureExtractor(Generic[S, A]):

    # Returns a list of feature names.
    # This will be used by the Approximate Q-Learning agent to initialize its weights dictionary.
    @property
    def feature_names(self) -> List[str]:
        return []

    # Given an enviroment and an observation (a state), return a set of features that represent the given state
    def extract_features(self, env: Environment[S, A], state: S) -> Features:
        return {}

# This is a class for a generic Q-learning agent


class ApproximateQLearningAgent(RLAgent[S, A]):
    weights: Dict[A, Features]    # The weights dictionary for this agent.
    # The first key is action and the second key is the feature name
    # The value is the weight
    # The feature extractor used to extract the features corresponding to a state
    feature_extractor: FeatureExtractor[S, A]

    def __init__(self,
                 feature_extractor: FeatureExtractor[S, A],
                 actions: List[A],
                 discount_factor: float = 0.99,
                 epsilon: float = 0.5,
                 learning_rate: float = 0.01,
                 seed: Optional[int] = None) -> None:
        super().__init__(actions, discount_factor, epsilon, learning_rate, seed)
        feature_names = feature_extractor.feature_names
        self.weights = {action: {feature: 0 for feature in feature_names}
                        for action in actions}  # Initialize weights to 0
        self.feature_extractor = feature_extractor

    def __compute_q_from_features(self, features: Dict[str, float], action: A) -> float:
        action_weights = self.weights[action]  # get the weight dictionary for the action
        return sum(action_weights[feature] * value for feature, value in features.items())  # compute weighted sum of feature values

    def __compute_utility_from_features(self, features: Dict[str, float]) -> float:
        return max(self.__compute_q_from_features(features, action) for action in self.actions)  # get max Q-value from all actions

    def compute_q(self, env: Environment[S, A], state: S, action: A) -> float:
        features = self.feature_extractor.extract_features(env, state)  
        return self.__compute_q_from_features(features, action) 

    def update(self, env: Environment[S, A], state: S, action: A, reward: float, next_state: S, done: bool):
        features = self.feature_extractor.extract_features(env, state)  # get features for current state
        q_current = self.__compute_q_from_features(features, action)  # calculate current Q-value for state-action pair


        # Compute target value
        if done:
            target = reward
        else:
            next_features = self.feature_extractor.extract_features(env, next_state)
            target = reward + self.discount_factor * self.__compute_utility_from_features(next_features)

        # Update weights
        for feature, value in features.items():
            self.weights[action][feature] += self.learning_rate * (target - q_current) * value

    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            weights = {env.format_action(action): w for action, w in self.weights.items()}
            json.dump(weights, f, indent=2, sort_keys=True)

    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            weights = json.load(f)
            self.weights = {env.parse_action(action): w for action, w in weights.items()}
