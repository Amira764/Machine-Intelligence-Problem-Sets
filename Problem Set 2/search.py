from typing import Tuple
from game import HeuristicFunction, Game, S, A
from helpers.utils import NotImplemented

#TODO: Import any built in modules you want to use

# All search functions take a problem, a state, a heuristic function and the maximum search depth.
# If the maximum search depth is -1, then there should be no depth cutoff (The expansion should not stop before reaching a terminal state) 

# All the search functions should return the expected tree value and the best action to take based on the search results

# This is a simple search function that looks 1-step ahead and returns the action that lead to highest heuristic value.
# This algorithm is bad if the heuristic function is weak. That is why we use minimax search to look ahead for many steps.
def greedy(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    agent = game.get_turn(state)
    
    terminal, values = game.is_terminal(state)
    if terminal: return values[agent], None

    actions_states = [(action, game.get_successor(state, action)) for action in game.get_actions(state)]
    value, _, action = max((heuristic(game, state, agent), -index, action) for index, (action , state) in enumerate(actions_states))
    return value, action

# Apply Minimax search and return the game tree value and the best action
# Hint: There may be more than one player, and in all the testcases, it is guaranteed that 
# game.get_turn(state) will return 0 (which means it is the turn of the player). All the other players
# (turn > 0) will be enemies. So for any state "s", if the game.get_turn(s) == 0, it should a max node,
# and if it is > 0, it should be a min node. Also remember that game.is_terminal(s), returns the values
# for all the agents. So to get the value for the player (which acts at the max nodes), you need to
# get values[0].

#recursive function -> game tree -> will exit when we are back to the root
def minimax_value(game: Game[S, A], state: S, heuristic: HeuristicFunction, depth: int, max_depth: int) -> float:
    terminal, values = game.is_terminal(state)
    if terminal: # Check if the state is terminal
        return values[0]  # Return the player's utility
    
    if depth == max_depth: # If max depth is reached, evaluate using heuristic
        return heuristic(game, state, 0) # Player is agent 0
    
    turn = game.get_turn(state) # know which one's turn is this
    if turn == 0:  # Maximize player's utility by choosing the best action from all available actions
        max_val = float('-inf')
        for action in game.get_actions(state): # try all actions
            successor = game.get_successor(state, action) # for each action get the successor
            max_val = max(max_val, minimax_value(game, successor, heuristic, depth + 1, max_depth)) 
        return max_val # return the action that yields max value
    else:  # Minimize opponent's utility by choosing the worst action from all available actions
        min_val = float('inf')
        for action in game.get_actions(state): # try all actions
            successor = game.get_successor(state, action)  # for each action get the successor
            min_val = min(min_val, minimax_value(game, successor, heuristic, depth + 1, max_depth))
        return min_val # return the action that yields min value


def minimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    terminal, values = game.is_terminal(state)
    if terminal: # Check if the state is terminal
        return values[0]  # Return the player's utility

    # Evaluate the root state itself as a max node (0)
    best_action = None
    best_value = float('-inf')
    # Explore actions at the root state by getting all it's children (calculate subtrees)
    # then evaluate the best subtree and return it
    for action in game.get_actions(state):
        successor = game.get_successor(state, action) # get successor
        value = minimax_value(game, successor, heuristic, 1, max_depth) # get value at successor node
        if value > best_value: # get the best value and its corresponding action
            best_value = value
            best_action = action
    
    return best_value, best_action

# Apply Alpha Beta pruning and return the tree value and the best action
# Hint: Read the hint for minimax.
#recursive function -> game tree -> will exit when we are back to the root
def alphabeta_value(game: Game[S, A], state: S, heuristic: HeuristicFunction, depth: int, max_depth: int, alpha: float, beta: float) -> float:
    terminal, values = game.is_terminal(state)
    if terminal: # Check if the state is terminal
        return values[0]  # Return the player's utility
    
    if depth == max_depth: # If max depth is reached, evaluate using heuristic
        return heuristic(game, state, 0) # Player is agent 0
    
    turn = game.get_turn(state) # know which one's turn is this
    if turn == 0:  # Maximize player's utility by choosing the best action from all available actions
        max_val = float('-inf')
        for action in game.get_actions(state): # try all actions
            max_val = max(max_val, alphabeta_value(game, game.get_successor(state, action), heuristic, depth + 1, max_depth, alpha, beta)) 
            if max_val >= beta:  # Beta cutoff
                return max_val # return the action that yields max value
            alpha = max(alpha, max_val)
        return max_val   
    else:  # Minimize opponent's utility by choosing the worst action from all available actions
        min_val = float('inf')
        for action in game.get_actions(state): # try all actions
            min_val = min(min_val, alphabeta_value(game, game.get_successor(state, action), heuristic, depth + 1, max_depth, alpha, beta))
            if min_val <= alpha:  # Alpha cutoff
                return min_val # return the action that yields min value
            beta = min(beta, min_val)
        return min_val
        

def alphabeta(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    terminal, values = game.is_terminal(state)
    if terminal: # Check if the state is terminal
        return values[0], None # Return the player's utility
    # Evaluate the root state itself as a max node (0)
    best_action = None
    best_value = float('-inf')
    alpha, beta = float('-inf'), float('inf')
    # Explore actions at the root state by getting all it's children (calculate subtrees)
    # then evaluate the best subtree and return it
    for action in game.get_actions(state):
        successor = game.get_successor(state, action) # get successor
        value = alphabeta_value(game, successor, heuristic, 1, max_depth, alpha, beta) # get value at successor node
        if value > best_value: # get the best value and its corresponding action
            best_value = value
            best_action = action
        alpha = max(alpha, best_value)  # update alpha
        if alpha >= beta:  # Beta cutoff at the root level
            break
    return best_value, best_action

# Apply Alpha Beta pruning with move ordering and return the tree value and the best action
# Hint: Read the hint for minimax.
def alphabeta_with_move_ordering_value(game: Game[S, A], state: S, heuristic: HeuristicFunction, depth: int, max_depth: int, alpha: float, beta: float) -> float:
    terminal, values = game.is_terminal(state)
    if terminal:  # Check if the state is terminal
        return values[0]  # Return the player's utility
    
    if depth == max_depth:  # If max depth is reached, evaluate using heuristic
        return heuristic(game, state, 0)  # Player is agent 0
    
    turn = game.get_turn(state)  # know whose turn it is
    if turn == 0:  # Maximize player's utility
        max_val = float('-inf')
        # Sort actions based on heuristic for the current state and explore best ones first
        actions = sorted(game.get_actions(state), 
                         key=lambda a: heuristic(game, game.get_successor(state, a), 0), 
                         reverse=True)
        for action in actions:
            successor = game.get_successor(state, action)
            max_val = max(max_val, alphabeta_with_move_ordering_value(game, successor, heuristic, depth + 1, max_depth, alpha, beta))
            if max_val >= beta:  # Beta cutoff
                return max_val  # Prune this branch
            alpha = max(alpha, max_val)  # Update alpha
        return max_val
    else:  # Minimize opponent's utility
        min_val = float('inf')
        # Sort actions based on heuristic for the current state and explore best ones first
        actions = sorted(game.get_actions(state), 
                         key=lambda a: heuristic(game, game.get_successor(state, a), 0), 
                         reverse=False)
        for action in actions:
            successor = game.get_successor(state, action)
            min_val = min(min_val, alphabeta_with_move_ordering_value(game, successor, heuristic, depth + 1, max_depth, alpha, beta))
            if min_val <= alpha:  # Alpha cutoff
                return min_val  # Prune this branch
            beta = min(beta, min_val)  # Update beta
        return min_val


def alphabeta_with_move_ordering(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    terminal, values = game.is_terminal(state)
    if terminal:  # Check if the state is terminal
        return values[0], None  # Return the utility and no action (game is over)
    
    best_action = None
    best_value = float('-inf')
    alpha, beta = float('-inf'), float('inf')
    
    # Sort actions based on heuristic to improve pruning effectiveness
    actions = sorted(game.get_actions(state), 
                     key=lambda a: heuristic(game, game.get_successor(state, a), 0), 
                     reverse=True)
    
    for action in actions:
        successor = game.get_successor(state, action)  # Get successor
        value = alphabeta_with_move_ordering_value(game, successor, heuristic, 1, max_depth, alpha, beta)
        if value > best_value:
            best_value = value
            best_action = action
        alpha = max(alpha, best_value)  # Update alpha
        if alpha >= beta:  # Beta cutoff at the root level
            break
    return best_value, best_action


# Apply Expectimax search and return the tree value and the best action
# Hint: Read the hint for minimax, but note that the monsters (turn > 0) do not act as min nodes anymore,
# they now act as chance nodes (they act randomly).
def expectimax_value(game: Game[S, A], heuristic: HeuristicFunction, state: S, depth: int, max_depth: int) -> float:
    terminal, values = game.is_terminal(state)
    if terminal:  # Check if the state is terminal
        return values[0]  # Return the player's utility
    
    if depth == max_depth:  # If max depth is reached, evaluate using heuristic
        return heuristic(game, state, 0)  # Player is agent 0
    
    turn = game.get_turn(state)  # know whose turn it is

    if turn==0: # Maximizing player: Try to maximize the agent's score
        max_val = float('-inf')
        for action in game.get_actions(state):
            max_val = max(max_val, expectimax_value(game, heuristic, game.get_successor(state, action), depth + 1, max_depth))
        return max_val
    else: # Chance node: Compute the expected value
        actions = game.get_actions(state)
        probability = 1 / len(actions)  # Uniform assumption for random agent.
        return sum(probability * expectimax_value(game, heuristic, game.get_successor(state, action), depth + 1, max_depth) for action in actions)


def expectimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    terminal, values = game.is_terminal(state)
    if terminal:  # Check if the state is terminal
        return values[0], None  # Return the utility and no action (game is over)
        
    # To get the best action, we need to evaluate each possible action and select the one with the highest value
    best_action = None
    best_value = float('-inf')
    for action in game.get_actions(state):
        value = expectimax_value(game, heuristic, game.get_successor(state, action), 1, max_depth)  # Start from the root with the maximizing player
        if value > best_value:
            best_value = value
            best_action = action

    return best_value, best_action
