from dungeon import DungeonProblem, DungeonState
from mathutils import Direction, Point, euclidean_distance, manhattan_distance
from helpers import utils
from collections import deque

# This heuristic returns the distance between the player and the exit as an estimate for the path cost
# While it is consistent, it does a bad job at estimating the actual cost thus the search will explore a lot of nodes before finding a goal
def weak_heuristic(problem: DungeonProblem, state: DungeonState):
    return euclidean_distance(state.player, problem.layout.exit)

#after searching on google (similar problem is pacman with eating fruits) and I found an answer on stack overflow
#we should be calculating the real distance

def mazeDistance(pos1: Point, pos2: Point, problem: DungeonProblem) -> float: #BFS to find real distance between 2 positions
    if pos1 == pos2:
        return 0
    frontier = deque([(pos1, 0)])
    exploredSet = set()
    exploredSet.add(pos1)
    while frontier:
        current_position, distance = frontier.popleft()
        if current_position == pos2:
            return distance
        for direction in Direction:
            neighbor = current_position + direction.to_vector()
            if neighbor in problem.layout.walkable and neighbor not in exploredSet:
                exploredSet.add(neighbor)
                frontier.append((neighbor, distance + 1))
    return float('inf')

# get the distance between the furthest 2 coins -> x
# get the distance between the player and the nearest of those -> y
# The interpretation of this x + y formula could be something like this:
# x - either way, you will have to travel this distance, at least at the end
# y - while you are at the some of the two furthest fruits, it's better to collect the food that is near to it so you don't have to go back

def strong_heuristic(problem: DungeonProblem, state: DungeonState) -> float:
    # Retrieve the player, coin, and exit positions from the state
    player_pos = state.player
    rem_pos = state.remaining_coins
    exit_pos = state.layout.exit
    cache = problem.cache()
    # If no coins remain, return distance from player to exit
    if not rem_pos: 
        return mazeDistance(player_pos, exit_pos, problem) 
    #if there is 1 coin remaining 
    if len(rem_pos) == 1:
        coin = next(iter(rem_pos))
        return mazeDistance(player_pos, coin, problem) + mazeDistance(coin, exit_pos, problem)
    #if 2 or more coins
    max_dist_coins = 0
    furthest_coin1, furthest_coin2 = None, None
    #find the distance between furtherst 2 coins and the furthest 2 coins
    for coin1 in rem_pos:
        for coin2 in rem_pos:
            if coin1 == coin2:
                continue
            if (coin1, coin2) not in cache: #use cache to speed up
                cache[(coin1, coin2)] = cache[(coin2, coin1)] = mazeDistance(coin1, coin2, problem)
            distance_2coins = cache[(coin1, coin2)]
            if distance_2coins >= max_dist_coins:
                max_dist_coins = distance_2coins
                furthest_coin1, furthest_coin2 = coin1, coin2
    player_to_furthest_coin1 = cache.get((player_pos, furthest_coin1), mazeDistance(player_pos, furthest_coin1, problem)) + 3
    player_to_furthest_coin2 = cache.get((player_pos, furthest_coin2), mazeDistance(player_pos, furthest_coin2, problem)) + 3
    nearest_coin_pos = min(player_to_furthest_coin1, player_to_furthest_coin2)

    return max_dist_coins + nearest_coin_pos