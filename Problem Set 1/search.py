from problem import HeuristicFunction, Problem, S, A, Solution
from collections import deque
import heapq
from helpers import utils

#TODO: Import any modules you want to use

# All search functions take a problem and a state
# If it is an informed search function, it will also receive a heuristic function
# S and A are used for generic typing where S represents the state type and A represents the action type

# All the search functions should return one of two possible type:
# 1. A list of actions which represent the path from the initial state to the final state
# 2. None if there is no solution

def BreadthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    #TODO: ADD YOUR CODE HERE
    #here we are using a queue for the frontier because we are doing breadth first and need to finish level by level
    frontier = deque([(initial_state, [])])  #queue has tuples of [state, path_to_state], initially the state is the start
    exploredSet = set()  #set is fast to check if node that was explored
    while frontier: #while frontier has elements in it
        state, path = frontier.popleft() #dequeue the state that should be processed now
        if problem.is_goal(state): #if it is the goal, the solution is the path to it
            return path
        exploredSet.add(state) #add it to the explored set        
        for action in problem.get_actions(state): #expand this node by adding the next possible actions to the frontier
            new_state = problem.get_successor(state, action) #new_state is the state if I took this action from my current state
            #if the node 'new_state' is not explored yet and is not in the frontier (ignoring path) -> we will add it
            if new_state not in exploredSet and new_state not in [s for s, _ in frontier]: 
                frontier.append((new_state, path + [action]))  #update the path to the new_state and add it
    return None #return none if the frontier is emptied and still goal not found

def DepthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    #TODO: ADD YOUR CODE HERE
    #here we are using a stack for the frontier because we are doing depth first and want to explore the deepest
    #I will use the imported deque as a stack by only using one end -> rest of the code is the same as prev
    frontier = deque([(initial_state, [])])  #queue has tuples of [state, path_to_state], initially the state is the start
    exploredSet = set()  #set is fast to check if node that was explored
    while frontier: #while frontier has elements in it
        state, path = frontier.pop() #dequeue the state that should be processed now
        if problem.is_goal(state): #if it is the goal, the solution is the path to it
            return path
        exploredSet.add(state) #add it to the explored set        
        for action in problem.get_actions(state): #expand this node by adding the next possible actions to the frontier
            new_state = problem.get_successor(state, action) #new_state is the state if I took this action from my current state
            #if the node 'new_state' is not explored yet and is not in the frontier (ignoring path) -> we will add it
            if new_state not in exploredSet and new_state not in [s for s, _ in frontier]: 
                frontier.append((new_state, path + [action]))  #update the path to the new_state and add it
    return None 
    
def UniformCostSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    #TODO: ADD YOUR CODE HERE
    #here we are using a priority queue for the frontier because we are doing uniform cost and need want to explore the cheapest
    #I imported 'heapq' for this 
    frontier = []
    counter = 0 #to maintain FIFO in case of equal costs
    heapq.heappush(frontier, (0, 0, initial_state, [])) #(cost, count, state, path) tuple
    exploredSet = set()  #set is fast to check if node that was explored
    while frontier: #while frontier has elements in it
        cost, _, state, path = heapq.heappop(frontier) #dequeue the state that should be processed now
        #if state is explored, we skip the processing
        if state in exploredSet: #checking that here so that same node is not added to frontier more than once
            continue
        if problem.is_goal(state): #if it is the goal, the solution is the path to it
            return path
        exploredSet.add(state) #add it to the explored set, then expand it
        for action in problem.get_actions(state): #expand this node by adding the next possible actions to the frontier
            new_state = problem.get_successor(state, action) #new_state is the state if I took this action from my current state
            if new_state not in exploredSet: #if the node 'new_state' is not explored yet -> we will add it
                new_cost = cost + problem.get_cost(state,action) #the new accumulative cost
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, new_state, path + [action])) #update the path and cost to the new_state and add it
    return None #return none if the frontier is emptied and still goal not found

def AStarSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    #TODO: ADD YOUR CODE HERE
    #greedy is a mix of greedy and UCS that uses the heuristic and the actual cost and recalculates the minimum
    frontier = []
    counter = 0  # To maintain FIFO in case of equal costs
    heapq.heappush(frontier, (0, 0, initial_state, []))  # Initialize with heuristic
    exploredSet = set()  #set is fast to check if node that was explored
    g_cost = {initial_state: 0}  #dictionary keeping actual costs for reaching a state -> to not expand nodes already explored at lower cost
    while frontier: #while frontier has elements in it
        _, _, state, path = heapq.heappop(frontier)  #dequeue the state that should be processed now
        #if state is explored, we skip the processing
        if state in exploredSet:
            continue
        if problem.is_goal(state): #if it is the goal, the solution is the path to it
            return path  #we do this at goal dequeuing because there might be a shorter path
        exploredSet.add(state)  #add it to the explored set, then expand it
        for action in problem.get_actions(state): #expand this node by adding the next possible actions to the frontier
            new_state = problem.get_successor(state, action) #new_state is the state if I took this action from my current state
            if new_state not in exploredSet:  # If the next state hasn't been explored yet
                new_g_cost = g_cost[state] + problem.get_cost(state, action) #cost to new_state is actual cost to current state + action
                #I faced an error here before adding the g_cost dictionary that I was calculating the cost as:
                #cost + heuristic + cost + heuristic and so on, while the actual cost only should have been saved,
                #and the heuristic is used for choosing pick the next state only -> adjusted at pushing it to the frontier 
                if new_state not in g_cost or new_g_cost < g_cost[new_state]:  #if this path is better or no actual cost calculated before
                    g_cost[new_state] = new_g_cost  #update the actual cost or add it
                    total_cost = new_g_cost + heuristic(problem, new_state) #the new accumulative cost
                    counter += 1
                    heapq.heappush(frontier, (total_cost, counter, new_state, path + [action])) #update the path and cost to the new_state and add it
    return None #return none if the frontier is emptied and still goal not found


def BestFirstSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    #TODO: ADD YOUR CODE HERE
    #greedy is like UCS but uses the heuristic only
    frontier = []
    counter = 0 #to maintain FIFO in case of equal costs
    heapq.heappush(frontier, (0, 0, initial_state, []))  #(cost, count, state, path) tuple
    exploredSet = set()  #set is fast to check if node that was explored
    while frontier: #while frontier has elements in it
        _, _, state, path = heapq.heappop(frontier) #dequeue the state that should be processed now
        #if state is explored, we skip the processing
        if state in exploredSet: #checking that here so that same node is not added to frontier more than once
            continue
        if problem.is_goal(state): #if it is the goal, the solution is the path to it
            return path
        exploredSet.add(state) #add it to the explored set, then expand it
        for action in problem.get_actions(state): #expand this node by adding the next possible actions to the frontier
            new_state = problem.get_successor(state, action) #new_state is the state if I took this action from my current state
            if new_state not in exploredSet: #if the node 'new_state' is not explored yet -> we will add it
                new_cost = heuristic(problem, new_state) #the evaluating criteria now is the heuristic alone
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, new_state, path + [action])) #update the path and cost to the new_state and add it
    return None #return none if the frontier is emptied and still goal not found