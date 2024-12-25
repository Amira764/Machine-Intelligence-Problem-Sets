from typing import Any, Dict, Set, Tuple, List
from problem import Problem
from mathutils import Direction, Point
from helpers import utils

#TODO: (Optional) Instead of Any, you can define a type for the parking state
ParkingState = Tuple[Point]
# An action of the parking problem is a tuple containing an index 'i' and a direction 'd' where car 'i' should move in the direction 'd'.
ParkingAction = Tuple[int, Direction]

# This is the implementation of the parking problem
class ParkingProblem(Problem[ParkingState, ParkingAction]):
    passages: Set[Point]    # A set of points which indicate where a car can be (in other words, every position except walls).
    cars: Tuple[Point]      # A tuple of points where state[i] is the position of car 'i'. 
    slots: Dict[Point, int] # A dictionary which indicate the index of the parking slot (if it is 'i' then it is the lot of car 'i') for every position.
                            # if a position does not contain a parking slot, it will not be in this dictionary.
    width: int              # The width of the parking lot.
    height: int             # The height of the parking lot.

    # This function should return the initial state
    def get_initial_state(self) -> ParkingState:
        #return the initial positions of cars
        return tuple(self.cars)
    
    # This function should return True if the given state is a goal. Otherwise, it should return False.
    def is_goal(self, state: ParkingState) -> bool:
        # every car 'i' is at its correct position slot 'i'
        # "cars" has current positions of cars
        # "slots" car index -> its pos
        # state is a tuple of car positions where each element in the position of its index
        for index, car_position in enumerate(state):
            if self.slots.get(car_position) != index: #we get the index of the slot that has that position, if it is not correct
                return False #we are not at the goal
        return True
    
    # This function returns a list of all the possible actions that can be applied to the given state
    def get_actions(self, state: ParkingState) -> List[ParkingAction]:
        #everywhere I can move where there is no wall
        #checking the 4 directions and no wall
        actions = []
        for index, carPos in enumerate(state):
            for direction in Direction:
                position = carPos + direction.to_vector()
                # Disallow walking into unwalkable positions
                if position not in self.passages or position in state: continue #we cannot go into # or crash another car
                actions.append([index, direction])
        return actions
    
    # This function returns a new state which is the result of applying the given action to the given state
    def get_successor(self, state: ParkingState, action: ParkingAction) -> ParkingState:
        #one action is a tuple of index and direction, action[0] is index and action[1] is direction
        index, direction = action #car chosen to be moved
        newPos = state[index] + direction.to_vector() #the car's new position
        if newPos not in self.passages or newPos in state:
            # If we try to walk into a wall or crash a car , the state does not change
            return state
        next_state = list(state) #because tuples are immutable
        next_state[index] = newPos #update the position of the chosen car to it's new position
        return tuple(next_state)
    
    # This function returns the cost of applying the given action to the given state
    def get_cost(self, state: ParkingState, action: ParkingAction) -> float:
        index , direction = action #car chosen to be moved
        next_state = self.get_successor(state, action) #if this action will lead to a state where we park in a slot that is not ours -> 101
        if next_state[index] in self.slots: #if the new position in a reserved slot for a car
            if self.slots[next_state[index]] != index: #if it is not my slot
                return 101 
        return 1
    
     # Read a parking problem from text containing a grid of tiles
    @staticmethod
    def from_text(text: str) -> 'ParkingProblem':
        passages =  set()
        cars, slots = {}, {}
        lines = [line for line in (line.strip() for line in text.splitlines()) if line]
        width, height = max(len(line) for line in lines), len(lines)
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char != "#":
                    passages.add(Point(x, y))
                    if char == '.':
                        pass
                    elif char in "ABCDEFGHIJ":
                        cars[ord(char) - ord('A')] = Point(x, y)
                    elif char in "0123456789":
                        slots[int(char)] = Point(x, y)
        problem = ParkingProblem()
        problem.passages = passages
        problem.cars = tuple(cars[i] for i in range(len(cars)))
        problem.slots = {position:index for index, position in slots.items()}
        problem.width = width
        problem.height = height
        return problem

    # Read a parking problem from file containing a grid of tiles
    @staticmethod
    def from_file(path: str) -> 'ParkingProblem':
        with open(path, 'r') as f:
            return ParkingProblem.from_text(f.read())
