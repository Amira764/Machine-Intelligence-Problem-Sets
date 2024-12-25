from typing import Any, Dict, List, Optional
from CSP import Assignment, BinaryConstraint, Problem, UnaryConstraint
from helpers.utils import NotImplemented

# This function applies 1-Consistency to the problem.
# In other words, it modifies the domains to only include values that satisfy their variables' unary constraints.
# Then all unary constraints are removed from the problem (they are no longer needed).
# The function returns False if any domain becomes empty. Otherwise, it returns True.
def one_consistency(problem: Problem) -> bool:
    remaining_constraints = []
    solvable = True
    for constraint in problem.constraints:
        if not isinstance(constraint, UnaryConstraint):
            remaining_constraints.append(constraint)
            continue
        variable = constraint.variable
        new_domain = {value for value in problem.domains[variable] if constraint.condition(value)}
        if not new_domain:
            solvable = False
        problem.domains[variable] = new_domain
    problem.constraints = remaining_constraints
    return solvable

# This function returns the variable that should be picked based on the MRV heuristic.
# NOTE: We don't use the domains inside the problem, we use the ones given by the "domains" argument 
#       since they contain the current domains of unassigned variables only.
# NOTE: If multiple variables have the same priority given the MRV heuristic, 
#       we order them in the same order in which they appear in "problem.variables".
def minimum_remaining_values(problem: Problem, domains: Dict[str, set]) -> str:
    _, _, variable = min((len(domains[variable]), index, variable) for index, variable in enumerate(problem.variables) if variable in domains)
    return variable

# This function should implement forward checking
# The function is given the problem, the variable that has been assigned and its assigned value and the domains of the unassigned values
# The function should return False if it is impossible to solve the problem after the given assignment, and True otherwise.
# In general, the function should do the following:
#   - For each binary constraints that involve the assigned variable:
#       - Get the other involved variable.
#       - If the other variable has no domain (in other words, it is already assigned), skip this constraint.
#       - Update the other variable's domain to only include the values that satisfy the binary constraint with the assigned variable.
#   - If any variable's domain becomes empty, return False. Otherwise, return True.
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def forward_checking(problem: Problem, assigned_variable: str, assigned_value: Any, domains: Dict[str, set]) -> bool:
    # Iterate through all binary constraints in the problem
    for constraint in problem.constraints:
        # If the constraint is Binary and 
        # The variable we want to assign value to is involved with the constrain
        if isinstance(constraint, BinaryConstraint) and assigned_variable in constraint.variables:
            # Get the other variable involved in the constraint
            other_variable = constraint.get_other(assigned_variable)
            # Skip if the other variable is not in the domains (already assigned)
            if other_variable not in domains:
                continue
            # The other variable is in domains (not assigned)
            new_domain = set() # create a new domain
            for value in domains[other_variable]: # Check for each value in the domain
                if constraint.condition(assigned_value, value): # If it satisfies the conditiom
                    new_domain.add(value) # If it does, add it to the domain
            # Assign the filtered domain back to the other variable
            domains[other_variable] = new_domain
            # If the domain becomes empty, the problem is unsolvable
            if not domains[other_variable]:
                return False
    # If no domains were emptied, return True
    return True

# This function should return the domain of the given variable order based on the "least restraining value" heuristic.
# IMPORTANT: This function should not modify any of the given arguments.
# Generally, this function is very similar to the forward checking function, but it differs as follows:
#   - You are not given a value for the given variable, since you should do the process for every value in the variable's
#     domain to see how much it will restrain the neigbors domain
#   - Here, you do not modify the given domains. But you can create and modify a copy.
# IMPORTANT: If multiple values have the same priority given the "least restraining value" heuristic, 
#            order them in ascending order (from the lowest to the highest value).
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def least_restraining_values(problem: Problem, variable_to_assign: str, domains: Dict[str, set]) -> List[Any]:
    # Dictionary to track the number of restrictions caused by each value
    restrictions_count = {}
    # Iterate over each possible value in the domain of the variable to assign
    for value in domains[variable_to_assign]:
        # Create a simulated copy of the domains
        simulated_domains = {}
        for var, dom in domains.items():
            simulated_domains[var] = dom.copy()
        simulated_domains[variable_to_assign] = {value} # Now we assign a value and see its impact
        total_restrictions = 0  # Initialize the total count of restrictions caused by this value
        # Iterate over all constraints in the problem
        for constraint in problem.constraints:
            # Focus only on binary constraints that involve the variable to assign
            if isinstance(constraint, BinaryConstraint) and variable_to_assign in constraint.variables:
                # Find the other variable involved in the constraint
                other_variable = constraint.get_other(variable_to_assign)
                # Skip if the other variable is already assigned (has no domain left to reduce)
                if other_variable not in simulated_domains:
                    continue
                # Get the original domain of the other variable
                original_domain = simulated_domains[other_variable]
                # Filter the domain of the other variable based on the constraint
                filtered_domain = set()  # Create a set to store valid values
                for other_value in original_domain:
                    # Add the value to the filtered domain if it satisfies the constraint
                    if constraint.condition(value, other_value):
                        filtered_domain.add(other_value)
                # Calculate how many values were removed from the domain
                restrictions_caused = len(original_domain) - len(filtered_domain)
                # Update the simulated domain for the other variable with the filtered values
                simulated_domains[other_variable] = filtered_domain
                # Add the number of restrictions caused by this value to the total count
                total_restrictions += restrictions_caused
        # Store the total number of restrictions caused by the current value
        restrictions_count[value] = total_restrictions
    # Sort the values by their restriction count in ascending order (least restrictive first)
    value_restrictions = []
    for val in domains[variable_to_assign]:
        count = restrictions_count[val]  # Get the restriction count for each value
        value_restrictions.append((val, count))  # Append the (value, count) pair to the list
    # Sort the list based on the restrictions count (second item in each tuple) then first
    value_restrictions.sort(key=lambda x: (x[1], x[0]))
    # Extract the sorted values from the list of tuples
    sorted_values = [val for val, _ in value_restrictions]
    return sorted_values

# This function should solve CSP problems using backtracking search with forward checking.
# The variable ordering should be decided by the MRV heuristic.
# The value ordering should be decided by the "least restraining value" heurisitc.
# Unary constraints should be handled using 1-Consistency before starting the backtracking search.
# This function should return the first solution it finds (a complete assignment that satisfies the problem constraints).
# If no solution was found, it should return None.
# IMPORTANT: To get the correct result for the explored nodes, you should check if the assignment is complete only once using "problem.is_complete"
#            for every assignment including the initial empty assignment, EXCEPT for the assignments pruned by the forward checking.
#            Also, if 1-Consistency deems the whole problem unsolvable, you shouldn't call "problem.is_complete" at all.
def solve(problem: Problem) -> Optional[Assignment]:
    # If unary constraint fails, problem has no solution
    if not one_consistency(problem):
        return None
    # Copy the domains and start with an empty assignment
    domains = {}
    for var in problem.variables:
        domains[var] = problem.domains[var].copy()
    assignment = {}

    def backtrack(domains) -> Optional[Assignment]:
        if problem.is_complete(assignment) and problem.satisfies_constraints(assignment): # base case : solved
            return assignment
        # Select the variable with MRV heuristic
        variable = minimum_remaining_values(problem, domains)
        # Get value ordering with LRV heuristic
        value_ordering = least_restraining_values(problem, variable, domains)
        # loop on each value trying to assign a value
        for value in value_ordering:
            # Create a new copy of domains for forward checking
            new_domains = {var: dom.copy() for var, dom in domains.items()}
            del new_domains[variable]  # Variable is now fully assigned
            # check the impact of this value on the problem
            if forward_checking(problem, variable, value, new_domains):
                # Assign the variable and remove it from the domains
                assignment[variable] = value
                # Recursively call backtrack with updated domains (without assigned variable)
                result = backtrack(new_domains)
                if result is not None:
                    return result
                # If result fails, backtrack
                del assignment[variable]
        # If forward checking yields false or if result is none
        return None
    return backtrack(domains)
