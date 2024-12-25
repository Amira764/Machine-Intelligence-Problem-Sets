from typing import Tuple
import re
from CSP import Assignment, Problem, UnaryConstraint, BinaryConstraint

#TODO (Optional): Import any builtin library or define any helper function you want to use

# This is a class to define for cryptarithmetic puzzles as CSPs
class CryptArithmeticProblem(Problem):
    LHS: Tuple[str, str]
    RHS: str

    # Convert an assignment into a string (so that is can be printed).
    def format_assignment(self, assignment: Assignment) -> str:
        LHS0, LHS1 = self.LHS
        RHS = self.RHS
        letters = set(LHS0 + LHS1 + RHS)
        formula = f"{LHS0} + {LHS1} = {RHS}"
        postfix = []
        valid_values = list(range(10))
        for letter in letters:
            value = assignment.get(letter)
            if value is None: continue
            if value not in valid_values:
                postfix.append(f"{letter}={value}")
            else:
                formula = formula.replace(letter, str(value))
        if postfix:
            formula = formula + " (" + ", ".join(postfix) +  ")" 
        return formula

    @staticmethod
    def from_text(text: str) -> 'CryptArithmeticProblem':
        # Given a text in the format "LHS0 + LHS1 = RHS", the following regex
        # matches and extracts LHS0, LHS1 & RHS
        # For example, it would parse "SEND + MORE = MONEY" and extract the
        # terms such that LHS0 = "SEND", LHS1 = "MORE" and RHS = "MONEY"
        pattern = r"\s*([a-zA-Z]+)\s*\+\s*([a-zA-Z]+)\s*=\s*([a-zA-Z]+)\s*"
        match = re.match(pattern, text)
        if not match: raise Exception("Failed to parse:" + text)
        LHS0, LHS1, RHS = [match.group(i+1).upper() for i in range(3)]

        problem = CryptArithmeticProblem()
        problem.LHS = (LHS0, LHS1)
        problem.RHS = RHS

        #TODO Edit and complete the rest of this function
        # problem.variables:    should contain a list of variables where each variable is string (the variable name)
        # problem.domains:      should be dictionary that maps each variable (str) to its domain (set of values)
        #                       For the letters, the domain can only contain integers in the range [0,9].
        # problem.constaints:   should contain a list of constraint (either unary or binary constraints).

        # Collect all unique variables (letters in the problem)
        letters = set(LHS0 + LHS1 + RHS)
        carries = [f"c{i}" for i in range(len(RHS) - 1)]  # Carries (cout) needed for each column except the last
        problem.variables = list(letters) + carries  # Combine letters and carry variables

        problem.domains = {letter: set(range(10)) for letter in letters}  # Domain for letters: digits 0-9
        for carry in carries:
            problem.domains[carry] = {0, 1}  # Domain for carries: either 0 or 1

        # First letter in any word cannot be zero
        for word in (LHS0, LHS1, RHS):
            # Apply unary constraint that the first letter of each word must not be zero
            problem.constraints.append(UnaryConstraint(variable=word[0], condition=lambda x: x != 0))

        # Binary constraint: no 2 letters can have the same number
        for letter1 in letters:
            for letter2 in letters:
                if letter1 != letter2:
                    problem.constraints.append(BinaryConstraint((letter1, letter2), lambda x, y: x != y))

        reversed_LHS0 = LHS0[::-1]  # Reverse LHS0
        reversed_LHS1 = LHS1[::-1]  # Reverse LHS1
        reversed_RHS = RHS[::-1]    # Reverse RHS

        # Loop through the columns (digits) in the reversed RHS term to start from units 
        for col in range(len(reversed_RHS)):
            # Determine the carry-in (from the previous column) and carry-out (to the next column)
            carry_in = f"c{col - 1}" if col > 0 else None
            carry_out = f"c{col}" if col < len(reversed_RHS) - 1 else None

            # For each column, get the corresponding digits from LHS0, LHS1, and RHS
            lhs0_digit = reversed_LHS0[col] if col < len(reversed_LHS0) else '0' # compensate difference in length with a zero
            lhs1_digit = reversed_LHS1[col] if col < len(reversed_LHS1) else '0' # compensate difference in length with a zero
            rhs_digit = reversed_RHS[col]

            # Function to enforce the column-wise sum with carry propagation
            temp_sum = (lhs0_digit + lhs1_digit + carry_in ) % 10
            def column_constraint1(temp_sum_fn, rhs_digit_fn):
                # The sum of the digits and the carry-in should match the right-hand side digit
                return (temp_sum_fn == rhs_digit_fn)

            def column_constraint2(temp_sum_fn, cout_fn):
                # The tens digit of the sum of the digits and the carry-in should match the carry out
                return (temp_sum_fn // 10 == cout_fn)

            # Add the binary constraint for the current column
            problem.constraints.append(BinaryConstraint((temp_sum, rhs_digit),column_constraint1)) 
            problem.constraints.append(BinaryConstraint((temp_sum, carry_out),column_constraint2)) 

        return problem

    # Read a cryptarithmetic puzzle from a file
    @staticmethod
    def from_file(path: str) -> "CryptArithmeticProblem":
        with open(path, 'r') as f:
            return CryptArithmeticProblem.from_text(f.read())