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
        # Parse the text into LHS0, LHS1, and RHS
        pattern = r"\s*([A-Z]+)\s*\+\s*([A-Z]+)\s*=\s*([A-Z]+)\s*"
        match = re.match(pattern, text)
        if not match:
            raise ValueError("Invalid input format. Expected 'LHS0 + LHS1 = RHS'.")
        LHS0, LHS1, RHS = [match.group(i + 1) for i in range(3)]

        problem = CryptArithmeticProblem()
        problem.LHS = (LHS0, LHS1)
        problem.RHS = RHS

        problem.variables = []
        problem.domains = {}
        problem.constraints = []

        # Collect variables (letters) and carries
        letters = set(LHS0 + LHS1 + RHS)
        carries = [f"c{i}" for i in range(len(RHS) - 1)]  # Carries for each column except the last
        aux_vars = [f"s{i}" for i in range(len(RHS))]  # Auxiliary variables for sums

        # Variables include letters, carries, and auxiliary variables
        problem.variables = list(letters) + carries + aux_vars

        # Domains: letters (digits 0-9), carries (0 or 1), auxiliary sums (0-999)
        problem.domains = {var: set(range(10)) for var in letters}
        for carry in carries:
            problem.domains[carry] = {0, 1}
        for aux in aux_vars:
            problem.domains[aux] = set(range(1000))  # Allow for large sums with weights

        # Unary constraints: First letters of each word cannot be zero
        for word in (LHS0, LHS1, RHS):
            problem.constraints.append(UnaryConstraint(word[0], lambda x: x != 0))

        # Binary constraints: All letters must map to unique digits
        for i, letter1 in enumerate(letters):
            for letter2 in list(letters)[i + 1:]:
                problem.constraints.append(BinaryConstraint((letter1, letter2), lambda x, y: x != y))

        # Reverse the strings for easier column-wise addition
        reversed_LHS0 = LHS0[::-1]
        reversed_LHS1 = LHS1[::-1]
        reversed_RHS = RHS[::-1]

        for col in range(len(reversed_RHS)):
            # Determine the carry-in, carry-out, and auxiliary sum variables
            carry_in = f"c{col - 1}" if col > 0 else None
            carry_out = f"c{col}" if col < len(reversed_RHS) - 1 else None
            aux_sum = f"s{col}"

            lhs0_digit = reversed_LHS0[col] if col < len(reversed_LHS0) else None
            lhs1_digit = reversed_LHS1[col] if col < len(reversed_LHS1) else None
            rhs_digit = reversed_RHS[col]

            # Constraint for sum decomposition with positional weights
            def sum_constraint(d0, d1, cin, s):
                return s == (d0 * 100 + d1 * 10 + (cin if cin is not None else 0))

            # Add constraints for the current column
            if lhs0_digit and lhs1_digit:
                problem.constraints.append(BinaryConstraint((lhs0_digit, aux_sum),
                                                            lambda d0, s: s >= d0 * 100))  # Weighted contribution from d0
                problem.constraints.append(BinaryConstraint((lhs1_digit, aux_sum),
                                                            lambda d1, s: s >= d1 * 10))  # Weighted contribution from d1

            if carry_in:
                problem.constraints.append(BinaryConstraint((carry_in, aux_sum),
                                                            lambda cin, s: s >= cin))  # Contribution from carry-in

            if rhs_digit:
                def rhs_constraint(s, r):
                    return s % 10 == r  # Units place of the sum matches the RHS digit
                problem.constraints.append(BinaryConstraint((aux_sum, rhs_digit), rhs_constraint))

            if carry_out:
                def carry_constraint(s, cout):
                    return s // 10 == cout  # Tens place of the sum matches carry-out
                problem.constraints.append(BinaryConstraint((aux_sum, carry_out), carry_constraint))

        return problem


    # Read a cryptarithmetic puzzle from a file
    @staticmethod
    def from_file(path: str) -> "CryptArithmeticProblem":
        with open(path, 'r') as f:
            return CryptArithmeticProblem.from_text(f.read())