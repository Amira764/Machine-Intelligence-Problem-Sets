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

        # Collect variables (letters) and carries
        letters = set(LHS0 + LHS1 + RHS)
        carries = [f"c{i}" for i in range(len(RHS))]

        # Variables include all letters and carry variables
        problem.variables = list(letters) + carries

        # Domains: letters (digits 0-9), carries (0 or 1)
        problem.domains = {var: set(range(10)) for var in letters}
        for carry in carries:
            problem.domains[carry] = {0, 1}

        # Unary constraints: First letters of each word cannot be zero
        for word in (LHS0, LHS1, RHS):
            problem.constraints.append(UnaryConstraint(word[0], lambda x: x != 0))

        # Binary constraints: All letters must map to unique digits
        for i, letter1 in enumerate(letters):
            for letter2 in list(letters)[i + 1:]:
                problem.constraints.append(BinaryConstraint((letter1, letter2), lambda x, y: x != y))

        # Add constraints for column-wise addition
        reversed_LHS0 = LHS0[::-1]
        reversed_LHS1 = LHS1[::-1]
        reversed_RHS = RHS[::-1]

        for col in range(len(reversed_RHS)):
            carry_in = f"c{col - 1}" if col > 0 else None
            carry_out = f"c{col}" if col < len(reversed_RHS) - 1 else None

            lhs0_digit = reversed_LHS0[col] if col < len(reversed_LHS0) else None
            lhs1_digit = reversed_LHS1[col] if col < len(reversed_LHS1) else None
            rhs_digit = reversed_RHS[col]

            # Binary constraint: LHS0_digit + LHS1_digit + carry_in = RHS_digit + 10 * carry_out
            def column_constraint(lhs0, lhs1, rhs, carry_in_val, carry_out_val):
                total = lhs0 + lhs1 + (carry_in_val if carry_in_val is not None else 0)
                return total % 10 == rhs and total // 10 == carry_out_val

            # Add constraints for the column
            if lhs0_digit and lhs1_digit:
                problem.constraints.append(BinaryConstraint((lhs0_digit, lhs1_digit),
                                                            lambda x, y: x + y <= 19))  # Ensure feasible sums
            if carry_in:
                problem.constraints.append(BinaryConstraint((lhs0_digit, carry_in),
                                                            lambda x, y: x + y <= 10))  # Feasible digit + carry
                problem.constraints.append(BinaryConstraint((lhs1_digit, carry_in),
                                                            lambda x, y: x + y <= 10))  # Feasible digit + carry
            if carry_out:
                problem.constraints.append(BinaryConstraint((carry_out, rhs_digit),
                                                            lambda carry, rhs: carry * 10 + rhs <= 19))  # Feasible carry

            if carry_in and carry_out:
                problem.constraints.append(BinaryConstraint((carry_in, carry_out),
                                                            lambda in_carry, out_carry: in_carry <= out_carry))

        return problem