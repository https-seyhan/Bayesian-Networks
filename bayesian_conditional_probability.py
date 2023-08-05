# Define the Bayesian network using dictionaries to represent conditional probabilities

# Prior probabilities
P_A = {
    'True': 0.3,    # P(A=True)
    'False': 0.7,   # P(A=False)
}

# Conditional probabilities P(B | A)
P_B_given_A = {
    'True|True': 0.9,    # P(B=True | A=True)
    'True|False': 0.6,   # P(B=True | A=False)
    'False|True': 0.1,   # P(B=False | A=True)
    'False|False': 0.4,  # P(B=False | A=False)
}

# Conditional probabilities P(C | B)
P_C_given_B = {
    'True|True': 0.8,    # P(C=True | B=True)
    'True|False': 0.3,   # P(C=True | B=False)
    'False|True': 0.2,   # P(C=False | B=True)
    'False|False': 0.7,  # P(C=False | B=False)
}

def get_prob(var, value, evidence={}):
    """Compute the probability of a variable given evidence using the Bayesian network."""
    if var == 'A':
        return P_A[value]
    elif var == 'B':
        parent_value = evidence.get('A', None)
        key = f"{value}|{parent_value}"
        return P_B_given_A[key]
    elif var == 'C':
        parent_value = evidence.get('B', None)
        key = f"{value}|{parent_value}"
        return P_C_given_B[key]
    else:
        raise ValueError(f"Variable '{var}' is not in the Bayesian network.")

# Example usage:
if __name__ == "__main__":
    evidence = {'A': 'True', 'B': 'True'}
    probability_c_true_given_evidence = get_prob('C', 'True', evidence)
    print(f"P(C=True | A=True, B=True) = {probability_c_true_given_evidence}")
