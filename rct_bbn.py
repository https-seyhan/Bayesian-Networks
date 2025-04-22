import numpy as np
from bayespy.nodes import Categorical, Bernoulli, Dirichlet, Beta
from bayespy.inference import VB
import bayespy.plot as bpplt

np.random.seed(42)

# Number of patients
N = 500

# Simulate categorical variables
treatment_data = np.random.randint(0, 2, N)
gender_data = np.random.randint(0, 2, N)
age_group_data = np.random.randint(0, 3, N)
smoker_data = np.random.randint(0, 2, N)
comorbidity_data = np.random.randint(0, 3, N)

# Simulate outcome based on weighted influence of all variables
def simulate_outcome(t, g, a, s, c):
    prob = 0.4 + 0.2*t - 0.05*g - 0.1*a - 0.1*s - 0.15*c
    return np.random.binomial(1, np.clip(prob, 0.05, 0.95))

outcome_data = np.array([
    simulate_outcome(t, g, a, s, c)
    for t, g, a, s, c in zip(treatment_data, gender_data, age_group_data, smoker_data, comorbidity_data)
])

# One-hot encode categorical variables for modeling
def one_hot(data, num_categories):
    return np.eye(num_categories)[data]

# One-hot inputs
X_treatment = one_hot(treatment_data, 2)
X_gender = one_hot(gender_data, 2)
X_age = one_hot(age_group_data, 3)
X_smoker = one_hot(smoker_data, 2)
X_comorbidity = one_hot(comorbidity_data, 3)

# Combine all features
X = np.hstack([X_treatment, X_gender, X_age, X_smoker, X_comorbidity])

# Create weights and outcome likelihood in Bayesian network
num_features = X.shape[1]

# Priors for weights
weights = Beta([1]*num_features, [1]*num_features)
outcome_prob = np.dot(X, weights.get_moments()[0])  # Expected value of weights

# Clip to [0.05, 0.95] to avoid extremes
outcome_prob = np.clip(outcome_prob, 0.05, 0.95)

# Binary outcome
outcome = Bernoulli(outcome_prob)
outcome.observe(outcome_data)

# Variational inference
Q = VB(weights, outcome)
Q.update(repeat=200)

# Plot posterior distributions for weights
bpplt.plot(weights)

# BBN

[Treatment Assigned] --> [Adherence to Treatment] --> [Blood Pressure Outcome]
                       \                          /
                        ---> [Side Effects] -----/
