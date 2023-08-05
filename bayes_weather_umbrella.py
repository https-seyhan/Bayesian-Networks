from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

# Create a Bayesian model
model = BayesianModel()

# Define nodes in the model
model.add_node('Weather')
model.add_node('Carries_Umbrella')

# Define edges in the model (directional dependencies)
model.add_edge('Weather', 'Carries_Umbrella')

# Sample data for training (Assuming 1: Rainy, 0: Sunny; 1: Carries umbrella, 0: Doesn't carry umbrella)
data = [
    {'Weather': 1, 'Carries_Umbrella': 1},
    {'Weather': 0, 'Carries_Umbrella': 1},
    {'Weather': 1, 'Carries_Umbrella': 0},
    {'Weather': 0, 'Carries_Umbrella': 0},
    {'Weather': 1, 'Carries_Umbrella': 1},
    {'Weather': 0, 'Carries_Umbrella': 0},
]

# Fit the model with data using Maximum Likelihood Estimation
estimator = MaximumLikelihoodEstimator(model, data)
model = estimator.estimate_cpd('Weather')
model = estimator.estimate_cpd('Carries_Umbrella')

# Print the Conditional Probability Distributions (CPDs)
print("CPD for Weather:")
print(model.get_cpds('Weather'))
print("\nCPD for Carries_Umbrella:")
print(model.get_cpds('Carries_Umbrella'))
