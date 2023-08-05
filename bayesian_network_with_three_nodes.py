# pip install pgmpy

from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Define the Bayesian network structure
model = BayesianModel([('A', 'C'), ('B', 'C')])

# Define the data
data = {
    'A': [0, 0, 1, 1],
    'B': [0, 1, 0, 1],
    'C': [0, 0, 0, 1]
}

# Create a MaximumLikelihoodEstimator
mle = MaximumLikelihoodEstimator(model, data)

# Estimate the parameters based on the data
model.fit(data, estimator=mle)

# Create a VariableElimination object for inference
inference = VariableElimination(model)

# Perform inference
# P(C|A=1, B=1)
result = inference.map_query(variables=['C'], evidence={'A': 1, 'B': 1})

# Print the result
print(result)
