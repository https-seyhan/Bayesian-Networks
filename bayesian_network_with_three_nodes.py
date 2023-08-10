# pip install pgmpy

from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd

# Define the Bayesian network structure
model = BayesianModel([('A', 'C'), ('B', 'C')])

# Define the data
data = {
    'A': [0, 0, 1, 1],
    'B': [0, 1, 0, 1],
    'C': [0, 0, 0, 1]
}

print("Model : ", model)
print("Type of Data ", type(data))
print("Data ", data)

# Convert dictionary to DataFrame
data = pd.DataFrame(data)

# Create a MaximumLikelihoodEstimator
mle = MaximumLikelihoodEstimator(model, data)

	
print("mle type ", type(mle))
# Estimate the parameters based on the data
#model.fit(Bayes( A= [0, 0, 1, 1], B= [0, 1, 0, 1], C= [0, 0, 0, 1]), estimator=mle)

#model.fit(data, estimator=mle)
model.fit(data)

# Create a VariableElimination object for inference
inference = VariableElimination(model)

# Perform inference
# P(C|A=1, B=1)
result = inference.map_query(variables=['C'], evidence={'A': 1, 'B': 1})

# Print the result
print(result)
