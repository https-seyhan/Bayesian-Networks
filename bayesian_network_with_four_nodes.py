#Create a simple Bayesian network with nodes A, B, C, and D. 
#Then provide some sample data to learn the parameters of the network using Maximum Likelihood Estimation (MLE). 
#Finally, use the VariableElimination class to perform inference and find the most probable value of node D given evidence that A=1 and B=1



from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian network
model = BayesianModel([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])

# Sample data for learning the parameters
data = [
    {'A': 0, 'B': 0, 'C': 0, 'D': 0},
    {'A': 0, 'B': 0, 'C': 0, 'D': 0},
    {'A': 0, 'B': 1, 'C': 1, 'D': 0},
    {'A': 1, 'B': 1, 'C': 1, 'D': 1},
    {'A': 1, 'B': 1, 'C': 0, 'D': 1},
    {'A': 1, 'B': 0, 'C': 1, 'D': 1},
    {'A': 1, 'B': 0, 'C': 0, 'D': 1}
]

# Use Maximum Likelihood Estimation (MLE) to estimate the parameters
mle = MaximumLikelihoodEstimator(model, data)

# Fit the model with the MLE estimates
model.fit(data, estimator=mle)

# Create an instance of the VariableElimination class for inference
inference = VariableElimination(model)

# Perform inference on the network
result = inference.map_query(variables=['D'], evidence={'A': 1, 'B': 1})
print("P(D|A=1, B=1):", result)
