from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

# Create the Bayesian network model
bayesian_model = BayesianModel([('Weather', 'Play')])

# Define the data (Weather, Play)
data = [
    ('Sunny', 'Yes'),
    ('Sunny', 'Yes'),
    ('Sunny', 'Yes'),
    ('Sunny', 'Yes'),
    ('Rainy', 'No'),
    ('Rainy', 'No'),
    ('Rainy', 'No'),
    ('Rainy', 'No')
]

# Fit the data to the model using Maximum Likelihood Estimation
estimator = MaximumLikelihoodEstimator(bayesian_model)
bayesian_model.fit(data, estimator)

# Calculate the probabilities for the variables
prob_weather_sunny = bayesian_model.get_cpds('Weather')['Sunny'].values[0]
prob_play_yes_given_sunny = bayesian_model.get_cpds('Play')['Sunny'].values[0]
prob_weather_rainy = bayesian_model.get_cpds('Weather')['Rainy'].values[0]
prob_play_yes_given_rainy = bayesian_model.get_cpds('Play')['Rainy'].values[0]

# Print the probabilities
print(f"P(Weather=Sunny) = {prob_weather_sunny}")
print(f"P(Play=Yes|Weather=Sunny) = {prob_play_yes_given_sunny}")
print(f"P(Weather=Rainy) = {prob_weather_rainy}")
print(f"P(Play=Yes|Weather=Rainy) = {prob_play_yes_given_rainy}")
