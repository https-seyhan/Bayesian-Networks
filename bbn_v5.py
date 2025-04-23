import numpy as np
import pandas as pd
from pomegranate import *

# 1. Simulate Dataset with RCT
np.random.seed(42)
N = 1000

# Simulate 5-valued treatment (RCT ensures it's random)
treatment_values = ['A', 'B', 'C', 'D', 'E']
T = np.random.choice(treatment_values, size=N)

# Simulate covariates
age = np.random.choice(['young', 'middle', 'old'], size=N, p=[0.3, 0.5, 0.2])
gender = np.random.choice(['male', 'female'], size=N)
ses = np.random.choice(['low', 'medium', 'high'], size=N, p=[0.4, 0.4, 0.2])

# Simulate behavioral_support as a function of treatment and covariates
def simulate_behavior(t, a, g, s):
    score = treatment_values.index(t) * 0.2
    score += {'young': 0.1, 'middle': 0.2, 'old': 0.05}[a]
    score += {'male': 0.1, 'female': 0.2}[g]
    score += {'low': 0.05, 'medium': 0.2, 'high': 0.3}[s]
    prob = min(score / 2.0, 1.0)
    return np.random.choice(['low', 'medium', 'high'], p=[1-prob, prob/2, prob/2])

behavioral_support = [
    simulate_behavior(t, a, g, s) for t, a, g, s in zip(T, age, gender, ses)
]

df = pd.DataFrame({
    'treatment': T,
    'age': age,
    'gender': gender,
    'ses': ses,
    'behavioral_support': behavioral_support
})

# 2. Encode Categorical Variables
def encode_column(col):
    return {val: i for i, val in enumerate(sorted(df[col].unique()))}

encodings = {col: encode_column(col) for col in df.columns}
df_encoded = df.replace({col: enc for col, enc in encodings.items()})

# 3. Build Discrete Distributions and Conditional Probabilities
# Create distributions for each node
treatment_dist = DiscreteDistribution(df_encoded['treatment'].value_counts(normalize=True).to_dict())
age_dist = DiscreteDistribution(df_encoded['age'].value_counts(normalize=True).to_dict())
gender_dist = DiscreteDistribution(df_encoded['gender'].value_counts(normalize=True).to_dict())
ses_dist = DiscreteDistribution(df_encoded['ses'].value_counts(normalize=True).to_dict())

# Conditional probability for behavioral_support
# Given: treatment, age, gender, ses
X = df_encoded[['treatment', 'age', 'gender', 'ses']]
y = df_encoded['behavioral_support']

# Create CPT manually
from collections import defaultdict, Counter

# Build counts
cpt_counts = defaultdict(Counter)
for row, label in zip(X.itertuples(index=False), y):
    cpt_counts[tuple(row)][label] += 1

# Normalize to get probabilities
cpt = {}
for cond, counter in cpt_counts.items():
    total = sum(counter.values())
    cpt[cond] = {k: v / total for k, v in counter.items()}

# Map CPT into format for ConditionalProbabilityTable
cpt_entries = []
for cond, probs in cpt.items():
    for bs_class, prob in probs.items():
        cpt_entries.append(list(cond) + [bs_class, prob])

# Create pomegranate nodes
treatment_node = Node(treatment_dist, name="treatment")
age_node = Node(age_dist, name="age")
gender_node = Node(gender_dist, name="gender")
ses_node = Node(ses_dist, name="ses")

behavior_cpt = ConditionalProbabilityTable(
    cpt_entries,
    [treatment_dist, age_dist, gender_dist, ses_dist]
)
behavior_node = Node(behavior_cpt, name="behavioral_support")

# 4. Build the Bayesian Network
model = BayesianNetwork("Behavioral Support BBN")
model.add_states(treatment_node, age_node, gender_node, ses_node, behavior_node)

model.add_edge(treatment_node, behavior_node)
model.add_edge(age_node, behavior_node)
model.add_edge(gender_node, behavior_node)
model.add_edge(ses_node, behavior_node)

model.bake()

# 5. Inference Example
# Predict behavioral_support given treatment = 'A' and SES = 'medium'
query_input = {
    "treatment": encodings['treatment']['A'],
    "ses": encodings['ses']['medium'],
    "age": encodings['age']['middle'],
    "gender": encodings['gender']['female']
}

beliefs = model.predict_proba(query_input)
decoded_support = beliefs[-1].parameters[0]
decoded_support_named = {
    list(encodings['behavioral_support'].keys())[list(encodings['behavioral_support'].values()).index(k)]: v
    for k, v in decoded_support.items()
}

print("Predicted Behavioral Support Probabilities:")
print(decoded_support_named)

import networkx as nx
import matplotlib.pyplot as plt

# Extract edges from the pomegranate model
edges = [(parent.name, child.name) for parent, child in model.edges]

# Create a directed graph
G = nx.DiGraph()
G.add_edges_from(edges)

# Plot
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)  # You can change layout here
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrowsize=20)
plt.title("Bayesian Belief Network Structure", fontsize=14)
plt.show()

treatment ─┐
           ├──► behavioral_support
     age ──┘
  gender ──┘
      ses ─┘
