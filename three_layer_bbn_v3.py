import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(42)

# Simulate data
n = 1000
data = pd.DataFrame({
    'Weather': np.random.choice(['Sunny', 'Rainy', 'Windy'], p=[0.4, 0.4, 0.2], size=n),
    'EquipmentStatus': np.random.choice(['Good', 'Average', 'Poor'], p=[0.5, 0.3, 0.2], size=n),
    'TeamSkill': np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2], size=n),
    'SiteAccess': np.random.choice(['Easy', 'Moderate', 'Difficult'], p=[0.4, 0.4, 0.2], size=n),
    'RiskLevel': np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2], size=n),
    'Delay': np.random.choice(['NoDelay', 'MinorDelay', 'MajorDelay'], p=[0.5, 0.3, 0.2], size=n)
})

# Define Bayesian Network structure
edges = [
    ('Weather', 'RiskLevel'),
    ('TeamSkill', 'RiskLevel'),
    ('Weather', 'SiteAccess'),
    ('RiskLevel', 'Delay'),
    ('EquipmentStatus', 'Delay'),
    ('Weather', 'EquipmentStatus'),
    ('SiteAccess', 'EquipmentStatus')
]

# Build the DAG
G = nx.DiGraph()
G.add_edges_from(edges)
nodes_topo = list(nx.topological_sort(G))

# Plot network
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, arrows=True, font_size=12)
plt.title("Three-Layer Bayesian Network Structure")
plt.show()

# CPT calculator
def compute_cpt(df, parents, child):
    cpt = df.groupby(parents + [child]).size().unstack().fillna(0)
    cpt = cpt.div(cpt.sum(axis=1), axis=0)
    return cpt

# Precompute all CPTs once
CPTs = {}
for node in G.nodes:
    parents = list(G.predecessors(node))
    CPTs[node] = compute_cpt(data, parents, node) if parents else data[node].value_counts(normalize=True)

# Likelihood Weighted Sampling
def likelihood_weighted_sampling(data, CPTs, G, query_var, evidence, n=5000):
    samples = []
    weights = []

    for _ in range(n):
        sample = {}
        weight = 1.0

        for node in nodes_topo:
            parents = list(G.predecessors(node))

            if node in evidence:
                sample[node] = evidence[node]
                if parents:
                    parent_vals = tuple(sample[p] for p in parents)
                    try:
                        weight *= CPTs[node].loc[parent_vals][evidence[node]]
                    except KeyError:
                        weight *= 0
                else:
                    weight *= CPTs[node].get(evidence[node], 0)
            else:
                if parents:
                    parent_vals = tuple(sample[p] for p in parents)
                    probs = CPTs[node].loc[parent_vals]
                    val = np.random.choice(probs.index, p=probs.values)
                else:
                    probs = CPTs[node]
                    val = np.random.choice(probs.index, p=probs.values)
                sample[node] = val

        samples.append(sample[query_var])
        weights.append(weight)

    result = pd.Series(samples)
    weighted_probs = result.groupby(result).apply(lambda x: np.sum([weights[i] for i in x.index]))
    weighted_probs = weighted_probs / weighted_probs.sum()
    return weighted_probs.sort_values(ascending=False)

# Example Query
posterior = likelihood_weighted_sampling(
    data=data,
    CPTs=CPTs,
    G=G,
    query_var='Delay',
    evidence={'Weather': 'Rainy', 'TeamSkill': 'Low', 'SiteAccess': 'Difficult', 'EquipmentStatus': 'Poor', 'RiskLevel': 'High'},
    n=5000
)

print("\nPosterior P(Delay | Weather=Rainy, TeamSkill=Low, SiteAccess=Difficult, EquipmentStatus=Poor, RiskLevel=High):")
print(posterior)
