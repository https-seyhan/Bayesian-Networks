import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

# Build DAG
G = nx.DiGraph()
G.add_edges_from(edges)
nodes_topo = list(nx.topological_sort(G))

# Plot the graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, arrows=True)
plt.title("Bayesian Network Structure")
plt.show()

# CPT computation
def compute_cpt(df, parents, child):
    if not parents:
        return df[child].value_counts(normalize=True)
    counts = df.groupby(parents + [child]).size().unstack(fill_value=0)
    probs = counts.div(counts.sum(axis=1), axis=0)
    return probs

# Precompute CPTs
CPTs = {}
for node in G.nodes:
    parents = list(G.predecessors(node))
    CPTs[node] = compute_cpt(data, parents, node)

# Likelihood Weighted Sampling with full dependency traversal
def likelihood_weighted_sampling(data, CPTs, G, query_var, evidence, n=5000):
    nodes_topo = list(nx.topological_sort(G))
    samples = []
    weights = []

    for _ in range(n):
        sample = {}
        weight = 1.0

        for node in nodes_topo:
            parents = list(G.predecessors(node))

            if node in evidence:
                val = evidence[node]
                sample[node] = val

                if parents:
                    parent_vals = tuple(sample[p] for p in parents)
                    try:
                        p = CPTs[node].loc[parent_vals][val]
                        weight *= p if not pd.isna(p) else 0
                    except:
                        weight *= 0
                else:
                    p = CPTs[node].get(val, 0)
                    weight *= p
            else:
                if parents:
                    parent_vals = tuple(sample[p] for p in parents)
                    try:
                        probs = CPTs[node].loc[parent_vals]
                        probs = probs[probs > 0]
                        probs = probs / probs.sum()
                        val = np.random.choice(probs.index, p=probs.values)
                    except:
                        # fallback: uniform over all
                        val = np.random.choice(CPTs[node].columns)
                else:
                    probs = CPTs[node]
                    val = np.random.choice(probs.index, p=probs.values)
                sample[node] = val

        samples.append(sample[query_var])
        weights.append(weight)

    result_df = pd.DataFrame({'value': samples, 'weight': weights})
    result = result_df.groupby('value')['weight'].sum()
    result = result / result.sum()
    return result.sort_values(ascending=False)

# Run correct inference
posterior = likelihood_weighted_sampling(
    data=data,
    CPTs=CPTs,
    G=G,
    query_var='Delay',
    evidence={
        'Weather': 'Rainy',
        'TeamSkill': 'Low',
        'SiteAccess': 'Difficult',
        'EquipmentStatus': 'Poor',
        'RiskLevel': 'High'
    },
    n=10000
)

print("\nCorrected Posterior for:")
print("P(Delay | Weather=Rainy, TeamSkill=Low, SiteAccess=Difficult, EquipmentStatus=Poor, RiskLevel=High):")
print(posterior)
