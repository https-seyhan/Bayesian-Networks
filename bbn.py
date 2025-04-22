import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(42)

# -----------------------------
# Step 1: Simulate Data
# -----------------------------
n = 1000
data = pd.DataFrame({
    'Weather': np.random.choice(['Sunny', 'Rainy', 'Windy'], p=[0.4, 0.4, 0.2], size=n),
    'EquipmentStatus': np.random.choice(['Good', 'Average', 'Poor'], p=[0.5, 0.3, 0.2], size=n),
    'TeamSkill': np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2], size=n),
    'SiteAccess': np.random.choice(['Easy', 'Moderate', 'Difficult'], p=[0.4, 0.4, 0.2], size=n),
    'RiskLevel': np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2], size=n),
    'Delay': np.random.choice(['NoDelay', 'MinorDelay', 'MajorDelay'], p=[0.5, 0.3, 0.2], size=n)
})

# -----------------------------
# Step 2: Define Dependencies
# -----------------------------
edges = [
    ('Weather', 'RiskLevel'),
    ('Weather', 'SiteAccess'),
    ('TeamSkill', 'RiskLevel'),
    ('SiteAccess', 'EquipmentStatus'),
    ('Weather', 'EquipmentStatus'),
    ('RiskLevel', 'Delay'),
    ('EquipmentStatus', 'Delay'),
]

# -----------------------------
# Step 3: Visualize Network
# -----------------------------
G = nx.DiGraph()
G.add_edges_from(edges)
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, arrows=True, font_size=12)
plt.title("Bayesian Network Structure")
plt.show()

# -----------------------------
# Step 4: Build CPT Function
# -----------------------------
def compute_cpt(df: pd.DataFrame, parents: list, child: str) -> pd.DataFrame:
    cpt = df.groupby(parents + [child]).size().unstack().fillna(0)
    cpt = cpt.div(cpt.sum(axis=1), axis=0)
    return cpt

# -----------------------------
# Step 5: Build CPTs
# -----------------------------
cpt_risk = compute_cpt(data, ['TeamSkill', 'Weather'], 'RiskLevel')
cpt_site = compute_cpt(data, ['Weather'], 'SiteAccess')
cpt_equipment = compute_cpt(data, ['Weather', 'SiteAccess'], 'EquipmentStatus')
cpt_delay = compute_cpt(data, ['RiskLevel', 'EquipmentStatus'], 'Delay')

# -----------------------------
# Step 6: Joint Probability Sampling
# -----------------------------
def sample_joint(data: pd.DataFrame, n_samples: int = 5) -> pd.DataFrame:
    samples = []
    for _ in range(n_samples):
        row = {}
        for col in ['Weather', 'TeamSkill', 'SiteAccess', 'RiskLevel', 'EquipmentStatus', 'Delay']:
            parents = {
                'RiskLevel': ['TeamSkill', 'Weather'],
                'SiteAccess': ['Weather'],
                'EquipmentStatus': ['Weather', 'SiteAccess'],
                'Delay': ['RiskLevel', 'EquipmentStatus']
            }.get(col, [])
            if parents:
                cpt = compute_cpt(data, parents, col)
                parent_vals = tuple(row[p] for p in parents)
                probs = cpt.loc[parent_vals]
                val = np.random.choice(probs.index, p=probs.values)
            else:
                probs = data[col].value_counts(normalize=True)
                val = np.random.choice(probs.index, p=probs.values)
            row[col] = val
        samples.append(row)
    return pd.DataFrame(samples)

print("\nSampled Joint Probability Rows:")
print(sample_joint(data, 5))

# -----------------------------
# Step 7: Likelihood-Weighted Sampling
# -----------------------------
def likelihood_weighted_sampling(data, query_var, evidence, n=1000):
    samples = []
    weights = []

    for _ in range(n):
        weight = 1.0
        sample = {}
        for col in ['Weather',
