import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(42)

# -----------------------------
# Step 1: Simulate Data
# -----------------------------
n = 10000
data = pd.DataFrame({
    'Weather': np.random.choice(['Sunny', 'Rainy', 'Windy'], p=[0.4, 0.4, 0.2], size=n),
    'TeamSkill': np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2], size=n),
})

# Dependencies
def generate_site_access(row):
    if row['Weather'] == 'Sunny':
        return np.random.choice(['Easy', 'Moderate', 'Difficult'], p=[0.7, 0.2, 0.1])
    elif row['Weather'] == 'Rainy':
        return np.random.choice(['Easy', 'Moderate', 'Difficult'], p=[0.2, 0.5, 0.3])
    else:
        return np.random.choice(['Easy', 'Moderate', 'Difficult'], p=[0.3, 0.4, 0.3])

def generate_risk(row):
    if row['Weather'] == 'Rainy' and row['TeamSkill'] == 'Low':
        return np.random.choice(['Low', 'Medium', 'High'], p=[0.1, 0.3, 0.6])
    elif row['Weather'] == 'Sunny' and row['TeamSkill'] == 'High':
        return np.random.choice(['Low', 'Medium', 'High'], p=[0.7, 0.2, 0.1])
    else:
        return np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.5, 0.3])

def generate_equipment(row):
    if row['SiteAccess'] == 'Difficult' or row['Weather'] == 'Rainy':
        return np.random.choice(['Good', 'Average', 'Poor'], p=[0.2, 0.5, 0.3])
    else:
        return np.random.choice(['Good', 'Average', 'Poor'], p=[0.6, 0.3, 0.1])

def generate_delay(row):
    if row['RiskLevel'] == 'High' and row['EquipmentStatus'] == 'Poor':
        return np.random.choice(['NoDelay', 'MinorDelay', 'MajorDelay'], p=[0.05, 0.15, 0.8])
    elif row['RiskLevel'] == 'Low' and row['EquipmentStatus'] == 'Good':
        return np.random.choice(['NoDelay', 'MinorDelay', 'MajorDelay'], p=[0.8, 0.15, 0.05])
    else:
        return np.random.choice(['NoDelay', 'MinorDelay', 'MajorDelay'], p=[0.4, 0.3, 0.3])

# Generate dependent columns
data['SiteAccess'] = data.apply(generate_site_access, axis=1)
data['RiskLevel'] = data.apply(generate_risk, axis=1)
data['EquipmentStatus'] = data.apply(generate_equipment, axis=1)
data['Delay'] = data.apply(generate_delay, axis=1)

# -----------------------------
# Step 2: Define Network Structure
# -----------------------------
edges = [
    ('Weather', 'RiskLevel'),
    ('TeamSkill', 'RiskLevel'),
    ('Weather', 'SiteAccess'),
    ('RiskLevel', 'Delay'),
    ('EquipmentStatus', 'Delay'),
    ('Weather', 'EquipmentStatus'),
    ('SiteAccess', 'EquipmentStatus')
]

# -----------------------------
# Step 3: Visualize Network
# -----------------------------
G = nx.DiGraph()
G.add_edges_from(edges)
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, arrows=True, font_size=12)
plt.title("Three-Layer Bayesian Network Structure")
plt.show()

# -----------------------------
# Step 4: Compute CPTs
# -----------------------------
def compute_cpt(df: pd.DataFrame, parents: list, child: str) -> pd.DataFrame:
    cpt = df.groupby(parents + [child]).size().unstack().fillna(0)
    cpt = cpt.div(cpt.sum(axis=1), axis=0)
    return cpt

# Build CPTs
cpt_risk = compute_cpt(data, ['TeamSkill', 'Weather'], 'RiskLevel')
cpt_site = compute_cpt(data, ['Weather'], 'SiteAccess')
cpt_equipment = compute_cpt(data, ['Weather', 'SiteAccess'], 'EquipmentStatus')
cpt_delay = compute_cpt(data, ['RiskLevel', 'EquipmentStatus'], 'Delay')

# -----------------------------
# Step 5: Likelihood-Weighted Sampling
# -----------------------------
def likelihood_weighted_sampling(data, query_var, evidence, n=5000):
    samples = []
    weights = []

    for _ in range(n):
        weight = 1.0
        sample = {}

        for col in ['Weather', 'TeamSkill', 'SiteAccess', 'RiskLevel', 'EquipmentStatus', 'Delay']:
            parents = {
                'SiteAccess': ['Weather'],
                'RiskLevel': ['TeamSkill', 'Weather'],
                'EquipmentStatus': ['Weather', 'SiteAccess'],
                'Delay': ['RiskLevel', 'EquipmentStatus']
            }.get(col, [])

            if col in evidence:
                sample[col] = evidence[col]
                if parents:
                    try:
                        cpt = compute_cpt(data, parents, col)
                        parent_vals = tuple(sample[p] for p in parents)
                        prob = cpt.loc[parent_vals][evidence[col]]
                        weight *= prob
                    except KeyError:
                        weight *= 0.0
                else:
                    prob = data[col].value_counts(normalize=True).get(evidence[col], 0.0)
                    weight *= prob
            else:
                if parents:
                    cpt = compute_cpt(data, parents, col)
                    try:
                        parent_vals = tuple(sample[p] for p in parents)
                        prob_row = cpt.loc[parent_vals]
                        val = np.random.choice(prob_row.index, p=prob_row.values)
                    except KeyError:
                        val = np.random.choice(data[col].unique())
                else:
                    val = np.random.choice(data[col].unique(), p=data[col].value_counts(normalize=True).values)
                sample[col] = val

        samples.append(sample[query_var])
        weights.append(weight)

    result = pd.Series(samples)
    weighted_probs = result.groupby(result).apply(lambda x: np.sum([weights[i] for i in x.index]))
    return (weighted_probs / weighted_probs.sum()).sort_values(ascending=False)

# -----------------------------
# Step 6: Run Query
# -----------------------------
posterior = likelihood_weighted_sampling(
    data=data,
    query_var='Delay',
    evidence={'RiskLevel': 'High', 'EquipmentStatus': 'Poor'},
    n=5000
)

print("\nPosterior P(Delay | RiskLevel=High, EquipmentStatus=Poor):")
print(posterior)
