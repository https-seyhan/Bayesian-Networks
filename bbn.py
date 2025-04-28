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
        for col in ['Weather', 'TeamSkill', 'SiteAccess', 'RiskLevel', 'EquipmentStatus', 'Delay']:
            parents = {
                'RiskLevel': ['TeamSkill', 'Weather'],
                'SiteAccess': ['Weather'],
                'EquipmentStatus': ['Weather', 'SiteAccess'],
                'Delay': ['RiskLevel', 'EquipmentStatus']
            }.get(col, [])

            if col in evidence:
                sample[col] = evidence[col]
                if parents:
                    cpt = compute_cpt(data, parents, col)
                    parent_vals = tuple(sample[p] for p in parents)
                    try:
                        weight *= cpt.loc[parent_vals][evidence[col]]
                    except:
                        weight *= 0
                else:
                    prob = data[col].value_counts(normalize=True)
                    weight *= prob.get(evidence[col], 0)
            else:
                if parents:
                    cpt = compute_cpt(data, parents, col)
                    parent_vals = tuple(sample[p] for p in parents)
                    prob_row = cpt.loc[parent_vals]
                    val = np.random.choice(prob_row.index, p=prob_row.values)
                else:
                    prob = data[col].value_counts(normalize=True)
                    val = np.random.choice(prob.index, p=prob.values)
                sample[col] = val

        samples.append(sample[query_var])
        weights.append(weight)

    result = pd.Series(samples)
    weighted_probs = result.groupby(result).apply(
        lambda x: np.sum([weights[i] for i in x.index])
    )
    return (weighted_probs / weighted_probs.sum()).sort_values(ascending=False)

# Example query
posterior = likelihood_weighted_sampling(
    data=data,
    query_var='Delay',
    evidence={'RiskLevel': 'High', 'EquipmentStatus': 'Poor'},
    n=5000
)

print("\nPosterior P(Delay | RiskLevel=High, EquipmentStatus=Poor):")
print(posterior)


🔄 Generates simulated categorical data

🧠 Defines and visualizes the BBN structure

📊 Creates CPTs directly from data


Creates a dataset of 1000 rows where:

    Weather is Sunny/Rainy/Windy

    Equipment status is Good/Average/Poor

    Team skill is High/Medium/Low

    Site access is Easy/Moderate/Difficult

    Risk level is Low/Medium/High

    Delay is NoDelay/MinorDelay/MajorDelay

Each column is generated using random choices with given probabilities.





🔁 Samples full joint distributions

🎯 Performs likelihood-weighted inference based on any evidence


compute_cpt(): Given parent columns and a child column, it:

    Groups data by parents and child.

    Counts occurrences.

    Normalizes to make it a probability table.

Builds Conditional Probability Tables:

    How RiskLevel depends on TeamSkill and Weather

    How SiteAccess depends on Weather

    etc.

sample_joint():

    Simulates new rows.

    Samples variables sequentially:

        If no parents → sample from marginal probability.

        If parents → sample according to CPT conditional on sampled parents.

Useful to generate synthetic scenarios consistent with the model.

likelihood_weighted_sampling():

    Given some known evidence (e.g., RiskLevel=High and EquipmentStatus=Poor), estimate the posterior probability of another variable (e.g., Delay).

    Process:

        Fix evidence variables.

        Adjust weight based on how likely the evidence is under sampled parents.

        Return a weighted estimate of the queried variable.

Why use weights?

    Because not all samples are equally likely given the evidence.

Concept | Meaning
Bayesian Network (BN) | Graph representing conditional dependencies
CPT (Conditional Probability Table) | Table defining probability of a variable given its parents
Joint Sampling | Sampling a full scenario consistent with BN
Likelihood-Weighted Sampling | Sampling while adjusting probability weights to account for evidence

If Weather is bad and Equipment is poor, there's a higher chance of delay.

Your model learns these relationships and can simulate or predict based on them.
