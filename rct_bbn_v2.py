import numpy as np
import pandas as pd

# 1. Define Parameters

# Probabilities
p_age_young = 0.6      # 60% are young
p_age_old = 0.4        # 40% are old

# Recovery probabilities:
# recovery_probs[treatment][age]
# treatment: 0 = no, 1 = yes
# age: 0 = young, 1 = old
recovery_probs = {
    0: {  # No treatment
        0: 0.3,   # Young, no treatment → 30% recover
        1: 0.4    # Old, no treatment → 40% recover
    },
    1: {  # Yes treatment
        0: 0.7,   # Young, treatment → 70% recover
        1: 0.6    # Old, treatment → 60% recover
    }
}

# 2. Sampling Functions

def sample_age(n):
    """Sample Age: 0 = Young, 1 = Old"""
    return np.random.choice([0, 1], size=n, p=[p_age_young, p_age_old])

def sample_recovery(treatment, age):
    """Given treatment and age, sample recovery"""
    prob = recovery_probs[treatment][age]
    return np.random.rand() < prob

# 3. Simulate Intervention (RCT)

def simulate_rct(n_samples, treatment_value):
    """
    Simulate randomized controlled trial with forced Treatment= treatment_value (0 or 1)
    """
    ages = sample_age(n_samples)
    recoveries = np.array([sample_recovery(treatment_value, age) for age in ages])
    df = pd.DataFrame({
        'Treatment': treatment_value,
        'Age': ages,
        'Recovery': recoveries.astype(int)
    })
    return df

# 4. Run Simulations

n_samples = 1000

# Group A: Treatment = 1 (Given)
data_treatment = simulate_rct(n_samples, treatment_value=1)

# Group B: Treatment = 0 (Control)
data_control = simulate_rct(n_samples, treatment_value=0)

# 5. Analyze

recovery_rate_treatment = data_treatment['Recovery'].mean()
recovery_rate_control = data_control['Recovery'].mean()

print("Recovery rate WITH treatment:", recovery_rate_treatment)
print("Recovery rate WITHOUT treatment:", recovery_rate_control)
print("Estimated Treatment Effect (difference):", recovery_rate_treatment - recovery_rate_control)


Recovery rate WITH treatment: 0.652
Recovery rate WITHOUT treatment: 0.364
Estimated Treatment Effect (difference): 0.288