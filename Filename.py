import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linprog

# Function to compute Mahalanobis distance
def compute_mahalanobis(u, v, cov_inv):
    return mahalanobis(u, v, cov_inv)

# Loading the data.csv file
data = pd.read_csv("data.csv")

data.loc[:, 'Treatment_Time'] = data['Treatment_Time'].fillna(0)

numeric_cols = ['Pain_Score', 'Urgency_Score', 'Frequency_Score', 'Treatment_Time']
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric)
data['Treated'] = data['Treated'].astype(int)
data['ID'] = data['ID'].astype(int)


treated = data[data['Treated'] == 1].reset_index(drop=True)
control = data[(data['Treated'] == 0) & (data['Treatment_Time'] < treated['Treatment_Time'].max())].reset_index(drop=True)

# Computing covariance matrix and its inverse
from numpy.linalg import LinAlgError

try:
    cov_matrix = np.cov(data[['Pain_Score', 'Urgency_Score', 'Frequency_Score']].values.T)
    cov_inv = np.linalg.inv(cov_matrix)
except LinAlgError:
    print("Warning: Covariance matrix is singular, adding small regularization term.")
    cov_inv = np.linalg.pinv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)  # Use pseudo-inverse
print(data[['Pain_Score', 'Urgency_Score', 'Frequency_Score']].describe())

# Computing distance matrix
distance_matrix = np.zeros((len(treated), len(control)))  # Defining matrix

for i, t in treated.iterrows():
    for j, c in control.iterrows():
        dist = compute_mahalanobis(
            t[['Pain_Score', 'Urgency_Score', 'Frequency_Score']].values,
            c[['Pain_Score', 'Urgency_Score', 'Frequency_Score']].values,
            cov_inv
        )
        distance_matrix[i, j] = dist

# Solve Integer Programming problem using linear programming
num_treated, num_control = distance_matrix.shape
cost = distance_matrix.flatten()
A_eq = np.zeros((num_treated, num_treated * num_control))  

for i in range(num_treated):
    A_eq[i, i * num_control:(i + 1) * num_control] = 1  

b_eq = np.ones(num_treated) 
print(f"Number of treated patients: {num_treated}")
print(f"Number of control patients: {num_control}")
print(f"Distance Matrix Shape: {distance_matrix.shape}")
print(f"Flattened Cost Array Shape: {cost.shape}")
print(f"Constraint Matrix Shape: {A_eq.shape}")
print(f"Constraint Vector Shape: {b_eq.shape}")
if num_treated > num_control:
    print("⚠️ Warning: More treated patients than available controls. Matching might be infeasible.")

res = linprog(cost, A_eq=A_eq, b_eq=b_eq, method="highs")
print(f"Linprog Status: {res.status}")
print(f"Linprog Message: {res.message}")
print(f"Optimal Solution Found? {res.success}")

#Optimal Matching
matched_pairs = []
if res.success and res.x is not None:
    matches = np.where(res.x.reshape(num_treated, num_control) > 0.5)
    matched_pairs = list(zip(treated.iloc[matches[0]]['ID'], control.iloc[matches[1]]['ID']))
    # Save matched pairs to matched_pairs.csv
    matched_df = pd.DataFrame(matched_pairs, columns=["Treated_ID", "Control_ID"])
    matched_df.to_csv("matched_pairs.csv", index=False)
    print("Matched pairs saved to matched_pairs.csv")

    for pair in matched_pairs:
        print(f"Treated: {pair[0]} <-> Control: {pair[1]}")
else:
    print("Linear programming failed to find a solution. Adjust constraints or check data distribution.")

# Analyze treatment effects
treated_symptoms = treated[['Pain_Score', 'Urgency_Score', 'Frequency_Score']].mean()
control_symptoms = control[['Pain_Score', 'Urgency_Score', 'Frequency_Score']].mean()
print("Symptom Changes - Treated vs Control:")
print(treated_symptoms - control_symptoms)

# Sensitivity analysis for different matching thresholds
for threshold in [0.1, 0.2, 0.5]:
    new_control = control[control['Treatment_Time'] < treated['Treatment_Time'].max() - threshold]
    print(f"Sensitivity Analysis with threshold {threshold}: {len(new_control)} control patients available.")

# Visualization of symptom changes
#pwede rani icomment out nga part if di needed
sns.boxplot(data=[treated['Pain_Score'], control['Pain_Score']], palette="Set2")
plt.title("Pain Score Distribution - Treated vs Control")
plt.xticks([0, 1], ["Treated", "Control"])
plt.show()