# Assignment_1
## Announcement
This assignment is a test of your algorithmic thinking. You can use any A.I. tools ot assist you in this coding assignment.

Instructions:

This is done by pair, preferably your thesis partner.
Each person should create a Github Repo titled 'Assignment_1_Data_Analytics'.
Read the journal provided.
Develop a Python implementation of the procedures in the journal.
Deadline is before premidterm week.


## Start
Python Installation
Link: https://www.python.org/downloads/
Verify in terminal
py --version

Install libraries
py -m pip install numpy pandas scipy networkx

py -m pip install matplotlib seaborn

Run code
py Filename.py

🔍 Balanced Risk Set Matching
Balanced Risk Set Matching (BRSM), a technique used in observational studies to create fair comparisons between treated and control groups when randomized trials are not possible.
The goal is to match treated patients (who received a treatment at a certain time) with control patients (who had a similar history but had not yet received the treatment at that time). This allows us to estimate treatment effects while reducing bias.

🔑 Key Concepts in Your Code
1️⃣ Matching Treated and Control Patients
We separate patients into:
Treated group (received treatment)
Control group (not yet treated)
Instead of random assignment (like in a clinical trial), we find the best possible match for each treated patient based on historical symptoms.

2️⃣ Mahalanobis Distance for Finding Best Matches
You use Mahalanobis distance, which measures how similar two patients are based on their symptoms.
It accounts for correlations between variables, making it better than simple Euclidean distance.
The goal is to minimize this distance when forming matches.

3️⃣ Solving the Optimal Matching Problem
The script computes Mahalanobis distances between each treated and control patient.
It then solves an optimization problem using the Hungarian algorithm (linear_sum_assignment from SciPy).
This ensures each treated patient is matched to the closest control patient while maintaining balance.

🔬 What is This Useful For?
Medical Studies (like in the paper)
Comparing treated vs. untreated patients when random trials are unethical or impractical.
Example: Evaluating the effect of cystoscopy and hydrodistention on interstitial cystitis.
Economics & Social Sciences
Estimating impact of policies (e.g., how minimum wage increases affect employment).
Matching companies receiving government grants with similar ones that didn’t.
Machine Learning & Causal Inference
Ensuring fairness in AI predictions by matching individuals across different groups.
Evaluating interventions in A/B testing scenarios.

🚀 To do
Python script: ✅ Loads patient data with symptoms and treatment history.
✅ Computes Mahalanobis distances to measure similarity between patients.
✅ Finds optimal matches between treated and control groups using a matching algorithm.
✅ Ensures fair comparisons by only using past symptom data (not future).
✅ Outputs matched pairs that are as similar as possible in symptoms before treatment.




















