"""
Exercise 3: Logistic Regression Parameter Finding

PROBLEM STATEMENT:
==================
Given a dataset with 6 samples, find logistic regression parameters that 
correctly classify all points.

Dataset Description:
- X1: Numeric attribute (continuous values: -5, 1, 2)
- X2: Categorical attribute (3 possible states: 'a', 'b', 'c')
- Y: Binary class label (0 or 1)

Data points:
+----+----+----+---+
| ID | X1 | X2 | Y |
+----+----+----+---+
| 1  | 2  | a  | 0 |
| 2  | 2  | a  | 1 |
| 3  | 1  | b  | 0 |
| 4  | -5 | b  | 0 |
| 5  | 2  | c  | 1 |
| 6  | 2  | b  | 0 |
+----+----+----+---+

OBJECTIVE:
==========
Find coefficients (w0, w1, w2, w3) such that:
  P(Y=1|X) = 1 / (1 + exp(-(w0 + w1*X1 + w2*X2_b + w3*X2_c)))

correctly predicts all 6 data points (where X2_b and X2_c are one-hot encoded
with 'a' as the reference category).

SOLUTION APPROACH:
==================
1. Encode categorical variable X2 using one-hot encoding (reference encoding)
2. Use logistic regression to fit the model
3. Verify the model correctly classifies all points
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 1: PREPARE THE DATA
# ============================================================================

# Original raw data
data = {
    'X1': [2, 2, 1, -5, 2, 2],
    'X2': ['a', 'a', 'b', 'b', 'c', 'b'],
    'Y': [0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print()

# ============================================================================
# STEP 2: ENCODE CATEGORICAL VARIABLE
# ============================================================================

# One-hot encode X2 with 'a' as reference (drop_first=True)
X2_encoded = pd.get_dummies(df['X2'], prefix='X2', drop_first=True)
print("Encoded Categorical Features:")
print(X2_encoded)
print()

# Combine numeric and encoded categorical features
X = pd.concat([df[['X1']], X2_encoded], axis=1)
y = df['Y']

print("Feature Matrix X:")
print(X)
print("\nTarget Vector y:")
print(y.values)
print()

# ============================================================================
# STEP 3: TRAIN LOGISTIC REGRESSION MODEL
# ============================================================================

# Fit logistic regression (using high regularization to potentially get cleaner solutions)
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
log_reg.fit(X, y)

# Extract coefficients
w0 = log_reg.intercept_[0]  # Bias term
w1 = log_reg.coef_[0][0]     # Coefficient for X1
w2 = log_reg.coef_[0][1]     # Coefficient for X2_b
w3 = log_reg.coef_[0][2]     # Coefficient for X2_c

print("=" * 70)
print("LEARNED LOGISTIC REGRESSION PARAMETERS:")
print("=" * 70)
print(f"w0 (intercept/bias):       {w0:.6f}")
print(f"w1 (coefficient for X1):   {w1:.6f}")
print(f"w2 (coefficient for X2_b): {w2:.6f}")
print(f"w3 (coefficient for X2_c): {w3:.6f}")
print()

# ============================================================================
# STEP 4: VERIFY PREDICTIONS
# ============================================================================

print("=" * 70)
print("CLASSIFICATION RESULTS:")
print("=" * 70)
predictions = log_reg.predict(X)
probabilities = log_reg.predict_proba(X)[:, 1]

results = pd.DataFrame({
    'X1': df['X1'],
    'X2': df['X2'],
    'True Y': y.values,
    'Predicted Y': predictions,
    'Probability (Y=1)': probabilities
})

print(results)
print()

# Check accuracy
accuracy = (predictions == y.values).sum() / len(y)
print(f"Accuracy: {accuracy * 100:.1f}% ({(predictions == y.values).sum()}/{len(y)} correct)")
print()

# ============================================================================
# STEP 5: DISPLAY DECISION FUNCTION
# ============================================================================

print("=" * 70)
print("DECISION FUNCTION:")
print("=" * 70)
print(f"log(odds) = {w0:.6f} + {w1:.6f}*X1 + {w2:.6f}*X2_b + {w3:.6f}*X2_c")
print()
print("Where:")
print("  - X2_b = 1 if X2='b', else 0")
print("  - X2_c = 1 if X2='c', else 0")
print("  - X2_a is the reference category (X2_b=0, X2_c=0)")
print()

# ============================================================================
# STEP 6: MANUAL VERIFICATION (EXAMPLE)
# ============================================================================

print("=" * 70)
print("MANUAL VERIFICATION EXAMPLES:")
print("=" * 70)

def logistic_prob(x1, x2):
    """Calculate P(Y=1) for given X1 and X2"""
    x2_b = 1 if x2 == 'b' else 0
    x2_c = 1 if x2 == 'c' else 0
    logit = w0 + w1 * x1 + w2 * x2_b + w3 * x2_c
    prob = 1 / (1 + np.exp(-logit))
    return prob

print(f"\nSample 1: X1=2, X2='a' (True Y=0)")
prob = logistic_prob(2, 'a')
pred = 1 if prob > 0.5 else 0
print(f"  P(Y=1) = {prob:.4f}, Predicted class = {pred} ✓" if pred == 0 else f"  P(Y=1) = {prob:.4f}, Predicted class = {pred} ✗")

print(f"\nSample 5: X1=2, X2='c' (True Y=1)")
prob = logistic_prob(2, 'c')
pred = 1 if prob > 0.5 else 0
print(f"  P(Y=1) = {prob:.4f}, Predicted class = {pred} ✓" if pred == 1 else f"  P(Y=1) = {prob:.4f}, Predicted class = {pred} ✗")

print("\n" + "=" * 70)
