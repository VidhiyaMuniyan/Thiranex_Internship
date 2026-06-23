# ============================================================
# CUSTOMER SPENDING PREDICTION
# HISTORICAL DATA ANALYSIS
# ============================================================

# ============================================================
# STEP 1 : IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt


# ============================================================
# STEP 2 : LOAD DATASET
# ============================================================

df = pd.read_csv("Mall_Customers.csv")

print(df.head())


# ============================================================
# STEP 3 : ENCODE CATEGORICAL DATA
# ============================================================

encoder = LabelEncoder()

df["Gender"] = encoder.fit_transform(df["Gender"])

# Male = 1
# Female = 0


# ============================================================
# STEP 4 : DEFINE FEATURES AND TARGET
# ============================================================

X = df[
    [
        "Gender",
        "Age",
        "Annual Income (k$)"
    ]
]

y = df["Spending Score (1-100)"]


# ============================================================
# STEP 5 : TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# STEP 6 : TRAIN MODEL
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed")


# ============================================================
# STEP 7 : MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

print("\nPredictions")
print(y_pred[:10])


# ============================================================
# STEP 8 : MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("MAE :", round(mae,2))

print("MSE :", round(mse,2))

print("RMSE:", round(rmse,2))

print("R2 Score:", round(r2,2))


# ============================================================
# STEP 9 : ACTUAL VS PREDICTED
# ============================================================

comparison = pd.DataFrame(
    {
        "Actual": y_test,
        "Predicted": y_pred
    }
)

print(comparison.head(20))


# ============================================================
# STEP 10 : VISUALIZATION
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Spending Score")

plt.ylabel("Predicted Spending Score")

plt.title("Actual vs Predicted")

plt.show()


# ============================================================
# STEP 11 : PREDICT NEW CUSTOMER
# ============================================================

# Example:
# Male = 1
# Age = 30
# Income = 80

new_customer = pd.DataFrame(
    {
        "Gender":[1],
        "Age":[30],
        "Annual Income (k$)":[80]
    }
)

prediction = model.predict(new_customer)

print(
    "\nPredicted Spending Score:",
    round(prediction[0],2)
)

# ============================================================
# END
# ============================================================
