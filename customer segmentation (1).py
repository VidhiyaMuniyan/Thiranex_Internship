# ============================================================
# CUSTOMER SEGMENTATION PROJECT
# ============================================================
# Objective:
# Segment customers based on their Annual Income and
# Spending Score using K-Means Clustering.
#
# Technologies:
# Python
# Pandas
# NumPy
# Matplotlib
# Seaborn
# Scikit-Learn
# ============================================================


# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

# Replace with your dataset path
df = pd.read_csv("Mall_Customers.csv")

print("\nDataset Preview\n")
print(df.head())


# ============================================================
# STEP 3: BASIC DATA EXPLORATION
# ============================================================

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())


# ============================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# ---------- AGE DISTRIBUTION ----------

plt.figure(figsize=(8,5))

plt.hist(df["Age"], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.show()


# ---------- GENDER DISTRIBUTION ----------

plt.figure(figsize=(6,4))

sns.countplot(
    x="Gender",
    data=df
)

plt.title("Gender Distribution")

plt.show()


# ---------- ANNUAL INCOME DISTRIBUTION ----------

plt.figure(figsize=(8,5))

plt.hist(
    df["Annual Income (k$)"],
    bins=20
)

plt.title("Annual Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")

plt.show()


# ---------- SPENDING SCORE DISTRIBUTION ----------

plt.figure(figsize=(8,5))

plt.hist(
    df["Spending Score (1-100)"],
    bins=20
)

plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Count")

plt.show()


# ============================================================
# STEP 5: CORRELATION ANALYSIS
# ============================================================

plt.figure(figsize=(8,5))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()


# ============================================================
# STEP 6: SELECT FEATURES FOR CLUSTERING
# ============================================================

# We choose Income and Spending Score

X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

print("\nSelected Features")
print(X.head())


# ============================================================
# STEP 7: FEATURE SCALING
# ============================================================

# K-Means works better when features are scaled

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Feature Sample")
print(X_scaled[:5])


# ============================================================
# STEP 8: FIND OPTIMAL NUMBER OF CLUSTERS
# USING ELBOW METHOD
# ============================================================

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

print("\nWCSS Values")
print(wcss)


# ============================================================
# STEP 9: ELBOW METHOD VISUALIZATION
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker="o"
)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.grid(True)

plt.show()


# ============================================================
# STEP 10: APPLY K-MEANS CLUSTERING
# ============================================================

# Based on elbow graph
# Usually K = 5 for Mall Customer Dataset

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataset

df["Cluster"] = clusters

print("\nCluster Labels Added")
print(df.head())


# ============================================================
# STEP 11: CLUSTER CENTERS
# ============================================================

centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

print("\nCluster Centers")

print(pd.DataFrame(
    centers,
    columns=[
        "Annual Income",
        "Spending Score"
    ]
))


# ============================================================
# STEP 12: VISUALIZE CUSTOMER SEGMENTS
# ============================================================

plt.figure(figsize=(10,6))

scatter = plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"]
)

plt.xlabel("Annual Income")

plt.ylabel("Spending Score")

plt.title("Customer Segments")

plt.colorbar(scatter)

plt.show()


# ============================================================
# STEP 13: CLUSTER WISE ANALYSIS
# ============================================================

cluster_summary = df.groupby("Cluster").agg(
    {
        "Age":"mean",
        "Annual Income (k$)":"mean",
        "Spending Score (1-100)":"mean",
        "CustomerID":"count"
    }
)

print("\nCluster Summary")
print(cluster_summary)


# ============================================================
# STEP 14: VISUALIZE NUMBER OF CUSTOMERS PER CLUSTER
# ============================================================

plt.figure(figsize=(8,5))

sns.countplot(
    x="Cluster",
    data=df
)

plt.title("Customers Per Cluster")

plt.xlabel("Cluster")

plt.ylabel("Number of Customers")

plt.show()


# ============================================================
# STEP 15: BUSINESS INTERPRETATION
# ============================================================

print("\n================ BUSINESS INSIGHTS ================\n")

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[df["Cluster"] == cluster]

    avg_income = cluster_data["Annual Income (k$)"].mean()

    avg_spending = cluster_data["Spending Score (1-100)"].mean()

    print(f"Cluster {cluster}")

    print(f"Average Income: {avg_income:.2f}")

    print(f"Average Spending Score: {avg_spending:.2f}")

    if avg_income > 70 and avg_spending > 70:

        print("Insight: Premium Customers")

    elif avg_income > 70 and avg_spending < 40:

        print("Insight: High Income but Low Spending")

    elif avg_income < 40 and avg_spending > 60:

        print("Insight: Low Income but High Spending")

    elif avg_income < 40 and avg_spending < 40:

        print("Insight: Budget Customers")

    else:

        print("Insight: Average Customers")

    print("-"*50)


# ============================================================
# STEP 16: SAVE FINAL DATASET
# ============================================================

df.to_csv(
    "Customer_Segmentation_Output.csv",
    index=False
)

print("\nClustered Dataset Saved Successfully")


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\nCustomer Segmentation Project Completed Successfully!")
