import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------
# 1. Load the Banking Dataset
# ---------------------------------------

# Banking.csv must be in the same folder as this Python file
file_path = r"C:\Users\SSC\OneDrive\Documents\Projects\Banking customer analysis\Banking.csv"
df = pd.read_csv(file_path)
sns.set_theme(style="whitegrid")

# ---------------------------------------
# 2. Initial Data Exploration
# ---------------------------------------

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# ---------------------------------------
# 3. Create Income Bands
# ---------------------------------------

bins = [0, 100000, 300000, float("inf")]
labels = ["Low", "Med", "High"]

df["Income Band"] = pd.cut(
    df["Estimated Income"],
    bins=bins,
    labels=labels,
    right=False
)

print("\nIncome Band Counts:")
print(df["Income Band"].value_counts())

plt.figure(figsize=(8, 5))
df["Income Band"].value_counts().reindex(labels).plot(
    kind="bar",
    color="steelblue"
)
plt.title("Customer Distribution by Income Band")
plt.xlabel("Income Band")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ---------------------------------------
# 4. Categorical Columns
# ---------------------------------------

categorical_cols = [
    "BRId",
    "GenderId",
    "IAId",
    "Amount of Credit Cards",
    "Nationality",
    "Occupation",
    "Fee Structure",
    "Loyalty Classification",
    "Properties Owned",
    "Risk Weighting",
    "Income Band"
]

# Check that the required columns exist
missing_cols = [
    col for col in categorical_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"Missing categorical columns in Banking.csv: {missing_cols}"
    )

# Display value counts
for col in categorical_cols:
    print(f"\nValue Counts for '{col}':")
    print(df[col].value_counts(dropna=False))

# ---------------------------------------
# 5. Univariate Analysis: Categorical Data
# ---------------------------------------

for col in categorical_cols:
    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x=col,
        order=df[col].value_counts().index,
        color="steelblue"
    )

    plt.title(f"Customer Distribution by {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ---------------------------------------
# 6. Bivariate Analysis
#    Categorical Columns vs Nationality
# ---------------------------------------

for col in categorical_cols:
    # Do not compare Nationality with itself
    if col == "Nationality":
        continue

    plt.figure(figsize=(10, 5))

    sns.countplot(
        data=df,
        x=col,
        hue="Nationality"
    )

    plt.title(f"{col} Distribution by Nationality")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Nationality", bbox_to_anchor=(1.02, 1),
               loc="upper left")
    plt.tight_layout()
    plt.show()

# ---------------------------------------
# 7. Numerical Columns
# ---------------------------------------

numerical_cols = [
    "Estimated Income",
    "Superannuation Savings",
    "Credit Card Balance",
    "Bank Loans",
    "Bank Deposits",
    "Checking Accounts",
    "Saving Accounts",
    "Foreign Currency Account",
    "Business Lending"
]

missing_numeric_cols = [
    col for col in numerical_cols
    if col not in df.columns
]

if missing_numeric_cols:
    raise ValueError(
        f"Missing numerical columns in Banking.csv: {missing_numeric_cols}"
    )

# Convert columns to numeric where possible
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------
# 8. Univariate Analysis: Numerical Data
# ---------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.histplot(
        data=df,
        x=col,
        kde=True,
        ax=axes[i],
        color="steelblue"
    )

    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# ---------------------------------------
# 9. Correlation Matrix and Heatmap
# ---------------------------------------

correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="crest",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Banking Features Correlation Matrix")
plt.tight_layout()
plt.show()

# ---------------------------------------
# 10. Finish
# ---------------------------------------

print("\nBanking EDA completed successfully!")
