
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# 1. LOAD THE BANKING DATASET
# ==================================================

# CHANGE THIS PATH TO YOUR Banking.csv FILE LOCATION
file_path = r"C:\Users\SSC\Downloads\Banking.csv"

df = pd.read_csv(file_path)

# Display first five rows
print("\nFirst Five Rows:")
print(df.head())

# ==================================================
# 2. DATAFRAME SHAPE AND INFORMATION
# ==================================================

print("\nShape of the DataFrame:", df.shape)

print("\nDataFrame Information:")
df.info()

# ==================================================
# 3. CREATE INCOME BANDS
# ==================================================

print("\nEstimated Income:")
print(df["Estimated Income"].head())

bins = [0, 100000, 300000, float("inf")]
labels = ["Low", "Mid", "High"]

df["Income Band"] = pd.cut(
    df["Estimated Income"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# ==================================================
# 4. CATEGORICAL DATA ANALYSIS
# ==================================================

categorical_cols = [
    "Risk Weighting",
    "Nationality",
    "Occupation",
    "Fee Structure",
    "Loyalty Classification",
    "Properties Owned",
    "Income Band"
]

for col in categorical_cols:
    print(f"\nValue Counts for '{col}':")
    print(df[col].value_counts(dropna=False))

# ==================================================
# 5. DESCRIPTIVE STATISTICS
# ==================================================

print("\nDescriptive Statistics:")
print(df.describe(include="all"))

# ==================================================
# 6. CHECK MISSING VALUES
# ==================================================

print("\nMissing Values Per Column:")
print(df.isnull().sum())

# ==================================================
# 7. CONVERT JOINED BANK TO DATETIME
# ==================================================

df["Joined Bank"] = pd.to_datetime(
    df["Joined Bank"],
    format="%d-%m-%Y",
    errors="coerce"
)

print("\nJoined Bank Data Type:")
print(df["Joined Bank"].dtype)

# ==================================================
# 8. UNIVARIATE ANALYSIS
# ==================================================

numerical_cols = [
    "Age",
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

fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.histplot(data=df, x=col, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")

# Hide unused subplots
for j in range(len(numerical_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# ==================================================
# 9. CORRELATION ANALYSIS
# ==================================================

correlation_cols = [
    "Age",
    "Estimated Income",
    "Superannuation Savings",
    "Credit Card Balance",
    "Bank Loans",
    "Bank Deposits",
    "Checking Accounts",
    "Saving Accounts",
    "Foreign Currency Account",
    "Business Lending",
    "Properties Owned"
]

correlation_matrix = df[correlation_cols].corr()

plt.figure(figsize=(12, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix of Banking Features")
plt.tight_layout()
plt.show()

# ==================================================
# 10. BIVARIATE ANALYSIS
# ==================================================

pairs_to_plot = [
    ("Bank Deposits", "Saving Accounts"),
    ("Checking Accounts", "Saving Accounts"),
    ("Checking Accounts", "Foreign Currency Account"),
    ("Age", "Superannuation Savings"),
    ("Estimated Income", "Checking Accounts"),
    ("Bank Loans", "Credit Card Balance"),
    ("Business Lending", "Bank Loans")
]

for x_col, y_col in pairs_to_plot:
    plt.figure(figsize=(8, 6))

    sns.regplot(
        data=df,
        x=x_col,
        y=y_col,
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"}
    )

    plt.title(f"{x_col} vs {y_col}", fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)

    plt.tight_layout()
    plt.show()

print("\nBanking Data Analysis Completed Successfully!")

