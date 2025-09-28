# ============================
# Data Analysis Project
# Using Pandas + Matplotlib + Seaborn
# Dataset: Iris (from sklearn)
# ============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ----------------------------
# Step 1: Load Dataset
# ----------------------------
iris = load_iris(as_frame=True)
df = iris.frame  # Convert to Pandas DataFrame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print("✅ Dataset loaded successfully!\n")

# ----------------------------
# Step 2: Explore Dataset
# ----------------------------
print("🔹 First 5 rows of dataset:")
print(df.head(), "\n")

print("🔹 Dataset Info:")
print(df.info(), "\n")

print("🔹 Missing Values:")
print(df.isnull().sum(), "\n")

# (If missing values existed, handle them)
df = df.dropna()

# ----------------------------
# Step 3: Basic Data Analysis
# ----------------------------
print("🔹 Summary Statistics:")
print(df.describe(), "\n")

print("🔹 Average Petal Length per Species:")
print(df.groupby("species")["petal length (cm)"].mean(), "\n")

# Find species with longest petals
longest_petal = df.groupby("species")["petal length (cm)"].mean().idxmax()
print(f"🌸 Species with longest average petals: {longest_petal}\n")

# ----------------------------
# Step 4: Data Visualizations
# ----------------------------

# 1. Line Chart (Trend of Sepal Length across index)
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length", color="blue")
plt.title("Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart (Average Petal Length per Species)
df.groupby("species")["petal length (cm)"].mean().plot(
    kind="bar", color=["skyblue", "lightgreen", "salmon"]
)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram (Sepal Length Distribution)
plt.hist(df["sepal length (cm)"], bins=20, color="purple", edgecolor="black")
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Count")
plt.show()

# 4. Scatter Plot (Sepal Length vs Petal Length)
plt.scatter(
    df["sepal length (cm)"],
    df["petal length (cm)"],
    c=df["target"],
    cmap="viridis",
)
plt.title("Sepal vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.colorbar(label="Species")
plt.show()

# ----------------------------
# Step 5: Error Handling Example
# ----------------------------
try:
    df2 = pd.read_csv("mydata.csv")  # Replace with your dataset
    print("✅ File loaded successfully!")
except FileNotFoundError:
    print("⚠️ Error: The file was not found.")
except Exception as e:
    print("⚠️ An error occurred:", e)

# ----------------------------
# Step 6: Observations
# ----------------------------
"""
📌 Observations:
1. Iris-setosa has the smallest petal length on average.
2. Iris-virginica has the longest petals overall.
3. Sepal length distribution is roughly normal.
4. There is a strong positive relationship between sepal length and petal length.
"""
