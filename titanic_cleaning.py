import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 1️⃣ Load Dataset
df = pd.read_csv("train.csv")   # make sure train.csv is in same folder

# 2️⃣ Basic Information
print("Dataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# 3️⃣ Handle Missing Values

# Fill Age with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Fill Embarked with mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
df.drop("Cabin", axis=1, inplace=True)

# 4️⃣ Encode Categorical Variables

# Label Encoding for Sex
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])

# One Hot Encoding for Embarked
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# 5️⃣ Visualize Age Distribution

plt.figure()
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution After Cleaning")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# 6️⃣ Save Cleaned Dataset

df.to_csv("titanic_cleaned.csv", index=False)

print("\nData cleaning completed. Cleaned file saved as titanic_cleaned.csv")
