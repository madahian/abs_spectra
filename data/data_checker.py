import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("./molecular_data.csv")

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Distribution Plots
plt.figure(figsize=(12, 5))
plt.hist(df["Wavelength"], bins=30, alpha=0.7, label="Wavelength")
plt.hist(df["Absorption Maxima"], bins=30, alpha=0.7, label="Absorption Maxima")
plt.legend()
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Distribution of Wavelength and Absorption Maxima")
plt.show()
