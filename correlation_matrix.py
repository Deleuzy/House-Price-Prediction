# Correlation Matrix where I can see which features are more important 

# Importing necessary libraries
import pandas as pd

# Load the dataset from CSV file in Google Colab
file_path = 'Housing.csv'  # Change the file name if needed
housing_data = pd.read_csv(file_path)

# Convert categorical variables to numerical using one-hot encoding
housing_data = pd.get_dummies(housing_data, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'])

# Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(housing_data.head())

# Perform correlation analysis
correlation_matrix = housing_data.corr()

# Display correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Optional: Visualize the correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Housing Data")
plt.show()

# Another one that puts them in like nummerical order from the most important to the least 

# Extract correlation coefficients between 'price' and all other variables
price_correlation = correlation_matrix['price'].drop('price')  # Drop 'price' to avoid self-correlation

# Sort the correlations in descending order
price_correlation_sorted = price_correlation.abs().sort_values(ascending=False)

# Display the top correlated variables with 'price'
print("Top correlated variables with 'price':")
print(price_correlation_sorted)
