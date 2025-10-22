# Feature importance with the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Define features and target variable
X = housing_data.drop('price', axis=1)
y = housing_data['price']

# Fit Random Forest model
model = RandomForestRegressor()
model.fit(X, y)

# Get feature importances
feature_importances = model.feature_importances_

# Print feature importances
print("Feature Importances:")
for i, feature_name in enumerate(X.columns):
    print(f"{feature_name}: {feature_importances[i]}")

# Feature importance for xgboost, random forest and decision trees

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Load the dataset from CSV file in Google Colab
file_path = 'Housing.csv'  # Change the file name if needed
housing_data = pd.read_csv(file_path)

# Define a dictionary to map 'yes' and 'no' to 1 and 0 respectively
binary_mapping = {'yes': 1, 'no': 0}

# Map 'yes' and 'no' to 1 and 0 in the specific columns
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
housing_data[binary_columns] = housing_data[binary_columns].replace(binary_mapping)

# Convert 'furnishingstatus' to binary (1 if 'furnished', 0 otherwise)
housing_data['furnishingstatus'] = (housing_data['furnishingstatus'] == 'furnished').astype(int)

# Define features and target variable
X = housing_data.drop('price', axis=1)
y = housing_data['price']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Random Forest": RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Multiple Linear Regression": LinearRegression(),
    "XGBoost": XGBRegressor()
}

# Train each model and store feature importances in a DataFrame
feature_importance_df = pd.DataFrame(index=X.columns)
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    # Get feature importances
    if name == 'Multiple Linear Regression':  # Linear regression doesn't provide feature importances
        print(f"{name} doesn't provide feature importances")
        feature_importance_df[name] = 0  # Placeholder for Linear Regression
    else:
        feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_
        total_importance = sum(feature_importances)
        feature_importances_percentage = (feature_importances / total_importance) * 100

        # Store feature importances in the DataFrame
        feature_importance_df[name] = feature_importances_percentage

# Display feature importances
print("\nFeature Importances (Percentage):")
print(feature_importance_df)
