import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor

# Load the data
data = pd.read_csv("housing_us.csv")

# Split data into features and target variable
X = data.drop(columns=['price'])
y = data['price']

# Identify categorical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CatBoost Regressor
model = CatBoostRegressor(cat_features=categorical_features, verbose=100)

# Define hyperparameters grid for tuning
param_grid = {
    'iterations': [50],            # Number of trees to fit
    'learning_rate': [0.1],         # Step size shrinkage to prevent overfitting
    'depth': [6],                   # Depth of the trees
    'loss_function': ['RMSE'],      # Loss function to optimize
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Perform GridSearchCV
grid_search.fit(X_train, y_train, eval_set=(X_test, y_test))

# Get the best estimator
best_model = grid_search.best_estimator_

# Make predictions
train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

# Calculate metrics
train_mse = mean_squared_error(y_train, train_preds)
train_mape = mean_absolute_error(y_train, train_preds) / y_train.mean() * 100
test_mse = mean_squared_error(y_test, test_preds)
test_mape = mean_absolute_error(y_test, test_preds) / y_test.mean() * 100

# Display results
print("Best CatBoost Model Parameters:")
print(grid_search.best_params_)
print("Training MSE:", train_mse)
print("Training MAPE:", train_mape)
print("Test MSE:", test_mse)
print("Test MAPE:", test_mape)
