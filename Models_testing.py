# I have tested several models on the data from Housing.csv to see which ones are doing a better one + LDA 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load the data
data = pd.read_csv("Housing.csv")

# Encode categorical variables
label_encoder = LabelEncoder()
data['mainroad'] = label_encoder.fit_transform(data['mainroad'])
data['guestroom'] = label_encoder.fit_transform(data['guestroom'])
data['basement'] = label_encoder.fit_transform(data['basement'])
data['hotwaterheating'] = label_encoder.fit_transform(data['hotwaterheating'])
data['airconditioning'] = label_encoder.fit_transform(data['airconditioning'])
data['prefarea'] = label_encoder.fit_transform(data['prefarea'])
data['furnishingstatus'] = label_encoder.fit_transform(data['furnishingstatus'])

# Split data into features and target variable
X = data.drop(columns=['price'])
y = data['price']

# Apply LDA to reduce dimensionality
lda = LDA()
X_lda = lda.fit_transform(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestRegressor(random_state=42)
lr_model = LinearRegression()
dt_model = DecisionTreeRegressor(random_state=42)
svr_model = SVR()
knn_model = KNeighborsRegressor()
xgb_model = XGBRegressor()

# Train models
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
svr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Make predictions on training and testing data
models = {
    "Random Forest": rf_model,
    "Multiple Linear Regression": lr_model,
    "Decision Tree": dt_model,
    "Support Vector Regression": svr_model,
    "K-Nearest Neighbors": knn_model,
    "XGBoost": xgb_model
}

overall_results = []

for name, model in models.items():
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_preds)
    train_mape = mean_absolute_error(y_train, train_preds) / y_train.mean() * 100

    test_mse = mean_squared_error(y_test, test_preds)
    test_mape = mean_absolute_error(y_test, test_preds) / y_test.mean() * 100

    overall_results.append({
        'Model': name,
        'Train MSE': train_mse,
        'Train MAPE': train_mape,
        'Test MSE': test_mse,
        'Test MAPE': test_mape
    })

# Display overall model results
overall_results_df = pd.DataFrame(overall_results)
print(overall_results_df)
