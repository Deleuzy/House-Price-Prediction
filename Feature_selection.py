# Feature selection for Linear Regression 

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Define features and target variable
X = housing_data.drop('price', axis=1)
y = housing_data['price']

# Create model
model = LinearRegression()

# Initialize RFE
rfe = RFE(model, n_features_to_select=5)  # Choose the number of features to select

# Fit RFE
rfe.fit(X, y)

# Print selected features
print("Selected Features:")
for i, feature_name in enumerate(X.columns):
    if rfe.support_[i]:
        print(feature_name)

# Feature selection via k-best

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Define features and target variable
X = housing_data.drop('price', axis=1)
y = housing_data['price']

# Create model
model = LinearRegression()

# Initialize RFE
rfe = RFE(model, n_features_to_select=5)  # Choose the number of features to select

# Fit RFE
rfe.fit(X, y)

# Print selected features
print("Selected Features:")
for i, feature_name in enumerate(X.columns):
    if rfe.support_[i]:
        print(feature_name)
