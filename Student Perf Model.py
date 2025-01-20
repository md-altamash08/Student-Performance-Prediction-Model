import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset
# Ensure the correct path for your CSV file
data = pd.read_csv('student_data.csv')

# Step 2: Inspect the first few rows and column names
print("First few rows of the dataset:")
print(data.head())

print("\nColumn Names in the Dataset:")
print(data.columns)

# Step 3: Clean the data (remove any extra spaces from column names)
data.columns = data.columns.str.strip()

# Step 4: Feature selection - Separating features (X) and target variable (y)
if 'final_grade' in data.columns:
    X = data.drop('final_grade', axis=1)  # Features
    y = data['final_grade']  # Target variable (final grade)
else:
    print("Error: 'final_grade' column not found.")
    exit()  # Exit if the target column doesn't exist

# Step 5: Handle categorical variables (if any) by encoding them using pd.get_dummies()
X = pd.get_dummies(X)

# Step 6: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Print the sizes of training and testing sets
print(f'\nTraining set size: {len(X_train)}')
print(f'Test set size: {len(X_test)}')

# Step 8: Model Training - Using RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Model Evaluation - Predictions and evaluation metrics
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Check if R² score is valid (i.e., more than 1 sample in test set)
if len(y_test) > 1:
    r2 = r2_score(y_test, y_pred)
else:
    r2 = 'Not enough data points for R² score'

# Step 10: Display results
print(f'\nModel Evaluation:')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R² Score: {r2}')


