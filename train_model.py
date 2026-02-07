import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("dataset/weather_dataset.csv")

# Convert 'Date' column to datetime → extract month
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')  # Corrected format
df['Month'] = df['Date'].dt.month
df.drop('Date', axis=1, inplace=True)

# Features & Target
X = df.drop("MaxTemp", axis=1)
y = df["MaxTemp"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models to compare
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_score = -1

print("MODEL COMPARISON RESULTS\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(name)
    print("MAE:", mae)
    print("R2 Score:", r2)
    print("---------------------")

    if r2 > best_score:
        best_score = r2
        best_model = model

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save best model
joblib.dump(best_model, "model/temperature_model.pkl")
print("✅ Best model saved successfully!")
