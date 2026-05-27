import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ---------------------------------
# LOAD CLEAN DATASET
# ---------------------------------

df = pd.read_csv("cleaned_jobs.csv")

print("Dataset Shape:", df.shape)

# ---------------------------------
# ENCODE CATEGORICAL FEATURES
# ---------------------------------

location_encoder = LabelEncoder()

experience_encoder = LabelEncoder()

df['location_encoded'] = location_encoder.fit_transform(
    df['location']
)

df['experience_encoded'] = experience_encoder.fit_transform(
    df['experience_level']
)

# ---------------------------------
# SELECT FEATURES
# ---------------------------------

X = df[
    [
        'skill_count',
        'remote',
        'location_encoded',
        'experience_encoded',
        'description_length'
    ]
]

# Target variable

y = df['salary']

# ---------------------------------
# TRAIN TEST SPLIT
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)

print("Testing Data Shape:", X_test.shape)

# ---------------------------------
# MODEL TRAINING
# ---------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ---------------------------------
# PREDICTION
# ---------------------------------

predictions = model.predict(X_test)

# ---------------------------------
# EVALUATION
# ---------------------------------
 

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = mse ** 0.5

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nMODEL PERFORMANCE")

print(f"RMSE : {rmse:.2f}")

print(f"MAE  : {mae:.2f}")

print(f"R² Score : {r2:.2f}")

# ---------------------------------
# FEATURE IMPORTANCE
# ---------------------------------

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:\n")

print(importance_df)

# ---------------------------------
# SAVE MODEL
# ---------------------------------

with open("salary_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel Saved Successfully")

print("File Name: salary_model.pkl")