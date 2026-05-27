import pandas as pd
import numpy as np

# -------------------------------
# LOAD DATA
# -------------------------------

df = pd.read_csv("raw_jobs_1000.csv")

print("Initial Shape:", df.shape)

# -------------------------------
# CHECK NULL VALUES
# -------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# REMOVE DUPLICATES
# -------------------------------

df.drop_duplicates(inplace=True)

print("\nShape After Removing Duplicates:", df.shape)

# -------------------------------
# HANDLE MISSING VALUES
# -------------------------------

df['location'].fillna('Unknown', inplace=True)

df['tags'].fillna('Not Specified', inplace=True)

df['job_types'].fillna('Unknown', inplace=True)

df['description'].fillna('No Description', inplace=True)

# -------------------------------
# STANDARDIZE TEXT
# -------------------------------

df['title'] = df['title'].str.strip()

df['company'] = df['company'].str.strip()

df['location'] = df['location'].str.title()

# -------------------------------
# CONVERT REMOTE COLUMN
# -------------------------------

# Convert True/False to 1/0

df['remote'] = df['remote'].astype(int)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

# Count number of skills

df['skill_count'] = df['tags'].apply(
    lambda x: len(str(x).split(','))
)

# Experience level based on skill count

def experience_level(skill_count):

    if skill_count <= 3:
        return 'Beginner'

    elif skill_count <= 6:
        return 'Intermediate'

    else:
        return 'Advanced'

df['experience_level'] = df['skill_count'].apply(
    experience_level
)

# -------------------------------
# CREATE SYNTHETIC SALARY COLUMN
# -------------------------------

# Since API doesn't provide salary
# we generate realistic salaries

np.random.seed(42)

df['salary'] = np.random.randint(
    30000,
    150000,
    size=len(df)
)

# -------------------------------
# CREATE SALARY BAND
# -------------------------------

def salary_band(x):

    if x < 50000:
        return 'Low'

    elif x < 100000:
        return 'Medium'

    else:
        return 'High'

df['salary_band'] = df['salary'].apply(
    salary_band
)

# -------------------------------
# DESCRIPTION LENGTH
# -------------------------------

df['description_length'] = df['description'].apply(
    lambda x: len(str(x))
)

# -------------------------------
# FINAL DATASET INFO
# -------------------------------

print("\nFinal Dataset Info:\n")

print(df.info())

print("\nSample Data:\n")

print(df.head())

# -------------------------------
# SAVE CLEAN DATASET
# -------------------------------

df.to_csv("cleaned_jobs.csv", index=False)

print("\nCleaned dataset saved successfully")
print("File Name: cleaned_jobs.csv")