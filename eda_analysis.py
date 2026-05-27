import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# LOAD CLEAN DATASET
# -------------------------------

df = pd.read_csv("cleaned_jobs.csv")

print("Dataset Shape:", df.shape)

print("\nColumns:\n", df.columns)

# -------------------------------
# BASIC STATISTICS
# -------------------------------

print("\nSummary Statistics:\n")

print(df.describe())

# -------------------------------
# STYLE SETTINGS
# -------------------------------

sns.set_style("whitegrid")

# -------------------------------
# 1. TOP HIRING LOCATIONS
# -------------------------------

plt.figure(figsize=(12,6))

df['location'].value_counts().head(10).plot(
    kind='bar'
)

plt.title("Top 10 Hiring Locations")

plt.xlabel("Location")

plt.ylabel("Number of Jobs")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# -------------------------------
# 2. SALARY DISTRIBUTION
# -------------------------------

plt.figure(figsize=(10,6))

sns.histplot(df['salary'], bins=20)

plt.title("Salary Distribution")

plt.xlabel("Salary")

plt.ylabel("Frequency")

plt.tight_layout()

plt.show()

# -------------------------------
# 3. REMOTE VS ONSITE JOBS
# -------------------------------

plt.figure(figsize=(6,6))

df['remote'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)

plt.title("Remote vs Onsite Jobs")

plt.ylabel("")

plt.tight_layout()

plt.show()

# -------------------------------
# 4. SKILL COUNT VS SALARY
# -------------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(
    x='skill_count',
    y='salary',
    data=df
)

plt.title("Skill Count vs Salary")

plt.xlabel("Skill Count")

plt.ylabel("Salary")

plt.tight_layout()

plt.show()

# -------------------------------
# 5. EXPERIENCE LEVEL DISTRIBUTION
# -------------------------------

plt.figure(figsize=(8,6))

sns.countplot(
    x='experience_level',
    data=df
)

plt.title("Experience Level Distribution")

plt.xlabel("Experience Level")

plt.ylabel("Count")

plt.tight_layout()

plt.show()

# -------------------------------
# 6. SALARY BAND DISTRIBUTION
# -------------------------------

plt.figure(figsize=(8,6))

sns.countplot(
    x='salary_band',
    data=df
)

plt.title("Salary Band Distribution")

plt.xlabel("Salary Band")

plt.ylabel("Count")

plt.tight_layout()

plt.show()

# -------------------------------
# 7. DESCRIPTION LENGTH ANALYSIS
# -------------------------------

plt.figure(figsize=(10,6))

sns.boxplot(
    x='experience_level',
    y='description_length',
    data=df
)

plt.title("Job Description Length by Experience Level")

plt.xlabel("Experience Level")

plt.ylabel("Description Length")

plt.tight_layout()

plt.show()

# -------------------------------
# 8. CORRELATION HEATMAP
# -------------------------------

plt.figure(figsize=(8,6))

correlation = df[
    [
        'salary',
        'skill_count',
        'remote',
        'description_length'
    ]
].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()

print("\nEDA Completed Successfully")