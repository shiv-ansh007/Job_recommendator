# JobPulse AI Dashboard

JobPulse is a Streamlit application for exploring job-market data and estimating salaries from job attributes. It combines data collection, cleaning, exploratory analysis, and machine-learning prediction in one project.

## Features

- Filter jobs by location, experience level, remote status, and salary band.
- View hiring locations, salary distribution, experience levels, remote work, and skill demand.
- Explore the filtered job dataset and download it as a CSV file.
- Predict an estimated salary using:
  - Skill count
  - Remote status
  - Location
  - Experience level
  - Job-description length

## Project workflow

1. `import requests.py` collects jobs from the [Arbeitnow Job Board API](https://www.arbeitnow.com/api/job-board-api) and saves them to `raw_jobs_1000.csv`.
2. `Data_cleaning.py` removes duplicates, fills missing values, standardizes fields, creates derived features, and saves `cleaned_jobs.csv`.
3. `eda_analysis.py` generates exploratory charts and summary statistics.
4. `model_training.py` trains a `RandomForestRegressor` and saves the model to `salary_model.pkl`.
5. `app.py` launches the interactive Streamlit dashboard.

## Requirements

- Python 3.9 or newer
- The dependencies listed in `requirements.txt`

Install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Run the dashboard

The repository includes the processed dataset and trained model, so the dashboard can be started directly:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Rebuild the data and model

To run the complete pipeline from fresh API data:

```bash
python "import requests.py"
python Data_cleaning.py
python model_training.py
```

To generate the exploratory analysis charts:

```bash
python eda_analysis.py
```

## Repository contents

| File | Description |
| --- | --- |
| `app.py` | Streamlit dashboard and salary-prediction interface |
| `import requests.py` | Fetches job listings from the Arbeitnow API |
| `Data_cleaning.py` | Cleans data and creates model features |
| `eda_analysis.py` | Produces exploratory visualizations |
| `model_training.py` | Trains and evaluates the salary model |
| `raw_jobs_1000.csv` | Raw collected job listings |
| `cleaned_jobs.csv` | Processed dataset used by the app |
| `salary_model.pkl` | Serialized trained Random Forest model |
| `requirements.txt` | Python dependencies |

## Important note about salary data

The source API does not provide salary information for these records. `Data_cleaning.py` therefore creates a reproducible synthetic `salary` column using a seeded random generator. Salary predictions and model metrics are consequently demonstrations of the project workflow, not verified market compensation estimates.

## Data source

Job listings are collected from the public Arbeitnow Job Board API. Please review and follow the API's current usage terms when collecting new data.
