import requests
import pandas as pd
import time

all_jobs = []

# Number of pages to scrape
# Increase if needed

TOTAL_PAGES = 20

for page in range(1, TOTAL_PAGES + 1):

    print(f"Scraping page {page}...")

    url = f"https://www.arbeitnow.com/api/job-board-api?page={page}"

    response = requests.get(url)

    # Check API status
    if response.status_code != 200:
        print(f"Failed at page {page}")
        continue

    data = response.json()

    jobs = data.get('data', [])

    # Stop if no jobs found
    if len(jobs) == 0:
        print("No more jobs available")
        break

    for job in jobs:

        all_jobs.append({
            'title': job.get('title'),
            'company': job.get('company_name'),
            'location': job.get('location'),
            'remote': job.get('remote'),
            'url': job.get('url'),
            'tags': ', '.join(job.get('tags', [])),
            'job_types': ', '.join(job.get('job_types', [])),
            'description': job.get('description')
        })

    # Avoid hitting API too fast
    time.sleep(1)

# Convert to DataFrame
df = pd.DataFrame(all_jobs)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Save CSV
df.to_csv('raw_jobs_1000.csv', index=False)

print("\nData Scraping Completed")
print(f"Total Records Collected: {len(df)}")
print("CSV File Saved Successfully")
print(df)