import csv
from datetime import datetime

# Sample dictionary
data = {
    "Index": [],
    "headline": [],
    "url": [],
    "publisher":[],
    "date":[],
    "stock":[]
}

import csv
import json

# Path to the CSV file
csv_file = "/Users/eunjinoh/Desktop/personal_project/vector_search/raw_partner_headlines.csv"

# List to store the transformed data
documents = []

with open(csv_file, mode="r", newline="", encoding="latin1") as file:
    reader = csv.DictReader(file)  # Automatically maps header to values in each row
    for row in reader:
        # Append the row as a dictionary to the documents list
        documents.append({
            "Index": row["Index"],
            "headline": row["headline"],
            "url": row["url"],
            "publisher": row["publisher"],
            "date": datetime.strptime(row["date"][:-6], "%d/%m/%Y").date().isoformat(),
            "stock": row["stock"]
        })

# Print the output (formatted as JSON for readability)
print(json.dumps(documents, indent=4))

# Save to a JSON file (optional)
with open("transformed_news_headlines.json", "w") as json_file:
    json.dump(documents, json_file, indent=4)