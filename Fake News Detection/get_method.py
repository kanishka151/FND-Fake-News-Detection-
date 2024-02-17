import requests
import csv



url = ('https://newsapi.org/v2/top-headlines?'
       'country=us&'
       'apiKey=f19bcb53eb724a1cb92ea8e0ba72cc7c')
response = requests.get(url)
json_response=response.json()
articles = json_response.get("articles", [])
data = [(article.get("title", ""), article.get("description", ""), article.get("publishedAt", "")) for article in articles]


# Saving data to a CSV file
true_data = "C:\\Users\\smyl2\\Downloads\\News.csv"

with open(true_data, mode="a", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    # Writing data
    csv_writer.writerows(data)

print(f"Data has been saved to {true_data}")