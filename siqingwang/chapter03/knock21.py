# 21. Lines with category names

import requests

# URL of the Wikipedia page for the United Kingdom
wiki_page_title = "United_Kingdom"

# API endpoint URL for fetching the categories of the page
api_url = "https://en.wikipedia.org/w/api.php"

# Parameters for the API request
params = {
    'action': 'query',
    'format': 'json',
    'titles': wiki_page_title,
    'prop': 'categories',
    'cllimit': 'max'
}

# Send a GET request to the API endpoint
response = requests.get(api_url, params=params)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    json_data = response.json()

    # Extract the categories from the JSON response
    pages = json_data['query']['pages']
    for page_id in pages:
        categories = pages[page_id].get('categories', [])
        category_titles = [category['title'] for category in categories]

    # Print the categories
    print("Categories of the Wikipedia article:")
    for title in category_titles:
        print(title)
else:
    print("Error:", response.status_code)
