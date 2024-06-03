# 22. Category names
# Extract the category names of the article.

import requests

# Title of the Wikipedia page
wiki_page_title = "United_Kingdom"

# API endpoint URL
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

    # Extract the page object
    page = next(iter(json_data['query']['pages'].values()))

    # Extract the categories
    categories = page.get('categories', [])

    # Extract the category names
    category_names = [category['title'] for category in categories]

    # Print the category names
    print("Categories of the Wikipedia article:")
    for name in category_names:
        print(name)
else:
    print("Error:", response.status_code)
