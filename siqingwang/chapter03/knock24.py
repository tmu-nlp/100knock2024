# 24. Media references

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
    'prop': 'images',
    'imlimit': 'max'
}

# Send a GET request to the API endpoint
response = requests.get(api_url, params=params)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    json_data = response.json()

    # Extract the pages dictionary
    pages = json_data['query']['pages']

    # Iterate over each page in the pages dictionary
    for page_id, page_info in pages.items():
        title = page_info.get('title', 'No Title')
        images = page_info.get('images', [])
        image_titles = [image['title'] for image in images]

        # Print the page title
        print(f"Title: {title}")
        print("Media Files:")
        for image in image_titles:
            print(f"  - {image}")
else:
    print(f"Error: {response.status_code}")
