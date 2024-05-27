# 23. Section structure

import requests

# Title of the Wikipedia page
wiki_page_title = "United_Kingdom"

# API endpoint URL
api_url = "https://en.wikipedia.org/w/api.php"

# Parameters for the API request
params = {
    'action': 'parse',
    'format': 'json',
    'page': wiki_page_title,
    'prop': 'sections'
}

# Send a GET request to the API endpoint
response = requests.get(api_url, params=params)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    json_data = response.json()

    # Extract the sections
    sections = json_data.get('parse', {}).get('sections', [])

    # Print the section names with their levels
    print("Sections in the Wikipedia article:")
    for section in sections:
        section_name = section['line']
        section_level = int(section['level'])
        print(f"Level {section_level}: {section_name}")
else:
    print("Error:", response.status_code)
