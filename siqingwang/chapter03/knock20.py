# 20. Read JSON documents
# Read the JSON documents and output the body of the article about the United Kingdom.

# action=query, format=json, and title=Bla_Bla_Bla are all standard MediaWiki API parameters
# prop=extracts makes us use the TextExtracts extension
# exintro limits the response to content before the first section heading
# explaintext makes the extract in the response be plain text instead of HTML
# Then parse the JSON response and extract the extract:

import requests
response = requests.get(
    'https://en.wikipedia.org/w/api.php',
    params={
        'action': 'query',
        'format': 'json',
        'titles': 'United_Kingdom',
        'prop': 'extracts',
        'exintro': True, # Extract only the introduction part of the page
        'explaintext': True, # Get plain text version
    }).json()
page = next(iter(response['query']['pages'].values()))
print(page['extract'])