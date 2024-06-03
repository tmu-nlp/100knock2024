# 29. Country flag

import wptools
import requests

# Fetch the page data for the given Wikipedia page
page = wptools.page("United Kingdom").get_parse()

# Extract the Infobox "country" data
infobox_country = page.data['infobox']

# Extract the filename of the country flag
flag_filename = infobox_country.get('image_flag')

# Call the MediaWiki API to get information about the image
api_url = "https://en.wikipedia.org/w/api.php"
params = {
    "action": "query",
    "titles": f"File:{flag_filename}",
    "prop": "imageinfo",
    "iiprop": "url",
    "format": "json"
}
response = requests.get(api_url, params=params)
data = response.json()

# Extract the URL of the country flag
flag_info = data['query']['pages']
flag_url = flag_info[next(iter(flag_info))]['imageinfo'][0]['url']

# Display the URL of the country flag
print("URL of the country flag:", flag_url)
