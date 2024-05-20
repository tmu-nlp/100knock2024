
# !pip install wptools

import wptools

# Fetch the page data for the given Wikipedia page
page = wptools.page("United Kingdom").get_parse()

# Extract the infobox data
infobox = page.data['infobox']

# Display the infobox data
print("Infobox data for 'United Kingdom':")
for field, value in infobox.items():
    print(f"{field}: {value}")

infobox_dict = dict(infobox)
print("\nInfobox dictionary:")
print(infobox_dict)