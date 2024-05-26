# 26. Remove emphasis markups

import wptools
import re

def clean_markup(text):
    # Remove emphasis markup like ''italic'' and '''bold'''
    text = re.sub(r"''+", '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove any other MediaWiki markup if needed
    # Add more rules as necessary

    return text

# Fetch the page data for the given Wikipedia page
page = wptools.page("United Kingdom").get_parse()

# Extract the infobox data
infobox = page.data['infobox']

# Clean the infobox values and store them in a dictionary
cleaned_infobox = {field: clean_markup(value) for field, value in infobox.items()}

# Display the cleaned infobox data
print("Cleaned Infobox data for 'United Kingdom':")
for field, value in cleaned_infobox.items():
    print(f"{field}: {value}")

# Store this data in a dictionary
infobox_dict = dict(cleaned_infobox)

# Print the dictionary
print("\nCleaned Infobox dictionary:")
print(infobox_dict)
