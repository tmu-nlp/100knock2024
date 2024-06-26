# 27. Remove internal links

import wptools
import re

def clean_markup(text):
    # Remove emphasis markup like ''italic'' and '''bold'''
    text = re.sub(r"''+", '', text)
    # Remove internal links like [[Link Text]] or [[Link Text|Displayed Text]]
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove any other MediaWiki markup if needed
    # Add more rules as necessary

    return text

# Fetch the page data for the given Wikipedia page
page = wptools.page("United Kingdom").get_parse()

# Extract the Infobox "country" data
infobox = page.data['infobox']

# Clean the values and store them in a dictionary object
infobox_cleaned = {}
for field, value in infobox.items():
    cleaned_value = clean_markup(value)
    infobox_cleaned[field] = cleaned_value

# Display the cleaned Infobox data
print("Cleaned Infobox data for 'United Kingdom':")
for field, value in infobox_cleaned.items():
    print(f"{field}: {value}")
