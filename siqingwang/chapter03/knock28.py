# 28. Remove MediaWiki markups
import wptools
import re

def clean_markup(text):
    # Remove emphasis markup like ''italic'' and '''bold'''
    text = re.sub(r"''+", '', text)
    # Remove internal links like [[Link Text]] or [[Link Text|Displayed Text]]
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove citations like [1], [2], etc.
    text = re.sub(r'\[.*?\]', '', text)
    # Remove templates like {{Template Name}} or {{Template Name|Parameter}}
    text = re.sub(r'\{\{.*?\}\}', '', text)
    # Remove lines with | symbols (often used for table syntax)
    text = re.sub(r'\n\|.*?\n', '\n', text)
    # Remove any other MediaWiki markup if needed
    # Add more rules as necessary

    return text.strip()

# Fetch the page data for the given Wikipedia page
page = wptools.page("United Kingdom").get_parse()

# Extract the Infobox "country" data
infobox_country = page.data['infobox']

# Clean the values and obtain basic information in plain text format
basic_info = {}
for field, value in infobox_country.items():
    cleaned_value = clean_markup(value)
    basic_info[field] = cleaned_value

# Display the basic information
print("Basic Information for 'United Kingdom':")
for field, value in basic_info.items():
    print(f"{field}: {value}")
