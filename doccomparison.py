import difflib
import requests
import nltk

nltk.download('punkt')
# from utils import generate_word_diff
import requests
from bs4 import BeautifulSoup


# Reference material: https://graphics.wsj.com/fed-statement-tracker-embed/

date1 = "20241107"
url_general = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{}a.htm"
url_date1 = url_general.format(date1)

date2 = "20240918"
url_date2 = url_general.format(date2)

response = requests.get(url_date1)
response2 = requests.get(url_date2)

# Parse the content with BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")
soup2 = BeautifulSoup(response2.content, "html.parser")

# Find the target div with the specified class
target_div = soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
target_div2 = soup2.find("div", class_="col-xs-12 col-sm-8 col-md-8")

# Extract and print the contents of the target div
if target_div and target_div2:
    print(target_div.text)
    print(target_div2.text)
else:
    print("Target div not found.")


def generate_word_diff(text1, text2):
    """
    Compares two texts word-by-word, highlighting insertions, deletions, and unchanged text.
    """
    # Split texts by words for comparison
    text1_words = text1.split()
    text2_words = text2.split()

    # Use difflib for a word-by-word comparison
    diff = list(difflib.ndiff(text1_words, text2_words))

    # HTML output
    html_output = "<div style='font-family: Arial, sans-serif; line-height: 1.6;'>"

    for word in diff:
        # Unchanged words (no formatting)
        if word.startswith(" "):
            html_output += f"{word[2:]} "
        # Deletion with strikethrough
        elif word.startswith("-"):
            html_output += f"<span style='color: red; text-decoration: line-through;'>{word[2:]}</span> "
        # Insertion with highlighted background
        elif word.startswith("+"):
            html_output += f"<span style='color: black; background-color: lightyellow;'>{word[2:]}</span> "

    html_output += "</div>"
    return html_output

with open("fomc_word_diff_output.html", "w") as file:
    file.write(generate_word_diff(target_div2.text, target_div.text))

