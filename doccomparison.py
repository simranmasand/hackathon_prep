import difflib
import streamlit as st
# import nltk
# nltk.download('punkt')
# from utils import generate_word_diff
import requests
from bs4 import BeautifulSoup


# Reference material: https://graphics.wsj.com/fed-statement-tracker-embed/
# Reference material2 : https://www.cnbc.com/2024/09/18/fed-rate-cut-heres-what-changed-in-the-central-banks-statement.html

url_general = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{}a.htm"


def fomc_statement(date):
    url_date = url_general.format(date)
    response = requests.get(url_date)
    soup = BeautifulSoup(response.content, "html.parser")
    target_div = soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
    if not target_div:
        print("Target div not found.")
    return target_div.text


def generate_word_diff(text1, text2):
    """
    Compares two texts word-by-word, highlighting insertions, deletions, and unchanged text; text1: old; text2: new.
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

def dateparser(str_input):
    date_pieces = str_input.split("-")
    final_date = date_pieces[2]+date_pieces[1]+date_pieces[0]
    return final_date

date1 = "20241107"
date2 = "20240918"
date3 = "20240731"


with open("fomc_word_diff_output.html", "w") as file:
    file.write(generate_word_diff(fomc_statement(date2),fomc_statement(date1)))

with open("fomc_word_diff_output_prev.html", "w") as file:
    file.write(generate_word_diff(fomc_statement(date3),fomc_statement(date2)))

def doccompst():
    st.title("Compare FOMC Statements")
    st.write("Pick two dates to generate a strikethrough comparison.") #order matters, select new first and then old benchmark.
    st.subheader("Document")
    button_cols_main = st.columns([1,1,1,1,4],gap="small")
    date1 = "20241107"
    date2 = "20240918"
    date3 = "20240731"
    options = ["07-11-2024", "18-09-2024", "31-07-2024","12-06-2024"]
    if "date_main" not in st.session_state:
        st.session_state.date_main = None
    # Assume date_main is selected by the user from a dropdown or other selector

    for i in range(len(options)):
        if button_cols_main[i].button(options[i],key=options[i]):
            date_main = options[i]
            st.session_state.date_main = options[i]

    date_main = st.session_state.date_main

    if date_main:
        # Set up columns for comparison buttons
        st.subheader("Benchmark")

        button_cols_comp = st.columns([1, 1, 1, 1, 4], gap="small")

        # Filter options to exclude the selected date_main
        options_filtered = [opt for opt in options if opt != date_main]

        # Initialize or retrieve date_comp in session state
        if "date_comp" not in st.session_state:
            st.session_state.date_comp = None

        # Loop through filtered options to create comparison buttons
        for i, opt in enumerate(options_filtered):
            if button_cols_comp[i].button(opt, key="second-" + opt):
                st.session_state.date_comp = opt  # Store selected date_comp in session state

        # Get the stored comparison date from session state
        date_comp = st.session_state.date_comp

        # Display and generate comparison if both dates are selected
        if date_main and date_comp:
            st.write(f"Compare the FOMC statement of {date_main} against {date_comp}.")
            if st.button("Go"):
                with st.spinner(f"Generating comparison for date {date_main} using date {date_comp} as a benchmark..."):
                    # Use dateparser or other functions as needed
                    # Example code for generating comparison
                    datenew = dateparser(date_main)
                    dateold = dateparser(date_comp)
                    st.markdown(generate_word_diff(fomc_statement(dateold), fomc_statement(datenew)), unsafe_allow_html=True)

                    st.write("\nComparison generated successfully.")

    if st.button("Reset Comparison"):
        st.session_state.date_comp = None
        st.session_state.markdown_content = ""
        st.experimental_rerun()  # Rerun to update the UI

    # for i in range(len(options)):
    #     if button_cols_main[i].button(options[i],key=options[i]):
    #         date_main = options[i]
    #
    # if date_main:
    #     button_cols_comp = st.columns([1,1,1,1,4],gap="small")
    #     options_filtered = [opt for opt in options if opt != date_main]
    #     date_comp = None
    #     for i in range(len(options_filtered)):
    #         if button_cols_comp[i].button(options_filtered[i],key="second-"+options_filtered[i]):
    #             print("Inside Loop 1")
    #             date_comp = options_filtered[i]
    #             print(date_comp)
    #             st.write("Compare the FOMC statement of {} against {}.".format(date_main,date_comp))
    #             if ((date_main is not None) and (date_comp is not None)):
    #                 print("Inside loop")
    #                 with st.spinner("Generating comparison for date {} using date {} as a benchmark...".format(date_main,date_comp)):
    #                     datenew = dateparser(date_main)
    #                     dateold = dateparser(date_comp)
    #
    #                     st.markdown(generate_word_diff(fomc_statement(dateold), fomc_statement(datenew)),unsafe_allow_html=True)



if __name__ == "__doccomparison__":
    doccompst()