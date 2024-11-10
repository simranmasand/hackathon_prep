# import streamlit as st
from main_module import *
from sidebar import sidebar
from ui import is_open_ai_key_valid, is_query_valid
import streamlit as st

st.title("Greenspan")

sidebar()
openai_api_key = st.session_state.get("OPENAI_API_KEY")

if not openai_api_key:
    st.warning(
        "Enter your OpenAI API key in the sidebar. You can get a key at"
        " https://platform.openai.com/account/api-keys."
    )



# Define functions for each tab
def main_tab_1():
    st.write("This is the content for Tab 1.")
    # Your code logic for Tab 1 goes here
    # Call main() or other specific functions as needed

def main_tab_2():
    st.write("This is the content for Tab 2.")
    # Your code logic for Tab 2 goes here

def main_tab_3():
    st.write("This is the content for Tab 3.")
    # Your code logic for Tab 3 goes here

# Streamlit tab structure
tabs = st.tabs(["Analytics", "Document Comparison", "HDS"])

# Initialize session states for tabs
if "tab1_active" not in st.session_state:
    st.session_state.tab1_active = False
if "tab2_active" not in st.session_state:
    st.session_state.tab2_active = False
if "tab3_active" not in st.session_state:
    st.session_state.tab3_active = False


# Link each tab to its respective function
with tabs[0]:
    if st.button("Activate Tab 1"):
        st.session_state.tab1_active = True

    if st.session_state.tab1_active:
        mainmodule(openai_api_key)

        if st.button("Close Analytics"):
            st.session_state.tab1_active = False

with tabs[1]:
    if st.button("Activate Tab 2"):
        st.session_state.tab2_active = True

    if st.session_state.tab2_active:
        from doccomparison import *
        doccompst()

        if st.button("Close Document Comparison"):
            st.session_state.tab2_active = False

with tabs[2]:
    main_tab_3()