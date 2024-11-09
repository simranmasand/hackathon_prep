from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
import os
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
import difflib



def process_documents_new(directory):
    '''
    This function uses the CharacterTextSplitter to make documents into smaller chunks admissible into LLMs

    :param pdf_files_list: this is the list of files from the directory
    :return: returns a doc list of processed documents
    '''
    loader = DirectoryLoader(directory, glob="./*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    #TODO add the metadata to the file here. This represents the
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 100, length_function = len, is_separator_regex = False)
    texts = text_splitter.split_documents(docs)

    return texts

def list_pdf_files(directory):
    '''
    Takes a directory and returns the list of pdfs in it
    :param directory: directory path for files you want to search
    :return: returns a list of all files names that are pdfs
    '''
    pdf_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    return pdf_files




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

