import os
import difflib
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader, PyPDFLoader




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

