import streamlit as st
from utils import process_documents_new
from utils import list_pdf_files
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.vectorstores import FAISS
from sidebar import sidebar
from ui import is_open_ai_key_valid, is_query_valid
from qa import query_folder
from plot import plot_gpt
import plotly.graph_objects as go
import os

def mainmodule(openai_api_key):
    # Title of the app
    st.title("Bank of England Report Simulation")

    # sidebar()
    # openai_api_key = st.session_state.get("OPENAI_API_KEY")
    #
    # if not openai_api_key:
    #     st.warning(
    #         "Enter your OpenAI API key in the sidebar. You can get a key at"
    #         " https://platform.openai.com/account/api-keys."
    #     )

    # Instructions for the user
    st.write("Press the button below to display projections.")


    MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]
    model = st.selectbox("Model", options=MODEL_LIST)  # type: ignore



    central_banks = ["FED", "BoE", "ECB"]
    selected_cb = st.multiselect("Central Bank", options=central_banks)  # type: ignore

    if not is_open_ai_key_valid(openai_api_key):
        st.stop()

    if not selected_cb:
        st.stop()


    if selected_cb:
        st.write("You selected:", ", ".join(selected_cb))


        if "BoE" in selected_cb:
            folder_path = "./BoE"
            process_documents_new(folder_path)

            pdf_files_list = list_pdf_files(folder_path)
            st.write('-----------------------------------')
            st.write('These are the files in this folder:')
            st.write('-----------------------------------')
            for pdf_file in pdf_files_list:
                st.write(pdf_file)
            st.write('-----------------------------------')







        with st.spinner("Indexing document... This may take a while⏳"): # TODO: Progress bar
            embeddings = OpenAIEmbeddings() # "text-embedding-ada-002"
            docsall = process_documents_new(folder_path,recursive=False,embeddings=embeddings) #process documents into a Document schema
            vector_store=FAISS.from_documents(docsall,embeddings) #using openai schema, we process documents into vector database
            retriever = vector_store.as_retriever(search_kwargs={"k": 5}) #get top k docs # this can be an argaparser requirement

    # Additional UI elements can go here
    # st.write("This is a synthetic representation for illustrative purposes.")

    with st.form(key="qa_form"):
        query = st.text_area("Ask a question about the document")
        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("Submit")
        with col2:
            plot = st.form_submit_button("Plot")


    if plot:
        fig = go.Figure()
        llm = OpenAI()
        fig = plot_gpt(query,"TBC",llm=llm,folder_index=vector_store)# check for valid query
        st.plotly_chart(fig)



    if submit:
        if not is_query_valid(query):
            st.stop()

        # Output Columns
        answer_col, sources_col = st.columns(2)


        llm = OpenAI()
        result = query_folder(
            folder_index=vector_store,
            query=query,
            llm=llm, return_all=False
        )


        #TODO: feature wishlist, ability to copy output; and re-run output; maybe with a guiding plea
        with answer_col:
            st.markdown("#### Answer")
            st.markdown(result.answer)

        with sources_col:
            st.markdown("#### Sources")
            for source in result.sources:
                st.markdown(source.page_content)
                st.markdown(source.metadata["source"])
                st.markdown("Page: "+str(source.metadata["page"]))
                st.markdown("---")






if __name__ == "__mainmodule__":
    mainmodule()