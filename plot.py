import plotly.graph_objects as go
import json
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from prompts import PLOT_PROMPT
from langchain.chains.qa_with_sources import load_qa_with_sources_chain



def plot_gpt(plot_prompt, plot_title,llm, folder_index):
    relevant_docs = folder_index.similarity_search(plot_prompt, k=5)
    if relevant_docs is not None:
        chunk1 = relevant_docs[0]
        chunk2 = relevant_docs[1]
        chunk3 = relevant_docs[2]
        chunk4 = relevant_docs[3]
        chunk5 = relevant_docs[4]

        chain = load_qa_with_sources_chain(
            llm=llm,
            chain_type="stuff",
            prompt=PLOT_PROMPT,
        )

        relevant_docs = folder_index.similarity_search(plot_prompt, k=5)
        response = chain(
            {"input_documents": relevant_docs, "question":plot_prompt},
            return_only_outputs=True
        )
        sources = relevant_docs
        # print(response)
        # print(response["output_text"])
        # print(response["output_text"][0])
        # print(type(response["output_text"][0]))
        # data_dict_initial = json.loads(response["output_text"])
        # print(data_dict_initial.keys())
        # print(data_dict_initial.values())

        assistant_reply = json.dumps(data_dict_initial["output_text"])
        plot_title = data_dict_initial["plot_title"]
        start = assistant_reply.find("{")
        end = assistant_reply.find("}") + 1
        json_data = assistant_reply[start:end]
        # Parse the JSON data into a Python dictionary
        data_dict = json.loads(json_data)
        # NOTE Pandas method if needed
        #df_plot = pd.DataFrame(list(data_dict.items()), columns=['Date', 'Value'])
        #fig = go.Figure(data=go.Scatter(x=df_plot['Date'], y=df_plot['Value'], mode='lines+markers', marker=dict(size=8), name='Scatter Plot'))
        #fig.update_layout(title=f"Time-Series Plot of {str(plot_title)}", xaxis_title='Date', yaxis_title=str(plot_title), template="simple_white")
        x_values = list(data_dict.keys())
        y_values = list(data_dict.values())
        fig = go.Figure(data=go.Scatter(x=x_values, y=y_values, mode='lines+markers', marker=dict(size=8), name='Scatter Plot'))
        fig.update_layout(title=f"Time-Series Plot of {str(plot_title)}", xaxis_title='Date', yaxis_title=str(plot_title), template="simple_white")
        return fig