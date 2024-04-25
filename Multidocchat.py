import logging
from models.loinc_class import loinccode
from models.program_builder import program
import streamlit as st
from llama_index.core import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.core.tools import QueryEngineTool
from PIL import Image
from llama_index.core import  ServiceContext, KnowledgeGraphIndex, VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle
# import openai
from dotenv import load_dotenv
import pandas as pd
import chromadb
import time
import json
import os
# from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llm.qna import ExlAI, QnA
from IPython.display import display, HTML
# from llama_index.core.storage import StorageContext, load_index_from_storage
from llama_index.core.tools import QueryPlanTool
from llama_index.core import get_response_synthesizer
# from llama_index.agent.openai import OpenAIAgent
# from llama_index.llms.openai import OpenAI
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.llms import AzureOpenAI
# from llama_index.embeddings.langchain import LangchainEmbedding
# Rebuild storage context

def main():
    # load_dotenv()
    # openai.api_type = os.getenv("OPENAI_API_TYPE")
    # openai.api_key = os.getenv("OPENAI_API_KEY")    
    # openai.api_base = os.getenv("OPENAI_API_BASE")
    # openai.api_version = os.getenv("OPENAI_API_VERSION")
    #add_logo("https://planetexl.exlservice.com/_catalogs/masterpage/EXL/images/icons/exlLogo.ico")

    exl,client=st.columns([1,5])
    # with client:
    #     image = Image.open(r'C:\Users\admin-1\Desktop\Projects\MultiDocChat\branding\exl_logo.png')
    #     new_image=image.resize((75,100))
    #     st.image(new_image)
    with exl:
        image = Image.open(r'C:\Users\admin-1\Desktop\Projects\MultiDocChat\branding\McKesson-Logo.png')
        new_image=image.resize((200,150))
        st.image(new_image)
 
    st.title('Multi Doc QnA Demo')
    # st.subheader("""Chat with Earnings transcripts""")
    llm=ExlAI()
    # llm = OpenAI(temperature=0, model="gpt-4-0613")
    # deployment_name = "gpt-4-0613"
    # llm = AzureOpenAI(deployment_name=deployment_name)

    # embed_model = HuggingFaceEmbedding(model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embed_model = LangchainEmbedding(OpenAIEmbeddings())
    Settings.embed_model=embed_model
    Settings.llm=llm
    # chroma_client=chromadb.PersistentClient(path='db_pubmedbert_v2/')
    # query_tool_fy16_q1 = createQueryEngineTool('vectordb/fy_16_q1','fy_16_q1',llm,embed_model)
    # query_tool_fy16_q2 = createQueryEngineTool('vectordb/fy_16_q1','fy_16_q2',llm,embed_model)
    # query_tool_fy16_q3 = createQueryEngineTool('vectordb/fy_16_q1','fy_16_q3',llm,embed_model)
    # query_tool_fy16_q4 = createQueryEngineTool('vectordb/fy_16_q1','fy_16_q4',llm,embed_model)

    # response_synthesizer = get_response_synthesizer()
    # query_plan_tool = QueryPlanTool.from_defaults(
    #     query_engine_tools=[query_tool_fy16_q1, query_tool_fy16_q2, query_tool_fy16_q3,query_tool_fy16_q4],
    #     response_synthesizer=response_synthesizer,
    # )
    # llm = ExlAI()

    # agent = OpenAIAgent.from_tools(
    #             [query_plan_tool],
    #             max_function_calls=10,
    #             llm=llm,
    #             verbose=True,
    #         )
    
    # question = "What was the revenue in Quarter 1, Quarter 2, Quarter 3 and Quarter 4 of FY16"
    # response = agent.query(question)
    # st.markdown(question)
    # st.markdown(response)
    db_path = 'vectordb/'
    collection_path = "fy_16_q1"
    chroma_client=chromadb.PersistentClient(path=db_path)
    chroma_collection=chroma_client.get_or_create_collection('earnings')
    vector_store=ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )
    query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[
        SentenceTransformerRerank(
            model="cross-encoder/ms-marco-TinyBERT-L-2-v2",
            top_n=3
        )
    ],
    response_mode="tree_summarize",
)
    
    new_query=index.as_query_engine()
    

    cont=st.container()
    program_object=program(loinccode,llm)
    prompt="""
You are an accountant assistant. You are given the information about the earning call transcripts.
Your task is to answer the user_query basis on the context given:
Context:
{loinc_description}
user_query:
{text_input}
REMINDER: DO NOT RETURN any answer outside the context information given.
"""
    with cont:
        input_col,button_col=st.columns([4,1])
              
        with input_col:
            text_input=st.text_input("Ask me a question")
        with button_col:
            st.markdown('')
            st.markdown('')
            # query_engine=st.session_state.query_engine
            ask_me_button=st.button("Enter")
            if "messages" not in st.session_state.keys():
                st.session_state.messages = [{"Query": "", "Response":""}]


            if ask_me_button:
                with cont:
                    # question_res=get_loinc_code(text_input,query_engine)
                    similar_nodes=get_similar_nodes(index,text_input)
                    reranked_nodes=get_retrieved_nodes(index,text_input,with_reranker=True)
                    with st.expander("Retrived Similar Text rows"):
                        for node in similar_nodes:
                            st.markdown(node.text)
                    
                    with st.expander("Retrived Reranked Text rows"):
                        for node in reranked_nodes:
                            st.markdown(node.text)
                    node_list=[node.text for node in reranked_nodes]
                    earning_insight='\n'.join(node_list)
                    prompt=f"""
You are an accountant assistant. You are given the information about the earning call transcripts.
Your task is to answer the user_query basis on the context given:
Context:
{earning_insight}
user_query:
{text_input}
REMINDER: DO NOT RETURN any answer outside the context information given.
"""
                    # program_respond=program_object.programrespond(prompt=prompt)
                    # question_res=program_respond(loinc_description=loinc_description,text_input=text_input)
                    # question_response=query_engine.query(text_input)
                    question_response=llm.complete(prompt)
                    # st.markdown(prompt)
                    st.markdown(question_response)
                    for message in reversed(st.session_state.messages):
                        st.markdown('<h4 class="font">'+ message["Query"] +'</h3>', unsafe_allow_html=True)                
                        st.markdown(message["Response"])
                        # print(st.session_state.data_list)

                    st.session_state.messages.append({"Query": text_input, "Response": question_response})
                

def get_loinc_code(
    query_str, query_engine
):
    query_bundle = QueryBundle(query_str)
    # configure retriever
    
    response = query_engine.query(
    query_str
    )


def get_similar_nodes(
    index,query_str, vector_top_k=20
):
    query_bundle = QueryBundle(query_str)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)
    return retrieved_nodes

def get_retrieved_nodes(
    index,query_str, vector_top_k=20, reranker_top_n=2, with_reranker=False
):
    query_bundle = QueryBundle(query_str)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)

    if with_reranker:
        # configure reranker
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-TinyBERT-L-2-v2",
            top_n=reranker_top_n
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

    return retrieved_nodes

def createQueryEngineTool(db_path,collectionpath,llm,embed_model):
    # llm=ExlAI()
    # llm = OpenAI(temperature=0, model="gpt-4-0613")

    # embed_model = HuggingFaceEmbedding(model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    # embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model=embed_model
    # Settings.llm=llm
    
    chroma_client_fy16_q1=chromadb.PersistentClient(path=db_path)
    chroma_collection_fy16_q1=chroma_client_fy16_q1.get_or_create_collection(collectionpath)
    vector_store=ChromaVectorStore(chroma_collection=chroma_collection_fy16_q1)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )
    # query_engine = index.as_query_engine(
    # similarity_top_k=10,
    # node_postprocessors=[
    #     SentenceTransformerRerank(
    #         model="cross-encoder/ms-marco-TinyBERT-L-2-v2",
    #         top_n=3
    #     )
    # ],
    # response_mode="tree_summarize",
    # )
    query_engine=index.as_query_engine(similarity_top_k=3, llm=llm)
    query_tool_fy16_q1 = QueryEngineTool.from_defaults(
                                        query_engine=query_engine,
                                        name=collectionpath,
                                        description=(
                                            f"Provides information about Mckasen quarterly financials for"
                                            f"{collectionpath}"
                                        ),
                                    )
    
    return query_tool_fy16_q1

    


main()

