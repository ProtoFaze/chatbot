import os
import ollama
import streamlit as st
import google.oauth2.id_token
from bson.objectid import ObjectId as oid 
from google.auth.transport.requests import Request as auth_req
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, VectorStoreIndex
from pymongo import MongoClient

#connection to ollama API
def setup_LLM():
    connection_type = st.selectbox("Select your ollama connection type", ["localhost ollama", 'modal ollama', "google cloud"], placeholder="localhost ollama")
    llm_client = None
    match connection_type:
        case "localhost ollama":
            llm_client = ollama
            Settings.embed_model = OllamaEmbedding(model_name='nomic-embed-text')
        case "google cloud":
            ollama_url = os.environ["GOOGLE_OLLAMA_API_URL"]
            token = google.oauth2.id_token.fetch_id_token(request=auth_req(), audience=ollama_url)
            headers = {"Authorization": f"Bearer {token}"}
            llm_client = ollama.Client(
                host = ollama_url,
                headers = headers
                )
            Settings.embed_model = OllamaEmbedding(model_name='nomic-embed-text',base_url=ollama_url,headers=headers)
        case 'modal ollama':
            ollama_url = os.environ["MODAL_OLLAMA_API_URL"]
            llm_client = ollama.Client(host=ollama_url)
            Settings.embed_model = OllamaEmbedding(model_name='nomic-embed-text', base_url=ollama_url)
    st.session_state['llm_client'] = llm_client
    return llm_client

def setup_mongo():
    # Connect to your MongoDB Atlas(Cloud) cluster
    collection_name = 'llamaIndexChunks'
    mongo_client = MongoClient(os.environ["MONGODB_URI"])
    st.session_state['mongo_client'] = mongo_client
    index_name = 'vector_index'
    vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=mongo_client,
            db_name = os.environ["MONGODB_DB"],
            collection_name = collection_name,
            vector_index_name = index_name
        )
    st.session_state['vector_store'] = vector_store
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)
    st.session_state['retriever'] = retriever
2
def fetch_id_list():
    ids = st.session_state['mongo_client'][os.environ["MONGODB_DB"]]['chat_session'].find({}, {"_id": 1})
    id_list = [str(doc['_id']) for doc in ids]
    st.session_state['id_list'] = id_list
    unique_id = oid().__str__()
    if unique_id not in id_list and "session_id" not in st.session_state:
        st.session_state['session_id'] = unique_id
        st.session_state['id_list'].append(unique_id)