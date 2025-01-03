import ollama
import google.oauth2.id_token
from bson.objectid import ObjectId as oid 
from google.auth.transport.requests import Request as auth_req
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, VectorStoreIndex
from pymongo import MongoClient
import streamlit as st

import requests
from typing import Literal



def login(password: str):
    st.session_state['IS_ADMIN'] = password == st.session_state['ADMIN_PASSWORD']
    if st.session_state['IS_ADMIN']:
        st.rerun()

# Initialize session state with defaults from secrets
def initialize_streamlit_session():
    for secret_key in st.secrets:
        if secret_key not in st.session_state and secret_key not in ['MONGODB_URI', 'ADMIN_EMAIL']:
            st.session_state[secret_key] = st.secrets[secret_key]

#connection to ollama API
def setup_LLM(connection_type: Literal['external ollama', 'localhost ollama'] = None):
    if connection_type is None:
        connection_type = st.selectbox("Select your ollama connection type", ['external ollama'])
    llm_client = None
    match connection_type:
        case "external ollama":
            token = None
            if st.session_state['GOOGLE_APPLICATION_CREDENTIALS'] != 'default':
                token = google.oauth2.id_token.fetch_id_token(request=auth_req(), audience=ollama_url)
            elif st.session_state['EXTERNAL_OLLAMA_API_KEY'] != 'default':
                token = st.session_state['EXTERNAL_OLLAMA_API_KEY']
            headers = {"Authorization": f"Bearer {token}"}
            ollama_url = st.session_state.get("EXTERNAL_OLLAMA_API_URL",st.secrets["EXTERNAL_OLLAMA_API_URL"])
            llm_client = ollama.Client(
                host = ollama_url,
                headers = headers
                )
            Settings.embed_model = OllamaEmbedding(model_name='nomic-embed-text',base_url=ollama_url,headers=headers)
        case "localhost ollama":
            llm_client = ollama
            Settings.embed_model = OllamaEmbedding(model_name='nomic-embed-text')
    st.session_state['llm_client'] = llm_client
    return llm_client

def warmup_LLM():
    url = f"{st.session_state.get('EXTERNAL_OLLAMA_API_URL',st.secrets["EXTERNAL_OLLAMA_API_URL"])}/api/warmup"
    default_models = ['intentClassifier','nomic-embed-text','C3']
    with st.status("Warming up models", state='running'):
        for model in default_models:
            warmup_command = requests.post(url= url, json={"model": model})
            response =  warmup_command.content.decode()
            if warmup_command.status_code == 200:
                st.success(response)
                continue
            else:
                st.error(f"{response}\n   {model} might run slower than expected upon first use")

def setup_mongo():
    # Connect to your MongoDB Atlas(Cloud) cluster
    mongo_client = MongoClient(st.secrets["MONGODB_URI"])
    st.session_state['mongo_client'] = mongo_client
    vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=mongo_client,
            db_name = st.session_state.get("MONGODB_DB",st.secrets["MONGODB_DB"]),
            collection_name = st.session_state.get("MONGODB_RAG_COLLECTION",st.secrets["MONGODB_RAG_COLLECTION"]),
            vector_index_name = 'vector_index'
        )
    st.session_state['vector_store'] = vector_store
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)
    st.session_state['retriever'] = retriever

def fetch_chat_ids():
    ids = st.session_state['mongo_client'][st.secrets["MONGODB_DB"]]['chat_session'].find({}, {"_id": 1})
    id_list = [str(doc['_id']) for doc in ids]
    st.session_state['id_list'] = id_list
    unique_id = oid().__str__()
    if unique_id not in id_list and "session_id" not in st.session_state:
        st.session_state['session_id'] = unique_id
        st.session_state['id_list'].append(unique_id)