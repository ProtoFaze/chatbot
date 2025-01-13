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
import extra_streamlit_components.CookieManager as CookieManager

import requests
from typing import Literal

ABUSE_CHANCE = 3

def get_cookies_manager():
    '''Get the cookie manager object'''
    if 'cookies_manager_object' not in st.session_state:
        manager = CookieManager()
        st.session_state['cookies_manager_object'] = manager
        return manager
    else:
        return st.session_state['cookies_manager_object']
    
def login(password: str):
    '''Verify the password entered by the user'''
    st.session_state['IS_ADMIN'] = password == st.session_state['ADMIN_PASSWORD']
    if st.session_state['IS_ADMIN']:
        st.rerun()

# Initialize session state with defaults from secrets
def initialize_streamlit_session():
    '''Initialize the session state with default values from the secrets'''
    for secret_key in st.secrets:
        if secret_key not in st.session_state and secret_key not in ['MONGODB_URI', 'ADMIN_EMAIL']:
            st.session_state[secret_key] = st.secrets[secret_key]

#connection to ollama API
def setup_LLM(connection_type: Literal['external ollama', 'localhost ollama'] = None):
    '''Setup connection to the ollama API'''
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
    '''Experimental function to warmup the models in the ollama API'''
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
    '''Setup connections to MongoDB Atlas as well as the llamaindex vector store and retriever'''
    if 'llm_client' not in st.session_state:
        setup_LLM('external ollama')
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

def get_chat_ids(fetch_only_ids: bool = True):
    '''Fetch chat ids from the database and store a reversed list in the session state'''
    mongo_client: MongoClient = st.session_state['mongo_client']
    chatlog = mongo_client[st.secrets["MONGODB_DB"]]['chat_session'].find({}, {"_id": 1, "created_on": 1})
    return_list: list = list(chatlog)
    return_list.reverse()
    if fetch_only_ids:
        id_list = [str(doc['_id']) for doc in return_list]
        st.session_state['id_list'] = id_list
        return id_list
    else:
        return return_list

def get_config(field: str,scope: Literal['public','admin'] = 'public'):
    '''Get the summary of the insurance product stored in the config collection of the relevant mongodb database'''
    if 'mongo_client' not in st.session_state:
        setup_mongo()
    mongo_client: MongoClient = st.session_state['mongo_client']
    requested_value: str =  mongo_client[st.session_state['MONGODB_DB']]['config'].find_one({'scope':scope, 'field':field})['value']
    return requested_value