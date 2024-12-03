import ollama
import google.oauth2.id_token
from google.auth.transport.requests import Request as auth_req
import streamlit as st
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, VectorStoreIndex
import os
from bson.objectid import ObjectId as oid 

from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient

# setup connections

#connection to ollama API
def setup_ollama():
    connection_type = st.selectbox("Select your ollama connection type", ["localhost", "google cloud"], placeholder="localhost")
    ollama_client = None
    match connection_type:
        case "localhost":
            ollama_client = ollama
            Settings.embed_model = OllamaEmbedding(model_name='nomic-embed-text')
        case "google cloud":
            ollama_url = os.environ["OLLAMA_API_URL"]
            token = google.oauth2.id_token.fetch_id_token(request=auth_req(), audience=ollama_url)
            headers = {"Authorization": f"Bearer {token}"}
            ollama_client = ollama.Client(
                host = ollama_url,
                headers = headers
                )
            Settings.embed_model = OllamaEmbedding(model_name='nomic-embed-text',base_url=ollama_url,headers=headers)
    return ollama_client

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
    st.write("MongoDB connection established")
    ids = mongo_client[os.environ["MONGODB_DB"]]['chat_session'].find({}, {"_id": 1})
    id_list = [str(doc['_id']) for doc in ids]
    st.session_state['id_list'] = id_list
    unique_id = oid().__str__()
    if unique_id not in id_list and "session_id" not in st.session_state:
        st.session_state['session_id'] = unique_id
        st.session_state['id_list'].append(unique_id)



def get_context(question: str, verbose: bool = False) -> list:
    """Retrieves text-based information for an insurance product only based on the user query.
does not answer quwstions about the chat agent"""
    context = []
    with st.spinner("retrieving context"):
        retriever = st.session_state['retriever']
        for node in retriever.retrieve(question):
            context.append(node.text)
    messages =  st.session_state['messages']+{'role': 'user', 'content':f"""fullfill the query with the provided information
Do not include greetings or thanks for providing relevant information
Query      :{question}
Information:{context}"""}
    return ollama_client.chat(
            model=st.session_state['model'],
            messages=messages,
            stream=True
        )

#setup a simple intent classifier
def classify_intent(user_input:str) -> str:
    """classify the intent of the user input
current implementation uses a lightweight model (llama3.2)
with chain of thought prompting examples
for classifying 'normal', 'register', 'RAG' intents
"""
    return ollama_client.chat(
            model = "intentClassifier",
            messages = [{"role":"user", "content":user_input}],
            stream = False
        )['message']['content']

def chat(user_input: str):
    intent = None
    with st.spinner("detecting intent..."):
        #substring the last word from classify_intent and remove any trailing symbols
        intent_response = classify_intent(user_input)
        intent = intent_response.split()[-1].strip('.,!?')
    match intent:
        case "rag":
            return get_context(user_input)
        case "register":
            with st.spinner("fetching registration instructions"):
                messages = [{"role":"system", "content":f"""The user is asking about information for registering to the insurnace scheme.
                             Always include this registration link (https://greatmultiprotect.com/gss315-spif/)
                             And this admin contact number (03-48133818) in your responses to the user's query."""},
                    {"role":"user","content":f"{user_input}"}]
            return ollama_client.chat(
                model = st.session_state['model'],
                messages = messages,
                stream = True)
        case "normal":
            with st.spinner("thinking about what to say"):
                messages = [{"role":"user", "content":f"Respond naturally and contextually to suitable work-related requests. For requests that are inappropriate or outside your design, politely decline to address them.\
                            Request: {user_input}"}]
            return ollama_client.chat(
                model = st.session_state['model'],
                messages = messages,
                stream = True)
        case "verify":
            with st.spinner("fetching verification instructions"):
                messages = [{"role":"user", "content":f"Please answer the given question with the following context:\
                    Question: {user_input}\
                    Context:\
                    user info are gathered from previous campaigns and stored in a secure database, we do not share your information with third parties.\
                    user can obtain verification by emailing this address: damonngkhaiweng@greateasternlife.com"}]
            return ollama_client.chat(
                model = st.session_state['model'],
                messages = messages,
                stream = True)
        case _:
            with st.spinner("fetching default response"):
                st.write('default')
                return ollama_client.chat(
                    model = st.session_state['model'],
                    messages = [{"role":"user", 
                                "content":user_input}],
                    stream = True)

st.set_page_config(
        page_title="Chat about GMBIS",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def initialize_messages():
    st.session_state['messages'] = []


def initialize_streamlit_session():
    defaults = {
        "name": "Reference LLM Chatbot implementation using Streamlit and Ollama",
        "model": "C3",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if 'messages' not in st.session_state:
        initialize_messages()

    st.header(f"{defaults['name']}")

def sl_module_chat():
    st.header("Chat")
    if st.button("Reset chat window"):
        initialize_messages()

### Streamlit app
initialize_streamlit_session()

with st.sidebar:
    st.header("How it works")
    st.write("The RAG chatbot works by embedding user inputs.",
            "The inputs are then used to query a mongodb index.",
            "The top closest matches are then used to query the relevant documents",
            "Which is finally used as context for the response generation."
            )
    sl_module_chat()
    ollama_client = setup_ollama()
    with st.expander("conversations") as expander:
        for message in st.session_state['messages']:
            if message["role"] == "system":
                continue
            else:
                st.write(f"{message['role']}: {message['content']}")
    setup_mongo()

for message in st.session_state['messages']:
    if message["role"] == "system":
        continue
    else:
        st.chat_message(message['role']).write(message['content'])

if prompt := st.chat_input("How can I help?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in chat(prompt):
            full_response += chunk['message']['content']
            response_placeholder.markdown(full_response + "▌")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    if st.session_state['session_id'] in st.session_state['id_list']:
        st.session_state['mongo_client'][os.environ["MONGODB_DB"]]['chat_session'].update_one({'_id': oid(st.session_state['session_id'])},{'$set': {'chatlog': st.session_state['messages']}})
    else:
        st.session_state['mongo_client'][os.environ["MONGODB_DB"]]['chat_session'].insert_one({'_id': oid(st.session_state['session_id']) , 'chatlog': st.session_state['messages']})