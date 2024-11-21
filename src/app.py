import ollama
import google.oauth2.id_token
from google.auth.transport.requests import Request as auth_req
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient

from pinecone import Pinecone, ServerlessSpec


# setup connections

#connection to ollama API
def setup_ollama():
    connection_type = st.selectbox("Select your ollama connection type", ["localhost", "google cloud"], placeholder="localhost")
    ollama_client = None
    match connection_type:
        case "localhost":
            ollama_client = ollama
        case "google cloud":
            ollama_url = os.environ["OLLAMA_API_URL"]
            token = google.oauth2.id_token.fetch_id_token(request=auth_req(), audience=ollama_url)
            ollama_client = ollama.Client(
                host=ollama_url,
                headers = {
                    "Authorization": f"Bearer {token}"
                    }
                )
    return ollama_client

# Connect to your MongoDB Atlas(Cloud) cluster
mongoclient = MongoClient(os.environ["MONGODB_URI"])
db = mongoclient[os.environ["MONGODB_DB"]]

# Connect to your pinecone vector database and specify the working index
pinecone_client = Pinecone(os.environ["PINECONE_API_KEY"])
index = None
if os.environ["PINECONE_INDEX_NAME"] not in [vector_index.name for vector_index in pinecone_client.list_indexes()]:
    index = pinecone_client.create_index(name = os.environ["PINECONE_INDEX_NAME"],
                            dimension = 768, 
                            metric = "cosine",
                            spec = ServerlessSpec(cloud = "aws", region = "us-east-1") 
                            )
else:
    index = pinecone_client.Index(os.environ["PINECONE_INDEX_NAME"])

#setup the semantic search function
def semantic_search(question, index=index, top_k=6, verbose=False):
    """vectorizes the question and queries the pinecone index for the top 3 closest matches.

current implementation does not support querying multiple namespaces or using mongoDB indexes.

Args:
    question (str): _description_
    debug (bool, optional): _description_. Defaults to False.
    embedding (SentenceTransformer, optional): _description_. Defaults to embedding.
    index (Pinecone.Index, optional): _description_. Defaults to index.

Returns:
    list(QueryResponse): the top closest query results from the pinecone index
    """
    if not index:
        print("Index not found, please specify your pinecone or mongo search index")
        return
    
    with st.spinner("fetching similar information..."):
        if verbose: print("Encoding question...")
        vector = ollama_client.embeddings(model="nomic-embed-text", prompt=question)['embedding']
        matches = []
        spaces = index.describe_index_stats()['namespaces']
        if verbose: print(f"""using namespaces: {spaces}""")
        for key, value in spaces.items():
            res = index.query(
                vector=vector,
                top_k=top_k,
                namespace=key,
                
            )
            matches.append(res)
        if len(matches) < 1:
            raise Exception("No matches found, please check your search index, it may be empty or not connected")
            
    return matches

#setup a function to receive semantic search results and return the collection name and ids
def get_collection_matches(response : list, verbose=False) -> list:
    """based on the response from a pinecone semantic search, extract and consolidate the collection name and ids of the matching documents.

    Args:
        response (list): a list of QueryResponse objects from a pinecone semantic search
        verbose (bool, optional): flag to use debug mode. Defaults to False.

    Returns:
        list: list of mongodb documents
    """
    def extract_info(match: str) -> dict:
        data_from_id = None
        try:
            data_from_id = match['id'].split("-")
        except TypeError as e: #probably a document object
            data_from_id = match.id.split("-")
        match_id = data_from_id[0]
        collection_name = data_from_id[1]
        return {'collection':collection_name, 'id':match_id}
    
    with st.spinner("filtering fetched data..."):
        if verbose: print("Getting collection matches...")
        document_metadata = []
        for namespaces_or_documents in response: #iterate over the namespaces
            if isinstance(namespaces_or_documents, list):
                if verbose: print("no pinecone namespace found, checking id or additional metadata")
                for metadata in namespaces_or_documents:
                    document = extract_info(metadata)
                    if document in document_metadata:
                        if verbose:
                            print(f"Duplicate id {document['id']} for {document['collection']} found in fetch list, ignoring")
                        continue
                    document_metadata.append(document)
            elif ('matches' in namespaces_or_documents): #probably a dictionary
                matches = namespaces_or_documents['matches']
                for match in matches:
                    document = extract_info(match)
                    if document in document_metadata:
                        if verbose: print(f"Duplicate id {document['id']} for {document['collection']} found in fetch list, ignoring")
                        continue
                    document_metadata.append(document)
    return document_metadata

#find the mongo documents based on their collection and ids
from bson.objectid import ObjectId as oid 
def find_documents(collection_matches : list, verbose=False, database=db) -> list:
    """use the collection matches to find the corresponding documents in the specified mongo database.

    Args:
        collection_matches (list): list of document metadata containing collection names and ids
        verbose (bool, optional): flag to turn on debug mode. Defaults to False.
        database (pymongo.synchronous.database.Database, optional): the mongodb database to use. Defaults to the db variable.

    Returns:
        list: list of mongodb documents
    """
    with st.spinner("pulling documents..."):
        if verbose: print(f"Finding documents using {len(collection_matches)} references")
        documents = []
        for collection_match in collection_matches:
            collection = database[collection_match['collection']]
            if 'ids' in collection_match:
                for id in collection_match['ids']:
                    documents.append(collection.find_one({"_id": oid(id)}))
            else:
                documents.append(collection.find_one({"_id": oid(collection_match['id'])}))
        if verbose:
            for document in documents:
                print(document)
    return documents

def get_context(question: str, verbose: bool = False) -> list:
    """Retrieves text-based information for an insurance product only based on the user query.
does not answer quwstions about the chat agent"""
    matches = semantic_search(question, verbose=verbose)
    document_data = get_collection_matches(matches, verbose=verbose)
    context = find_documents(document_data, verbose=verbose)
    messages = [{'role': 'user', 'content':f"""fullfill the query with the provided information
Do not include greetings or thanks for providing relevant information
Query      :{question}
Information:{context}"""}]
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

def chat(user_input: str, verbose: bool = False):
    intent = None
    with st.spinner("detecting intent..."):
        #substring the last word from classify_intent and remove any trailing symbols
        intent_response = classify_intent(user_input)
        intent = intent_response.split()[-1].strip('.,!?')
    match intent:
        case "rag":
            return get_context(user_input, verbose=verbose)
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
                messages = [{"role":"user", "content":f"address the request if it is suitable for work,\
                            otherwise apologise and state that you are not design to address these requests.\
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

def initialize_chat_messages():
    st.session_state['chat_messages'] = [
        {"role": "system", "content": st.session_state['setup_prompt']}
    ]


def initialize_streamlit_session():
    defaults = {
        "name": "Reference LLM Chatbot implementation using Streamlit and Ollama",
        "model": "C3",
        "setup_prompt": """Your name is C3, a top customer service chatbot at Great Eastern Life Assurance Malaysia. 
You only answer questions from the user to help better understand a specific product that you sell.""" ,
        "temperature": 0.5
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if 'chat_messages' not in st.session_state:
        initialize_chat_messages()

    st.header(f"{defaults['name']}")

def sl_module_chat():
    st.header("Chat")
    if st.button("Reset chat window"):
        initialize_chat_messages()

### Streamlit app
initialize_streamlit_session()

with st.sidebar:
    st.header("How it works")
    st.write("The RAG chatbot works by embedding user inputs.",
            "The inputs are then used to query a pinecone index.",
            "The top closest matches are then used to query the relevant documents from a mongoDB database.",
            "Which is finally used as context for the response generation."
            )
    sl_module_chat()
    ollama_client = setup_ollama()
    with st.expander("conversations") as expander:
        for message in st.session_state['chat_messages']:
            if message["role"] == "system":
                continue
            else:
                st.write(f"{message['role']}: {message['content']}")

for message in st.session_state['chat_messages']:
    if message["role"] == "system":
        continue
    else:
        st.chat_message(message['role']).write(message['content'])

if prompt := st.chat_input("How can I help?"):
    
    st.session_state['chat_messages'].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in chat(prompt, verbose=True):
            full_response += chunk['message']['content']
            response_placeholder.markdown(full_response + "▌")

    st.session_state.chat_messages.append({"role": "assistant", "content": full_response})   