import json
import ollama
import requests
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
import nltk
import re
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient
from pinecone import Pinecone, ServerlessSpec

# setup connections
ollama_client = ollama.Client(host='http://ollama:11434')

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

try:
    nltk.data.find('corpora/wordnet')
except:
    # print("sam ting wong")
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

#initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
#initialize the stopwords, remove specific stopwords
stop_words = set(stopwords.words('english')).remove('both')
# print("file has been rerun")

def preprocess_text(text):
    pattern = re.compile(r"[;@#&*!()\[\]]")
    def get_wordnet_pos(tag):
        match tag[0]:
            case 'J':
                return wordnet.ADJ
            case 'V':
                return wordnet.VERB
            case 'R':
                return wordnet.ADV
            case _:
                return wordnet.NOUN
    # Tokenize
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    # Remove stop words and lemmatize
    processed_tokens = [
        lemmatizer.lemmatize(word.lower(), get_wordnet_pos(tag)) 
        for word, tag in pos_tags 
        if word.lower() not in stop_words and not pattern.match(word)
    ]
    # Join tokens back to a string
    return ' '.join(processed_tokens)

# embedding = Embeddings(model="nomic-embed-text")
#                     #    , model_kwargs={"trust_remote_code":True}
#                     #    )
# number_of_dimensions = 768 #the embedding dimensions according to the model documentation
# from langchain_pinecone import PineconeVectorStore
# vector_store = PineconeVectorStore(index=index, embedding=embedding)
# def vectorize(texts):
#     "vectorize the provided texts using a globally declared model."    
#     vectors = embedding.encode(texts)
#     return vectors

# #using local(ollama) base model as chatbot model
# # intent_classifier = HuggingFace(model_id='llama3.2',#:1B
# #                            temperature=0,
# #                            num_predict=4,
# #                            request_timeout=10,
# #                            verbose=False )
# # chatbot_model = HuggingFace(model_id='C3', request_timeout=60)
# intent_classifier = Ollama(model='llama3.2',#:1B
#                            temperature=0,
#                            num_predict=4,
#                            request_timeout=10,
#                            verbose=False )
# chatbot_model = Ollama(model='llama3.2', request_timeout=60)

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

    print("Encoding question...") if verbose else None
        
    matches = []
    spaces = index.describe_index_stats()['namespaces']
    print(f"""using namespaces: {spaces}""") if verbose else None
    for key, value in spaces.items():
        res = index.query(
            query=question,
            k=top_k
        #     namespace=key,
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
    
    print("Getting collection matches...") if verbose else None
    document_metadata = []
    for namespaces_or_documents in response:
        is_using_default_namespace = False
        if isinstance(namespaces_or_documents, list):
            if verbose:
                print("no pinecone namespace found, checking id or additional metadata")
            for metadata in namespaces_or_documents:
                document = extract_info(metadata)
                if document in document_metadata:
                    if verbose:
                        print(f"Duplicate id {document['id']} for {document['collection']} found in fetch list, ignoring")
                    continue
                document_metadata.append(document)
        else: #probably a dictionary
            document = extract_info(namespaces_or_documents)
            if document in document_metadata:
                if verbose:
                    print(f"Duplicate id {document['id']} for {document['collection']} found in fetch list, ignoring")
                continue
            ids = []
            for match in collection:
                if verbose:
                    print(match)
                data_from_id = match['id'].split("-")
                match_id = data_from_id[0]
                if is_using_default_namespace:
                    collection = data_from_id[1]
                #filter out duplicate ids
                if match_id not in ids:
                    ids.append(match_id)
                else:
                    if verbose:
                        print(f"Duplicate id {document['id']} for {document['collection']} found in fetch list, ignoring")
                    continue
                if is_using_default_namespace:
                    document_metadata.append({'collection':collection, 'id':match_id})
            document_metadata.append({'collection':collection, 'ids':ids})
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
    print(f"Finding documents using {len(collection_matches)} references") if verbose else None
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

# #setup the chatbot model for tool use
# from langchain_core.tools import tool
# import re

# def get_context(question: str, verbose: bool = False) -> list:
#     """Retrieves text-based information for an insurance product only based on the user query.
# does not answer quwstions about the chat agent"""
#     print(question)
#     matches = semantic_search(question, verbose=verbose)
#     document_data = get_collection_matches(matches, verbose=verbose)
#     context = find_documents(document_data, verbose=verbose)
#     # if verbose:
#     #     for message in messages:
#     #         print(message)
#     template = ChatPromptTemplate.from_template(f"""fullfill the query with the provided information
# Do not include greetings or thanks for providing relevant information
# Query      :{{question}}
# Information:{{context}}""")

#     # RAG pipeline
#     chain = {
#         "context": lambda x: context , "question": RunnablePassthrough()
#         } | template | chatbot_model | StrOutputParser()
#     return chain.stream(question)


# #setup the RAG workflow
# from pydantic import ValidationError

# # def RAG(query : str, verbose : bool | None = False):
# #     if query in ["exit", "quit", "bye", "end", "stop", ""]:
# #         return
# #     if verbose:
# #         tool_query=query+" (Turn on debug mode)"
# #     else:
# #         tool_query=query
# #     messages = [query]
    
# #     ai_msg = toolPicker.invoke(tool_query)
# #     for tool_call in ai_msg.tool_calls:
# #         selected_tool = {"fetch_metrics": fetch_metrics,
# #                         "get_context": get_context
# #                         }[tool_call["name"].lower()]

# #         tool_msg = None 
# #         try:
# #             tool_msg = selected_tool.invoke(tool_call)
# #         except ValidationError as e:
# #             print(f"Error: {e.json()}")
# #         messages.append(f"connected {tool_call['name'].lower()} function returned {tool_msg.content}")

# #     if verbose:
# #         for message in messages:
# #             print(message)

# #     return chatbot_model.stream(f"""fullfill the query with the provided information
# # query      :{query}
# # information:{messages}""")

#setup a simple intent classifier
def classify_intent(user_input:str) -> str:
    """classify the intent of the user input

current implementation uses a lightweight model (llama3.2:1B)
with few-shot prompting examples
for classifying 'normal', 'register', 'RAG' intents

"""
    return intent_classifier.invoke(f"""
Classify the given input, use RAG if it is asking about insurance products,answer only with 'normal','register','RAG','verify':

example:

Input: "Is there a contact number", Intent: RAG
Input: "How do I create a new account", Intent: register
Input: "How do I make a claim", Intent: RAG
Input: "Search the web for cat videos", Intent: normal
Input: "Help me register for this service", Intent: register
Input: "Where can I get started", Intent: register
Input: "What is your name", Intent: normal
Input: "What entities are attached to this service", Intent: RAG
Input: "What is your purpose", Intent: normal
Input: "Can I see some fund performance metrics", Intent: RAG
Input: "What is the weather in your country", Intent: normal
Input: "How can i register for an account", Intent: register
Input: "Who owns the product", Intent: RAG
Input: "Tell me how to subscribe", Intent: register
Input: "Guide me through the registration process", Intent: register
Input: "How do I sign up for the trial", Intent: register
Input: "Can you explain your features", Intent: normal
Input: "Sign me up", Intent: register
Input: "Can you assist me with enrolling", Intent: register
Input: "What services does the product provide", Intent: RAG
Input: "Where can i wash my dog", Intent: normal
Input: "Goodbye", Intent: normal
Input: "How do i pay for the service", Intent: RAG
Input: "how is my premium allocated", Intent: RAG
Input: "Hello", Intent: normal
Input: "What are the coverage options", Intent: RAG
Input: "What's the first step to register", Intent: register
Input: "What funds are involved", Intent: RAG
Input: "Where are you located", Intent: normal
Input: "Who do I contact for help", Intent: RAG
Input: "Can I pay with a credit card", Intent: RAG
Input: "Why should i trust you", Intent: verify
Input: "What should i get ready for enrollment", Intent: register
Input: "How are you", Intent": normal
Input: "What company distributes this service", Intent: RAG
Input: "How do i know you are not a scam", Intent: verify
Input: "Are there any additional charges", Intent: RAG
Input: "What can you tell me about the available insurance plans", Intent: RAG
Input: "How much do i need to pay for the insurance scheme", Intent: RAG
Input: "Tell me about the available insurance plans", Intent: RAG
Input: "How did you get my number", Intent: verify
Input: "Show me verification so i know this isn't a scam", Intent: verify
Input: "How do i know you are not scamming me", Intent: verify
Input: "Why should I sign up for this plan", Intent: RAG

Input: {user_input}""").content

def chat(user_input: str, verbose: bool = False):
    intent = classify_intent(user_input)
    print(intent)
    match intent:
        case "RAG":
            return get_context(user_input, verbose=verbose)
        case "register":
            prompt = ChatPromptTemplate.from_template(f"Please provide this link https://greatmultiprotect.com/gss315-spif/ to address the given query \
                \nQuery: {{input}}")
            chain = prompt | chatbot_model | StrOutputParser()
            return chain.stream({"input":user_input})
        case "normal":
            chain = chatbot_model | StrOutputParser()
            return chain.stream(user_input)
        case "verify":
            prompt = ChatPromptTemplate.from_template(f"Please answer the given question\
                Question: {{input}}\
                Context:\
                user info are gathered from previous campaigns and stored in a secure database, we do not share your information with third parties.\
                user can obtain verification by emailing this address: damonngkhaiweng@greateasternlife.com")
            print(prompt)
            chain = prompt | chatbot_model | StrOutputParser()
            return chain.stream({"input":user_input})
        case _:
            chain = chatbot_model | StrOutputParser()
            return chain.stream(user_input)
            

# def main() -> None:
#     """
#     Main function to run the Streamlit chat application.

#     This function sets up the page configuration, initializes the session state,
#     creates the sidebar, and handles the chat interaction loop. It manages the
#     chat history, processes user inputs, and displays AI responses.
#     """
#     st.set_page_config(
#         page_title="Chat about GMBIS",
#         page_icon="💬",
#         layout="wide",
#         initial_sidebar_state="expanded",
#     )
#     st.write(
#     "This is a simple chatbot that uses a Llama 3.2:3B model to generate responses.",
#     "the model works by checking your input and searching for relevant documents in a connected database.",
#     "To use this app, simply start typing in the chat input field below.",
# )
    
#     initialize_session_state()

#     chat_model = Ollama(model="llama3.2:latest")

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", st.session_state.system_prompt),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"),
#         ]
#     )
#     chain = prompt | chat_model

#     msgs = StreamlitChatMessageHistory(key="special_app_key")
#     if not msgs.messages:
#         msgs.add_ai_message("How can I help you?")

#     chain_with_history = RunnableWithMessageHistory(
#         chain,
#         lambda _: msgs,
#         input_messages_key="input",
#         history_messages_key="chat_history",
#     )

#     for msg in msgs.messages:
#         st.chat_message(msg.type).write(msg.content)

#     if prompt := st.chat_input("Type your message here..."):
#         st.chat_message("human").write(prompt)

#         with st.spinner("Thinking..."):
#             config = {"configurable": {"session_id": "any"}}
#             try:
#                 # response = get_context(prompt, verbose=True)
#                 response = chat(prompt, verbose=True)
#                 st.chat_message("ai").write_stream(response)
#             except ConnectionError as e:
#                 st.error(f"Connection error: {e}", icon="😎")
#     if st.button("Clear Chat History"):
#         msgs.clear()
#         st.rerun()


# if __name__ == "__main__":
#     main()

package_data = {
   "name": "Reference LLM Chatbot implementation using Streamlit and Ollama",
   "version": "1.0.0-alpha.2",
}

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
        "model": "llama3.2",
        "setup_prompt": """Your name is C3, a top customer service chatbot at Great Eastern Life Assurance Malaysia. 
You answer questions from the user to help better understand a specific product that you sell.""" ,
        "temperature": 0.5
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if 'chat_messages' not in st.session_state:
        initialize_chat_messages()

# Loads available models from JSON file
# def load_models():
#     with open('./data/models.json', 'r') as file:
#         data = json.load(file)
#     return data['models']

# Downloads an up-to-date list of models and saves it to 'models.json'
# def update_model_list():
#     url = "https://ollama-models.zwz.workers.dev/"
#     response = requests.get(url)
    
#     if response.status_code == 200:
#         models_data = response.json()
        
#         # Save the models data to a file
#         with open('./data/models.json', 'w') as file:
#             json.dump(models_data, file)
        
#         st.success("Models updated successfully.")
#     else:
#         st.error(f"Failed to update models. Status code: {response.status_code}")

# def pull_ollama_model():
#     model_name = st.session_state['model']
#     response = ollama_client.pull(model=model_name, stream=True)
    
#     # Initialize a placeholder for progress update
#     progress_bar = st.progress(0)
#     progress_status = st.empty()
    
#     try:
#         for progress in response:
#             if 'completed' in progress and 'total' in progress:
#                 # Calculate the percentage of completion
#                 completed = progress['completed']
#                 total = progress['total']
#                 progress_percentage = int((completed / total) * 100)
#                 progress_bar.progress(progress_percentage)
#             if 'status' in progress:
#                 if progress['status'] == 'success':
#                     progress_status.success("Model pulled successfully!")
#                     break
#                 elif progress['status'] == 'error':
#                     progress_status.error("Error pulling the model: " + progress.get('message', 'No specific error message provided.'))
#                     break
#     except Exception as e:
#         progress_status.error(f"Failed to pull model: {str(e)}")
    
# Model selection module to be used within the Streamlit App layout.
# def sl_module_model_selection():
#     st.subheader("Model")

#     models = load_models()
#     model_options = {model['name']: (model['name'], model['description']) for model in models}
#     model_identifier = st.selectbox(
#         "Choose a model",
#         options=list(model_options.keys()),
#         format_func=lambda x: model_options[x][0]
#     )

#     if model_identifier:
#         st.session_state['model'] = model_identifier
        
#         st.write(model_options[model_identifier][1])

#         col1, col2 = st.columns(2, gap="small")
#         with col1:
#             if st.button("Update list", use_container_width=True):
#               update_model_list()
#         with col2:
#             if st.button("Pull model", use_container_width=True):
#               pull_ollama_model()

def sl_module_chat():
  st.subheader("Chat")
  if st.button("Reset chat window"):
      initialize_chat_messages()

### Streamlit app
initialize_streamlit_session()

with st.sidebar:
    st.header("Preferences")
    # sl_module_model_selection()
    sl_module_chat()

st.header(f"{package_data['name']}")

for message in st.session_state['chat_messages']:
    if message["role"] == "system":
        continue
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("How can I help?"):
    st.session_state['chat_messages'].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in ollama_client.chat(
            model=st.session_state['model'],
            messages=st.session_state['chat_messages'],
            stream=True,
        ):
            full_response += chunk['message']['content']
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

    st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
    