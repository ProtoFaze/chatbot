from langchain_ollama.chat_models import ChatOllama as Ollama
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import nltk
import re
nltk.download('wordnet')

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
#initialize the stopwords
stop_words = set(stopwords.words('english'))
#allow specific stopwords
stop_words.remove("both")

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

client = MongoClient("mongodb+srv://damonngkhaiweng:jhT8LGE0qi6XsKfz@chatbotcluster.noyfa.mongodb.net/?retryWrites=true&w=majority&appName=chatbotcluster")
db = client["product1"]
from pinecone import Pinecone
pc = Pinecone("ce2e7e04-18d6-4408-a9ab-7527162af1d7")
index = pc.Index("product1")

global_embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

#setup the semantic search function
def semantic_search(question, model=global_embedding_model, index=index, top_k=6, debug=False):
    """vectorizes the question and queries the pinecone index for the top 3 closest matches.

current implementation does not support querying multiple namespaces or using mongoDB indexes.

Args:
    question (str): _description_
    debug (bool, optional): _description_. Defaults to False.
    model (SentenceTransformer, optional): _description_. Defaults to global_embedding_model.
    index (Pinecone.Index, optional): _description_. Defaults to index.

Returns:
    list(QueryResponse): the top closest query results from the pinecone index
    """
    if not index:
        print("Index not found, please specify your pinecone or mongo search index")
        return
    if not model:
        print("Model not found")
        return
    if debug:
        print("Encoding question...")
        
    vectors = model.encode(question)
    
    if not isinstance(vectors, list): # Ensure vectors is a list of floats
        vectors = vectors.tolist()
    matches = []
    spaces = index.describe_index_stats()['namespaces']
    for key, value in spaces.items():
        res = index.query(
            namespace=key,
            vector=vectors,
            top_k=top_k,
            include_metadata=False,
            include_values=False
        )
        matches.append(res)
        print(res['usage'])
    return matches

#setup a function to receive semantic search results and return the collection name and ids
def get_collection_matches(response, debug=False):
    if debug:
        print("Getting collection matches...")
    document_metadata = []
    for collection_matches in response:
        collection = collection_matches['namespace']
        is_using_default_namespace = False
        if collection == "":
            is_using_default_namespace = True
            if debug:
                print("no pinecone namespace found, checking id or additional metadata")
        ids = []
        for match in collection_matches['matches']:
            if debug:
                print(match)
            data_from_id = match['id'].split("-")
            match_id = data_from_id[0]
            if is_using_default_namespace:
                collection = data_from_id[1]
            #filter out duplicate ids
            if match_id not in ids:
                ids.append(match_id)
            else:
                if debug:
                    print(f"Duplicate id {match_id} for {collection} found in fetch list, ignoring")
                continue
            if is_using_default_namespace:
                document_metadata.append({'collection':collection, 'id':match_id})
        if not is_using_default_namespace:
            document_metadata.append({'collection':collection, 'ids':ids})
        
    return document_metadata

from bson.objectid import ObjectId as oid 
def find_documents(collection_matches, debug=False, database=db):
    if debug:
        print("Finding documents...")
    documents = []
    for collection_match in collection_matches:
        collection = database[collection_match['collection']]
        if 'ids' in collection_match:
            for id in collection_match['ids']:
                documents.append(collection.find_one({"_id": oid(id)}))
        else:
            documents.append(collection.find_one({"_id": oid(collection_match['id'])}))
    return documents

#setup the chatbot model for tool use
from langchain_core.tools import tool
import re

@tool
def fetch_metrics(question: str) -> list:
    """Retrieves JSON table data for either premium plans, total investment estimations, fund performance, or premium allocation based on keywords inthe user question."""
    cleaned_question = preprocess_text(question)
    documents = []
    has_fetched_preimum_plans = False
    has_fetched_total_investment_estimation = False
    has_fetched_fund = False
    has_fetched_premium_allocation = False
    for keyword in re.findall(r'\b\w+\b', cleaned_question):
        match keyword:
            case "premium" | "plan" | "coverage" | "price" | "pricing":
                if not has_fetched_preimum_plans:
                    print("fetching premium plans")
                    
                    documents.append(list(db["premium_plans"].find()))
                    has_fetched_preimum_plans = True
            case "investment" | "value" | "allocation":
                if not has_fetched_total_investment_estimation:
                    print("fetching investment plans")
                    
                    documents.append(list(db["total_investment_estimation"].find()))
                    has_fetched_total_investment_estimation = True
            case "performance" | "perform" | "fund":
                if not has_fetched_fund:
                    print("fetching fund performance")
                    
                    # documents.append(list(db["fund"].find()))
                    has_fetched_fund = True
            case "allocation":
                if not has_fetched_premium_allocation:
                    print("fetching premium allocation")
                    
                    documents.append(list(db["premium_allocation"].find()))
                    has_fetched_premium_allocation = True
            case _:
                continue

    return documents

@tool
def get_context(question: str, debug: bool = False) -> list:
    """Retrieves text-based information for an insurance product based on the user query."""
    print(question)
    matches = semantic_search(question, debug=debug)
    document_data = get_collection_matches(matches, debug=debug)
    context = find_documents(document_data, debug=debug)
    return context

tools = [fetch_metrics,
        get_context
        ]
toolPicker = Ollama(model='llama3.2').bind_tools(tools)

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
prompt = hub.pull("hwchase17/openai-tools-agent")

def chat(query, debug=False):
    if query in ["exit", "quit", "bye", "end", "stop", ""]:
        return
    # if debug:
    #     tool_query=query+" (Turn on debug mode)"
    # else:
    #     tool_query=query
    # messages = [query]
    agent = create_tool_calling_agent(toolPicker, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke({"input": query})

chat(input("Enter your query: "))
