import streamlit as st
import asyncio
import datetime
from bson.objectid import ObjectId as oid 
from shared.Models import Intent
from shared.Setup import setup_LLM, setup_mongo, get_chat_ids, get_config, get_cookies_manager, ABUSE_CHANCE
from typing import Literal
from ollama._types import ChatResponse, Message, ResponseError
from pymongo.errors import ConnectionFailure


TIMEOUT_MESSAGE_DELAY: float = 60

### initialization functions
def initialize_session_id():
    '''Set the session ID to a random object ID'''
    id_list = get_chat_ids()
    unique_id = oid().__str__()
    if unique_id not in id_list and "session_id" not in st.session_state:
        st.session_state['session_id'] = unique_id
        st.session_state['id_list'].append(unique_id)
    
def initialize_messages():
    '''Set chat messages to the initial message'''
    st.session_state['messages'] = [{"role": "assistant", "content": "Hello! I'm C3, your friendly customer service chatbot at Great Eastern Life Assurance Malaysia. I'm here to help answer any questions you may have about our Group Multiple Benefit Insurance Scheme (GMBIS). Feel free to ask me anything, and I'll do my best to provide you with accurate and helpful information. How can I assist you today?"}]

def st_module_chat():
    if st.button("Reset chat window", use_container_width=True):
        initialize_messages()

### chatbot functions
def get_context(question: str):
    """Retrieves text-based information for an insurance product only based on the user query.
does not answer questions about the chat agent"""
    context = []
    with st.spinner("retrieving context"):
        retriever = st.session_state['retriever']
        for node in retriever.retrieve(question):
            context.append(node.text)
    messages =  st.session_state['messages'][-3:-1]+[{'role': 'user', 'content':f"""fullfill the query with the provided information
Do not include greetings or thanks for providing relevant information, answer the question first before explaining
Query      :{question}
Information:{context}"""}]
    return llm_client.chat(
            model=st.session_state['MODEL'],
            messages=messages,
            stream=True
        )

def classify_intent() -> str:
    """classify the intent of the user input
current implementation uses a lightweight model (llama3.2)
with chain of thought prompting examples
for classifying 'normal', 'register', 'rag', 'verify', 'abuse', 'end' intents
"""
    response = llm_client.chat(
            model = "intentClassifier",
            messages = st.session_state['messages'][-3:],
            stream = False,
            format=Intent.model_json_schema()
        )
    IntentModel: Intent
    try:
        IntentModel = Intent.model_validate_json(response.message.content)
        return IntentModel.intent
    except ResponseError as e: #fallback in case of potential invalid json schema
        return 'default'

def get_response(user_input: str, intent: Literal["normal","register","rag","verify","abuse","end"] = None):
    '''Get the response from the chatbot based on the user input and intent'''
    if intent is None:
        with st.spinner("detecting intent..."):
            intent = classify_intent()
    product_summary = get_config(field='PRODUCT_SUMMARY')
    match intent:
        case "rag":
            return (get_context(user_input), intent)
        case "register":
            with st.spinner("fetching registration instructions"):
                messages = [{"role":"system", "content":f"""The user is asking about information for registering to the insurance scheme.
                             Always include this registration link [Great Multiple Benefits Insurance Scheme Promotional page]({get_config('PROMOTIONAL_LINK')}/?utm_source=chatbot&utm_medium=cpc&utm_campaign=boost&utm_id=spif&utm_content=message_link)
                             And this admin contact number (03-48133818) in your responses to the user's query.
                             Add a note where they can leave their contact details and our agent will contact them soon."""},
                    {"role":"user","content":f"{user_input}"}]
            return (llm_client.chat(
                model = st.session_state['MODEL'],
                messages = messages,
                stream = True), intent)
        case "normal":
            with st.spinner("thinking about what to say"):
                messages = st.session_state['messages'][-3:-1]+[{"role":"user", "content":f"""Respond naturally and contextually to suitable work-related requests. For requests that are inappropriate or outside your design, politely decline to address them.
if the user asks about the insurance product, here is a summary
summary:
{product_summary}
Request: {user_input}"""}]
            return (llm_client.chat(
                model = st.session_state['MODEL'],
                messages = messages,
                stream = True), intent)
        case "verify":
            with st.spinner("fetching verification instructions"):
                messages = st.session_state['messages'][-3:-1]+[{"role":"user", "content":f"Please answer the given question with the following context:\
                    Question: {user_input}\
                    Context:\
                    user info are gathered from previous campaigns and stored in a secure database, we do not share your information with third parties.\
                    if the user asks about insurance product, here is a summary\
                    summary:\
                    {product_summary}"}]
            return (llm_client.chat(
                model = st.session_state['MODEL'],
                messages = messages,
                stream = True), intent)
        case "abuse":
            response = []
            for word in "The system has detected abusive language in the user input, please refrain from asking about things unrelated to the service.".split(' '):
                response.append(ChatResponse(message=Message(role='assistant',content=word+" ")))
            return (response, intent)
        case "end":
            response = []
            end_message: str
            try:
                end_message = get_config("TIMEOUT_MESSAGE_FINAL")
            except ConnectionFailure:
                end_message = "Thanks for chatting with me. Have a great day!"
            for word in end_message.split(' '):
                response.append(ChatResponse(message=Message(role='assistant',content=word+" ")))
            return (response, intent)
        case _:
            with st.spinner("fetching default response"):
                return (llm_client.chat(
                    model = st.session_state['MODEL'],
                    messages = [{"role":"user", 
                                "content":user_input}],
                    stream = True), intent)

### Streamlit app
st.header('Chatbot for the Great Multiple Benefits Insurance Scheme')

if 'messages' not in st.session_state:
    initialize_messages()

with st.sidebar:
    st.subheader("Chat")
    st.page_link(icon="📣",
                 label=":orange[link to product]",
                 page=f'{get_config('PROMOTIONAL_LINK')}/?utm_source=chatbot&utm_medium=cpc&utm_campaign=boost&utm_id=spif&utm_content=sidebar_link',
                 use_container_width=True)
    llm_client = setup_LLM('localhost ollama')
    setup_mongo()
    st_module_chat()    
    initialize_session_id()

for message in st.session_state['messages']:
    if message["role"] == "system":
        continue
    else:
        st.chat_message(message['role']).write(message['content'])

def chat(prompt: str):
    '''Generate message boxes to simulate a chat conversation'''
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        generator, intent = get_response(prompt)
        for chunk in generator:
            full_response += chunk.message.content
            response_placeholder.markdown(full_response + "▌")

    st.session_state.messages.append({"role": "assistant", "content": full_response, "detected_user_intent": intent})
    return intent

def message_helper(intent: str):
    'store the chat logs in the database'
    cookie_manager = get_cookies_manager()
    if intent == 'abuse':
        counter = cookie_manager.get("IS_ABUSIVE")
        if counter > 2:
            expiry_time = datetime.datetime.now() + datetime.timedelta(days=1)
        else:
            expiry_time = datetime.datetime.now() + datetime.timedelta(minutes=5)
        cookie_manager.set(cookie='IS_ABUSIVE',
                           val=counter+1,
                           expires_at=expiry_time)
    else:
        session_messages = st.session_state['messages'][1:]
        interactions = (len(session_messages)/2)
        current_timestamp = datetime.datetime.now()
        if st.session_state['session_id'] in st.session_state['id_list']:
            st.session_state['mongo_client'][st.session_state["MONGODB_DB"]]['chat_session'].update_one({'_id': oid(st.session_state['session_id'])},
                                                                                                {'$set':
                                                                                                    {'chatlog': session_messages,
                                                                                                    'interactions': interactions,
                                                                                                    'updated_on': current_timestamp,
                                                                                                    'is_abusive': (False if cookie_manager.get("IS_ABUSIVE")<ABUSE_CHANCE else True)
                                                                                                    }})
        else:
            st.session_state['mongo_client'][st.session_state["MONGODB_DB"]]['chat_session'].insert_one({'_id': oid(st.session_state['session_id']), 
                                                                                                'chatlog': session_messages,
                                                                                                'interactions': interactions,
                                                                                                'created_on': current_timestamp,
                                                                                                'updated_on': None,
                                                                                                'is_abusive': False
                                                                                                })  

async def a_timeout_message():
    '''Run a timer to check if the user is still active and display 2 timeout messages accordingly'''
    await asyncio.sleep(delay = TIMEOUT_MESSAGE_DELAY)  # first message delay
    message : str
    with st.chat_message("assistant"):
        try:
            message = get_config("TIMEOUT_MESSAGE_INITIAL")
        except ConnectionFailure:
            message = 'Anyone still there?'
        message
    # Wait for the timer to finish
    await asyncio.sleep(TIMEOUT_MESSAGE_DELAY*2)  # final message delay
    with st.chat_message("assistant"):
        message = 'I guess that\'s all for now.  '
        try:
            message += get_config("TIMEOUT_MESSAGE_FINAL")
        except ConnectionFailure:
            message += 'Thanks for using our services, have a nice day!'
        message

# Chat input and activity tracking
if prompt := st.chat_input("How can I help?", disabled=(True if st.session_state.get("IS_ABUSIVE", 0) > ABUSE_CHANCE else False)):
    latest_message_intent = chat(prompt)
    message_helper(latest_message_intent)
    if latest_message_intent != 'end':
        asyncio.run(a_timeout_message())

# Abuse warning
with st.sidebar:
    if get_cookies_manager().get("IS_ABUSIVE") > 0:
        st.warning(f'If abusive behavior continues {(ABUSE_CHANCE+1)-(get_cookies_manager().get("IS_ABUSIVE"))} more times, you will be blocked from using the chatbot')

#sample questions
questions = list(st.session_state['mongo_client'][st.session_state['MONGODB_DB']]['sample_questions'].find({},{'question':1}))
sidebar_expander = st.sidebar.expander('Sample questions')
for question in questions:
    if sidebar_expander.button(label=question['question'], use_container_width=True):
        chat(question['question'])