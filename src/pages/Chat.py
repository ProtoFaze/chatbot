import streamlit as st
from bson.objectid import ObjectId as oid 
import datetime
from shared.Models import Intent
from shared.Setup import setup_LLM, setup_mongo, fetch_chat_ids
from typing import Literal

def initialize_messages():
    st.session_state['messages'] = [{"role": "assistant", "content": "Hello! I'm C3, your friendly customer service chatbot at Great Eastern Life Assurance Malaysia. I'm here to help answer any questions you may have about our Group Multiple Benefit Insurance Scheme (GMBIS). Feel free to ask me anything, and I'll do my best to provide you with accurate and helpful information. How can I assist you today?"}]
    
# setup connections
def get_context(question: str):
    """Retrieves text-based information for an insurance product only based on the user query.
does not answer questions about the chat agent"""
    context = []
    with st.spinner("retrieving context"):
        retriever = st.session_state['retriever']
        for node in retriever.retrieve(question):
            context.append(node.text)
    messages =  st.session_state['messages'][-3:-1]+[{'role': 'user', 'content':f"""fullfill the query with the provided information
Do not include greetings or thanks for providing relevant information
Query      :{question}
Information:{context}"""}]
    return llm_client.chat(
            model=st.session_state['MODEL'],
            messages=messages,
            stream=True
        )

#setup a simple intent classifier
def classify_intent(user_input:str) -> str:
    """classify the intent of the user input
current implementation uses a lightweight model (llama3.2)
with chain of thought prompting examples
for classifying 'normal', 'register', 'rag' intents
"""
    response = llm_client.chat(
            model = "intentClassifier",
            messages = [{"role":"user", "content":user_input}],
            stream = False,
            format=Intent.model_json_schema()
        )
    response: Intent
    try:
        response = Intent.model_validate_json(response.message.content)
        return response.intent
    except Exception as e: #fallback in case of potential invalid json schema
        return 'default'

def get_response(user_input: str, intent: Literal["normal","register","rag","verify"] = None):
    if intent is None:
        with st.spinner("detecting intent..."):
            intent = classify_intent(user_input)
    match intent:
        case "rag":
            return (get_context(user_input), intent)
        case "register":
            with st.spinner("fetching registration instructions"):
                messages = [{"role":"system", "content":f"""The user is asking about information for registering to the insurnace scheme.
                             Always include this registration link [Great Multiple Benefits Insurance Scheme Promotional page](https://greatmultiprotect.com/gss315-spif/?utm_source=chatbot&utm_medium=cpc&utm_campaign=boost&utm_id=spif&utm_content=message_link)
                             And this admin contact number (03-48133818) in your responses to the user's query."""},
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
Group Multiple Benefit Insurance Scheme (GMBIS)
Offered by Great Eastern Life Assurance (Malaysia) Berhad in partnership with Axiata Digital Ecode Sdn Bhd, this scheme provides financial coverage for employees, their legal spouses, and children. Key features include:
Coverage
    Critical Illnesses: Covers 45 critical illnesses with benefits ranging from RM20,000 to RM30,000.
    Death and Accidental Death: Benefits range from RM20,000 to RM60,000 depending on premium.
    Total and Permanent Disability (TPD): Coverage for illness and accidents, ranging from RM20,000 to RM60,000.
    Hospitalisation: Daily income benefits up to RM30 per day for 500 days.
    Funeral Expenses: A lump sum of RM5,000.

Key Advantages
    Fixed premiums for all ages and genders.
    Low-cost premiums (starting at RM30/month).
    Investment-linked with redemption options after 12 months.
    Participation eligibility up to age 65.
    Includes Shariah-approved securities (but not Shariah-compliant).

Eligibility
    Employees/members aged 19-60 years.
    Spouses and children (unmarried, unemployed students up to age 23).

Exclusions
    Pre-existing conditions, suicide within the first year, and conditions resulting from activities like racing, scuba diving, or assault are excluded.

Additional Benefits
    Critical Illness/TPD Claims: Partial payout (50%) for claims within six months; full payout thereafter.
    Retirement Fund: Redeemable at age 65 based on fund value.
    Surrender Value: Redeemable before age 65, subject to minimum participation.

Fees and Charges
    Policy fee: RM5/month.
    Fund management charge: 0.5% per annum.

Important Notes:
    The scheme is not Shariah-compliant.
    Early termination may result in losses due to market volatility and charges.

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
                    user info are gathered from previous campaigns and stored in a secure database, we do not share your information with third parties."}]
            return (llm_client.chat(
                model = st.session_state['MODEL'],
                messages = messages,
                stream = True), intent)
        case _:
            with st.spinner("fetching default response"):
                return (llm_client.chat(
                    model = st.session_state['MODEL'],
                    messages = [{"role":"user", 
                                "content":user_input}],
                    stream = True), intent)

def initialize_chat_session():    
    if 'messages' not in st.session_state:
        initialize_messages()

st.header('Chatbot for the Great Multiple Benefits Insurance Scheme')

def st_module_chat():
    if st.button("Reset chat window", use_container_width=True):
        initialize_messages()

### Streamlit app
initialize_chat_session()

with st.sidebar:
    st.subheader("Chat")
    st.page_link(icon="📣",
                 label=":orange[link to product]",
                 page='https://greatmultiprotect.com/gss315-spif/?utm_source=chatbot&utm_medium=cpc&utm_campaign=boost&utm_id=spif&utm_content=sidebar_link',
                 use_container_width=True)
    llm_client = setup_LLM('external ollama')
    st_module_chat()    
    setup_mongo()
    fetch_chat_ids()



for message in st.session_state['messages']:
    if message["role"] == "system":
        continue
    else:
        st.chat_message(message['role']).write(message['content'])

def chat(prompt: str):
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
    interactions = (len(st.session_state['messages'])/2)
    if st.session_state['session_id'] in st.session_state['id_list']:
        st.session_state['mongo_client'][st.session_state["MONGODB_DB"]]['chat_session'].update_one({'_id': oid(st.session_state['session_id'])},
                                                                                              {'$set':
                                                                                                {'chatlog': st.session_state['messages'],
                                                                                                 'interactions': interactions,
                                                                                                 'updated_on': datetime.datetime.now(),
                                                                                                }})
    else:
        st.session_state['mongo_client'][st.session_state["MONGODB_DB"]]['chat_session'].insert_one({'_id': oid(st.session_state['session_id']), 
                                                                                               'chatlog': st.session_state['messages'],
                                                                                               'interactions': interactions,
                                                                                               'created_on': datetime.datetime.now(),
                                                                                               'updated_on': None,
                                                                                               })

if prompt := st.chat_input("How can I help?"):
    chat(prompt)

# with st.sidebar:
#     if 'last_warmup' not in st.session_state or (datetime.datetime.now() - st.session_state['last_warmup']).seconds > 180:
#         st.session_state['last_warmup'] = datetime.datetime.now()
#         warmup_LLM()

questions = list(st.session_state['mongo_client'][st.session_state['MONGODB_DB']]['sample_questions'].find({},{'question':1}))
sidebar_expander = st.sidebar.expander('Sample questions')
for question in questions:
    if sidebar_expander.button(label=question['question'], use_container_width=True):
        chat(question['question'])
