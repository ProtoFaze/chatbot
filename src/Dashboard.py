import datetime
import streamlit as st
from shared.Setup import setup_LLM, setup_mongo, initialize_streamlit_session, login, get_chat_ids
from shared.Models import AnalysisResults
import time
initialize_streamlit_session()

@st.dialog('confirm delete chatlog?')
def confirm_delete_dialog():
    st.write("Are you sure you want to delete this chatlog?  \nthis action is irreversable.")
    yes, no = st.columns(2)
    if yes.button("Yes, delete"):
        #send delete request to db, refresh chat ids and update the displayed chatlog
        st.session_state['mongo_client'][st.session_state["MONGODB_DB"]]['chat_session'].delete_one({'_id': st.session_state['inspecting_chat_collection']['_id']})
        get_chat_ids()
        show_chat_details()
        st.rerun()
    if no.button("No, cancel"):
        st.rerun()

@st.dialog("Chatlog will not be analysed")
def show_analysis_dialog(chat_record):
    '''Show alert for blocking analysis'''
    if len(chat_record['chatlog'])>2:
        st.write("This chatlog has already been analysed.")
    else:
        st.write("Analysis blocked, lack of messages would waste resources")
    if st.button("Close"):
        st.rerun()

def show_chat_details(item: dict = None):
    '''Set the chat collection to inspect'''
    if item is None:
        item = get_chat_ids(fetch_only_ids=False)[0]
    st.session_state['inspecting_chat_collection'] = st.session_state['mongo_client'][st.session_state["MONGODB_DB"]]['chat_session'].find_one({'_id': item['_id']})

def analyse_chatlog(chat_record):
    '''Analyse the chatlog based on a set of metrics'''
    if 'llm_client' not in st.session_state:
        llm_client = setup_LLM('external ollama')
    else:
        llm_client = st.session_state['llm_client']
    chatlog = ''
    with st.spinner("Preprocessing chatlog"):
        for seq,message in enumerate(chat_record['chatlog']):
            if message['role'] == 'user':
                chatlog += str(seq)+". { USER_MSG:"+message['content']+"},\n"
            else:
                chatlog +=  str(seq)+". { AI_MSG:"+message['content']+"},\n"
        print(chatlog)
    with st.spinner("Analyzing chatlog"):
        prompt = f"""{chatlog}

Analyze the provided chat log and return the following details in JSON format:
1. Overall User Sentiment: Analyze the user's tone throughout the conversation. Options: positive, neutral, or negative.
2. Overall User Interest Level: Indicate the user's level of interest in the advertised product. Options: low, low-medium, medium, medium-high, high.
3. Key Topics Discussed: Identify key topics or questions raised by the user or addressed by the AI.
4. Sign-up disinterest: Determine if the user is not likely interested in signing up. Indicate "False" if: the user explicitly asks to sign up, OR shares personal details (e.g., name, age, contact information), OR exchanges more than 4 messages and demonstrates interest in the product. If not interested, provide reasons such as negative sentiment, irrelevant inquiries, or unmet eligibility criteria.
5. Reasons for Lack of Interest (if applicable): Provide detailed reasons if the user is not interested in signing up.
"""

        response = llm_client.generate(model = 'llama3.2',
                                       prompt = prompt,
                                       format = AnalysisResults.model_json_schema(),
                                       options={'temperature':0},
                                       stream=False
                                       )
        results = AnalysisResults.model_validate_json(response.response)
    return results

def display_results(label, value):
    st.write(f":orange[{label}] ", value)

def display_key_topics(topics):
    for topic in topics:
        st.markdown("- " + topic)



if 'IS_ADMIN' not in st.session_state:
    st.session_state['IS_ADMIN'] = False
    
if st.session_state['IS_ADMIN']:
    if 'llm_client' not in st.session_state:
        llm_client = setup_LLM('external ollama')
    else:
        llm_client = st.session_state['llm_client']
    if 'mongo_client' not in st.session_state:
        setup_mongo()
    chat_ids = get_chat_ids(fetch_only_ids=False)
    with st.sidebar:
        st.subheader('Admin Dashboard')
        setup_LLM('external ollama')
        "Chatlogs"
        for item in chat_ids:
            if st.button(label = "Chatlog - "+item['created_on'].strftime("%Y-%b-%d %H:%M:%S"),use_container_width=True): 
                show_chat_details(item)
        if 'inspecting_chat_collection' not in st.session_state:
            show_chat_details(chat_ids[0])

    header_column, action_column = st.columns(spec=[0.9, 0.1], vertical_alignment='bottom')
    header_column.header(st.session_state['inspecting_chat_collection']['created_on'].strftime("%Y-%B-%d %H:%M:%S"))

    if action_column.button("Delete", use_container_width=True):
        confirm_delete_dialog()
    with st.expander(label = "messages"):
        specific_chatlog = st.session_state['inspecting_chat_collection']['chatlog']
        for message in specific_chatlog:
            st.chat_message(name=message['role']).markdown(message['content'])
            if 'detected_user_intent' in message:
                st.write("Intent: ", message['detected_user_intent'])

    st.header("Analysis")
    analysis_results = st.session_state['inspecting_chat_collection'].get('analysis_results', None)
    if analysis_results is not None:
        analysis_results = AnalysisResults.model_validate(analysis_results)
    if st.button(label = "Analyse Chatlog"):
        if analysis_results is None and len(st.session_state['inspecting_chat_collection']['chatlog'])>2:
            analysis_results = analyse_chatlog(st.session_state['inspecting_chat_collection'])
            st.session_state['inspecting_chat_collection']['analysis_results'] = analysis_results
            st.session_state['mongo_client'][st.session_state["MONGODB_DB"]]['chat_session'].update_one({'_id': st.session_state['inspecting_chat_collection']['_id']},
                                                                                            {'$set':
                                                                                                {'updated_on': datetime.datetime.now(),
                                                                                                'analysis_results': analysis_results.model_dump(mode='json')
                                                                                            }})
        else:
            show_analysis_dialog(st.session_state['inspecting_chat_collection'])

    left, right = st.columns(2)
    with left:
        display_results("Overall Sentiment", getattr(analysis_results, 'overall_user_sentiment', None))
        display_results("Interest in product", getattr(analysis_results, 'interest_in_product', None))
        display_results("Sign-up Disengagement Flag", getattr(analysis_results, 'is_user_not_interested_in_signing_up', False))
        display_results("Interactions", int(st.session_state['inspecting_chat_collection'].get('interactions', 0)))
    with right:
        st.write(":orange[Key Topics:] ")
        display_key_topics(getattr(analysis_results, 'key_topics', []))
        display_results("Sign-up Disengagement Reason", getattr(analysis_results, 'disengagement_reason', None))


else:   
    with st.sidebar:
        password = st.text_input(label='Enter admin password', type='password')
        st.button('login', on_click=login(password))
    st.write("Please enter the admin password to view the chatlogs")