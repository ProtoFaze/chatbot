import datetime
import streamlit as st
from shared.Setup import setup_LLM, setup_mongo, initialize_streamlit_session
from shared.Models import AnalysisResults
initialize_streamlit_session()
st.set_page_config(page_title="Dashboard", page_icon=":material/dashboard:",layout="wide", menu_items={"about":"#Dashboard with a simple chatlog analysis function ,locked behind admin credentials"})
@st.dialog("Chatlog will not be analysed")
def show_analysis_dialog(chat_record):
    if len(chat_record['chatlog'])>2:
        st.write("This chatlog has already been analysed.")
    else:
        st.write("Analysis blocked, lack of messages would waste resources")
    if st.button("Close"):
        st.rerun()

def show_chat_details(item: dict):
    st.session_state['inspecting_chat_collection'] = st.session_state['mongo_client'][st.session_state["MONGODB_DB"]]['chat_session'].find_one({'_id': item['_id']})

def analyse_chatlog(chat_record):
    if 'llm_client' not in st.session_state:
        llm_client = setup_LLM()
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

Analyse the chatlog, tell me the overall user sentiment, overal user interest in the product the AI is advertising, key topics in the conversation, is the user not interested in signing up, and the disengagement reason.
please reply in json format.
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
    if 'mongo_client' not in st.session_state:
        setup_mongo()
    chat_ids = list(st.session_state['mongo_client'][st.session_state["MONGODB_DB"]]['chat_session'].find({}, {"_id": 1, "created_on": 1}))
    chat_ids.reverse()
    with st.sidebar:
        setup_LLM()
        '---'
        st.write("Chatlogs")
        for item in chat_ids:
            if st.button(label = "Chatlog - "+item['created_on'].strftime("%Y-%b-%d %H:%M:%S"),use_container_width=True): 
                show_chat_details(item)
        if 'inspecting_chat_collection' not in st.session_state:
            show_chat_details(chat_ids[0])

    st.header(st.session_state['inspecting_chat_collection']['created_on'].strftime("%Y-%B-%d %H:%M:%S"))
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
        display_results("Sign-up Disengagement Flag", getattr(analysis_results, 'is_user_not_interested_in_signing_up', None))
        display_results("Interactions", int(st.session_state['inspecting_chat_collection'].get('interactions', 0)))
    with right:
        st.write(":orange[Key Topics:] ")
        display_key_topics(getattr(analysis_results, 'key_topics', []))
        display_results("Sign-up Disengagement Reason", getattr(analysis_results, 'disengagement_reason', None))


else:
    def login():
        st.session_state['IS_ADMIN'] = password == st.session_state['ADMIN_PASSWORD']
        
    with st.sidebar:
        password = st.text_input(label='Enter admin password', type='password')
        st.button('login', on_click=login())
    st.write("Please enter the admin password to view the chatlogs")