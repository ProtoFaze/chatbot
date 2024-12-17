import os
import datetime
import streamlit as st
from bson import objectid as oid
from shared.Connection import setup_LLM, setup_mongo
from shared.Analysis import AnalysisResults
from dotenv import load_dotenv
load_dotenv()

@st.dialog("Chatlog Already Analysed")
def show_analysis_dialog():
    st.write("This chatlog has already been analysed.")
    if st.button("Close"):
        st.rerun()

def show_chat_details(item: dict):
    st.session_state['inspecting_chat_collection'] = dict(item)

def analyse_chatlog(chat_collection):
    if 'llm_client' not in st.session_state:
        llm_client = setup_LLM()
    else:
        llm_client = st.session_state['llm_client']
    with st.spinner("Analyzing chatlog"):
        chatlog = chat_collection['chatlog']

        prompt = f'Analyse the chatlog: {chatlog}'
        response = llm_client.generate(model = 'llama3.2',
                                       prompt = prompt,
                                       format = AnalysisResults.model_json_schema(),
                                       options={'temperature':0},
                                       stream=False)
        results = AnalysisResults.model_validate_json(response.response)
    return results

if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False
    
if st.session_state['is_admin']:
    if 'mongo_client' not in st.session_state:
        setup_mongo()
    chat_collection = list(st.session_state['mongo_client'][os.environ["MONGODB_DB"]]['chat_session'].find())
    
    with st.sidebar:
        with st.expander('Chatlogs'):
            for item in chat_collection:
                if st.button(label = "Chatlog - "+str(item['_id'])) : 
                    show_chat_details(item)
            if 'inspecting_chat_collection' not in st.session_state:
                show_chat_details(chat_collection[0])


    st.header(st.session_state['inspecting_chat_collection']['_id'])
    with st.expander(label = "messages"):
        specific_chatlog = st.session_state['inspecting_chat_collection']['chatlog']
        for message in specific_chatlog:
            st.chat_message(name=message['role']).markdown(message['content'])
            if 'intent' in message:
                st.write("Intent: ", message['intent'])

    st.header("Analysis")
    analysis_results = st.session_state['inspecting_chat_collection'].get('analysis_results', None)
    if analysis_results is not None:
        analysis_results = AnalysisResults.model_validate(analysis_results)
    # st.write(st.session_state['inspecting_chat_collection']['_id'])
    if st.button(label = "Analyse Chatlog"):
        if analysis_results is None:
            analysis_results = analyse_chatlog(st.session_state['inspecting_chat_collection'])
            
            st.session_state['inspecting_chat_collection']['analysis_results'] = analysis_results
            st.session_state['mongo_client'][os.environ["MONGODB_DB"]]['chat_session'].update_one({'_id': st.session_state['inspecting_chat_collection']['_id']},
                                                                                            {'$set':
                                                                                                {'updated_on': datetime.datetime.now(),
                                                                                                'is_analysed': True,
                                                                                                'analysis_results': analysis_results.model_dump(mode='json')
                                                                                            }})
        else:
            show_analysis_dialog()

    col1, col2 = st.columns(2)
    if analysis_results is not None:
        with col1:
            st.write(":orange[Overall Sentiment:] ", analysis_results.overall_sentiment)
            st.write(":orange[Interest in product:] ", analysis_results.interest_in_product)
            st.write(":orange[Intent Classification Accuracy:] ", analysis_results.intent_classification_accuracy)
            st.write(":orange[Conversation Turns:] ", analysis_results.conversation_turns)
            st.write(":orange[Sign-up Disengagement Flag:] ", analysis_results.signup_disengagement_flag)
        with col2:
            st.write(":orange[Key Topics: ]")
            for topic in analysis_results.key_topics:
                st.markdown("- "+topic)
            st.write(":orange[Sign-up Disengagement Reason:] ", analysis_results.signup_disengagement_reason)
    else:
        with col1:
            st.write(":orange[Overall Sentiment:] ", None)
            st.write(":orange[Interest in product:] ", None)
            st.write(":orange[Intent Classification Accuracy:] ", None)
            st.write(":orange[Conversation Turns:] ", None)
            st.write(":orange[Disengagement Flag:] ", None)
        with col2:
            st.write(":orange[Key Topics:] ", None)
            st.write(":orange[Disengagement Reason:] ", None)

else:
    def login():
        st.session_state['is_admin'] = password==os.environ['ADMIN_PASSWORD']
        
    with st.sidebar:
        password = st.text_input(label='Enter admin password', type='password')
        st.button('login', on_click=login())
    st.write("Please enter the admin password to view the chatlogs")