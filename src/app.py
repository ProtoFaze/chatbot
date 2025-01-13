import streamlit as st

from shared.Setup import initialize_streamlit_session, login, get_cookies_manager, ABUSE_CHANCE
from streamlit.navigation.page import StreamlitPage
import datetime

### Streamlit app
initialize_streamlit_session()

### setup global config
st.set_page_config(layout="wide",
                   initial_sidebar_state="expanded",
                   #if you want to format display text, use markdown
                   menu_items={"About": "Chat with the RAG chatbot about the GMBIS insurance scheme\n### How it works  \nThe RAG chatbot works by embedding user inputs.  \nThe inputs are then used to query a mongodb index.  \nThe top closest matches are then used to fetch the relevant document sections,  \nWhich is finally used as context for the response generation."}
                )

### empty page initialization
pg: StreamlitPage

### Setup abuse counter
abuse_counter_coookie: int
cookie_manager = get_cookies_manager()
if cookie_manager.get("IS_ABUSIVE") is None:
    cookie_manager.set(cookie = "IS_ABUSIVE",
                       val = 0, 
                       expires_at = datetime.datetime.now() + datetime.timedelta(minutes=5))
abuse_counter_coookie = cookie_manager.get("IS_ABUSIVE")

### Page routing
if st.session_state.get("IS_ADMIN", False):
    Chat = st.Page("Chat.py",title="Chat", icon="💬")
    Dashboard = st.Page("Dashboard.py",title="Dashboard", icon="📊")
    Setting = st.Page("Setting.py",title="Settings", icon="⚙️")

    pg = st.navigation(pages=[Chat, Dashboard, Setting],position="sidebar",expanded=True)

elif abuse_counter_coookie > ABUSE_CHANCE:
    Abuse = st.Page(page='./Abuse.py', title='Abuse', icon='🚫')
    pg = st.navigation([Abuse])

else:
    Chat = st.Page("Chat.py",title="Chat about GMBIS", icon="💬")
    pg = st.navigation([Chat])
    st.sidebar.subheader("Admin Login")
    password = st.sidebar.text_input("password", type="password")
    if st.sidebar.button("Login"):
        login(password)

### Run the app
pg.run()