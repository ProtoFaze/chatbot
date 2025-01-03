import streamlit as st

from shared.Setup import initialize_streamlit_session, login

### Streamlit app
initialize_streamlit_session()
Chat = st.Page("pages/Chat.py",title="Chat", icon="💬")

if st.session_state.get("IS_ADMIN", False):

    Dashboard = st.Page("pages/Dashboard.py",title="Dashboard", icon="📊")
    Setting = st.Page("pages/Setting.py",title="Settings", icon="⚙️")

    pg = st.navigation(pages=[Chat, Dashboard, Setting],position="sidebar",expanded=True)
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": "Chat with the RAG chatbot about the GMBIS insurance scheme\n### How it works  \nThe RAG chatbot works by embedding user inputs.  \nThe inputs are then used to query a mongodb index.  \nThe top closest matches are then used to fetch the relevant document sections,  \nWhich is finally used as context for the response generation."}
    )
    pg.run()
else:
    pg = st.navigation([Chat])
    st.set_page_config(
        page_title="Chat about GMBIS",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": "Chat with the RAG chatbot about the GMBIS insurance scheme\n### How it works  \nThe RAG chatbot works by embedding user inputs.  \nThe inputs are then used to query a mongodb index.  \nThe top closest matches are then used to fetch the relevant document sections,  \nWhich is finally used as context for the response generation."}
    )
    st.sidebar.subheader("Admin Login")
    password = st.sidebar.text_input("password", type="password")
    if st.sidebar.button("Login"):
        login(password)
    pg.run()