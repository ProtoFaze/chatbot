import streamlit as st
from shared.Setup import initialize_streamlit_session, login
st.set_page_config(page_title="Settings", page_icon=":gear:",layout="wide")
initialize_streamlit_session()

if not st.session_state.get("IS_ADMIN", False):
    st.sidebar.subheader("Admin Login")
    password = st.sidebar.text_input("password", type="password")
    st.sidebar.button("Login", on_click=login(password))

def change_variable(environment_variable: str, new_value: str):
    """Update session state with new endpoint or notify if no changes."""
    default_value = st.secrets.get(environment_variable, "default_value")
    if new_value == "" or new_value == default_value:
        st.warning(f"No changes made to {environment_variable.lower().replace('_',' ')}.")
    else:
        st.session_state[environment_variable] = new_value
        st.success(f"{environment_variable.lower().replace('_',' ')} updated to: {new_value}")

def use_default(environment_variable: str):
    """Reset session state to the default secret value."""
    default_value = st.secrets.get(environment_variable, "default_value")
    st.session_state[environment_variable] = default_value
    st.success(f"{environment_variable.lower().replace('_',' ')} reverted to default")

def set_display_value(key_value: str, default_value: str) -> str:
    """Set display value for text input based on session state or default secret."""
    return "default" if st.session_state[key_value] == st.secrets.get(key_value, default_value) else st.session_state[key_value]

# Display current values or 'default' as placeholder in textboxes
display_values = {
    "EXTERNAL_OLLAMA_API_URL": set_display_value("EXTERNAL_OLLAMA_API_URL", "default_value"),
    "EXTERNAL_OLLAMA_API_KEY": set_display_value("EXTERNAL_OLLAMA_API_KEY", "default_value"),
    "GOOGLE_APPLICATION_CREDENTIALS": set_display_value("GOOGLE_APPLICATION_CREDENTIALS", "default_value"),
    "ADMIN_PASSWORD": set_display_value("ADMIN_PASSWORD", "default_value")
}

@st.dialog('unauthorized')
def unauthorized_dialog():
    st.write("If you wish to change admin settings, please specify your own settings for other default settings first.")
    if st.button("Ok"):
        st.rerun()

@st.dialog('Total reset warning')
def total_reset_warning():
    st.write("This will reset all settings to their default values [llm endpoints and database configs]. Are you sure you want to proceed?")
    if st.button("Yes, reset all settings"):
        for key in display_values.keys():
            st.session_state[key] = st.secrets.get(key, "default_value")
        st.rerun()

def setup_text_field(key: str, label: str, type:str = 'default') -> str:
    """Create a text input with a default value as well as its, submit, reset to default buttons and message box."""
    new_field = st.text_input(label='Specify your '+label,
                              type=type,
                              value=('' if display_values[key] == 'default' else display_values[key]),
                              placeholder=(display_values[key] if 'admin' not in key.lower() else 'your password'))

    left, right = st.columns(2)  # Add buttons in a row
    if left.button("Change "+label, use_container_width=True):
        if 'admin' not in key.lower():
            change_variable(key, new_field)
        else:
            is_all_changed = True
            for identifier in display_values:
                if display_values[identifier] == "" and 'admin' not in identifier.lower():
                    is_all_changed = False
                    break
            if is_all_changed:
                change_variable(key, new_field)
            else:
                unauthorized_dialog()
    if right.button("Use default "+label, use_container_width=True):
        if 'admin' not in key.lower():
            use_default(key)
        else:
            is_all_default = True
            for identifier in display_values:
                if display_values[identifier] != "":
                    is_all_default = False
                    break
            if is_all_default:
                use_default(key)
            else:
                total_reset_warning()

st.subheader("External Ollama API")
setup_text_field("EXTERNAL_OLLAMA_API_URL", "external Ollama endpoint")
setup_text_field("EXTERNAL_OLLAMA_API_KEY", "optional API key")
setup_text_field("GOOGLE_APPLICATION_CREDENTIALS", "optional Google App Credentials")

st.subheader("admin password")
setup_text_field("ADMIN_PASSWORD", "admin password", "password")
