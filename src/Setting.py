import streamlit as st
from shared.Setup import initialize_streamlit_session, get_config
initialize_streamlit_session()

def change_variable(key: str, new_value: str, is_referencing_secret: bool = True):
    """Update session state with new endpoint or notify if no changes."""
    default_value:str
    st.write('hi')
    if is_referencing_secret:
        default_value = st.secrets.get(key, "default_value")
    else:
        db_value = get_config(field=key)
        default_value = st.session_state.get(key, db_value)
    if new_value == "" or new_value == default_value:
        st.warning(f"No changes made to {key.lower().replace('_',' ')}.")
    else:
        st.session_state[key] = new_value
        st.success(f"{key.lower().replace('_',' ')} updated to: {new_value}")

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
    "PROMOTIONAL_LINK": get_config(field='PROMOTIONAL_LINK'),
    "PRODUCT_SUMMARY": get_config(field='PRODUCT_SUMMARY'),
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

def setup_text_field(key: str, label: str, text_input_type:str = 'default', use_text_area:bool = False, help:str = None, is_referencing_secret = True) -> str:
    """Create a text input with a default value as well as its, submit, reset to default buttons and message box."""
    new_field: str
    if use_text_area:
        new_field = st.text_area(label='Specify your '+label,         
                            value=('' if display_values[key] == 'default' else display_values[key]),
                            placeholder=(display_values[key] if 'admin' not in key.lower() else f'your {label}'),
                            help=help)
    else:
        new_field = st.text_input(label='Specify your '+label,
                            type=text_input_type,
                            value=('' if display_values[key] == 'default' else display_values[key]),
                            placeholder=(display_values[key] if 'admin' not in key.lower() else f'your {label}'),
                            help=help)

    left, right = st.columns(2)  # Add buttons in a row
    if left.button("Change "+label, use_container_width=True):
        if 'admin' not in key.lower():
            change_variable(key, new_field, is_referencing_secret)
        else:
            is_all_changed = True
            for identifier in display_values:
                if display_values[identifier] == "" and 'admin' not in identifier.lower():
                    is_all_changed = False
                    break
            if is_all_changed:
                change_variable(key, new_field, is_referencing_secret)
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

st.subheader("Product specific instructions")
setup_text_field(key="PROMOTIONAL_LINK", label="promotional link",
                 help='The link to the promotional page for the product, please omit the last "/" in the link',
                 is_referencing_secret=False)
setup_text_field(key="PRODUCT_SUMMARY", label="product summary", use_text_area=True,
                 help='''The summary of the product to be used in the chatbot\'s instructions,   
                 to save cost, please keep list items in 1 line and refrain from adding new paragraphs''',
                 is_referencing_secret=False)

st.subheader("External Ollama API")
setup_text_field(key="EXTERNAL_OLLAMA_API_URL", label="external Ollama endpoint")
setup_text_field(key="EXTERNAL_OLLAMA_API_KEY", label="optional API key")
setup_text_field(key="GOOGLE_APPLICATION_CREDENTIALS", label="optional Google App Credentials")

st.subheader("Admin credentials")
setup_text_field(key="ADMIN_PASSWORD", label="admin password", text_input_type="password")

with st.sidebar:
    st.subheader("Admin Settings")
    st.session_state['IS_ADMIN']