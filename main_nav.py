import streamlit as st

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Page definitions
analysis_page = st.Page("app.py", title="Analysis")
food_log_page = st.Page("food_logging.py", title="Food Logging")
coaching_page = st.Page("coaching.py", title="Coaching")
login_page = st.Page("login.py", title="Login")

# Conditional navigation based on login status
if st.session_state.logged_in:
    # Show main app pages only when logged in
    pg = st.navigation([analysis_page, food_log_page, coaching_page])
else:
    # Show only login page when not logged in
    pg = st.navigation([login_page])

st.set_page_config(page_title="Patient Portal")
pg.run()